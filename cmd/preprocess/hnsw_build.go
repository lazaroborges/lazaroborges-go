package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"sync"
	"time"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/index"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/index/hnsw"
)

const (
	hnswM              uint8 = 6
	hnswEfConstruction       = 200
	hnswEdgeSlotBytes        = 14 // for M=6: 1+1+6*2
)

// writeIndexV2 takes the float32 vectors+labels+centroid assignments and
// writes the v2 format. It:
//
//  1. Computes int8 centroids + per-member int8 residuals padded to 16.
//  2. Builds per-cluster HNSW graphs (parallel).
//  3. Serializes to `dstPath`.
func writeIndexV2(
	dstPath string,
	vecs []float32, // N*14 row-major
	labels []uint8,
	assign []uint32, // cluster id per vector
	centroids []float32, // K*14 row-major
	seed int64,
) error {
	N := len(vecs) / 14
	K := len(centroids) / 14
	dim := 14

	log.Printf("v2: %d vectors, %d clusters", N, K)

	// 1. Cluster offsets + member id ordering (group by cluster, ascending id).
	clusterMembers := make([][]uint32, K) // cluster -> global vector ids
	for vid := 0; vid < N; vid++ {
		c := assign[vid]
		clusterMembers[c] = append(clusterMembers[c], uint32(vid))
	}
	offsets := make([]uint32, K+1)
	var run uint32
	for c := 0; c < K; c++ {
		offsets[c] = run
		run += uint32(len(clusterMembers[c]))
	}
	offsets[K] = run
	log.Printf("largest cluster: %d", largestCluster(clusterMembers))

	// 2. int8 centroids padded to 16.
	centInt8 := make([]int8, K*16)
	for c := 0; c < K; c++ {
		for d := 0; d < dim; d++ {
			v := centroids[c*dim+d] * 127
			if v > 127 {
				v = 127
			} else if v < -128 {
				v = -128
			}
			centInt8[c*16+d] = int8(math.Round(float64(v)))
		}
	}

	// 3. int8 member residuals padded to 16, in cluster-grouped order.
	residuals := make([]int8, N*16)
	memberLabels := make([]uint8, N)
	for c := 0; c < K; c++ {
		members := clusterMembers[c]
		for i, vid := range members {
			localGlobalIdx := int(offsets[c]) + i
			memberLabels[localGlobalIdx] = labels[vid]
			for d := 0; d < dim; d++ {
				vf := vecs[int(vid)*dim+d] - centroids[c*dim+d]
				q := vf * 127
				if q > 127 {
					q = 127
				} else if q < -128 {
					q = -128
				}
				residuals[localGlobalIdx*16+d] = int8(math.Round(float64(q)))
			}
		}
	}

	// 4. Build per-cluster HNSW graphs in parallel.
	graphs := make([]*hnsw.Graph, K)
	var wg sync.WaitGroup
	sem := make(chan struct{}, 8) // limit parallelism
	t0 := time.Now()
	for c := 0; c < K; c++ {
		c := c
		members := clusterMembers[c]
		size := uint16(len(members))
		if size == 0 {
			graphs[c] = &hnsw.Graph{N: 0, M: hnswM}
			continue
		}
		base := int(offsets[c]) * 16
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer func() { <-sem; wg.Done() }()
			df := func(a, b uint16) int32 {
				var s int32
				ai, bi := base+int(a)*16, base+int(b)*16
				for d := 0; d < 14; d++ {
					dd := int32(residuals[ai+d]) - int32(residuals[bi+d])
					s += dd * dd
				}
				return s
			}
			graphs[c] = hnsw.Build(size, hnswM, hnswEfConstruction, seed+int64(c), df)
		}()
	}
	wg.Wait()
	log.Printf("v2: built %d HNSW graphs in %s", K, time.Since(t0))

	// 5. Layout HnswEdges block. For each cluster, per-level: NodeIds (L>=1)
	//    then edge slots. Compute total byte size first.
	headers := make([]index.HnswHeader, K)
	edgeBlock := []byte{} // append progressively
	var curEdgeOffset uint32
	for c := 0; c < K; c++ {
		g := graphs[c]
		h := &headers[c]
		h.EntryPoint = g.Entry
		h.MaxLevel = g.MaxLevel
		h.EdgeOffset = curEdgeOffset
		for L := uint8(0); L <= uint8(index.HnswMaxLevel); L++ {
			if int(L) <= int(g.MaxLevel) {
				h.LevelCount[L] = uint32(len(g.Degree[L]))
			}
		}
		// Serialize this cluster's edges into edgeBlock.
		for L := uint8(0); L <= g.MaxLevel; L++ {
			// NodeIds for L>=1.
			if L >= 1 {
				for _, id := range g.NodeIds[L] {
					var b [2]byte
					binary.LittleEndian.PutUint16(b[:], id)
					edgeBlock = append(edgeBlock, b[:]...)
				}
			}
			// Edge slots.
			countAtL := len(g.Degree[L])
			for i := 0; i < countAtL; i++ {
				slot := make([]byte, hnswEdgeSlotBytes)
				slot[0] = g.Degree[L][i]
				slot[1] = 0
				baseEdge := i * int(g.M)
				for j := 0; j < int(g.M); j++ {
					binary.LittleEndian.PutUint16(slot[2+j*2:], g.Edges[L][baseEdge+j])
				}
				edgeBlock = append(edgeBlock, slot...)
			}
		}
		curEdgeOffset = uint32(len(edgeBlock))
	}
	log.Printf("v2: edge block %d bytes", len(edgeBlock))

	// 6. Write the file.
	f, err := os.Create(dstPath)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer f.Close()
	w := newBufWriter(f)

	// Header (48 bytes).
	hdr := make([]byte, index.HeaderBytesV2)
	binary.LittleEndian.PutUint32(hdr[0:], index.MagicV2)
	binary.LittleEndian.PutUint32(hdr[4:], index.VersionV2)
	binary.LittleEndian.PutUint32(hdr[8:], uint32(N))
	binary.LittleEndian.PutUint32(hdr[12:], uint32(K))
	binary.LittleEndian.PutUint32(hdr[16:], uint32(dim))
	binary.LittleEndian.PutUint32(hdr[20:], uint32(hnswM))
	// maxLevel across all clusters.
	var globalMax uint32
	for c := 0; c < K; c++ {
		if uint32(graphs[c].MaxLevel) > globalMax {
			globalMax = uint32(graphs[c].MaxLevel)
		}
	}
	binary.LittleEndian.PutUint32(hdr[24:], globalMax)
	w.Write(hdr)

	// Centroids float32.
	writeFloats(w, centroids)
	// CentroidsPadded float32 (pad to 16 lanes).
	padded := make([]float32, K*16)
	for c := 0; c < K; c++ {
		copy(padded[c*16:c*16+14], centroids[c*14:(c+1)*14])
	}
	writeFloats(w, padded)
	// CentroidsInt8.
	writeBytes(w, int8Slice(centInt8))
	// CentroidNorms float32.
	norms := make([]float32, K)
	for c := 0; c < K; c++ {
		var s float32
		for d := 0; d < 14; d++ {
			v := centroids[c*14+d]
			s += v * v
		}
		norms[c] = s
	}
	writeFloats(w, norms)
	// ClusterOffsets uint32.
	writeUint32s(w, offsets)
	// MemberResiduals.
	writeBytes(w, int8Slice(residuals))
	// Labels.
	writeBytes(w, memberLabels)
	// HnswPerClusterHeaders.
	for c := 0; c < K; c++ {
		writeHeader(w, &headers[c])
	}
	// HnswEdges.
	writeBytes(w, edgeBlock)

	return w.Flush()
}

type bufWriter struct {
	w   io.Writer
	buf []byte
}

func newBufWriter(w io.Writer) *bufWriter { return &bufWriter{w: w} }
func (b *bufWriter) Write(p []byte)       { b.buf = append(b.buf, p...) }
func (b *bufWriter) Flush() error         { _, err := b.w.Write(b.buf); return err }

func writeFloats(w *bufWriter, xs []float32) {
	buf := make([]byte, len(xs)*4)
	for i, x := range xs {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(x))
	}
	w.Write(buf)
}

func writeUint32s(w *bufWriter, xs []uint32) {
	buf := make([]byte, len(xs)*4)
	for i, x := range xs {
		binary.LittleEndian.PutUint32(buf[i*4:], x)
	}
	w.Write(buf)
}

func writeBytes(w *bufWriter, xs []byte) { w.Write(xs) }

func int8Slice(xs []int8) []byte {
	out := make([]byte, len(xs))
	for i, v := range xs {
		out[i] = byte(v)
	}
	return out
}

func writeHeader(w *bufWriter, h *index.HnswHeader) {
	buf := make([]byte, index.HeaderSizeV2)
	binary.LittleEndian.PutUint16(buf[0:], h.EntryPoint)
	buf[2] = h.MaxLevel
	buf[3] = 0
	binary.LittleEndian.PutUint32(buf[4:], h.EdgeOffset)
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(buf[8+i*4:], h.LevelCount[i])
	}
	w.Write(buf)
}

func largestCluster(cm [][]uint32) int {
	m := 0
	for _, c := range cm {
		if len(c) > m {
			m = len(c)
		}
	}
	return m
}
