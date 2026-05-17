package index

import (
	"encoding/binary"
	"unsafe"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/index/hnsw"
)

// SearchIVFHNSW runs the v2 search: centroid pass → top-N cells → per-cell
// HNSW (K=8 each) → merge to top-10 by int8 distance → float32 re-rank →
// final top-5 written to `out`.
//
// nCells must be ≤ len(scratch.PerCell) (currently 32).
// The caller is responsible for providing scratch.CellBuf sized to NClusters
// and scratch.Visited sized to the cluster's max member count.
func (idx *Index) SearchIVFHNSW(
	qFloat *[16]float32,
	qInt8 *[14]int8,
	nCells int,
	ef int,
	scratch *SearchScratch,
	out *Top5Final,
) {
	out.Reset()
	scratch.Merged.Reset()

	// 1. Centroid pass: distance from qFloat to each padded centroid.
	centroidPassAvx2(qFloat, &idx.CentroidsPadded[0], &idx.CentroidNorms[0],
		&scratch.CellBuf[0], uint64(idx.NClusters))

	// 2. Select top-N cells (linear for small N, quickselect otherwise).
	if nCells <= 4 {
		for i := 0; i < nCells; i++ {
			bestIdx := i
			for j := i + 1; j < idx.NClusters; j++ {
				if scratch.CellBuf[j].Dist < scratch.CellBuf[bestIdx].Dist {
					bestIdx = j
				}
			}
			scratch.CellBuf[i], scratch.CellBuf[bestIdx] = scratch.CellBuf[bestIdx], scratch.CellBuf[i]
		}
	} else {
		selectTopK(scratch.CellBuf[:idx.NClusters], nCells)
	}

	// 3. For each selected cell, scan members + drain top-K into Merged.
	//    Reuses a single HnswOut MaxHeap across cells — no PerCell array needed.
	const K = 8
	for k := 0; k < nCells; k++ {
		cellIdx := scratch.CellBuf[k].Cluster
		idx.searchOneCell(cellIdx, qInt8, K, ef, scratch, &scratch.HnswOut)
		for _, e := range scratch.HnswOut.Items() {
			from := idx.ClusterOffsets[cellIdx]
			global := from + uint32(e.ID)
			scratch.Merged.Insert(Candidate{
				Dist:    e.Dist,
				Cluster: cellIdx,
				LocalID: e.ID,
				Label:   idx.Labels[global],
			})
		}
	}

	// 4. Float32 re-rank.
	idx.rerankTop5(&scratch.Merged, qFloat, out)
}

func (idx *Index) searchOneCell(
	cellIdx uint32,
	qInt8 *[14]int8,
	K, ef int, // ef unused in IVFADC mode (kept for API compat with HNSW path)
	scratch *SearchScratch,
	out *hnsw.MaxHeap,
) {
	_ = ef
	// Translate query into this cell's residual int16 space.
	centBase := int(cellIdx) * 16
	for i := 0; i < 14; i++ {
		scratch.QRes[i] = int16(qInt8[i]) - int16(idx.CentroidsInt8[centBase+i])
	}
	scratch.QRes[14] = 0
	scratch.QRes[15] = 0

	// IVFADC pivot: exhaustively scan all members in the cell with int8
	// residual SIMD. HNSW navigation hit a recall ceiling on this data
	// (small residual magnitudes → poor graph navigation precision); a
	// linear scan with the same kernel gives true top-K and pays only a
	// constant-factor cost over the per-cluster size (~3000 dist evals).
	from, to := idx.ClusterOffsets[cellIdx], idx.ClusterOffsets[cellIdx+1]
	count := int(to - from)
	mBase := int(from) * 16

	out.Reset()
	for i := 0; i < count; i++ {
		row := mBase + i*16
		d := int8ResidualSquaredDistance(&scratch.QRes,
			(*[16]int8)(unsafe.Pointer(&idx.MemberResiduals[row])))
		if out.Len() < K {
			out.Push(hnsw.Entry{Dist: d, ID: uint16(i)})
		} else if d < out.Top().Dist {
			out.Pop()
			out.Push(hnsw.Entry{Dist: d, ID: uint16(i)})
		}
	}
}

// cellGraph constructs a *hnsw.Graph view backed by the mmap'd edge bytes
// for this cluster, without copying. mBase is the element offset into
// MemberResiduals for the first member of this cluster (each member uses 16
// int8 elements).
func (idx *Index) cellGraph(cellIdx uint32) (*hnsw.Graph, int) {
	h := idx.HnswHeaders[cellIdx]
	from, to := idx.ClusterOffsets[cellIdx], idx.ClusterOffsets[cellIdx+1]
	n := uint16(to - from)
	const M = uint8(6) // pinned at build time

	g := &hnsw.Graph{
		N:        n,
		M:        M,
		MaxLevel: h.MaxLevel,
		Entry:    h.EntryPoint,
		NodeIds:  make([][]uint16, int(h.MaxLevel)+1),
		Edges:    make([][]uint16, int(h.MaxLevel)+1),
		Degree:   make([][]uint8, int(h.MaxLevel)+1),
	}

	cursor := int(h.EdgeOffset)
	for L := uint8(0); L <= h.MaxLevel; L++ {
		count := int(h.LevelCount[L])
		if L >= 1 {
			g.NodeIds[L] = readUint16Slice(idx.HnswEdges, cursor, count)
			cursor += count * 2
		}
		edges := make([]uint16, count*int(M))
		deg := make([]uint8, count)
		base := cursor
		for i := 0; i < count; i++ {
			slot := base + i*hnswEdgeSlotBytes
			deg[i] = idx.HnswEdges[slot]
			for j := 0; j < int(M); j++ {
				edges[i*int(M)+j] = binary.LittleEndian.Uint16(idx.HnswEdges[slot+2+j*2:])
			}
		}
		g.Edges[L] = edges
		g.Degree[L] = deg
		cursor += count * hnswEdgeSlotBytes
	}

	mBase := int(from) * 16
	return g, mBase
}

// hnswEdgeSlotBytes is the byte width of one edge slot in the serialized edge
// block: degree(1) + pad(1) + M×2 neighbors = 14 bytes for M=6.
const hnswEdgeSlotBytes = 14

// readUint16Slice reads n uint16 values from data starting at byte offset off.
func readUint16Slice(data []byte, off, n int) []uint16 {
	out := make([]uint16, n)
	for i := 0; i < n; i++ {
		out[i] = binary.LittleEndian.Uint16(data[off+i*2:])
	}
	return out
}

// rerankTop5 reconstructs float32 coordinates for each merged candidate and
// writes the top-5 by exact float32 squared distance into `out`, sorted
// ascending by Dist.
func (idx *Index) rerankTop5(merged *TopK10, qFloat *[16]float32, out *Top5Final) {
	var dists [MergedTopKCap]float32
	for i := 0; i < merged.N; i++ {
		c := merged.Items[i]
		centBase := int(c.Cluster) * Dim
		resBase := (int(idx.ClusterOffsets[c.Cluster]) + int(c.LocalID)) * 16
		var s float32
		for d := 0; d < Dim; d++ {
			rf := float32(idx.MemberResiduals[resBase+d]) / 127
			mf := idx.Centroids[centBase+d] + rf
			diff := qFloat[d] - mf
			s += diff * diff
		}
		dists[i] = s
	}
	// Select top-5 by ascending float32 dist. Partial insertion sort over the
	// first 5 positions of a paired index array.
	var indices [MergedTopKCap]int
	for i := 0; i < merged.N; i++ {
		indices[i] = i
	}
	limit := 5
	if merged.N < limit {
		limit = merged.N
	}
	for i := 1; i < merged.N; i++ {
		for j := i; j > 0 && dists[indices[j]] < dists[indices[j-1]]; j-- {
			indices[j], indices[j-1] = indices[j-1], indices[j]
		}
	}
	out.N = limit
	for i := 0; i < limit; i++ {
		out.Dist[i] = dists[indices[i]]
		out.Label[i] = merged.Items[indices[i]].Label
	}
}
