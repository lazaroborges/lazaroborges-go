package index

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
	"unsafe"
)

// Index is the in-memory (mmap'd) representation of index.bin.
type Index struct {
	data    []byte
	buckets [NumBuckets]bucketView
	nprobe  int
}

type bucketView struct {
	nDims      int
	nCentroids int
	nVectors   int
	dims       []int     // comparison dim indices (cached from DimsForBucket)
	centroids  []float32 // nCentroids * nDims
	cellSizes  []uint32  // nCentroids
	cellStart  []int     // prefix sum of cellSizes, len = nCentroids+1
	vectors    []int8    // nVectors * nDims, grouped by cell
	labels     []uint8   // nVectors, 0=legit 1=fraud
}

// Open loads index.bin via mmap. The Index must outlive all SearchK5 calls.
func Open(path string, nprobe int) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(fi.Size())

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	if string(data[0:4]) != Magic {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("bad magic: %q", data[0:4])
	}
	ver := binary.LittleEndian.Uint32(data[4:8])
	if ver != Version {
		_ = syscall.Munmap(data)
		return nil, fmt.Errorf("unknown version %d", ver)
	}

	idx := &Index{data: data, nprobe: nprobe}

	for bi := 0; bi < NumBuckets; bi++ {
		off := HeaderSize + bi*BucketDescSize
		desc := parseBucketDesc(data[off : off+BucketDescSize])

		nDims := int(desc.NDims)
		nCentroids := int(desc.NCentroids)
		nVectors := int(desc.NVectors)

		if nVectors == 0 {
			continue
		}

		centSlice := bytesToFloat32(data[desc.CentroidsOff : desc.CentroidsOff+uint64(nCentroids*nDims*4)])
		szSlice := bytesToUint32(data[desc.CellSzOff : desc.CellSzOff+uint64(nCentroids*4)])
		vecSlice := bytesToInt8(data[desc.VectorsOff : desc.VectorsOff+uint64(nVectors*nDims)])
		lblSlice := data[desc.LabelsOff : desc.LabelsOff+uint64(nVectors)]

		cellStart := make([]int, nCentroids+1)
		for i, sz := range szSlice {
			cellStart[i+1] = cellStart[i] + int(sz)
		}

		idx.buckets[bi] = bucketView{
			nDims:      nDims,
			nCentroids: nCentroids,
			nVectors:   nVectors,
			dims:       DimsForBucket(bi),
			centroids:  centSlice,
			cellSizes:  szSlice,
			cellStart:  cellStart,
			vectors:    vecSlice,
			labels:     lblSlice,
		}
	}

	return idx, nil
}

// Close unmaps the index data. Only safe to call at process shutdown.
func (idx *Index) Close() {
	_ = syscall.Munmap(idx.data)
}

// SearchK5 finds the 5 nearest neighbors of query (a 14-dim float32 vector)
// and returns the fraud count among them and the fraud_score (count/5).
func (idx *Index) SearchK5(query [14]float32) (fraudCount int, fraudScore float32) {
	bid := bucketIDFromVec(query)
	bv := &idx.buckets[bid]

	if bv.nVectors == 0 {
		return 0, 0
	}

	dims := bv.dims
	nDims := bv.nDims
	nprobe := idx.nprobe
	if nprobe > bv.nCentroids {
		nprobe = bv.nCentroids
	}

	// Extract query's comparison dims as float32 (stack-allocated)
	var qfArr [11]float32
	qf := qfArr[:nDims]
	for i, d := range dims {
		qf[i] = query[d]
	}

	// Find nprobe nearest centroids using float32 Euclidean distance (stack-allocated)
	type centDist struct {
		idx  int
		dist float32
	}
	var topCentArr [MaxNProbe]centDist
	topCent := topCentArr[:nprobe]
	for i := range topCent {
		topCent[i].dist = math.MaxFloat32
	}
	worstInTop := float32(math.MaxFloat32)

	for ci := 0; ci < bv.nCentroids; ci++ {
		cent := bv.centroids[ci*nDims : ci*nDims+nDims]
		d := sqDistF32(qf, cent)
		if d < worstInTop {
			// Replace the worst entry
			wi := 0
			for j := 1; j < nprobe; j++ {
				if topCent[j].dist > topCent[wi].dist {
					wi = j
				}
			}
			topCent[wi] = centDist{ci, d}
			worstInTop = 0
			for _, cd := range topCent {
				if cd.dist > worstInTop {
					worstInTop = cd.dist
				}
			}
		}
	}

	// Quantize query dims to int8 for cell scanning (stack-allocated)
	var qiArr [11]int8
	qi := qiArr[:nDims]
	for i, v := range qf {
		qi[i] = QuantizeF32(v)
	}

	// Track top-5 nearest by int8 squared distance
	const K = 5
	type neighbor struct {
		dist  int32
		fraud bool
	}
	var top5 [K]neighbor
	for i := range top5 {
		top5[i].dist = math.MaxInt32
	}
	top5Worst := int32(math.MaxInt32)
	top5Count := 0

	for _, cd := range topCent {
		if bv.cellSizes[cd.idx] == 0 {
			continue
		}
		start := bv.cellStart[cd.idx]
		end := bv.cellStart[cd.idx+1]
		vecs := bv.vectors[start*nDims : end*nDims]
		lbls := bv.labels[start:end]

		for vi := 0; vi < end-start; vi++ {
			vec := vecs[vi*nDims : vi*nDims+nDims]
			d := sqDistInt8(qi, vec)
			if top5Count < K {
				top5[top5Count] = neighbor{d, lbls[vi] == 1}
				top5Count++
				if top5Count == K {
					top5Worst = 0
					for _, nb := range top5 {
						if nb.dist > top5Worst {
							top5Worst = nb.dist
						}
					}
				}
			} else if d < top5Worst {
				wi := 0
				for j := 1; j < K; j++ {
					if top5[j].dist > top5[wi].dist {
						wi = j
					}
				}
				top5[wi] = neighbor{d, lbls[vi] == 1}
				top5Worst = 0
				for _, nb := range top5 {
					if nb.dist > top5Worst {
						top5Worst = nb.dist
					}
				}
			}
		}
	}

	for i := 0; i < top5Count; i++ {
		if top5[i].fraud {
			fraudCount++
		}
	}
	if top5Count == 0 {
		return 0, 0
	}
	fraudScore = float32(fraudCount) / float32(top5Count)
	return fraudCount, fraudScore
}

func sqDistInt8(a, b []int8) int32 {
	var sum int32
	for i := range a {
		d := int32(a[i]) - int32(b[i])
		sum += d * d
	}
	return sum
}

func parseBucketDesc(b []byte) BucketDesc {
	return BucketDesc{
		NDims:        b[0],
		NCentroids:   binary.LittleEndian.Uint32(b[4:]),
		NVectors:     binary.LittleEndian.Uint32(b[8:]),
		CentroidsOff: binary.LittleEndian.Uint64(b[16:]),
		CellSzOff:    binary.LittleEndian.Uint64(b[24:]),
		VectorsOff:   binary.LittleEndian.Uint64(b[32:]),
		LabelsOff:    binary.LittleEndian.Uint64(b[40:]),
	}
}

func bytesToFloat32(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func bytesToUint32(b []byte) []uint32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func bytesToInt8(b []byte) []int8 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), len(b))
}
