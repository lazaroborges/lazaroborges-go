package index

// Top5 is the fixed-size top-5 maintained without heap allocation. Entries are
// kept sorted ascending by distance.
type Top5 struct {
	Dist  [5]int32
	Label [5]uint8
	N     int // current count, ≤ 5
}

// Reset clears the buffer between queries.
func (t *Top5) Reset() {
	t.N = 0
}

// FraudCount returns how many of the (up to 5) entries are labeled fraud.
func (t *Top5) FraudCount() int {
	n := 0
	for i := 0; i < t.N; i++ {
		if t.Label[i] == LabelFraud {
			n++
		}
	}
	return n
}

// insert adds (d, lab) into the sorted top-5 if it qualifies. Branch-light
// insertion sort over a fixed 5-element array.
func (t *Top5) insert(d int32, lab uint8) {
	if t.N < 5 {
		// Append, then bubble down.
		t.Dist[t.N] = d
		t.Label[t.N] = lab
		t.N++
		for i := t.N - 1; i > 0 && t.Dist[i] < t.Dist[i-1]; i-- {
			t.Dist[i], t.Dist[i-1] = t.Dist[i-1], t.Dist[i]
			t.Label[i], t.Label[i-1] = t.Label[i-1], t.Label[i]
		}
		return
	}
	if d >= t.Dist[4] {
		return
	}
	// Find insertion point and shift right.
	pos := 4
	for pos > 0 && t.Dist[pos-1] > d {
		t.Dist[pos] = t.Dist[pos-1]
		t.Label[pos] = t.Label[pos-1]
		pos--
	}
	t.Dist[pos] = d
	t.Label[pos] = lab
}

// SearchIVF runs the IVF query. `qVec` is the int16-quantized query and
// `qVecFloat` is the same value as float32 (used for centroid distance). The
// caller is responsible for both. `nprobe` controls how many cells to scan.
// Results are written into `out`.
//
// SearchIVF runs the IVF query. `qVec` is the int16-quantized query and
// `qVecFloat` is the same value as float32 (used for centroid distance). The
// caller is responsible for both. `nprobe` controls how many cells to scan.
// `cellBuf` is a scratch buffer of length ≥ nClusters used for centroid
// distance ranking. Results are written into `out`.
func (idx *Index) SearchIVF(qVec *[Dim]int16, qVecFloat *[Dim]float32, nprobe int, cellBuf []CentroidDist, out *Top5) {
	out.Reset()

	// Distance from query to every centroid via the norm trick:
	//   ||q-c||² = ||q||² + ||c||² - 2·(q·c)
	// We pre-computed ||c||² at load. ||q||² is identical across centroids so
	// it doesn't affect ranking — we can drop it entirely (the same constant
	// added to every distance). What remains: dot product + add - shift. That
	// halves the FP op count vs the subtract-square version.
	q0, q1, q2, q3, q4, q5, q6 := qVecFloat[0], qVecFloat[1], qVecFloat[2], qVecFloat[3], qVecFloat[4], qVecFloat[5], qVecFloat[6]
	q7, q8, q9, q10, q11, q12, q13 := qVecFloat[7], qVecFloat[8], qVecFloat[9], qVecFloat[10], qVecFloat[11], qVecFloat[12], qVecFloat[13]
	cents := idx.Centroids
	norms := idx.CentroidNorms
	nC := idx.NClusters
	for c := 0; c < nC; c++ {
		base := c * Dim
		_ = cents[base+13] // hoist the bounds check
		dot := q0*cents[base+0] + q1*cents[base+1] + q2*cents[base+2] + q3*cents[base+3] +
			q4*cents[base+4] + q5*cents[base+5] + q6*cents[base+6] + q7*cents[base+7] +
			q8*cents[base+8] + q9*cents[base+9] + q10*cents[base+10] + q11*cents[base+11] +
			q12*cents[base+12] + q13*cents[base+13]
		// rank-equivalent distance: ||c||² - 2·(q·c). ||q||² omitted (constant).
		cellBuf[c] = CentroidDist{Cluster: uint32(c), Dist: norms[c] - 2*dot}
	}
	// Partial sort: find the nprobe smallest using selection.
	selectTopK(cellBuf[:idx.NClusters], nprobe)

	// Re-sort the selected nprobe cells by cluster index (ascending) so the
	// member scan reads MemberVecs in increasing offset order. This is much
	// friendlier to the L2 prefetcher than the arbitrary order quickselect
	// leaves behind. Insertion sort is fine — nprobe is small (≤ ~96).
	for i := 1; i < nprobe; i++ {
		for j := i; j > 0 && cellBuf[j].Cluster < cellBuf[j-1].Cluster; j-- {
			cellBuf[j], cellBuf[j-1] = cellBuf[j-1], cellBuf[j]
		}
	}

	// Iterate members of those top nprobe cells, computing int16 squared
	// distance and feeding into the fixed-size top-5 buffer. The distance
	// kernel is AVX2 assembly on amd64 (see distance_amd64.s); see
	// distance_other.go for the portable fallback.
	mem := idx.MemberVecs
	for k := 0; k < nprobe; k++ {
		c := cellBuf[k].Cluster
		from := idx.ClusterOffsets[c]
		to := idx.ClusterOffsets[c+1]
		for v := from; v < to; v++ {
			base := int(v) * Dim
			d := int16SqDist14(qVec, &mem[base])
			out.insert(d, idx.Labels[v])
		}
	}
}

// CentroidDist is the scratch element used by SearchIVF for centroid-distance
// ranking. Exported so callers can pre-allocate the buffer.
type CentroidDist struct {
	Cluster uint32
	Dist    float32
}

// selectTopK partitions `arr` so that the K smallest entries are in arr[:K]
// (unordered). O(n) average via quickselect-style partial sort. The full
// permutation order beyond K is undefined.
func selectTopK(arr []CentroidDist, k int) {
	if k >= len(arr) {
		return
	}
	lo, hi := 0, len(arr)-1
	for lo < hi {
		// Median-of-three pivot.
		mid := (lo + hi) / 2
		if arr[lo].Dist > arr[mid].Dist {
			arr[lo], arr[mid] = arr[mid], arr[lo]
		}
		if arr[lo].Dist > arr[hi].Dist {
			arr[lo], arr[hi] = arr[hi], arr[lo]
		}
		if arr[mid].Dist > arr[hi].Dist {
			arr[mid], arr[hi] = arr[hi], arr[mid]
		}
		pivot := arr[mid].Dist
		i, j := lo, hi
		for i <= j {
			for arr[i].Dist < pivot {
				i++
			}
			for arr[j].Dist > pivot {
				j--
			}
			if i <= j {
				arr[i], arr[j] = arr[j], arr[i]
				i++
				j--
			}
		}
		// The k-th element lies in one of the two partitions.
		if k <= j {
			hi = j
		} else if k >= i {
			lo = i
		} else {
			return
		}
	}
}

// int32SqDist computes squared Euclidean distance between two int16 14-d
// vectors. We widen to int32 only for the subtract+multiply (each component
// d² ≤ 65534² ≈ 4.3e9, which fits in uint32 but not int32). Accumulating
// 14 such terms peaks around 6e10, requiring int64 for the running sum.
//
// On amd64 a 32-bit IMUL is a single cycle, where int64 MUL is also 1 cycle
// but typically with worse register pressure; on Apple Silicon arm64 the
// difference is small either way. The previous all-int64 version unnecessarily
// extended every load, costing two extra MOVSX per iteration. Manual unroll
// helps the Go compiler keep the accumulator in a register.
func int32SqDist(a *[Dim]int16, b []int16) int32 {
	_ = b[Dim-1] // bounds-check hint: eliminates 14 checks inside the loop
	d0 := int32(a[0]) - int32(b[0])
	d1 := int32(a[1]) - int32(b[1])
	d2 := int32(a[2]) - int32(b[2])
	d3 := int32(a[3]) - int32(b[3])
	d4 := int32(a[4]) - int32(b[4])
	d5 := int32(a[5]) - int32(b[5])
	d6 := int32(a[6]) - int32(b[6])
	d7 := int32(a[7]) - int32(b[7])
	d8 := int32(a[8]) - int32(b[8])
	d9 := int32(a[9]) - int32(b[9])
	d10 := int32(a[10]) - int32(b[10])
	d11 := int32(a[11]) - int32(b[11])
	d12 := int32(a[12]) - int32(b[12])
	d13 := int32(a[13]) - int32(b[13])
	sum := int64(d0)*int64(d0) + int64(d1)*int64(d1) +
		int64(d2)*int64(d2) + int64(d3)*int64(d3) +
		int64(d4)*int64(d4) + int64(d5)*int64(d5) +
		int64(d6)*int64(d6) + int64(d7)*int64(d7) +
		int64(d8)*int64(d8) + int64(d9)*int64(d9) +
		int64(d10)*int64(d10) + int64(d11)*int64(d11) +
		int64(d12)*int64(d12) + int64(d13)*int64(d13)
	if sum > 0x7fffffff {
		return 0x7fffffff
	}
	return int32(sum)
}
