package index

// Top5 is the fixed-size top-5 maintained without heap allocation. Entries are
// kept sorted ascending by distance.
//
// Distances are int64 because the dot-product distance form (qNorm + memberNorm
// − 2·dot) doesn't fit in int32 at full quantization scale, and we want
// ranking to remain exact (no saturation).
type Top5 struct {
	Dist  [5]int64
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
func (t *Top5) insert(d int64, lab uint8) {
	if t.N < 5 {
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
// `cellBuf` is a scratch buffer of length ≥ nClusters used for centroid
// distance ranking. `distBuf` is a scratch buffer of length ≥ maxCellSize
// used for per-cell member distances. Results are written into `out`.
func (idx *Index) SearchIVF(qVec *[Dim]int16, qVecFloat *[Dim]float32, nprobe int, cellBuf []CentroidDist, distBuf []int64, out *Top5) {
	out.Reset()

	// Distance from query to every centroid via the norm trick:
	//   ||q-c||² = ||q||² + ||c||² - 2·(q·c)
	// We pre-computed ||c||² at load. ||q||² is identical across centroids so
	// it doesn't affect ranking — we can drop it entirely.
	var qPad [16]float32
	copy(qPad[:Dim], qVecFloat[:])
	centroidPassAvx2(&qPad, &idx.CentroidsPadded[0], &idx.CentroidNorms[0], &cellBuf[0], uint64(idx.NClusters))
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

	// Precompute ||q||² in int64 — constant for this query, used by every
	// member-distance computation in the kernel.
	var qNorm int64
	var qPadI16 [16]int16
	for i := 0; i < Dim; i++ {
		v := int64(qVec[i])
		qNorm += v * v
		qPadI16[i] = qVec[i]
	}

	// Iterate members of those top nprobe cells. Per cell we batch the
	// distance kernel as one VPMADDWD-based asm call that keeps q in YMM
	// registers across all members in the cell and writes int64 distances to
	// distBuf; then a tight Go loop feeds them into the fixed-size top-5.
	mem := idx.MemberVecs
	labels := idx.Labels
	norms := idx.MemberNorms
	for k := 0; k < nprobe; k++ {
		c := cellBuf[k].Cluster
		from := idx.ClusterOffsets[c]
		to := idx.ClusterOffsets[c+1]
		count := int(to - from)
		if count == 0 {
			continue
		}
		memberScanAvx2(&qPadI16, &mem[int(from)*Dim], &norms[from], qNorm, &distBuf[0], uint64(count))
		labelBase := labels[from:to]
		for v := 0; v < count; v++ {
			out.insert(distBuf[v], labelBase[v])
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
