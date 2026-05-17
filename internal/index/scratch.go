package index

import "github.com/lazaroborges/rinha-de-backend-2026/internal/index/hnsw"

// Candidate is a single result entry, tagged with its source cluster + local
// member id so the float32 re-rank step can reconstruct the float
// coordinates from CentroidsInt8 + MemberResiduals.
type Candidate struct {
	Dist    int32  // int8 residual squared distance, pre-rerank
	Cluster uint32
	LocalID uint16
	Label   uint8
}

// MergedTopKCap is the size of the merged candidate pool fed into float32
// re-rank. Wider = more chances to recover from int8 ranking noise; cost is
// re-rank arithmetic (still trivial vs the cell scan).
const MergedTopKCap = 64

// TopK10 holds up to MergedTopKCap candidates as a max-heap by Dist (root =
// farthest, eviction target when full). Fixed-size to avoid allocation. Named
// TopK10 for historical reasons; capacity controlled by MergedTopKCap.
type TopK10 struct {
	Items [MergedTopKCap]Candidate
	N     int
}

func (t *TopK10) Reset() { t.N = 0 }

// Insert respects max-heap invariant on Dist (root = farthest).
func (t *TopK10) Insert(c Candidate) {
	if t.N < MergedTopKCap {
		t.Items[t.N] = c
		t.N++
		i := t.N - 1
		for i > 0 {
			p := (i - 1) / 2
			if t.Items[p].Dist >= t.Items[i].Dist {
				return
			}
			t.Items[p], t.Items[i] = t.Items[i], t.Items[p]
			i = p
		}
		return
	}
	if c.Dist >= t.Items[0].Dist {
		return
	}
	t.Items[0] = c
	// Sift down.
	i := 0
	for {
		l := 2*i + 1
		if l >= t.N {
			break
		}
		r := l + 1
		s := l
		if r < t.N && t.Items[r].Dist > t.Items[l].Dist {
			s = r
		}
		if t.Items[i].Dist >= t.Items[s].Dist {
			break
		}
		t.Items[i], t.Items[s] = t.Items[s], t.Items[i]
		i = s
	}
}

// Top5Final holds the post-rerank top-5 (sorted ascending by float distance).
type Top5Final struct {
	Dist  [5]float32
	Label [5]uint8
	N     int
}

func (t *Top5Final) Reset() { t.N = 0 }

func (t *Top5Final) FraudCount() int {
	n := 0
	for i := 0; i < t.N; i++ {
		if t.Label[i] == LabelFraud {
			n++
		}
	}
	return n
}

// SearchScratch bundles all per-query buffers so the server pool does one
// Get/Put per request.
type SearchScratch struct {
	CellBuf []CentroidDist // length NClusters (caller pre-allocates)
	QRes    [16]int16      // per-cell residual query (padded)
	Merged  TopK10
	Visited []uint64      // bitmap sized to max cluster
	Cand    hnsw.MinHeap
	HnswOut hnsw.MaxHeap
	PerCell [32]hnsw.MaxHeap // per-cell K=8 results (supports nCells up to 32)
}
