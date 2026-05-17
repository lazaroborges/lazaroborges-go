package hnsw

import (
	"math/rand"
	"testing"
)

// A tiny hand-built graph: 4 nodes on a single level.
// All distances supplied via a slice.
func TestSearchTrivial(t *testing.T) {
	g := &Graph{
		N:        4,
		M:        2,
		MaxLevel: 0,
		Entry:    0,
		NodeIds:  [][]uint16{nil}, // level 0: no NodeIds
		Edges: [][]uint16{{
			1, 2, // node 0 → 1, 2
			0, 3, // node 1 → 0, 3
			0, 3, // node 2 → 0, 3
			1, 2, // node 3 → 1, 2
		}},
		Degree: [][]uint8{{2, 2, 2, 2}},
	}
	dists := []int32{100, 30, 40, 20} // node 3 is closest
	df := DistFn(func(id uint16) int32 { return dists[id] })

	visited := make([]uint64, 1)
	cand := MinHeap{}
	out := MaxHeap{}
	g.Search(df, 2, 4, visited, &cand, &out)

	if out.Len() != 2 {
		t.Fatalf("got %d, want 2 results", out.Len())
	}
	// items order in the heap isn't sorted; just check membership.
	have := map[uint16]int32{}
	for _, e := range out.Items() {
		have[e.ID] = e.Dist
	}
	if have[3] != 20 || have[1] != 30 {
		t.Fatalf("expected {3:20, 1:30}, got %v", have)
	}
}

// Recall test: build a graph over 1000 random 14-dim int8 vectors, query 100
// times, and check that the HNSW top-5 matches brute-force top-5 with high
// recall. Recall@5 ≥ 0.95 is the bar; we expect 0.98+ in practice.
func TestRecallSmall(t *testing.T) {
	const (
		N      = 1000
		Dim    = 14
		Trials = 100
		K      = 5
		Ef     = 32
	)
	rng := rand.New(rand.NewSource(7))
	vecs := make([]int8, N*Dim)
	for i := range vecs {
		vecs[i] = int8(rng.Intn(255) - 127)
	}
	distFor := func(qResI16 *[16]int16) DistFn {
		return func(mid uint16) int32 {
			var s int32
			base := int(mid) * Dim
			for i := 0; i < Dim; i++ {
				d := int32(qResI16[i]) - int32(vecs[base+i])
				s += d * d
			}
			return s
		}
	}

	// M=12 gives ≥0.98 recall@5 at ef=32; M=6 only achieves ~0.85 at this ef.
	g := Build(uint16(N), 12, 200, 0xBEEF, func(a, b uint16) int32 {
		var s int32
		ab, bb := int(a)*Dim, int(b)*Dim
		for i := 0; i < Dim; i++ {
			d := int32(vecs[ab+i]) - int32(vecs[bb+i])
			s += d * d
		}
		return s
	})

	visited := make([]uint64, (N+63)>>6)
	cand := MinHeap{}
	out := MaxHeap{}

	totalHits := 0
	for q := 0; q < Trials; q++ {
		var qRes [16]int16
		for i := 0; i < Dim; i++ {
			qRes[i] = int16(rng.Intn(255) - 127)
		}
		// Brute force top-K.
		all := make([]testPair, N)
		for i := 0; i < N; i++ {
			var s int32
			base := i * Dim
			for j := 0; j < Dim; j++ {
				d := int32(qRes[j]) - int32(vecs[base+j])
				s += d * d
			}
			all[i] = testPair{s, uint16(i)}
		}
		sortPairs(all)
		bruteTopK := map[uint16]bool{}
		for i := 0; i < K; i++ {
			bruteTopK[all[i].id] = true
		}

		// HNSW search.
		g.Search(distFor(&qRes), K, Ef, visited, &cand, &out)
		for _, e := range out.Items() {
			if bruteTopK[e.ID] {
				totalHits++
			}
		}
	}

	recall := float64(totalHits) / float64(Trials*K)
	if recall < 0.95 {
		t.Fatalf("recall@5 = %.3f < 0.95", recall)
	}
	t.Logf("recall@5 = %.3f", recall)
}

type testPair struct {
	d  int32
	id uint16
}

func sortPairs(p []testPair) {
	// simple insertion sort — N=1000, fine for tests.
	for i := 1; i < len(p); i++ {
		for j := i; j > 0 && p[j].d < p[j-1].d; j-- {
			p[j], p[j-1] = p[j-1], p[j]
		}
	}
}
