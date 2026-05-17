package hnsw

import "testing"

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
