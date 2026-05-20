package index

import (
	"math/rand"
	"testing"
)

func TestKMeansBasic(t *testing.T) {
	// Two clear clusters: one around 0.1 and one around 0.9 (1-dim for simplicity)
	b := &BucketedData{
		BucketID: 0,
		NDims:    1,
		Dims:     []int{0},
	}
	for i := 0; i < 100; i++ {
		b.Vecs = append(b.Vecs, 0.1)
		b.Labels = append(b.Labels, false)
	}
	for i := 0; i < 100; i++ {
		b.Vecs = append(b.Vecs, 0.9)
		b.Labels = append(b.Labels, true)
	}

	rng := rand.New(rand.NewSource(42))
	cr := KMeans(b, 2, 20, rng)

	c0 := cr.Centroids[0]
	c1 := cr.Centroids[1]
	if !((c0 < 0.5 && c1 > 0.5) || (c0 > 0.5 && c1 < 0.5)) {
		t.Errorf("k-means failed to separate clusters: c0=%.3f c1=%.3f", c0, c1)
	}
	if len(cr.Assignments) != 200 {
		t.Errorf("expected 200 assignments, got %d", len(cr.Assignments))
	}

	var total uint32
	for _, sz := range cr.CellSizes {
		total += sz
	}
	if total != 200 {
		t.Errorf("cell sizes sum %d != 200", total)
	}
}

func TestSortVectorsByCell(t *testing.T) {
	b := &BucketedData{
		BucketID: 0,
		NDims:    1,
		Dims:     []int{0},
		Vecs:     []float32{0.9, 0.1, 0.9, 0.1},
		Labels:   []bool{true, false, true, false},
	}
	cr := ClusterResult{
		Centroids:   []float32{0.1, 0.9},
		Assignments: []int32{1, 0, 1, 0}, // vecs 0,2 → cell1; vecs 1,3 → cell0
		CellSizes:   []uint32{2, 2},
		NDims:       1,
		NCentroids:  2,
	}

	sortedVecs, sortedLabels := SortVectorsByCell(b, cr)

	// Cell 0 first (the 0.1 vectors, legit), then cell 1 (the 0.9 vectors, fraud)
	expected := []float32{0.1, 0.1, 0.9, 0.9}
	for i, v := range expected {
		if sortedVecs[i] != v {
			t.Errorf("sortedVecs[%d]: got %.1f, want %.1f", i, sortedVecs[i], v)
		}
	}
	expectedLabels := []bool{false, false, true, true}
	for i, l := range expectedLabels {
		if sortedLabels[i] != l {
			t.Errorf("sortedLabels[%d]: got %v, want %v", i, sortedLabels[i], l)
		}
	}
}
