package index

import (
	"math/rand"
	"testing"
)

func TestTopK10_KeepsKSmallest(t *testing.T) {
	const total = 200
	var h TopK10
	rng := rand.New(rand.NewSource(42))
	dists := make([]int32, 0, total)
	for i := 0; i < total; i++ {
		d := rng.Int31n(10000)
		dists = append(dists, d)
		h.Insert(Candidate{Dist: d, Cluster: 0, LocalID: uint16(i), Label: uint8(i & 1)})
	}
	if h.N != MergedTopKCap {
		t.Fatalf("got N=%d want %d", h.N, MergedTopKCap)
	}
	// Sort all distances; the heap should contain exactly the K smallest.
	var copy []int32
	copy = append(copy, dists...)
	for i := 1; i < len(copy); i++ {
		for j := i; j > 0 && copy[j] < copy[j-1]; j-- {
			copy[j], copy[j-1] = copy[j-1], copy[j]
		}
	}
	want := map[int32]int{}
	for _, d := range copy[:MergedTopKCap] {
		want[d]++
	}
	have := map[int32]int{}
	for i := 0; i < h.N; i++ {
		have[h.Items[i].Dist]++
	}
	for d, n := range want {
		if have[d] != n {
			t.Fatalf("dist %d: have %d want %d", d, have[d], n)
		}
	}
}

func TestTopK10_RootIsFarthest(t *testing.T) {
	var h TopK10
	for _, d := range []int32{10, 50, 20, 80, 30, 40, 5, 90, 25, 35, 15} {
		h.Insert(Candidate{Dist: d, LocalID: uint16(d)})
	}
	// Root must be the largest of the 10 kept.
	max := h.Items[0].Dist
	for i := 1; i < h.N; i++ {
		if h.Items[i].Dist > max {
			t.Fatalf("root not the max: root=%d items=%v", max, h.Items[:h.N])
		}
	}
}

func TestTop5Final_FraudCount(t *testing.T) {
	var t5 Top5Final
	t5.N = 5
	t5.Label = [5]uint8{0, 1, 0, 1, 1}
	if got := t5.FraudCount(); got != 3 {
		t.Fatalf("got %d want 3", got)
	}
}
