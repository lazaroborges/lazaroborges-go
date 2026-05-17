package hnsw

import (
	"math/rand"
	"sort"
	"testing"
)

func TestMinHeapOrder(t *testing.T) {
	h := MinHeap{}
	rng := rand.New(rand.NewSource(1))
	want := make([]int32, 0, 200)
	for i := 0; i < 200; i++ {
		d := rng.Int31n(10000)
		h.Push(Entry{Dist: d, ID: uint16(i)})
		want = append(want, d)
	}
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	for i := 0; i < 200; i++ {
		got := h.Pop().Dist
		if got != want[i] {
			t.Fatalf("step %d: got %d want %d", i, got, want[i])
		}
	}
}

func TestMaxHeapOrder(t *testing.T) {
	h := MaxHeap{}
	rng := rand.New(rand.NewSource(2))
	want := make([]int32, 0, 200)
	for i := 0; i < 200; i++ {
		d := rng.Int31n(10000)
		h.Push(Entry{Dist: d, ID: uint16(i)})
		want = append(want, d)
	}
	sort.Slice(want, func(i, j int) bool { return want[i] > want[j] })
	for i := 0; i < 200; i++ {
		got := h.Pop().Dist
		if got != want[i] {
			t.Fatalf("step %d: got %d want %d", i, got, want[i])
		}
	}
}
