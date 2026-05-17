package index_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/index"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/index/hnsw"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/indexwriter"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/vector"
)

func TestSearchIVFHNSW_Smoke(t *testing.T) {
	tmp := t.TempDir()
	dst := filepath.Join(tmp, "syn.bin")
	defer os.Remove(dst)

	const N, K = 500, 4
	vecs := make([]float32, N*14)
	labels := make([]uint8, N)
	for i := 0; i < N; i++ {
		labels[i] = uint8(i % 2)
		for d := 0; d < 14; d++ {
			vecs[i*14+d] = float32(((i*13)+d*7)%101)/101*2 - 1
		}
	}
	centroids := make([]float32, K*14)
	assign := make([]uint32, N)
	for c := 0; c < K; c++ {
		for d := 0; d < 14; d++ {
			centroids[c*14+d] = float32(c)/8 - 0.25
		}
	}
	for i := 0; i < N; i++ {
		assign[i] = uint32(i % K)
	}

	if err := indexwriter.WriteV2(dst, vecs, labels, assign, centroids, 13); err != nil {
		t.Fatal(err)
	}
	idx, err := index.Load(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	scratch := &index.SearchScratch{
		CellBuf: make([]index.CentroidDist, idx.NClusters),
		Visited: make([]uint64, 1024),
		Cand:    hnsw.MinHeap{},
		HnswOut: hnsw.MaxHeap{},
	}
	for i := 0; i < 4; i++ {
		scratch.PerCell[i] = hnsw.MaxHeap{}
	}

	var qFloat [16]float32
	for d := 0; d < 14; d++ {
		qFloat[d] = 0.1 * float32(d)
	}
	var qInt8 [14]int8
	vector.QuantizeInt8(&qFloat, &qInt8)

	var out index.Top5Final
	idx.SearchIVFHNSW(&qFloat, &qInt8, 2, 32, scratch, &out)

	if out.N != 5 {
		t.Fatalf("expected 5 results, got %d", out.N)
	}
	for i := 1; i < 5; i++ {
		if out.Dist[i] < out.Dist[i-1] {
			t.Fatalf("results not sorted ascending: %v", out.Dist)
		}
	}
}
