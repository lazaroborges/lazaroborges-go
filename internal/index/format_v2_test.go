package index_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/index"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/indexwriter"
)

func TestFormatV2RoundTrip(t *testing.T) {
	tmp := t.TempDir()
	dst := filepath.Join(tmp, "tiny.bin")
	defer os.Remove(dst)

	const N, K = 200, 4
	vecs := make([]float32, N*14)
	labels := make([]uint8, N)
	for i := 0; i < N; i++ {
		labels[i] = uint8(i % 2)
		for d := 0; d < 14; d++ {
			vecs[i*14+d] = float32((i*7+d*3)%101)/101 - 0.5
		}
	}
	centroids := make([]float32, K*14)
	for c := 0; c < K; c++ {
		for d := 0; d < 14; d++ {
			centroids[c*14+d] = float32(c)/10 - 0.2
		}
	}
	assign := make([]uint32, N)
	for i := 0; i < N; i++ {
		assign[i] = uint32(i % K)
	}

	if err := indexwriter.WriteV2(dst, vecs, labels, assign, centroids, 42); err != nil {
		t.Fatal(err)
	}
	idx, err := index.Load(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	if idx.Version != index.VersionV2 {
		t.Fatalf("version mismatch: got %d want %d", idx.Version, index.VersionV2)
	}
	if idx.NVectors != N || idx.NClusters != K {
		t.Fatalf("shape mismatch: got %dv/%dc want %dv/%dc",
			idx.NVectors, idx.NClusters, N, K)
	}
	// Spot-check centroids round-tripped.
	for c := 0; c < K; c++ {
		for d := 0; d < 14; d++ {
			got := idx.Centroids[c*14+d]
			want := centroids[c*14+d]
			if got != want {
				t.Fatalf("centroid[%d][%d]: got %v want %v", c, d, got, want)
			}
		}
	}
	// Labels are reordered into cluster groups. Spot-check legitimate range.
	for i := 0; i < N; i++ {
		if idx.Labels[i] > 1 {
			t.Fatalf("bad label %d at %d", idx.Labels[i], i)
		}
	}
	// Per-cluster sanity: offsets monotonic and end at N.
	for c := 0; c < K; c++ {
		if idx.ClusterOffsets[c] > idx.ClusterOffsets[c+1] {
			t.Fatalf("offsets non-monotonic at cluster %d", c)
		}
	}
	if int(idx.ClusterOffsets[K]) != N {
		t.Fatalf("final offset %d != N=%d", idx.ClusterOffsets[K], N)
	}
}
