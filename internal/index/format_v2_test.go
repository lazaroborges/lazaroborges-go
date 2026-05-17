package index_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/index"
)

// This test compiles only after both writer (in cmd/preprocess/hnsw_build.go)
// and reader (in internal/index/index.go) exist. It exercises the round trip
// on a tiny synthetic dataset.
func TestFormatV2RoundTrip(t *testing.T) {
	tmp := t.TempDir()
	dst := filepath.Join(tmp, "tiny.bin")

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

	// writeIndexV2 lives in package main of cmd/preprocess; for the test,
	// expose it via a small wrapper there (see Task 8) OR replicate the call
	// here in a helper. For now this test is a placeholder until Task 8
	// extracts the writer into a callable from tests.
	t.Skip("activated in Task 8 after extracting writeIndexV2")

	_ = dst
	_ = assign
	_ = os.Remove
	_ = vecs
	_ = labels
	_ = centroids
	_ = index.VersionV2
	// idx, err := index.Load(dst)
	// require.NoError(t, err)
	// require.Equal(t, index.VersionV2, idx.Version)
	// require.Equal(t, N, idx.NVectors)
	// require.Equal(t, K, idx.NClusters)
}
