//go:build amd64

package index

import (
	"math"
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestCentroidPassAvx2_AgreesWithScalar fuzzes the inlined centroid pass
// kernel against the simple per-centroid loop. Both should produce
// CentroidDist entries with identical Cluster (0..n-1) and Dist within a
// small float tolerance (different reduction order).
func TestCentroidPassAvx2_AgreesWithScalar(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(0xC0FFEE))
	const n = 1024
	cents := make([]float32, n*16)
	norms := make([]float32, n)
	for c := 0; c < n; c++ {
		var sq float32
		for i := 0; i < Dim; i++ {
			v := rng.Float32()*2 - 1
			cents[c*16+i] = v
			sq += v * v
		}
		norms[c] = sq
	}

	var q [16]float32
	for i := 0; i < Dim; i++ {
		q[i] = rng.Float32()*2 - 1
	}

	wantOut := make([]CentroidDist, n)
	for c := 0; c < n; c++ {
		cp := (*[16]float32)(cents[c*16 : c*16+16 : c*16+16])
		_ = cp
		var dot float32
		for i := 0; i < Dim; i++ {
			dot += q[i] * cents[c*16+i]
		}
		wantOut[c] = CentroidDist{Cluster: uint32(c), Dist: norms[c] - 2*dot}
	}

	gotOut := make([]CentroidDist, n)
	centroidPassAvx2(&q, &cents[0], &norms[0], &gotOut[0], uint64(n))

	for c := 0; c < n; c++ {
		if gotOut[c].Cluster != wantOut[c].Cluster {
			t.Fatalf("cluster %d: got %d want %d", c, gotOut[c].Cluster, wantOut[c].Cluster)
		}
		if math.Abs(float64(gotOut[c].Dist-wantOut[c].Dist)) > 1e-4 {
			t.Fatalf("c=%d: got dist=%g want %g", c, gotOut[c].Dist, wantOut[c].Dist)
		}
	}
}

func BenchmarkCentroidPass_PerCall(b *testing.B) {
	if !cpu.X86.HasAVX2 {
		b.Skip("AVX2 not available on this CPU")
	}
	const n = 1024
	cents := make([]float32, n*16)
	norms := make([]float32, n)
	for i := range cents {
		cents[i] = 0.1
	}
	for i := range norms {
		norms[i] = 1.0
	}
	var q [16]float32
	for i := 0; i < Dim; i++ {
		q[i] = 0.5
	}
	out := make([]CentroidDist, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for c := 0; c < n; c++ {
			cp := (*[16]float32)(cents[c*16 : c*16+16 : c*16+16])
			dot := dot14Avx2(&q, cp)
			out[c] = CentroidDist{Cluster: uint32(c), Dist: norms[c] - 2*dot}
		}
	}
}

func BenchmarkCentroidPass_Inlined(b *testing.B) {
	if !cpu.X86.HasAVX2 {
		b.Skip("AVX2 not available on this CPU")
	}
	const n = 1024
	cents := make([]float32, n*16)
	norms := make([]float32, n)
	for i := range cents {
		cents[i] = 0.1
	}
	for i := range norms {
		norms[i] = 1.0
	}
	var q [16]float32
	for i := 0; i < Dim; i++ {
		q[i] = 0.5
	}
	out := make([]CentroidDist, n)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		centroidPassAvx2(&q, &cents[0], &norms[0], &out[0], uint64(n))
	}
}
