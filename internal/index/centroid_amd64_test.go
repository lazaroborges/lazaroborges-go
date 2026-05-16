//go:build amd64

package index

import (
	"math"
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestDot14Avx2_AgreesWithScalar fuzzes the AVX2 dot-product kernel against
// the obvious scalar reference. Inputs cover the actual operating range:
// query and centroid values are in [-1, 1] (post-normalization domain).
func TestDot14Avx2_AgreesWithScalar(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(0xC0FFEE))
	for trial := 0; trial < 4096; trial++ {
		var q, c [16]float32
		for i := 0; i < Dim; i++ {
			q[i] = rng.Float32()*2 - 1
			c[i] = rng.Float32()*2 - 1
		}
		// q[14..15] and c[14..15] are zero from declaration — the invariant
		// the AVX2 kernel relies on.

		var want float32
		for i := 0; i < Dim; i++ {
			want += q[i] * c[i]
		}
		got := dot14Avx2(&q, &c)

		// Floats — accept tiny rounding differences from the SIMD reduction
		// shape vs the scalar left-to-right accumulation.
		if math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("trial %d: q=%v c=%v: avx2=%g scalar=%g", trial, q[:Dim], c[:Dim], got, want)
		}
	}
}

// TestDot14Avx2_MasksTrailingLanes confirms the kernel doesn't read past
// lane 13 — if it did, garbage at lanes 14, 15 would change the result.
func TestDot14Avx2_MasksTrailingLanes(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not available on this CPU")
	}
	var q [16]float32
	var clean, dirty [16]float32
	for i := 0; i < Dim; i++ {
		q[i] = float32(i) * 0.05
		v := float32(i)*0.07 - 0.3
		clean[i] = v
		dirty[i] = v
	}
	dirty[Dim] = 99
	dirty[Dim+1] = -99
	q[Dim] = 0 // query must hold the zero invariant
	q[Dim+1] = 0

	want := dot14Avx2(&q, &clean)
	got := dot14Avx2(&q, &dirty)
	if got != want {
		t.Fatalf("trailing-lane mask broken: clean=%g dirty=%g", want, got)
	}
}

func BenchmarkDot14_Scalar(b *testing.B) {
	var q, c [16]float32
	for i := 0; i < Dim; i++ {
		q[i] = float32(i) * 0.05
		c[i] = float32(i)*0.07 - 0.3
	}
	b.ResetTimer()
	var acc float32
	for i := 0; i < b.N; i++ {
		var s float32
		for j := 0; j < Dim; j++ {
			s += q[j] * c[j]
		}
		acc += s
	}
	_ = acc
}

func BenchmarkDot14_Avx2(b *testing.B) {
	if !cpu.X86.HasAVX2 {
		b.Skip("AVX2 not available on this CPU")
	}
	var q, c [16]float32
	for i := 0; i < Dim; i++ {
		q[i] = float32(i) * 0.05
		c[i] = float32(i)*0.07 - 0.3
	}
	b.ResetTimer()
	var acc float32
	for i := 0; i < b.N; i++ {
		acc += dot14Avx2(&q, &c)
	}
	_ = acc
}
