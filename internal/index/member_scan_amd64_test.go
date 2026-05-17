//go:build amd64

package index

import (
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

// refDist computes squared Euclidean distance against the scalar Go reference,
// returning int64 (matches the kernel output type — no saturation).
func refDist(q *[Dim]int16, m []int16) int64 {
	var sum int64
	for i := 0; i < Dim; i++ {
		d := int64(q[i]) - int64(m[i])
		sum += d * d
	}
	return sum
}

func computeNorm(v []int16) int64 {
	var s int64
	for _, x := range v {
		xi := int64(x)
		s += xi * xi
	}
	return s
}

func runMemberScan(t *testing.T, q [Dim]int16, members []int16, n int, offset int) {
	t.Helper()
	want := make([]int64, n)
	for i := 0; i < n; i++ {
		want[i] = refDist(&q, members[(offset+i)*Dim:(offset+i+1)*Dim])
	}

	norms := make([]int64, n)
	for i := 0; i < n; i++ {
		norms[i] = computeNorm(members[(offset+i)*Dim : (offset+i+1)*Dim])
	}
	qNorm := computeNorm(q[:])

	var qPad [16]int16
	copy(qPad[:Dim], q[:])

	got := make([]int64, n)
	memberScanAvx2(&qPad, &members[offset*Dim], &norms[0], qNorm, &got[0], uint64(n))

	for i := 0; i < n; i++ {
		if got[i] != want[i] {
			t.Fatalf("i=%d offset=%d: got %d want %d", i, offset, got[i], want[i])
		}
	}
}

// TestMemberScanAvx2_AgreesWithScalar fuzzes the kernel against a scalar
// reference. Covers the common case (random members) plus three edge cases
// the VPMADDWD/dot-product implementation must handle: queries that exercise
// the quant-scale extremes, an all-zero query (verifies the dist == memberNorm
// path), and members reordered to put the highest cluster offset last (so the
// kernel's 4-byte overrun on the final member reads into the test slice's
// internal allocation tail).
func TestMemberScanAvx2_AgreesWithScalar(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(0xC0FFEE))
	const n = 1024
	const scale = 32767
	// Allocate +2 trailing int16 so the 4-byte overrun on the last member is
	// always safe in the test harness regardless of allocator behaviour.
	members := make([]int16, n*Dim+2)
	for i := 0; i < n*Dim; i++ {
		members[i] = int16(rng.Intn(2*scale+1) - scale)
	}

	t.Run("random", func(t *testing.T) {
		var q [Dim]int16
		for i := 0; i < Dim; i++ {
			q[i] = int16(rng.Intn(2*scale+1) - scale)
		}
		runMemberScan(t, q, members, n, 0)
	})

	t.Run("zero-query", func(t *testing.T) {
		var q [Dim]int16 // all zero
		runMemberScan(t, q, members, n, 0)
	})

	t.Run("sentinel-lanes", func(t *testing.T) {
		// −32767 is the quantized representation of the `-1` null sentinel
		// (see internal/vector). Members carry the same sentinel for
		// last_transaction-null entries; the kernel must produce the same
		// distance as scalar on these.
		var q [Dim]int16
		for i := 0; i < Dim; i++ {
			q[i] = -32767
		}
		runMemberScan(t, q, members, n, 0)
	})

	t.Run("last-member-overrun", func(t *testing.T) {
		// Run a single member at the very end of the slice. The kernel's
		// 32-byte load goes 4 bytes past the 28-byte member into the
		// allocator-owned trailing bytes; q[14]=q[15]=0 means those bytes
		// can't affect the result.
		var q [Dim]int16
		for i := 0; i < Dim; i++ {
			q[i] = int16(rng.Intn(2*scale+1) - scale)
		}
		runMemberScan(t, q, members, 1, n-1)
	})
}

func BenchmarkMemberScan_PerCall(b *testing.B) {
	if !cpu.X86.HasAVX2 {
		b.Skip("AVX2 not available on this CPU")
	}
	const n = 3000
	members := make([]int16, n*Dim)
	for i := range members {
		members[i] = int16(i * 7)
	}
	var q [Dim]int16
	for i := 0; i < Dim; i++ {
		q[i] = int16(i * 13)
	}
	out := make([]int32, n)

	b.ResetTimer()
	for it := 0; it < b.N; it++ {
		for i := 0; i < n; i++ {
			out[i] = int16SqDist14(&q, &members[i*Dim])
		}
	}
}

func BenchmarkMemberScan_Inlined(b *testing.B) {
	if !cpu.X86.HasAVX2 {
		b.Skip("AVX2 not available on this CPU")
	}
	const n = 3000
	members := make([]int16, n*Dim+2)
	for i := 0; i < n*Dim; i++ {
		members[i] = int16(i * 7)
	}
	var q [Dim]int16
	for i := 0; i < Dim; i++ {
		q[i] = int16(i * 13)
	}
	norms := make([]int64, n)
	for i := 0; i < n; i++ {
		norms[i] = computeNorm(members[i*Dim : (i+1)*Dim])
	}
	qNorm := computeNorm(q[:])
	var qPad [16]int16
	copy(qPad[:Dim], q[:])
	out := make([]int64, n)

	b.ResetTimer()
	for it := 0; it < b.N; it++ {
		memberScanAvx2(&qPad, &members[0], &norms[0], qNorm, &out[0], uint64(n))
	}
}
