//go:build amd64

package index

import (
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestMemberScanAvx2_AgreesWithScalar fuzzes the inlined member-scan kernel
// against repeated calls to int16SqDist14. Both must produce the same int32
// for every member.
func TestMemberScanAvx2_AgreesWithScalar(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not available on this CPU")
	}
	rng := rand.New(rand.NewSource(0xC0FFEE))
	const n = 1024
	// Quant scale matches vector.QuantScale (32767) — bounds inputs so the
	// kernel's int32-accumulator path doesn't saturate spuriously and we
	// can compare against scalar exactly.
	const scale = 32767
	members := make([]int16, n*Dim)
	for i := 0; i < n*Dim; i++ {
		members[i] = int16(rng.Intn(2*scale+1) - scale)
	}
	var q [Dim]int16
	for i := 0; i < Dim; i++ {
		q[i] = int16(rng.Intn(2*scale+1) - scale)
	}

	want := make([]int32, n)
	for i := 0; i < n; i++ {
		want[i] = int16SqDist14(&q, &members[i*Dim])
	}

	got := make([]int32, n)
	memberScanAvx2(&q, &members[0], &got[0], uint64(n))

	for i := 0; i < n; i++ {
		if got[i] != want[i] {
			t.Fatalf("i=%d: got %d want %d", i, got[i], want[i])
		}
	}
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
		memberScanAvx2(&q, &members[0], &out[0], uint64(n))
	}
}
