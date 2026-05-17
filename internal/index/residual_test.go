package index

import "testing"

// Known answer: q=(1,2,3,...,14) - m=(0,0,...) → sum of squares 1+4+9+...+196 = 1015
func TestInt8ResidualSquaredDistance_Known(t *testing.T) {
	var qRes [16]int16
	var mRes [16]int8
	for i := 0; i < 14; i++ {
		qRes[i] = int16(i + 1)
	}
	// mRes all zeros (allocated by var).

	got := int8ResidualSquaredDistance(&qRes, &mRes)
	want := int32(0)
	for i := 1; i <= 14; i++ {
		want += int32(i * i)
	}
	if got != want {
		t.Fatalf("dist mismatch: got %d want %d", got, want)
	}
}

func TestInt8ResidualSquaredDistance_Negative(t *testing.T) {
	var qRes [16]int16
	var mRes [16]int8
	qRes[0] = -3
	mRes[0] = 5
	// diff = -8, sq = 64. Other lanes contribute 0.
	got := int8ResidualSquaredDistance(&qRes, &mRes)
	if got != 64 {
		t.Fatalf("dist mismatch: got %d want 64", got)
	}
}

func TestInt8ResidualSquaredDistance_PadIgnored(t *testing.T) {
	var qRes [16]int16
	var mRes [16]int8
	qRes[14] = 100 // padded lane should NOT contribute
	mRes[15] = 100 // padded lane should NOT contribute
	got := int8ResidualSquaredDistance(&qRes, &mRes)
	if got != 0 {
		t.Fatalf("padded lanes leaked: got %d want 0", got)
	}
}

// Boundary: qRes filled with +255, mRes filled with -128. Per-lane diff is
// 383, squared 146689, summed over 14 lanes ≈ 2.05e6. Confirms no overflow
// at the max expected input range produced by int8-residual subtraction.
func TestInt8ResidualSquaredDistance_MaxRange(t *testing.T) {
	var qRes [16]int16
	var mRes [16]int8
	for i := 0; i < 14; i++ {
		qRes[i] = 255
		mRes[i] = -128
	}
	got := int8ResidualSquaredDistance(&qRes, &mRes)
	want := int32(14 * 383 * 383) // 2,053,646
	if got != want {
		t.Fatalf("got %d want %d", got, want)
	}
}

func TestInt8ResidualSquaredDistance_RandomCrossCheck(t *testing.T) {
	// Brute-force reference for comparison.
	ref := func(q *[16]int16, m *[16]int8) int32 {
		var s int64
		for i := 0; i < 14; i++ {
			d := int64(q[i]) - int64(m[i])
			s += d * d
		}
		return int32(s)
	}
	rng := newDeterministicRNG(0xC0FFEE)
	for trial := 0; trial < 1000; trial++ {
		var q [16]int16
		var m [16]int8
		for i := 0; i < 14; i++ {
			// Stay in the documented ±255 range for q, full int8 range for m.
			q[i] = int16(rng.intn(511) - 255)
			m[i] = int8(rng.intn(255) - 127)
		}
		got := int8ResidualSquaredDistance(&q, &m)
		want := ref(&q, &m)
		if got != want {
			t.Fatalf("trial %d: got %d want %d (q=%v m=%v)", trial, got, want, q, m)
		}
	}
}

// newDeterministicRNG is a tiny xorshift, avoids pulling in math/rand state.
type detRNG struct{ s uint64 }

func newDeterministicRNG(seed uint64) *detRNG { return &detRNG{s: seed | 1} }
func (r *detRNG) intn(n int) int {
	r.s ^= r.s << 13
	r.s ^= r.s >> 7
	r.s ^= r.s << 17
	return int(r.s % uint64(n))
}

func BenchmarkInt8ResidualSquaredDistance(b *testing.B) {
	var q [16]int16
	var m [16]int8
	for i := 0; i < 14; i++ {
		q[i] = int16(i*17 - 50)
		m[i] = int8(i*11 - 70)
	}
	b.ResetTimer()
	var sink int32
	for i := 0; i < b.N; i++ {
		sink = int8ResidualSquaredDistance(&q, &m)
	}
	_ = sink
}
