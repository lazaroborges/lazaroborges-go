package index

import (
	"math/rand"
	"testing"
)

// goRef is the pure-Go reference computation; identical to int32SqDist in
// search.go but kept here so the test is self-contained and not subject to
// the search.go signature.
func goRef(q *[Dim]int16, m *[Dim]int16) int32 {
	var sum int64
	for i := 0; i < Dim; i++ {
		d := int64(q[i]) - int64(m[i])
		sum += d * d
	}
	if sum > 0x7fffffff {
		return 0x7fffffff
	}
	return int32(sum)
}

func TestInt16SqDist14_Reference(t *testing.T) {
	cases := []struct {
		name string
		q, m [Dim]int16
	}{
		{"zero", [Dim]int16{}, [Dim]int16{}},
		{"identity-positive", [Dim]int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, [Dim]int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}},
		{"unit-shift", [Dim]int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, [Dim]int16{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
		{"signs", [Dim]int16{-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7}, [Dim]int16{1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7}},
		{"large", [Dim]int16{32000, -32000, 32000, -32000, 32000, -32000, 32000, -32000, 32000, -32000, 32000, -32000, 32000, -32000}, [Dim]int16{-32000, 32000, -32000, 32000, -32000, 32000, -32000, 32000, -32000, 32000, -32000, 32000, -32000, 32000}},
		{"sentinels", [Dim]int16{-32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768}, [Dim]int16{}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			want := goRef(&c.q, &c.m)
			got := int16SqDist14(&c.q, &c.m[0])
			if got != want {
				t.Errorf("got %d, want %d (q=%v m=%v)", got, want, c.q, c.m)
			}
		})
	}
}

func TestInt16SqDist14_Random(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		var q, m [Dim]int16
		for j := 0; j < Dim; j++ {
			q[j] = int16(r.Intn(65536) - 32768)
			m[j] = int16(r.Intn(65536) - 32768)
		}
		want := goRef(&q, &m)
		got := int16SqDist14(&q, &m[0])
		if got != want {
			t.Fatalf("iter %d: got %d want %d (q=%v m=%v)", i, got, want, q, m)
		}
	}
}

func BenchmarkInt16SqDist14(b *testing.B) {
	var q [Dim]int16
	mem := make([]int16, Dim*1024) // 1024 contiguous members
	r := rand.New(rand.NewSource(1))
	for i := range mem {
		mem[i] = int16(r.Intn(65536) - 32768)
	}
	for i := 0; i < Dim; i++ {
		q[i] = int16(r.Intn(65536) - 32768)
	}
	b.ResetTimer()
	var sink int32
	for i := 0; i < b.N; i++ {
		sink += int16SqDist14(&q, &mem[(i&1023)*Dim])
	}
	runtime_KeepAlive(sink)
}

// runtime_KeepAlive avoids the benchmark loop being optimised away.
//
//go:noinline
func runtime_KeepAlive(int32) {}
