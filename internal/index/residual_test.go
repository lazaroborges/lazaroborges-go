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
