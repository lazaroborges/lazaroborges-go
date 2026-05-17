package vector

import "testing"

func TestQuantizeInt8_Bounds(t *testing.T) {
	in := [16]float32{
		-1.0, -0.5, 0.0, 0.5, 1.0,
		0.99, -0.99, -1.0, 1.0, 0.1,
		-0.1, 0.0, 0.0, 0.0,
		0.0, 0.0, // padded lanes — must be ignored
	}
	var out [14]int8
	QuantizeInt8(&in, &out)

	wants := []int8{-127, -64, 0, 64, 127, 126, -126, -127, 127, 13, -13, 0, 0, 0}
	for i, w := range wants {
		if out[i] != w {
			t.Errorf("[%d] got %d want %d", i, out[i], w)
		}
	}
}

func TestQuantizeInt8_ClampOverrange(t *testing.T) {
	in := [16]float32{}
	in[0] = 1.5  // out of [-1, 1]
	in[1] = -1.5
	var out [14]int8
	QuantizeInt8(&in, &out)
	if out[0] != 127 {
		t.Errorf("expected clamp to 127, got %d", out[0])
	}
	if out[1] != -128 {
		t.Errorf("expected clamp to -128, got %d", out[1])
	}
}
