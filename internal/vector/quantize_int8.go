package vector

// QuantizeInt8 maps a 14-d float vector in [-1, 1] to int8 by scaling × 127
// and clamping to [-128, 127]. Padded lanes [14:16] in the input are ignored.
func QuantizeInt8(in *[16]float32, out *[14]int8) {
	for i := 0; i < 14; i++ {
		v := in[i] * 127
		if v >= 127 {
			out[i] = 127
		} else if v <= -128 {
			out[i] = -128
		} else if v >= 0 {
			out[i] = int8(v + 0.5)
		} else {
			out[i] = int8(v - 0.5)
		}
	}
}
