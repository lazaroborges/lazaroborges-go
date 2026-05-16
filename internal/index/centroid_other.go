//go:build !amd64

package index

func dot14Avx2(q, c *[16]float32) float32 {
	var s float32
	for i := 0; i < Dim; i++ {
		s += q[i] * c[i]
	}
	return s
}
