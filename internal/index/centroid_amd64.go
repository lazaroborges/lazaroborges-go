//go:build amd64

package index

//go:noescape
func dot14Avx2(q, c *[16]float32) float32
