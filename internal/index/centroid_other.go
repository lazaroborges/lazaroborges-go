//go:build !amd64

package index

import "unsafe"

func dot14Avx2(q, c *[16]float32) float32 {
	var s float32
	for i := 0; i < Dim; i++ {
		s += q[i] * c[i]
	}
	return s
}

func centroidPassAvx2(q *[16]float32, cents *float32, norms *float32, out *CentroidDist, n uint64) {
	centsSlice := unsafe.Slice(cents, int(n)*16)
	normsSlice := unsafe.Slice(norms, int(n))
	outSlice := unsafe.Slice(out, int(n))
	for c := uint64(0); c < n; c++ {
		cp := (*[16]float32)(unsafe.Pointer(&centsSlice[c*16]))
		dot := dot14Avx2(q, cp)
		outSlice[c] = CentroidDist{Cluster: uint32(c), Dist: normsSlice[c] - 2*dot}
	}
}
