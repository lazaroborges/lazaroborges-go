//go:build amd64

package index

//go:noescape
func dot14Avx2(q, c *[16]float32) float32

// centroidPassAvx2 runs the full centroid distance loop in assembly,
// avoiding the per-iteration Go→ASM call overhead of dot14Avx2.
//
//go:noescape
func centroidPassAvx2(q *[16]float32, cents *float32, norms *float32, out *CentroidDist, n uint64)
