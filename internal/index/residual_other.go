//go:build !amd64

package index

// int8ResidualSquaredDistance computes Σ_{i=0..13} (qRes[i] - mRes[i])²
// where qRes is the query in residual int16 space (already pre-translated
// by subtracting the cluster centroid in int8) and mRes is the stored
// member residual (int8). Padded lanes [14:16] in both inputs MUST be zero
// — callers are responsible.
func int8ResidualSquaredDistance(qRes *[16]int16, mRes *[16]int8) int32 {
	var sum int32
	for i := 0; i < 14; i++ {
		d := int32(qRes[i]) - int32(mRes[i])
		sum += d * d
	}
	return sum
}
