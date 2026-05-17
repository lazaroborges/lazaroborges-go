//go:build !amd64

package index

// int8ResidualSquaredDistance computes Σ_{i=0..13} (qRes[i] - mRes[i])²
// where qRes is the per-query residual in int16 space (produced as
// int16(qInt8[i]) - int16(centroidInt8[i]); range ±255) and mRes is the
// stored member residual (int8). Padded lanes [14:16] in both inputs MUST
// be zero — callers are responsible.
//
// Accumulator is int64 for safety; at the documented input range the sum
// always fits in int32, so the final narrowing is loss-free.
func int8ResidualSquaredDistance(qRes *[16]int16, mRes *[16]int8) int32 {
	var sum int64
	for i := 0; i < 14; i++ {
		d := int64(qRes[i]) - int64(mRes[i])
		sum += d * d
	}
	return int32(sum)
}
