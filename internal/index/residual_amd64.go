//go:build amd64

package index

// int8ResidualSquaredDistance is implemented in residual_amd64.s.
// Inputs: qRes (16 × int16, padded; range ±255), mRes (16 × int8, padded).
// Output: Σ (qRes[i] - mRes[i])² for i in [0,14), as int32.
// Padded lanes [14:16] MUST be zero on input.
//
//go:noescape
func int8ResidualSquaredDistance(qRes *[16]int16, mRes *[16]int8) int32
