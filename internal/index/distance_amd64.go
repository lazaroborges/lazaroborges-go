package index

// int16SqDist14 returns the squared Euclidean distance between two 14-d int16
// vectors. Implementation lives in distance_amd64.s and uses SSE2/SSSE3
// PMADDWD; both q and m must point to at least 14 int16 (28 bytes) of
// readable memory. Saturates at int32 max on overflow.
//
//go:noescape
func int16SqDist14(q *[Dim]int16, m *int16) int32
