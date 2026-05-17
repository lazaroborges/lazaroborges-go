package index

// int16SqDist14 returns the squared Euclidean distance between two 14-d int16
// vectors. Implementation lives in distance_amd64.s and uses SSE2/SSSE3
// PMADDWD; both q and m must point to at least 14 int16 (28 bytes) of
// readable memory. Saturates at int32 max on overflow.
//
//go:noescape
func int16SqDist14(q *[Dim]int16, m *int16) int32

// memberScanAvx2 computes n consecutive squared distances between q and the
// 14-int16 members at `members` (stride 28 bytes), writing each result as
// an int32 to out[]. Inlining the loop in assembly keeps q sign-extended in
// YMM registers across iterations — saves the Go→ASM boundary cost paid by
// int16SqDist14 per call (~10-20 ns × n).
//
//go:noescape
func memberScanAvx2(q *[Dim]int16, members *int16, out *int32, n uint64)
