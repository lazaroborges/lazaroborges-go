package index

// int16SqDist14 returns the squared Euclidean distance between two 14-d int16
// vectors. Implementation lives in distance_amd64.s and uses SSE2/SSSE3
// PMADDWD; both q and m must point to at least 14 int16 (28 bytes) of
// readable memory. Saturates at int32 max on overflow.
//
//go:noescape
func int16SqDist14(q *[Dim]int16, m *int16) int32

// memberScanAvx2 computes n squared Euclidean distances between q and each
// of n consecutive 14-int16 members at `members` (stride 28 bytes), writing
// the int64 result for member i to out[i].
//
// Uses the dot-product identity ‖q-m‖² = qNorm + norms[i] − 2·(q·m), where
// `norms` points at the precomputed per-member ‖m‖² array (one int64 per
// member, same indexing as members) and qNorm is the caller-computed ‖q‖².
//
// q is padded to 16 int16 (trailing 2 lanes must be zero). The kernel loads
// 32 bytes per member, which reads 4 bytes past the 28-byte member boundary;
// because q[14]=q[15]=0, the resulting VPMADDWD contribution from those lanes
// is zero, so the overrun is harmless. The mmap region continues into Labels
// for the very last member, so the read is page-safe.
//
//go:noescape
func memberScanAvx2(q *[16]int16, members *int16, norms *int64, qNorm int64, out *int64, n uint64)
