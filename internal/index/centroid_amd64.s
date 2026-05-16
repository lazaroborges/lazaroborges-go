//go:build amd64

#include "textflag.h"

// func dot14Avx2(q, c *[16]float32) float32
//
// Returns sum_{i=0..13} q[i]*c[i] as a float32. Both q and c are 16-float
// arrays with the trailing 2 lanes (indices 14, 15) zeroed so the AVX2
// products at those lanes contribute 0 to the horizontal sum.
//
// 14 of 16 lanes are valid; the trailing-zero invariant is enforced by:
//   - the query: cmd/api padded copy of qFloat (last two stay zero)
//   - the centroids: Load() pads each centroid into idx.CentroidsPadded
//
// Plan 9 op order is (src2, src1, dst) → dst = src1 op src2.
TEXT ·dot14Avx2(SB), NOSPLIT, $0-20
	MOVQ q+0(FP), AX
	MOVQ c+8(FP), BX

	VMOVUPS (AX), Y0           // q[0..7]
	VMOVUPS 32(AX), Y1         // q[8..15]
	VMULPS  (BX), Y0, Y0       // Y0 = q[0..7] * c[0..7]
	VMOVUPS 32(BX), Y3
	VMULPS  Y3, Y1, Y1         // Y1 = q[8..15] * c[8..15]
	VADDPS  Y1, Y0, Y0         // 8 lane partial sums

	// Horizontal sum of 8 float32 in Y0 → scalar in X0.
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0    // 4 lanes
	VHADDPS      X0, X0, X0    // 2 lanes
	VHADDPS      X0, X0, X0    // 1 lane

	VZEROUPPER
	MOVSS X0, ret+16(FP)
	RET
