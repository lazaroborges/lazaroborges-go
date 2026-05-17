#include "textflag.h"

// func int8ResidualSquaredDistance(qRes *[16]int16, mRes *[16]int8) int32
//
// AVX2 squared distance between a 16-lane int16 query residual and a 16-lane
// int8 member residual. Only the first 14 lanes contribute to the sum; lanes
// 14 and 15 are masked off after subtraction so callers don't need to clear
// the padding (matches the pure-Go reference which iterates i=0..13).
//
// Strategy:
//   VMOVDQU   load 16 × int16 query (32 bytes)
//   VPMOVSXBW load 16 × int8 member, sign-extend to 16 × int16 in one shot
//   VPSUBW    16 × int16 difference (range ±383 fits in int16)
//   VPAND     zero lanes 14, 15 of the difference
//   VPMADDWD  pairwise multiply-add → 8 × int32 (each = a²+b²)
//   horizontal reduce → scalar int32
//
// Plan 9 VEX op order is `src2, src1, dst` → dst = src1 op src2.
TEXT ·int8ResidualSquaredDistance(SB), NOSPLIT, $0-20
	MOVQ qRes+0(FP), AX
	MOVQ mRes+8(FP), BX

	VMOVDQU (AX), Y0           // Y0 = qRes as 16 × int16
	VPMOVSXBW (BX), Y1         // Y1 = mRes sign-extended to 16 × int16
	VPSUBW Y1, Y0, Y0          // Y0 = qRes - mRes (16 × int16)
	VPAND lane14Mask<>(SB), Y0, Y0  // zero lanes 14, 15 so they don't contribute
	VPMADDWD Y0, Y0, Y0        // Y0 = 8 × int32 pair-sums (a²+b²)

	// Horizontal sum 8 × int32 → scalar int32.
	VEXTRACTI128 $1, Y0, X1    // X1 = upper 4 × int32
	VPADDD X1, X0, X0          // X0 = lane-wise sum of halves (4 × int32)
	VPHADDD X0, X0, X0         // X0 = [a+b, c+d, a+b, c+d]
	VPHADDD X0, X0, X0         // X0 lane 0 = a+b+c+d
	VMOVD X0, AX
	VZEROUPPER
	MOVL AX, ret+16(FP)
	RET

// 32-byte mask: 14 × 0xFFFF followed by 2 × 0x0000. ANDed against the int16
// difference vector to zero out the padded lanes [14:16] before VPMADDWD.
DATA lane14Mask<>+0x00(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA lane14Mask<>+0x08(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA lane14Mask<>+0x10(SB)/8, $0xFFFFFFFFFFFFFFFF
DATA lane14Mask<>+0x18(SB)/8, $0x00000000FFFFFFFF
GLOBL lane14Mask<>(SB), RODATA|NOPTR, $32
