//go:build amd64

#include "textflag.h"

// func memberScanAvx2(q *[16]int16, members *int16, out *int32, n uint64)
//
// Computes the squared Euclidean distance between q (a 14-d int16 vector,
// passed as a 16-int16 array — the trailing lanes 14, 15 are not read by
// this kernel) and each of `n` consecutive 14-int16 members at `members`,
// writing the int32 results to out[].
//
// Members are tightly packed (stride 14 int16 = 28 bytes), matching the
// on-disk layout of MemberVecs. Like int16SqDist14, this uses two overlapping
// 8-lane loads at offsets 0 and 12 to cover the 14 lanes without reading
// past the member's 28-byte boundary; the doubled contribution from lanes
// 6, 7 is subtracted as a scalar correction.
//
// Q is sign-extended to int32 lanes ONCE before the loop and kept in YMM
// registers across iterations — that's the win over calling int16SqDist14
// in a Go loop, which would re-load and re-extend q every iteration.
//
// Plan 9 VEX op order: (src2, src1, dst) → dst = src1 op src2.
TEXT ·memberScanAvx2(SB), NOSPLIT, $0-32
	MOVQ q+0(FP), AX
	MOVQ members+8(FP), BX
	MOVQ out+16(FP), CX
	MOVQ n+24(FP), R8

	// Pre-load q into YMM regs (sign-extended to int32).
	VPMOVSXWD (AX), Y0          // Y0 = q[0..7]   as 8×int32  (kept)
	VPMOVSXWD 12(AX), Y10       // Y10 = q[6..13] as 8×int32  (kept; overlaps Y0 at lanes 6,7)

	MOVQ $0x7fffffff, R12       // saturation constant

	XORQ R9, R9                 // i = 0
loop:
	CMPQ R9, R8
	JE   done

	// ---- low half: q[0..7] - m[0..7] ----
	VPMOVSXWD (BX), Y1          // m[0..7]
	VPSUBD    Y1, Y0, Y2        // Y2 = q[0..7] - m[0..7]
	VPSRLQ    $32, Y2, Y3       // odd int32 lanes → low halves
	VPMULDQ   Y2, Y2, Y4        // (d0², d2², d4², d6²) as 4×int64
	VPMULDQ   Y3, Y3, Y5        // (d1², d3², d5², d7²) as 4×int64
	VPADDQ    Y5, Y4, Y4        // 4×int64 pair sums

	// ---- high half: q[6..13] - m[6..13] ----
	VPMOVSXWD 12(BX), Y6
	VPSUBD    Y6, Y10, Y7
	VPSRLQ    $32, Y7, Y8
	VPMULDQ   Y7, Y7, Y9
	VPMULDQ   Y8, Y8, Y11
	VPADDQ    Y11, Y9, Y9
	VPADDQ    Y9, Y4, Y4        // sum both halves (lanes 6,7 counted twice)

	// ---- horizontal sum of 4 int64 in Y4 → scalar in DX ----
	VEXTRACTI128 $1, Y4, X12
	VPADDQ       X12, X4, X4
	VPSHUFD      $0x4E, X4, X13
	VPADDQ       X13, X4, X4
	MOVQ         X4, DX

	// ---- subtract doubled d[6]² + d[7]² (lanes 0,1 of X7 = q[6..13]-m[6..13]) ----
	VPEXTRD $0, X7, R14
	MOVLQSX R14, R14
	IMULQ   R14, R14
	VPEXTRD $1, X7, R15
	MOVLQSX R15, R15
	IMULQ   R15, R15
	SUBQ    R14, DX
	SUBQ    R15, DX

	// ---- saturate at int32 max ----
	CMPQ DX, R12
	JLE  no_sat
	MOVQ R12, DX
no_sat:
	MOVL DX, (CX)               // out[i] = (int32) clamped(distance)

	ADDQ $28, BX                // members += 14 int16 = 28 bytes
	ADDQ $4, CX                 // out += 1 int32 = 4 bytes
	INCQ R9
	JMP  loop
done:
	VZEROUPPER
	RET
