#include "textflag.h"

// func int16SqDist14(q *[14]int16, m *int16) int32
//
// AVX2 squared Euclidean distance between two 14-d int16 vectors. Sign-extends
// int16 → int32 before subtract (so the difference can't wrap), then squares
// into int64 via VPMULDQ on even/odd lanes, accumulates in int64, finally
// saturates at int32 max to match the Go reference.
//
// Members are stride 14 (28 bytes), so we cover all 14 lanes with two
// overlapping 8-lane loads (lanes 0..7 and 6..13) and subtract the doubled
// d[6]²+d[7]² as a scalar correction at the end.
//
// Plan 9 VEX op order is `src2, src1, dst` → dst = src1 op src2.
TEXT ·int16SqDist14(SB), NOSPLIT, $0-24
	MOVQ q+0(FP), AX
	MOVQ m+8(FP), BX

	// ---- Low half: d[0..7] = q[0..7] - m[0..7] ----
	VPMOVSXWD (AX), Y0          // Y0 = q[0..7] as 8×int32
	VPMOVSXWD (BX), Y1          // Y1 = m[0..7] as 8×int32
	VPSUBD Y1, Y0, Y0           // Y0 = differences (no wrap; int32 holds [-65535, 65535] easily)

	// Square 8 int32 lanes → 4 int64 pair-sums via VPMULDQ on even / odd lanes.
	VPSRLQ $32, Y0, Y2          // Y2: odd int32 lanes promoted into low 32 of each 64-bit slot
	VPMULDQ Y0, Y0, Y3          // Y3 = (d0², d2², d4², d6²) as 4×int64
	VPMULDQ Y2, Y2, Y4          // Y4 = (d1², d3², d5², d7²) as 4×int64
	VPADDQ Y4, Y3, Y3           // Y3 = (d0²+d1², d2²+d3², d4²+d5², d6²+d7²)

	// ---- High half: d[6..13] (overlap at 6,7) ----
	VPMOVSXWD 12(AX), Y5
	VPMOVSXWD 12(BX), Y6
	VPSUBD Y6, Y5, Y5

	VPSRLQ $32, Y5, Y7
	VPMULDQ Y5, Y5, Y8
	VPMULDQ Y7, Y7, Y9
	VPADDQ Y9, Y8, Y8           // Y8 = (d6²+d7², d8²+d9², d10²+d11², d12²+d13²)

	VPADDQ Y8, Y3, Y3           // total with d[6]²+d[7]² counted twice

	// ---- Horizontal sum of 4 int64s in Y3 → scalar ----
	VEXTRACTI128 $1, Y3, X10    // X10 = upper 2 int64 lanes
	VPADDQ X10, X3, X3          // X3 = (a+c, b+d) of original 4 int64s
	VPSHUFD $0x4E, X3, X11      // X11 = swap halves of X3 → (b+d, a+c)
	VPADDQ X11, X3, X3          // X3 lane 0 = full int64 total
	MOVQ X3, DX                 // DX = total (signed int64)

	// ---- Correction: subtract d[6]² + d[7]² (lanes 0,1 of X5 = q[6..13]-m[6..13]) ----
	VPEXTRD $0, X5, R8
	MOVLQSX R8, R8              // sign-extend int32 → int64
	IMULQ R8, R8
	VPEXTRD $1, X5, R9
	MOVLQSX R9, R9
	IMULQ R9, R9
	SUBQ R8, DX
	SUBQ R9, DX

	// ---- Saturate at int32 max ----
	MOVQ $0x7fffffff, CX
	CMPQ DX, CX
	JLE done
	MOVQ CX, DX
done:
	VZEROUPPER
	MOVL DX, ret+16(FP)
	RET
