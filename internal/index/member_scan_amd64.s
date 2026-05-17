//go:build amd64

#include "textflag.h"

// func memberScanAvx2(q *[16]int16, members *int16, norms *int64, qNorm int64, out *int64, n uint64)
//
// Per-member squared distance via the dot-product identity:
//   ‖q-m‖² = qNorm + norms[i] − 2·(q·m)
//
// Why this kernel and not subtract-then-square: VPMADDWD on Haswell has
// 1-cycle throughput on port 0 and turns 16 int16 lanes into 8 int32
// partial sums of the form (a₀b₀+a₁b₁, a₂b₂+a₃b₃, …) in one instruction.
// That replaces the prior widen-subtract-multiply chain entirely. Memory
// bandwidth is also lower: one 32-byte member load per iteration vs the
// previous two overlapping 16-byte sign-extended loads.
//
// Q layout: padded to 16 int16 (last 2 lanes zero). Because q[14]=q[15]=0,
// the VPMADDWD pair for lanes (14,15) contributes 0 regardless of what's
// loaded for m[14..15] — so we can do a full 32-byte VMOVDQU per member
// even though member stride is only 28 bytes. The 4-byte overrun lands in
// the next member (or in the Labels region for the very last member); the
// mmap is one contiguous mapping so the read is page-safe.
//
// Overflow: each VPMADDWD lane = 2·(int16²) ≤ 2·32767² ≈ 2.147e9 < INT32_MAX
// by ~130k. Horizontal sum of 8 such lanes can reach ~1.7e10, so we widen to
// int64 before accumulating. Final distance is int64.
//
// Plan 9 VEX op order: (src2, src1, dst) → dst = src1 op src2.
TEXT ·memberScanAvx2(SB), NOSPLIT, $0-48
	MOVQ q+0(FP),       AX
	MOVQ members+8(FP), BX
	MOVQ norms+16(FP),  CX
	MOVQ qNorm+24(FP),  DX
	MOVQ out+32(FP),    R8
	MOVQ n+40(FP),      R9

	VMOVDQU (AX), Y0              // Y0 = q (16 int16, q[14]=q[15]=0)

	XORQ R10, R10                 // i = 0
loop:
	CMPQ R10, R9
	JE   done

	VMOVDQU (BX), Y1              // 32-byte member load (overruns 4B; lanes 14,15 zeroed by q)
	VPMADDWD Y0, Y1, Y2           // Y2 = 8 int32 partial sums: (q·m)_pair

	// Horizontal sum 8 int32 → 1 int64 via widen-then-add (avoid overflow).
	VPMOVSXDQ X2, Y3              // low 4 int32 → 4 int64 in Y3
	VEXTRACTI128 $1, Y2, X4
	VPMOVSXDQ X4, Y4              // high 4 int32 → 4 int64 in Y4
	VPADDQ Y4, Y3, Y3             // 4 int64 sums
	VEXTRACTI128 $1, Y3, X5
	VPADDQ X5, X3, X3             // 2 int64 sums in X3
	VPSHUFD $0x4E, X3, X6         // swap qwords of X3 → X6
	VPADDQ X6, X3, X3             // X3.qword[0] = full dot product
	VMOVQ X3, R11                 // R11 = dot

	// dist = qNorm + norms[i] − 2·dot
	MOVQ (CX), R12
	ADDQ DX, R12                  // R12 = qNorm + norms[i]
	SHLQ $1, R11                  // R11 = 2·dot
	SUBQ R11, R12                 // R12 = qNorm + norms[i] − 2·dot
	MOVQ R12, (R8)                // out[i] = dist

	ADDQ $28, BX                  // members += 14 int16
	ADDQ $8,  CX                  // norms   += 1 int64
	ADDQ $8,  R8                  // out     += 1 int64
	INCQ R10
	JMP  loop
done:
	VZEROUPPER
	RET
