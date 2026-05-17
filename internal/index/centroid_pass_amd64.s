//go:build amd64

#include "textflag.h"

// func centroidPassAvx2(q *[16]float32, cents *float32, norms *float32, out *CentroidDist, n uint64)
//
// Inlines the centroid distance loop in a single ASM call:
//   for c := 0; c < n; c++ {
//       dot = q · cents[c]
//       out[c] = { Cluster: c, Dist: norms[c] - 2*dot }
//   }
//
// Saves the Go→ASM boundary cost paid per centroid by the dot14Avx2 stub:
// ~10-20ns × n. At n=1024 that's 10-20µs of pure overhead reclaimed.
//
// Layout: cents is row-major nClusters × 16 float32 (padded; lanes 14, 15
// zero per row). norms is nClusters float32. out is a *CentroidDist array
// where CentroidDist = { Cluster uint32; Dist float32 } (8 bytes each).
//
// Plan 9 VEX order: (src2, src1, dst) → dst = src1 op src2.
TEXT ·centroidPassAvx2(SB), NOSPLIT, $0-40
	MOVQ q+0(FP), AX
	MOVQ cents+8(FP), BX
	MOVQ norms+16(FP), CX
	MOVQ out+24(FP), DX
	MOVQ n+32(FP), R8

	VMOVUPS (AX), Y0           // Y0 = q[0..7]   (kept across iterations)
	VMOVUPS 32(AX), Y1         // Y1 = q[8..15]  (kept across iterations)

	XORQ R9, R9                // c = 0
loop:
	CMPQ R9, R8
	JE   done

	// dot = q · cents[c]  using two 256-bit lanes
	VMULPS  (BX), Y0, Y2       // Y2 = q[0..7] * cents[c*16+0..7]
	VMOVUPS 32(BX), Y3
	VMULPS  Y3, Y1, Y3         // Y3 = q[8..15] * cents[c*16+8..15]
	VADDPS  Y3, Y2, Y2         // Y2 = 8 lane partial sums

	// Horizontal sum of 8 floats → scalar in X2
	VEXTRACTF128 $1, Y2, X3
	VADDPS       X3, X2, X2    // 4 lanes
	VHADDPS      X2, X2, X2    // 2 lanes
	VHADDPS      X2, X2, X2    // 1 lane (X2[0] = dot)

	// dist = norms[c] - 2*dot   (rank-equivalent; ||q||² omitted as constant)
	VMOVSS (CX), X4            // X4 = norms[c]
	VSUBSS X2, X4, X4          // X4 = norm - dot
	VSUBSS X2, X4, X4          // X4 = norm - 2*dot

	// Store CentroidDist{Cluster: uint32(c), Dist: dist}
	MOVL  R9, (DX)             // Cluster (low 32 bits of c)
	VMOVSS X4, 4(DX)           // Dist

	// Advance pointers
	ADDQ $64, BX               // cents += 16 float32 = 64 bytes
	ADDQ $4, CX                // norms += 1 float32  = 4 bytes
	ADDQ $8, DX                // out   += 1 CentroidDist = 8 bytes
	INCQ R9
	JMP  loop
done:
	VZEROUPPER
	RET
