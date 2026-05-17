//go:build !amd64

package index

import "unsafe"

// int16SqDist14 fallback for non-amd64. Pure-Go reference; the amd64 build
// uses an SSE2/SSSE3 assembly version (see distance_amd64.s).
func int16SqDist14(q *[Dim]int16, m *int16) int32 {
	mp := (*[Dim]int16)(unsafe.Pointer(m))
	d0 := int32(q[0]) - int32(mp[0])
	d1 := int32(q[1]) - int32(mp[1])
	d2 := int32(q[2]) - int32(mp[2])
	d3 := int32(q[3]) - int32(mp[3])
	d4 := int32(q[4]) - int32(mp[4])
	d5 := int32(q[5]) - int32(mp[5])
	d6 := int32(q[6]) - int32(mp[6])
	d7 := int32(q[7]) - int32(mp[7])
	d8 := int32(q[8]) - int32(mp[8])
	d9 := int32(q[9]) - int32(mp[9])
	d10 := int32(q[10]) - int32(mp[10])
	d11 := int32(q[11]) - int32(mp[11])
	d12 := int32(q[12]) - int32(mp[12])
	d13 := int32(q[13]) - int32(mp[13])
	sum := int64(d0)*int64(d0) + int64(d1)*int64(d1) +
		int64(d2)*int64(d2) + int64(d3)*int64(d3) +
		int64(d4)*int64(d4) + int64(d5)*int64(d5) +
		int64(d6)*int64(d6) + int64(d7)*int64(d7) +
		int64(d8)*int64(d8) + int64(d9)*int64(d9) +
		int64(d10)*int64(d10) + int64(d11)*int64(d11) +
		int64(d12)*int64(d12) + int64(d13)*int64(d13)
	if sum > 0x7fffffff {
		return 0x7fffffff
	}
	return int32(sum)
}

func memberScanAvx2(q *[Dim]int16, members *int16, out *int32, n uint64) {
	for i := uint64(0); i < n; i++ {
		mp := (*int16)(unsafe.Pointer(uintptr(unsafe.Pointer(members)) + uintptr(i)*Dim*2))
		op := (*int32)(unsafe.Pointer(uintptr(unsafe.Pointer(out)) + uintptr(i)*4))
		*op = int16SqDist14(q, mp)
	}
}
