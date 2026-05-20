package index

import "unsafe"

const (
	Magic   = "RIVF"
	Version = uint32(1)

	HeaderSize     = 8
	BucketDescSize = 48
	NumBuckets     = 16

	NCentroids = 512
	NProbe     = 64
	MaxNProbe  = 128
)

// BucketDesc describes one partition in the binary index.
// All offsets are absolute byte positions within index.bin.
// Stored at: HeaderSize + bucketID * BucketDescSize.
type BucketDesc struct {
	NDims       uint8
	_pad        [3]byte
	NCentroids  uint32
	NVectors    uint32
	_pad2       [4]byte
	CentroidsOff uint64 // float32[NCentroids * NDims]
	CellSzOff   uint64 // uint32[NCentroids]
	VectorsOff  uint64 // int8[NVectors * NDims], grouped by cell
	LabelsOff   uint64 // uint8[NVectors], 0=legit 1=fraud
}

// Compile-time assertion that BucketDesc is exactly BucketDescSize bytes.
var _ [0]struct{} = [unsafe.Sizeof(BucketDesc{}) - BucketDescSize]struct{}{}

// DimsForBucket returns which of the 14 dims are compared within a given bucket.
// Drops dims 9, 10, 11 (binary, constant within every bucket).
// Drops dims 5, 6 in no-history buckets (bit0=0) where they equal −1.
func DimsForBucket(bucketID int) []int {
	if bucketID&1 == 0 {
		return []int{0, 1, 2, 3, 4, 7, 8, 12, 13} // 9 dims
	}
	return []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13} // 11 dims
}

// QuantizeF32 converts a float32 in [0,1] to int8 in [0,127].
func QuantizeF32(v float32) int8 {
	q := int32(v*127 + 0.5)
	if q < 0 {
		return 0
	}
	if q > 127 {
		return 127
	}
	return int8(q)
}
