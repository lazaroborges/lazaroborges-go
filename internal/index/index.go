package index

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Index holds the IVF index in memory. All slices are zero-copy views into a
// single mmap'd region — the GC never moves them, the kernel pages them in on
// demand.
type Index struct {
	NVectors  int
	NClusters int

	Centroids       []float32 // nClusters × Dim, row-major (mmap view, kept for invariants)
	CentroidsPadded []float32 // nClusters × 16, row-major; last 2 lanes per row zeroed for AVX2 loads
	CentroidNorms   []float32 // length nClusters: ||c||² per centroid (computed once)
	ClusterOffsets []uint32  // length nClusters+1
	MemberVecs     []int16   // nVectors × Dim, row-major (reordered)  [v1 only]
	MemberNorms    []int64   // length nVectors: ||m||² per member (computed once for the dot-product
	                         // distance trick: ||q-m||² = ||q||² + ||m||² − 2·(q·m))  [v1 only]
	Labels         []uint8   // nVectors

	// New fields for v2 (zero/nil for v1 indices).
	Version         uint32
	CentroidsInt8   []int8       // K*16 padded
	MemberResiduals []int8       // N*16 padded
	HnswHeaders     []HnswHeader // length K
	HnswEdges       []byte       // packed edge block

	mmapData []byte
}

// Load mmaps `path`, reads the magic bytes, and dispatches to loadV1 or loadV2.
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open index: %w", err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat index: %w", err)
	}
	size := int(st.Size())
	if size < 4 {
		return nil, errors.New("index file too small")
	}
	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	magic := binary.LittleEndian.Uint32(data[0:4])
	switch magic {
	case Magic:
		return loadV1(data)
	case MagicV2:
		return loadV2(data)
	default:
		_ = unix.Munmap(data)
		return nil, fmt.Errorf("bad index header: magic=%x", magic)
	}
}

// loadV1 parses the v1 ('RIVF') index format from an already-mmap'd byte slice.
func loadV1(data []byte) (*Index, error) {
	size := len(data)
	if size < HeaderBytes {
		return nil, errors.New("index file too small")
	}

	version := binary.LittleEndian.Uint32(data[4:8])
	nVectors := int(binary.LittleEndian.Uint32(data[8:12]))
	nClusters := int(binary.LittleEndian.Uint32(data[12:16]))
	dim := int(binary.LittleEndian.Uint32(data[16:20]))
	if version != Version {
		_ = unix.Munmap(data)
		return nil, fmt.Errorf("bad index header: version=%d", version)
	}
	if dim != Dim {
		_ = unix.Munmap(data)
		return nil, fmt.Errorf("dim mismatch: got %d want %d", dim, Dim)
	}

	off := HeaderBytes

	centBytes := nClusters * Dim * 4
	cent := unsafe.Slice((*float32)(unsafe.Pointer(&data[off])), nClusters*Dim)
	off += centBytes

	offsetsBytes := (nClusters + 1) * 4
	offsets := unsafe.Slice((*uint32)(unsafe.Pointer(&data[off])), nClusters+1)
	off += offsetsBytes

	memBytes := nVectors * Dim * 2
	mem := unsafe.Slice((*int16)(unsafe.Pointer(&data[off])), nVectors*Dim)
	off += memBytes

	lab := unsafe.Slice((*uint8)(unsafe.Pointer(&data[off])), nVectors)
	off += nVectors

	if off > size {
		_ = unix.Munmap(data)
		return nil, errors.New("index file truncated")
	}

	// Advise the kernel that we'll touch the whole thing — speeds up first
	// few queries by avoiding cold-page latency.
	_ = unix.Madvise(data, unix.MADV_WILLNEED)

	// Pre-compute squared centroid norms. ||q-c||² = ||q||² + ||c||² - 2·q·c
	// lets the hot search loop do a dot product (cheaper) instead of subtract-
	// then-square. The float32 norms cost nClusters·4 bytes (~16KB), trivial.
	norms := make([]float32, nClusters)
	for c := 0; c < nClusters; c++ {
		base := c * Dim
		var n float32
		for i := 0; i < Dim; i++ {
			n += cent[base+i] * cent[base+i]
		}
		norms[c] = n
	}

	// 16-float padded copy of the centroids. The AVX2 dot-product kernel
	// reads 32 bytes from each side, so the source layout has to provide
	// 8 valid + 8 zero (or 14 valid + 2 zero) per row. ~256 KB at 4096
	// clusters — small enough to stay in L2 and avoid TLB pressure on
	// the centroid pass.
	centPadded := make([]float32, nClusters*16)
	for c := 0; c < nClusters; c++ {
		copy(centPadded[c*16:c*16+Dim], cent[c*Dim:(c+1)*Dim])
	}

	// Per-member squared norms for the dot-product distance kernel. One-shot
	// at startup; 3M × 8 bytes = 24 MB heap, well within the container limit.
	// Reading int16 from the mmap view and accumulating to int64 sidesteps the
	// overflow concern at large quantization scales.
	memNorms := make([]int64, nVectors)
	for i := 0; i < nVectors; i++ {
		base := i * Dim
		var s int64
		for d := 0; d < Dim; d++ {
			v := int64(mem[base+d])
			s += v * v
		}
		memNorms[i] = s
	}

	return &Index{
		NVectors:        nVectors,
		NClusters:       nClusters,
		Version:         Version,
		Centroids:       cent,
		CentroidsPadded: centPadded,
		CentroidNorms:   norms,
		ClusterOffsets:  offsets,
		MemberVecs:      mem,
		MemberNorms:     memNorms,
		Labels:          lab,
		mmapData:        data,
	}, nil
}

// loadV2 parses the v2 ('RIVH') IVF-HNSW index format from an already-mmap'd byte slice.
func loadV2(data []byte) (*Index, error) {
	if len(data) < HeaderBytesV2 {
		return nil, errors.New("v2 index file too small")
	}
	version := binary.LittleEndian.Uint32(data[4:8])
	if version != VersionV2 {
		return nil, fmt.Errorf("v2 unsupported version %d", version)
	}
	nVectors := int(binary.LittleEndian.Uint32(data[8:12]))
	nClusters := int(binary.LittleEndian.Uint32(data[12:16]))
	dim := int(binary.LittleEndian.Uint32(data[16:20]))
	if dim != Dim {
		return nil, fmt.Errorf("v2 dim mismatch: got %d want %d", dim, Dim)
	}

	off := HeaderBytesV2

	cent := unsafeFloat32Slice(data, off, nClusters*Dim)
	off += nClusters * Dim * 4

	centPadded := unsafeFloat32Slice(data, off, nClusters*16)
	off += nClusters * 16 * 4

	centInt8 := unsafeInt8Slice(data, off, nClusters*16)
	off += nClusters * 16

	centNorms := unsafeFloat32Slice(data, off, nClusters)
	off += nClusters * 4

	offsets := unsafeUint32Slice(data, off, nClusters+1)
	off += (nClusters + 1) * 4

	residuals := unsafeInt8Slice(data, off, nVectors*16)
	off += nVectors * 16

	labels := unsafeUint8Slice(data, off, nVectors)
	off += nVectors

	headers := make([]HnswHeader, nClusters)
	for c := 0; c < nClusters; c++ {
		base := off + c*HeaderSizeV2
		headers[c].EntryPoint = binary.LittleEndian.Uint16(data[base:])
		headers[c].MaxLevel = data[base+2]
		// data[base+3] is pad byte — skip
		headers[c].EdgeOffset = binary.LittleEndian.Uint32(data[base+4:])
		for i := 0; i < 8; i++ {
			headers[c].LevelCount[i] = binary.LittleEndian.Uint32(data[base+8+i*4:])
		}
	}
	off += nClusters * HeaderSizeV2

	edges := data[off:]

	_ = unix.Madvise(data, unix.MADV_WILLNEED)

	return &Index{
		NVectors:        nVectors,
		NClusters:       nClusters,
		Version:         VersionV2,
		Centroids:       cent,
		CentroidsPadded: centPadded,
		CentroidsInt8:   centInt8,
		CentroidNorms:   centNorms,
		ClusterOffsets:  offsets,
		MemberResiduals: residuals,
		Labels:          labels,
		HnswHeaders:     headers,
		HnswEdges:       edges,
		mmapData:        data,
	}, nil
}

// Close releases the mmap. Safe to skip in tests but important in test cleanup.
func (idx *Index) Close() error {
	if idx.mmapData != nil {
		err := unix.Munmap(idx.mmapData)
		idx.mmapData = nil
		return err
	}
	return nil
}

// unsafeFloat32Slice returns a []float32 aliasing data[off : off+n*4].
func unsafeFloat32Slice(data []byte, off, n int) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[off])), n)
}

// unsafeInt8Slice returns a []int8 aliasing data[off : off+n].
func unsafeInt8Slice(data []byte, off, n int) []int8 {
	return unsafe.Slice((*int8)(unsafe.Pointer(&data[off])), n)
}

// unsafeUint32Slice returns a []uint32 aliasing data[off : off+n*4].
func unsafeUint32Slice(data []byte, off, n int) []uint32 {
	return unsafe.Slice((*uint32)(unsafe.Pointer(&data[off])), n)
}

// unsafeUint8Slice returns a []uint8 aliasing data[off : off+n].
func unsafeUint8Slice(data []byte, off, n int) []uint8 {
	return unsafe.Slice((*uint8)(unsafe.Pointer(&data[off])), n)
}
