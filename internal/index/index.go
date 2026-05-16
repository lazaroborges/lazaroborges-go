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
	MemberVecs     []int16   // nVectors × Dim, row-major (reordered)
	Labels         []uint8   // nVectors

	mmapData []byte
}

// Load mmaps `path` and verifies the header.
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
	if size < HeaderBytes {
		return nil, errors.New("index file too small")
	}
	data, err := unix.Mmap(int(f.Fd()), 0, size, unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	magic := binary.LittleEndian.Uint32(data[0:4])
	version := binary.LittleEndian.Uint32(data[4:8])
	nVectors := int(binary.LittleEndian.Uint32(data[8:12]))
	nClusters := int(binary.LittleEndian.Uint32(data[12:16]))
	dim := int(binary.LittleEndian.Uint32(data[16:20]))
	if magic != Magic || version != Version {
		_ = unix.Munmap(data)
		return nil, fmt.Errorf("bad index header: magic=%x version=%d", magic, version)
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

	return &Index{
		NVectors:        nVectors,
		NClusters:       nClusters,
		Centroids:       cent,
		CentroidsPadded: centPadded,
		CentroidNorms:   norms,
		ClusterOffsets:  offsets,
		MemberVecs:      mem,
		Labels:          lab,
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
