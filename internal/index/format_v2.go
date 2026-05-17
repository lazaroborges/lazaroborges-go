package index

// V2 index format. Magic bytes are 'RIVH' (Rinha IVF-HNSW); version is 2.
// See docs/superpowers/specs/2026-05-17-ivf-hnsw-design.md for the full
// binary layout description.

const (
	MagicV2       uint32 = 0x52495648 // 'RIVH'
	VersionV2     uint32 = 2
	HeaderBytesV2        = 48
	HnswMaxLevel         = 7 // global cap on per-node level

	// Per-EdgeSlot byte size for M=6: degree(1) + pad(1) + neighbors(6*2) = 14.
	// Helpers compute this from g.M, but we pin M=6 in the shipped index.
)

// HnswHeader is the per-cluster header. 40 bytes packed.
type HnswHeader struct {
	EntryPoint uint16
	MaxLevel   uint8
	_pad       uint8
	EdgeOffset uint32 // byte offset into the HnswEdges block
	LevelCount [8]uint32
}

// HeaderSizeV2 is the byte size of HnswHeader on disk. Must stay synced.
const HeaderSizeV2 = 2 + 1 + 1 + 4 + 32 // = 40
