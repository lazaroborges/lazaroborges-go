package index

// Binary layout of index.bin. Written by cmd/preprocess, mmap-read at startup.
//
//	+-------------------------------------------------------------+
//	| Header (32 bytes)                                           |
//	|   magic        uint32  = 'RIVF' (0x52495646)                |
//	|   version      uint32  = 1                                  |
//	|   nVectors     uint32                                       |
//	|   nClusters    uint32                                       |
//	|   dim          uint32  = 14                                 |
//	|   _pad         uint32  × 3                                  |
//	+-------------------------------------------------------------+
//	| Centroids:  nClusters × Dim × float32                       |
//	+-------------------------------------------------------------+
//	| ClusterOffsets:  (nClusters + 1) × uint32                   |
//	|   offsets[i]..offsets[i+1] = vector indices in cluster i    |
//	+-------------------------------------------------------------+
//	| MemberVecs:  nVectors × Dim × int16  (quantized, reordered  |
//	|              into cluster groups — index `j` here is unrelated   |
//	|              to the original reference index)               |
//	+-------------------------------------------------------------+
//	| Labels:  nVectors × uint8  (0 = legit, 1 = fraud), in the   |
//	|          same cluster-reordered order as MemberVecs         |
//	+-------------------------------------------------------------+

const (
	Magic        uint32 = 0x52495646 // 'RIVF'
	Version      uint32 = 1
	HeaderBytes         = 32
	Dim                 = 14

	LabelLegit uint8 = 0
	LabelFraud uint8 = 1
)
