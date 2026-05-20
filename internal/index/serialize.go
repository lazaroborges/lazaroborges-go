package index

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"time"
)

// Build runs the full offline pipeline:
// load → partition → cluster → sort by cell → quantize → write index.bin.
func Build(inPath, outPath string) error {
	t0 := time.Now()

	fmt.Fprintf(os.Stderr, "[build] loading %s\n", inPath)
	buckets, err := LoadAndPartition(inPath)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}
	fmt.Fprintf(os.Stderr, "[build] loaded in %s\n", time.Since(t0).Round(time.Millisecond))

	rng := rand.New(rand.NewSource(42))

	type bucketSerialized struct {
		desc      BucketDesc
		centroids []byte // float32 LE
		cellSizes []byte // uint32 LE
		vectors   []byte // int8
		labels    []byte // uint8
	}

	serialized := make([]bucketSerialized, NumBuckets)

	for bi, b := range buckets {
		t1 := time.Now()
		fmt.Fprintf(os.Stderr, "[build] bucket %2d: %7d vectors — clustering...\n", bi, b.NVecs())

		k := NCentroids
		if b.NVecs() < k {
			k = b.NVecs()
		}

		cr := KMeans(b, k, 50, rng)

		var sortedVecs []float32
		var sortedLabels []bool
		if b.NVecs() > 0 {
			sortedVecs, sortedLabels = SortVectorsByCell(b, cr)
		}

		// Encode centroids as float32 LE
		centBytes := make([]byte, cr.NCentroids*b.NDims*4)
		for i, v := range cr.Centroids {
			binary.LittleEndian.PutUint32(centBytes[i*4:], math.Float32bits(v))
		}

		// Encode cell sizes as uint32 LE
		szBytes := make([]byte, cr.NCentroids*4)
		for i, sz := range cr.CellSizes {
			binary.LittleEndian.PutUint32(szBytes[i*4:], sz)
		}

		// Quantize and encode vectors as int8
		vecBytes := make([]byte, len(sortedVecs))
		for i, v := range sortedVecs {
			vecBytes[i] = byte(QuantizeF32(v))
		}

		// Encode labels as uint8 (0=legit, 1=fraud)
		lblBytes := make([]byte, len(sortedLabels))
		for i, fraud := range sortedLabels {
			if fraud {
				lblBytes[i] = 1
			}
		}

		serialized[bi] = bucketSerialized{
			desc: BucketDesc{
				NDims:      uint8(b.NDims),
				NCentroids: uint32(cr.NCentroids),
				NVectors:   uint32(b.NVecs()),
			},
			centroids: centBytes,
			cellSizes: szBytes,
			vectors:   vecBytes,
			labels:    lblBytes,
		}

		fmt.Fprintf(os.Stderr, "[build] bucket %2d done in %s\n", bi, time.Since(t1).Round(time.Millisecond))
	}

	// Compute absolute offsets for each bucket's data sections
	offset := uint64(HeaderSize + NumBuckets*BucketDescSize)
	for bi := range serialized {
		s := &serialized[bi]
		s.desc.CentroidsOff = offset
		offset += uint64(len(s.centroids))
		s.desc.CellSzOff = offset
		offset += uint64(len(s.cellSizes))
		s.desc.VectorsOff = offset
		offset += uint64(len(s.vectors))
		s.desc.LabelsOff = offset
		offset += uint64(len(s.labels))
	}

	fmt.Fprintf(os.Stderr, "[build] writing %s (%.1f MB)\n", outPath, float64(offset)/(1<<20))

	out, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer out.Close()

	// Header: magic + version
	if _, err := io.WriteString(out, Magic); err != nil {
		return err
	}
	if err := binary.Write(out, binary.LittleEndian, Version); err != nil {
		return err
	}

	// All bucket descriptors
	for bi := range serialized {
		if err := writeBucketDesc(out, serialized[bi].desc); err != nil {
			return fmt.Errorf("bucket desc %d: %w", bi, err)
		}
	}

	// Data sections in order: centroids, cellSizes, vectors, labels for each bucket
	for bi := range serialized {
		s := &serialized[bi]
		for _, section := range [][]byte{s.centroids, s.cellSizes, s.vectors, s.labels} {
			if _, err := out.Write(section); err != nil {
				return fmt.Errorf("bucket %d data: %w", bi, err)
			}
		}
	}

	fmt.Fprintf(os.Stderr, "[build] done in %s\n", time.Since(t0).Round(time.Second))
	return nil
}

func writeBucketDesc(w io.Writer, d BucketDesc) error {
	buf := make([]byte, BucketDescSize)
	buf[0] = d.NDims
	// buf[1:4] padding — zero
	binary.LittleEndian.PutUint32(buf[4:], d.NCentroids)
	binary.LittleEndian.PutUint32(buf[8:], d.NVectors)
	// buf[12:16] padding — zero
	binary.LittleEndian.PutUint64(buf[16:], d.CentroidsOff)
	binary.LittleEndian.PutUint64(buf[24:], d.CellSzOff)
	binary.LittleEndian.PutUint64(buf[32:], d.VectorsOff)
	binary.LittleEndian.PutUint64(buf[40:], d.LabelsOff)
	_, err := w.Write(buf)
	return err
}
