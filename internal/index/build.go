package index

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// RefRecord is one record from references.json.gz.
type RefRecord struct {
	Vector [14]float32 `json:"vector"`
	Label  string      `json:"label"` // "fraud" or "legit"
}

// BucketedData holds all vectors assigned to one bucket.
type BucketedData struct {
	BucketID int
	NDims    int
	Dims     []int     // which of the 14 dims are compared
	Vecs     []float32 // flat: len = NVecs * NDims
	Labels   []bool    // len = NVecs, true = fraud
}

func (b *BucketedData) NVecs() int { return len(b.Labels) }

// LoadAndPartition reads references.json.gz and partitions all records
// into 16 BucketedData structs (one per bucket_id).
func LoadAndPartition(path string) ([NumBuckets]*BucketedData, error) {
	f, err := os.Open(path)
	if err != nil {
		return [NumBuckets]*BucketedData{}, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return [NumBuckets]*BucketedData{}, err
	}
	defer gz.Close()

	var buckets [NumBuckets]*BucketedData
	for i := range buckets {
		dims := DimsForBucket(i)
		buckets[i] = &BucketedData{
			BucketID: i,
			NDims:    len(dims),
			Dims:     dims,
		}
	}

	dec := json.NewDecoder(gz)

	tok, err := dec.Token()
	if err != nil {
		return [NumBuckets]*BucketedData{}, err
	}
	if tok != json.Delim('[') {
		return [NumBuckets]*BucketedData{}, fmt.Errorf("expected '[', got %v", tok)
	}

	count := 0
	var rec RefRecord
	for dec.More() {
		if err := dec.Decode(&rec); err != nil {
			return [NumBuckets]*BucketedData{}, fmt.Errorf("record %d: %w", count, err)
		}

		bid := bucketIDFromVec(rec.Vector)
		b := buckets[bid]
		for _, d := range b.Dims {
			b.Vecs = append(b.Vecs, rec.Vector[d])
		}
		b.Labels = append(b.Labels, rec.Label == "fraud")

		count++
		if count%500_000 == 0 {
			fmt.Fprintf(os.Stderr, "  loaded %d records\n", count)
		}
	}

	if _, err := dec.Token(); err != nil && err != io.EOF {
		return [NumBuckets]*BucketedData{}, err
	}

	fmt.Fprintf(os.Stderr, "  loaded %d total records\n", count)
	return buckets, nil
}

// bucketIDFromVec computes the 4-bit bucket ID from a 14-dim reference vector.
// bit3=is_online, bit2=card_present, bit1=unknown_merchant, bit0=has_last_tx
func bucketIDFromVec(v [14]float32) int {
	var id int
	if v[9] >= 0.5 {
		id |= 8
	}
	if v[10] >= 0.5 {
		id |= 4
	}
	if v[11] >= 0.5 {
		id |= 2
	}
	if v[5] != -1 {
		id |= 1
	}
	return id
}
