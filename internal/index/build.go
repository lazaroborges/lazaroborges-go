package index

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
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

// ClusterResult holds k-means output for one bucket.
type ClusterResult struct {
	Centroids  []float32 // flat: NCentroids * NDims (float32 for query-time distance)
	Assignments []int32  // which centroid each vector belongs to
	CellSizes  []uint32  // number of vectors per centroid
	NDims      int
	NCentroids int
}

// KMeans runs k-means with random initialization on b's vectors.
func KMeans(b *BucketedData, k, maxIter int, rng *rand.Rand) ClusterResult {
	n := b.NVecs()
	nDims := b.NDims

	if n == 0 {
		return ClusterResult{NDims: nDims, NCentroids: 0}
	}
	if n < k {
		k = n
	}

	// Random init: pick k distinct vector indices
	centroids := make([]float32, k*nDims)
	perm := rng.Perm(n)
	for i := 0; i < k; i++ {
		copy(centroids[i*nDims:], b.Vecs[perm[i]*nDims:perm[i]*nDims+nDims])
	}

	assignments := make([]int32, n)
	counts := make([]int32, k)

	for iter := 0; iter < maxIter; iter++ {
		// Assignment step
		changed := false
		for vi := 0; vi < n; vi++ {
			vec := b.Vecs[vi*nDims : vi*nDims+nDims]
			best := int32(0)
			bestDist := sqDistF32(vec, centroids[:nDims])
			for ci := 1; ci < k; ci++ {
				d := sqDistF32(vec, centroids[ci*nDims:ci*nDims+nDims])
				if d < bestDist {
					bestDist = d
					best = int32(ci)
				}
			}
			if assignments[vi] != best {
				assignments[vi] = best
				changed = true
			}
		}
		if !changed {
			break
		}

		// Update step: recompute centroids as mean of assigned vectors
		newC := make([]float32, k*nDims)
		for i := range counts {
			counts[i] = 0
		}
		for vi := 0; vi < n; vi++ {
			ci := int(assignments[vi])
			counts[ci]++
			vec := b.Vecs[vi*nDims : vi*nDims+nDims]
			dst := newC[ci*nDims : ci*nDims+nDims]
			for d := range dst {
				dst[d] += vec[d]
			}
		}
		for ci := 0; ci < k; ci++ {
			if counts[ci] == 0 {
				// Re-seed empty centroid from a random vector
				ri := rng.Intn(n)
				copy(newC[ci*nDims:], b.Vecs[ri*nDims:ri*nDims+nDims])
				continue
			}
			inv := 1.0 / float32(counts[ci])
			dst := newC[ci*nDims : ci*nDims+nDims]
			for d := range dst {
				dst[d] *= inv
			}
		}
		centroids = newC
	}

	cellSizes := make([]uint32, k)
	for _, a := range assignments {
		cellSizes[a]++
	}

	return ClusterResult{
		Centroids:   centroids,
		Assignments: assignments,
		CellSizes:   cellSizes,
		NDims:       nDims,
		NCentroids:  k,
	}
}

// SortVectorsByCell reorders Vecs and Labels so that cell 0 comes first, then 1, etc.
func SortVectorsByCell(b *BucketedData, cr ClusterResult) (sortedVecs []float32, sortedLabels []bool) {
	n := b.NVecs()
	nDims := b.NDims

	cellStart := make([]int, cr.NCentroids+1)
	for i, sz := range cr.CellSizes {
		cellStart[i+1] = cellStart[i] + int(sz)
	}

	sortedVecs = make([]float32, n*nDims)
	sortedLabels = make([]bool, n)
	pos := make([]int, cr.NCentroids)
	copy(pos, cellStart[:cr.NCentroids])

	for vi := 0; vi < n; vi++ {
		ci := int(cr.Assignments[vi])
		wp := pos[ci]
		copy(sortedVecs[wp*nDims:], b.Vecs[vi*nDims:vi*nDims+nDims])
		sortedLabels[wp] = b.Labels[vi]
		pos[ci]++
	}

	return sortedVecs, sortedLabels
}

func sqDistF32(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
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
