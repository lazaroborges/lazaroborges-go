// preprocess decompresses references.json.gz, runs mini-batch k-means to
// produce N centroids, assigns every vector to its nearest centroid, quantizes
// the data into int16, and writes the packed index.bin consumed at runtime by
// the API.
//
// Designed to run inside the Docker build, where time is forgiving but memory
// matters. With 3M vectors × 14 floats = ~168 MB transient state, we keep the
// working set in float32 and only quantize when writing the final file.
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

const dim = 14

func main() {
	src := flag.String("src", "/data/references.json.gz", "gzipped references dataset")
	dst := flag.String("dst", "/index.bin", "output index file")
	nClusters := flag.Int("clusters", 1024, "k-means cluster count")
	iters := flag.Int("iters", 25, "k-means iterations")
	batch := flag.Int("batch", 200000, "mini-batch sample size per iteration")
	seed := flag.Int64("seed", 42, "RNG seed")
	flag.Parse()

	t0 := time.Now()
	log.Printf("loading %s", *src)
	vecs, labels, err := loadDataset(*src)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	n := len(vecs) / dim
	log.Printf("loaded %d vectors in %s", n, time.Since(t0))

	rng := rand.New(rand.NewSource(*seed))

	// Sampled k-means++ init: full k-means++ over 3M × 4096 is prohibitive,
	// but k-means++ over a fixed sample gives a much better starting point
	// than uniform random while staying affordable.
	log.Printf("sampled k-means++ init (k=%d)", *nClusters)
	t0 = time.Now()
	centroids := sampledKMeansPlusPlusInit(vecs, n, *nClusters, 50000, rng)
	log.Printf("init done in %s", time.Since(t0))

	// Mini-batch k-means: each iteration samples `batch` points, assigns them
	// to nearest centroid, and updates centroids with a learning rate per
	// cluster (1/count). Far cheaper than full Lloyd's at 3M points.
	log.Printf("mini-batch k-means: %d iters, batch=%d", *iters, *batch)
	t0 = time.Now()
	miniBatchKMeans(vecs, n, centroids, *nClusters, *iters, *batch, rng)
	log.Printf("training done in %s", time.Since(t0))

	// Final full assignment so every point lands in its current-nearest cell.
	log.Printf("final assignment pass")
	t0 = time.Now()
	assignments := assignAll(vecs, n, centroids, *nClusters)
	log.Printf("assignment done in %s", time.Since(t0))

	// Group by cluster, write index.
	log.Printf("writing %s", *dst)
	t0 = time.Now()
	if err := writeIndex(*dst, vecs, labels, centroids, assignments, *nClusters); err != nil {
		log.Fatalf("write: %v", err)
	}
	log.Printf("wrote index in %s", time.Since(t0))
}

// reference matches one element of references.json.gz.
type reference struct {
	Vector [dim]float32 `json:"vector"`
	Label  string       `json:"label"`
}

// loadDataset streams the gzipped JSON array and returns flat float32 vectors
// (row-major, length n*dim) and per-vector labels.
func loadDataset(path string) ([]float32, []uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, err
	}
	defer gz.Close()

	dec := json.NewDecoder(bufio.NewReaderSize(gz, 1<<20))
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '[' {
		return nil, nil, fmt.Errorf("expected array, got %v", tok)
	}

	vecs := make([]float32, 0, 3_000_000*dim)
	labels := make([]uint8, 0, 3_000_000)
	var entry reference
	for dec.More() {
		if err := dec.Decode(&entry); err != nil {
			return nil, nil, err
		}
		vecs = append(vecs, entry.Vector[:]...)
		var lab uint8 = 0
		if entry.Label == "fraud" {
			lab = 1
		}
		labels = append(labels, lab)
	}
	return vecs, labels, nil
}

// sampledKMeansPlusPlusInit runs full k-means++ over a uniform random sample
// of `sampleSize` points. Output is k centroids drawn from that sample (each
// is an actual data point), which by k-means++ properties gives an
// O(log k)-competitive starting point with the optimal clustering of the
// sample. Cost scales with sampleSize × k instead of n × k.
func sampledKMeansPlusPlusInit(vecs []float32, n, k, sampleSize int, rng *rand.Rand) []float32 {
	if sampleSize > n {
		sampleSize = n
	}
	// Pick a sample by reservoir-style index list.
	sample := make([]float32, sampleSize*dim)
	for i := 0; i < sampleSize; i++ {
		src := rng.Intn(n)
		copy(sample[i*dim:(i+1)*dim], vecs[src*dim:(src+1)*dim])
	}

	centroids := make([]float32, k*dim)
	first := rng.Intn(sampleSize)
	copy(centroids[0:dim], sample[first*dim:(first+1)*dim])

	minDist := make([]float32, sampleSize)
	for i := range minDist {
		minDist[i] = float32(math.MaxFloat32)
	}
	updateMinDistSeq(sample, sampleSize, centroids[0:dim], minDist)

	for c := 1; c < k; c++ {
		var total float64
		for _, d := range minDist {
			total += float64(d)
		}
		if total == 0 {
			idx := rng.Intn(sampleSize)
			copy(centroids[c*dim:(c+1)*dim], sample[idx*dim:(idx+1)*dim])
			continue
		}
		target := rng.Float64() * total
		var acc float64
		idx := sampleSize - 1
		for i, d := range minDist {
			acc += float64(d)
			if acc >= target {
				idx = i
				break
			}
		}
		copy(centroids[c*dim:(c+1)*dim], sample[idx*dim:(idx+1)*dim])
		updateMinDistSeq(sample, sampleSize, centroids[c*dim:(c+1)*dim], minDist)
	}
	return centroids
}

func updateMinDistSeq(vecs []float32, n int, centroid []float32, minDist []float32) {
	for i := 0; i < n; i++ {
		d := sqDist(vecs[i*dim:(i+1)*dim], centroid)
		if d < minDist[i] {
			minDist[i] = d
		}
	}
}

func sqDist(a, b []float32) float32 {
	var s float32
	for i := 0; i < dim; i++ {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// miniBatchKMeans runs Sculley-style mini-batch k-means.
func miniBatchKMeans(vecs []float32, n int, centroids []float32, k, iters, batch int, rng *rand.Rand) {
	counts := make([]int64, k)
	idxs := make([]int, batch)
	assignments := make([]int, batch)

	for it := 0; it < iters; it++ {
		// Sample without replacement: shuffle a prefix.
		for i := 0; i < batch; i++ {
			idxs[i] = rng.Intn(n)
		}

		// Assign batch points to nearest centroid (parallel).
		assignBatch(vecs, idxs, centroids, k, assignments)

		// Update centroids one point at a time (Sculley 2010): for each
		// (point, cluster) pair, c ← (1 - 1/count) * c + (1/count) * point.
		for bi, p := range idxs {
			c := assignments[bi]
			counts[c]++
			lr := 1.0 / float32(counts[c])
			cBase := c * dim
			pBase := p * dim
			for d := 0; d < dim; d++ {
				centroids[cBase+d] = (1-lr)*centroids[cBase+d] + lr*vecs[pBase+d]
			}
		}
		if (it+1)%5 == 0 || it == iters-1 {
			log.Printf("  iter %d/%d", it+1, iters)
		}
	}
}

func assignBatch(vecs []float32, idxs []int, centroids []float32, k int, out []int) {
	workers := runtime.NumCPU()
	chunk := (len(idxs) + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > len(idxs) {
			end = len(idxs)
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for bi := s; bi < e; bi++ {
				p := idxs[bi]
				out[bi] = nearest(vecs[p*dim:(p+1)*dim], centroids, k)
			}
		}(start, end)
	}
	wg.Wait()
}

func nearest(v []float32, centroids []float32, k int) int {
	best := 0
	bestD := float32(math.MaxFloat32)
	for c := 0; c < k; c++ {
		base := c * dim
		var d float32
		for i := 0; i < dim; i++ {
			diff := v[i] - centroids[base+i]
			d += diff * diff
		}
		if d < bestD {
			bestD = d
			best = c
		}
	}
	return best
}

// assignAll returns the cluster id of every vector. Parallel over workers.
func assignAll(vecs []float32, n int, centroids []float32, k int) []uint32 {
	out := make([]uint32, n)
	var done int64
	workers := runtime.NumCPU()
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > n {
			end = n
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				out[i] = uint32(nearest(vecs[i*dim:(i+1)*dim], centroids, k))
				if atomic.AddInt64(&done, 1)%500000 == 0 {
					log.Printf("  assigned %d/%d", done, n)
				}
			}
		}(start, end)
	}
	wg.Wait()
	return out
}

// writeIndex emits the index.bin in the format documented in index/format.go.
func writeIndex(path string, vecs []float32, labels []uint8, centroids []float32, assignments []uint32, k int) error {
	n := len(labels)

	// Build per-cluster member lists.
	clusterLists := make([][]int, k)
	for i, c := range assignments {
		clusterLists[c] = append(clusterLists[c], i)
	}

	// Empty clusters break IVF — they reduce effective `nprobe` and bias
	// retrieval. Move a few outliers from oversized clusters to empties.
	splitEmptyClusters(clusterLists, vecs, centroids)

	// Sort cluster contents by original index for deterministic builds.
	for c := range clusterLists {
		sort.Ints(clusterLists[c])
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	defer func() {
		_ = w.Flush()
		_ = f.Close()
	}()

	// Header.
	hdr := make([]byte, 32)
	binary.LittleEndian.PutUint32(hdr[0:4], 0x52495646) // 'RIVF'
	binary.LittleEndian.PutUint32(hdr[4:8], 1)
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(n))
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(k))
	binary.LittleEndian.PutUint32(hdr[16:20], uint32(dim))
	if _, err := w.Write(hdr); err != nil {
		return err
	}

	// Centroids: k × dim × float32.
	centBuf := make([]byte, k*dim*4)
	for i, v := range centroids {
		binary.LittleEndian.PutUint32(centBuf[i*4:i*4+4], math.Float32bits(v))
	}
	if _, err := w.Write(centBuf); err != nil {
		return err
	}

	// Cluster offsets: prefix sums.
	offsets := make([]uint32, k+1)
	for c := 0; c < k; c++ {
		offsets[c+1] = offsets[c] + uint32(len(clusterLists[c]))
	}
	offBuf := make([]byte, (k+1)*4)
	for i, v := range offsets {
		binary.LittleEndian.PutUint32(offBuf[i*4:i*4+4], v)
	}
	if _, err := w.Write(offBuf); err != nil {
		return err
	}

	// Member vectors, reordered into cluster groups, quantized to int16.
	memBuf := make([]byte, dim*2)
	for c := 0; c < k; c++ {
		for _, origIdx := range clusterLists[c] {
			base := origIdx * dim
			for d := 0; d < dim; d++ {
				v := vecs[base+d] * 32767
				if v > 32767 {
					v = 32767
				} else if v < -32767 {
					v = -32767
				}
				binary.LittleEndian.PutUint16(memBuf[d*2:d*2+2], uint16(int16(v)))
			}
			if _, err := w.Write(memBuf); err != nil {
				return err
			}
		}
	}

	// Labels in the same cluster-reordered order.
	labBuf := make([]byte, 4096)
	off := 0
	for c := 0; c < k; c++ {
		for _, origIdx := range clusterLists[c] {
			labBuf[off] = labels[origIdx]
			off++
			if off == len(labBuf) {
				if _, err := w.Write(labBuf); err != nil {
					return err
				}
				off = 0
			}
		}
	}
	if off > 0 {
		if _, err := w.Write(labBuf[:off]); err != nil {
			return err
		}
	}

	return nil
}

// splitEmptyClusters reassigns the most-distant member of the largest cluster
// into each empty cluster, until none remain. Cheap heuristic to avoid the
// pathology where a few centroids end up dead.
func splitEmptyClusters(clusters [][]int, vecs, centroids []float32) {
	for c, list := range clusters {
		if len(list) > 0 {
			continue
		}
		// Find the largest cluster.
		big := 0
		for i := range clusters {
			if len(clusters[i]) > len(clusters[big]) {
				big = i
			}
		}
		if len(clusters[big]) < 2 {
			continue
		}
		// Move the member farthest from its own centroid.
		bigCentroid := centroids[big*dim : (big+1)*dim]
		farIdx := 0
		farD := float32(-1)
		for j, pi := range clusters[big] {
			d := sqDist(vecs[pi*dim:(pi+1)*dim], bigCentroid)
			if d > farD {
				farD = d
				farIdx = j
			}
		}
		moved := clusters[big][farIdx]
		clusters[big] = append(clusters[big][:farIdx], clusters[big][farIdx+1:]...)
		clusters[c] = append(clusters[c], moved)
		// Update the centroid to land on the moved point (so future queries
		// route there).
		copy(centroids[c*dim:(c+1)*dim], vecs[moved*dim:(moved+1)*dim])
	}
}
