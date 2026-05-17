// accuracy is a development-only probe: loads index.bin + test-data.json,
// runs the full handler pipeline in-process, and prints FP/FN counts. Lets
// us iterate on nprobe / cluster count without paying the docker + k6
// round-trip on every change. NOT a replacement for run.sh — k6 is the
// authoritative validation; this just gates accuracy before we get there.
//
// Supports both v1 (IVF + int16) and v2 (IVF-HNSW + int8 residuals) index
// formats. Use -search=ivf for v1 and -search=ivfhnsw for v2.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/index"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/vector"
)

type entry struct {
	Request          json.RawMessage `json:"request"`
	ExpectedApproved bool            `json:"expected_approved"`
}

type testFile struct {
	Entries []entry `json:"entries"`
}

func main() {
	indexPath  := flag.String("index", "tmp/index.bin", "path to index file")
	testPath   := flag.String("test", "data/test-data.json", "path to test-data.json")
	limit      := flag.Int("limit", 0, "process at most N entries (0=all)")
	baseN      := flag.Int("nprobe", 2, "base nprobe for v2 (or base for v1)")
	retryN     := flag.Int("retry-nprobe", 4, "retry nprobe for ambiguous queries")
	ef         := flag.Int("ef", 32, "ef for HNSW search (v2 only)")
	retryEF    := flag.Int("retry-ef", 64, "retry ef for ambiguous queries (v2 only)")
	searchMode := flag.String("search", "ivfhnsw", "search mode: ivf or ivfhnsw")
	flag.Parse()

	idx, err := index.Load(*indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}
	defer idx.Close()
	log.Printf("loaded index: %d vectors, %d clusters (version %d)", idx.NVectors, idx.NClusters, idx.Version)

	// Validate that search mode matches index version.
	switch *searchMode {
	case "ivfhnsw":
		if idx.Version != index.VersionV2 {
			log.Fatalf("search=ivfhnsw requires a v2 index (version %d), but loaded version %d",
				index.VersionV2, idx.Version)
		}
	case "ivf":
		if idx.Version != index.Version {
			log.Fatalf("search=ivf requires a v1 index (version %d), but loaded version %d",
				index.Version, idx.Version)
		}
	default:
		log.Fatalf("unknown search mode %q; use ivf or ivfhnsw", *searchMode)
	}

	f, err := os.Open(*testPath)
	if err != nil {
		log.Fatalf("open test: %v", err)
	}
	defer f.Close()
	var tf testFile
	if err := json.NewDecoder(f).Decode(&tf); err != nil {
		log.Fatalf("decode: %v", err)
	}
	n := len(tf.Entries)
	if *limit > 0 && *limit < n {
		n = *limit
	}
	log.Printf("evaluating %d entries with search=%s nprobe=%d retry-nprobe=%d ef=%d retry-ef=%d",
		n, *searchMode, *baseN, *retryN, *ef, *retryEF)

	var tp, tn, fp, fn, errs int
	t0 := time.Now()

	switch *searchMode {
	case "ivf":
		runV1(idx, tf.Entries[:n], *baseN, *retryN, &tp, &tn, &fp, &fn, &errs)
	case "ivfhnsw":
		runV2(idx, tf.Entries[:n], *baseN, *retryN, *ef, *retryEF, &tp, &tn, &fp, &fn, &errs)
	}

	dur := time.Since(t0)
	perReq := dur / time.Duration(n)

	weightedE := fp*1 + fn*3 + errs*5
	failureRate := float64(fp+fn+errs) / float64(n)
	fmt.Printf("\n=== detection summary ===\n")
	fmt.Printf("search mode        %s\n", *searchMode)
	fmt.Printf("entries            %d\n", n)
	fmt.Printf("tp (caught fraud)  %d\n", tp)
	fmt.Printf("tn (passed legit)  %d\n", tn)
	fmt.Printf("fp (blocked legit) %d\n", fp)
	fmt.Printf("fn (missed fraud)  %d\n", fn)
	fmt.Printf("errs               %d\n", errs)
	fmt.Printf("weighted E         %d\n", weightedE)
	fmt.Printf("failure rate       %.4f\n", failureRate)
	fmt.Printf("total time         %s\n", dur)
	fmt.Printf("per-request avg    %s\n", perReq)
}

// runV1 exercises the v1 IVF + int16 search path.
func runV1(idx *index.Index, entries []entry, baseN, retryN int, tp, tn, fp, fn, errs *int) {
	var qFloat [16]float32
	var qInt [16]int16
	var top index.Top5
	cellBuf := make([]index.CentroidDist, idx.NClusters)
	maxCell := 0
	for c := 0; c < idx.NClusters; c++ {
		size := int(idx.ClusterOffsets[c+1] - idx.ClusterOffsets[c])
		if size > maxCell {
			maxCell = size
		}
	}
	distBuf := make([]int64, maxCell)

	for _, e := range entries {
		if err := vector.NormalizePayload(e.Request, &qFloat); err != nil {
			*errs++
			continue
		}
		vector.Quantize(&qFloat, &qInt)
		idx.SearchIVF(&qInt, &qFloat, baseN, cellBuf, distBuf, &top)
		count := top.FraudCount()
		if !decisive(count) {
			idx.SearchIVF(&qInt, &qFloat, retryN, cellBuf, distBuf, &top)
			count = top.FraudCount()
		}
		approved := count < 3
		tally(approved, e.ExpectedApproved, tp, tn, fp, fn)
	}
}

// runV2 exercises the v2 IVF-HNSW + int8 residual search path.
func runV2(idx *index.Index, entries []entry, baseN, retryN, ef, retryEF int, tp, tn, fp, fn, errs *int) {
	var qFloat [16]float32
	var qInt8 [14]int8
	var out index.Top5Final

	// Pre-compute the Visited bitmap size. Size to the largest cluster.
	maxCell := 0
	for c := 0; c < idx.NClusters; c++ {
		size := int(idx.ClusterOffsets[c+1] - idx.ClusterOffsets[c])
		if size > maxCell {
			maxCell = size
		}
	}
	visitedSize := (maxCell + 63) / 64

	scratch := &index.SearchScratch{
		CellBuf: make([]index.CentroidDist, idx.NClusters),
		Visited: make([]uint64, visitedSize),
	}

	for _, e := range entries {
		if err := vector.NormalizePayload(e.Request, &qFloat); err != nil {
			*errs++
			continue
		}
		vector.QuantizeInt8(&qFloat, &qInt8)
		idx.SearchIVFHNSW(&qFloat, &qInt8, baseN, ef, scratch, &out)
		count := out.FraudCount()
		if !decisive(count) {
			idx.SearchIVFHNSW(&qFloat, &qInt8, retryN, retryEF, scratch, &out)
			count = out.FraudCount()
		}
		approved := count < 3
		tally(approved, e.ExpectedApproved, tp, tn, fp, fn)
	}
}

// decisive returns true when the vote is unambiguous.
// Asymmetric trigger: FN weighs 3× FP, so retry whenever leaning
// "approve" with any fraud signal at all. Skip retry only on truly
// unanimous "approved" (count == 0) and strong "fraud" (count >= 4).
func decisive(fraudCount int) bool {
	return fraudCount == 0 || fraudCount >= 4
}

// tally increments the appropriate tp/tn/fp/fn counter.
func tally(approved, expectedApproved bool, tp, tn, fp, fn *int) {
	if approved == expectedApproved {
		if approved {
			*tn++
		} else {
			*tp++
		}
	} else {
		if approved {
			*fn++
		} else {
			*fp++
		}
	}
}
