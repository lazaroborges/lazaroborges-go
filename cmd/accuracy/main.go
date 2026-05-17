// accuracy is a development-only probe: loads index.bin + test-data.json,
// runs the full handler pipeline in-process, and prints FP/FN counts. Lets
// us iterate on nprobe / cluster count without paying the docker + k6
// round-trip on every change. NOT a replacement for run.sh — k6 is the
// authoritative validation; this just gates accuracy before we get there.
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
	indexPath := flag.String("index", "tmp/index.bin", "")
	testPath := flag.String("test", "data/test-data.json", "")
	limit := flag.Int("limit", 0, "process at most N entries (0=all)")
	baseN := flag.Int("nprobe", 12, "")
	retryN := flag.Int("retry-nprobe", 48, "")
	flag.Parse()

	idx, err := index.Load(*indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}
	defer idx.Close()
	log.Printf("loaded index: %d vectors, %d clusters", idx.NVectors, idx.NClusters)

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
	log.Printf("evaluating %d entries", n)

	var qFloat [vector.Dim]float32
	var qInt [vector.Dim]int16
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

	var tp, tn, fp, fn, errs int
	t0 := time.Now()
	for i := 0; i < n; i++ {
		e := tf.Entries[i]
		if err := vector.NormalizePayload(e.Request, &qFloat); err != nil {
			errs++
			continue
		}
		vector.Quantize(&qFloat, &qInt)
		idx.SearchIVF(&qInt, &qFloat, *baseN, cellBuf, distBuf, &top)
		count := top.FraudCount()
		if !decisive(count) {
			idx.SearchIVF(&qInt, &qFloat, *retryN, cellBuf, distBuf, &top)
			count = top.FraudCount()
		}
		approved := count < 3
		if approved == e.ExpectedApproved {
			if approved {
				tn++
			} else {
				tp++
			}
		} else {
			if approved {
				fn++
			} else {
				fp++
			}
		}
	}
	dur := time.Since(t0)
	perReq := dur / time.Duration(n)

	weightedE := fp*1 + fn*3 + errs*5
	failureRate := float64(fp+fn+errs) / float64(n)
	fmt.Printf("\n=== detection summary ===\n")
	fmt.Printf("entries          %d\n", n)
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

func decisive(fraudCount int) bool {
	return fraudCount <= 1 || fraudCount >= 4
}
