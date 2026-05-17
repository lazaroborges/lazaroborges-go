package main

import (
	"fmt"
	"log"
	"net/http"
	"sort"
	"sync/atomic"
	"time"
)

// Tail-latency instrumentation. Records up to 16384 samples per stage in a
// lock-free ring; /debug/timings dumps them. Enabled via -debug.
//
// We deliberately keep the recording path free of allocations and locks. A
// single atomic.AddInt64 + array store per sample is well under 50 ns on a
// modern x86.

const ringSize = 16384

type stage struct {
	name    string
	idx     atomic.Int64
	samples [ringSize]int64 // nanoseconds
}

var (
	stReadBody  = &stage{name: "readBody"}
	stParse     = &stage{name: "parse+quantize"}
	stSearchOne = &stage{name: "searchBase"}
	stSearchTwo = &stage{name: "searchRetry"}
	stWrite     = &stage{name: "writeResp"}
	stTotal     = &stage{name: "total"}
)

func (s *stage) record(ns int64) {
	// Recording is always on. Cost is a single atomic increment + a single
	// array write per stage (~30 ns total per request), well under 0.05% of
	// budget at 900 RPS. The /debug/timings HTTP endpoint is still gated on
	// the -debug flag, but the in-process timing dump emitted on SIGTERM
	// (see dumpTimings in main.go) needs unconditional data.
	i := s.idx.Add(1) - 1
	s.samples[int(i)%ringSize] = ns
}

func (s *stage) snapshot() []int64 {
	n := int(s.idx.Load())
	if n > ringSize {
		n = ringSize
	}
	out := make([]int64, n)
	copy(out, s.samples[:n])
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func pct(sorted []int64, p float64) int64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(float64(len(sorted)-1) * p)
	return sorted[i]
}

func handleDebugTimings(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	stages := []*stage{stReadBody, stParse, stSearchOne, stSearchTwo, stWrite, stTotal}
	fmt.Fprintf(w, "%-16s %8s %8s %8s %8s %8s\n", "stage", "n", "p50", "p90", "p99", "p999")
	for _, s := range stages {
		ss := s.snapshot()
		fmt.Fprintf(w, "%-16s %8d %8s %8s %8s %8s\n",
			s.name, len(ss),
			fmtDur(pct(ss, 0.50)),
			fmtDur(pct(ss, 0.90)),
			fmtDur(pct(ss, 0.99)),
			fmtDur(pct(ss, 0.999)),
		)
	}
}

func fmtDur(ns int64) string {
	return time.Duration(ns).String()
}

func handleDebugReset(w http.ResponseWriter, _ *http.Request) {
	for _, s := range []*stage{stReadBody, stParse, stSearchOne, stSearchTwo, stWrite, stTotal} {
		s.idx.Store(0)
	}
	w.WriteHeader(http.StatusOK)
}

// dumpTimings logs a one-line-per-stage timing summary to stdout. Called from
// a SIGTERM handler so the contest infrastructure captures it in container
// logs after the load test finishes — no need for an exposed debug port.
func dumpTimings() {
	stages := []*stage{stReadBody, stParse, stSearchOne, stSearchTwo, stWrite, stTotal}
	log.Printf("timing dump: %-16s %8s %8s %8s %8s %8s", "stage", "n", "p50", "p90", "p99", "p999")
	for _, s := range stages {
		ss := s.snapshot()
		log.Printf("timing dump: %-16s %8d %8s %8s %8s %8s",
			s.name, len(ss),
			fmtDur(pct(ss, 0.50)),
			fmtDur(pct(ss, 0.90)),
			fmtDur(pct(ss, 0.99)),
			fmtDur(pct(ss, 0.999)),
		)
	}
}
