package main

import (
	"flag"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
	"time"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/index"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/response"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/vector"
)

func init() {
	runtime.GOMAXPROCS(1)
	debug.SetGCPercent(-1)
	debug.SetMemoryLimit(150 << 20)
	runtime.GC()
}

var contentTypeJSON = []string{"application/json"}

var (
	idx *index.Index

	bufPool = sync.Pool{New: func() any {
		b := make([]byte, 0, 4096)
		return &b
	}}
	qFloatPool = sync.Pool{New: func() any {
		return new([vector.Dim]float32)
	}}
	qInt16Pool = sync.Pool{New: func() any {
		return new([vector.Dim]int16)
	}}
	top5Pool = sync.Pool{New: func() any {
		return new(index.Top5)
	}}
	cellBufPool *sync.Pool // initialised after we know NClusters

	baseNprobe  int
	retryNprobe int
)

// readBody reads the entire body into the pooled buffer, growing as needed.
// Avoids io.ReadAll's fresh-slice allocation per request.
func readBody(r io.Reader, dst *[]byte) ([]byte, error) {
	*dst = (*dst)[:0]
	for {
		if cap(*dst)-len(*dst) < 512 {
			tmp := make([]byte, len(*dst), cap(*dst)*2+512)
			copy(tmp, *dst)
			*dst = tmp
		}
		n, err := r.Read((*dst)[len(*dst):cap(*dst)])
		*dst = (*dst)[:len(*dst)+n]
		if err == io.EOF {
			return *dst, nil
		}
		if err != nil {
			return *dst, err
		}
	}
}

func handleFraudScore(w http.ResponseWriter, r *http.Request) {
	t0 := time.Now()
	bufPtr := bufPool.Get().(*[]byte)
	defer bufPool.Put(bufPtr)

	body, err := readBody(r.Body, bufPtr)
	t1 := time.Now()
	stReadBody.record(t1.Sub(t0).Nanoseconds())
	if err != nil || len(body) == 0 {
		writeBody(w, response.Fallback)
		return
	}

	qFloat := qFloatPool.Get().(*[vector.Dim]float32)
	defer qFloatPool.Put(qFloat)
	if err := vector.NormalizePayload(body, qFloat); err != nil {
		writeBody(w, response.Fallback)
		return
	}

	qInt := qInt16Pool.Get().(*[vector.Dim]int16)
	defer qInt16Pool.Put(qInt)
	vector.Quantize(qFloat, qInt)
	t2 := time.Now()
	stParse.record(t2.Sub(t1).Nanoseconds())

	top := top5Pool.Get().(*index.Top5)
	defer top5Pool.Put(top)
	cellBuf := cellBufPool.Get().(*[]index.CentroidDist)
	defer cellBufPool.Put(cellBuf)

	idx.SearchIVF(qInt, qFloat, baseNprobe, *cellBuf, top)
	t3 := time.Now()
	stSearchOne.record(t3.Sub(t2).Nanoseconds())
	if !decisive(top.FraudCount()) {
		idx.SearchIVF(qInt, qFloat, retryNprobe, *cellBuf, top)
		stSearchTwo.record(time.Since(t3).Nanoseconds())
	}
	t4 := time.Now()

	count := top.FraudCount()
	writeBody(w, response.Bodies[count])
	t5 := time.Now()
	stWrite.record(t5.Sub(t4).Nanoseconds())
	stTotal.record(t5.Sub(t0).Nanoseconds())
}

// decisive reports whether the top-5 vote is far enough from the 0.6 decision
// threshold that re-running is unlikely to flip the verdict. Retry on counts
// 2 and 3 — these are the boundary cases where extra cells can change the
// majority. Aligned with cmd/accuracy so behavior matches the gate.
func decisive(fraudCount int) bool {
	return fraudCount <= 1 || fraudCount >= 4
}

func writeBody(w http.ResponseWriter, body []byte) {
	h := w.Header()
	h["Content-Type"] = contentTypeJSON
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

func handleReady(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func main() {
	sockPath := flag.String("socket", "/var/run/api.sock", "UDS path to listen on (when -listen is empty)")
	listenAddr := flag.String("listen", "", "if non-empty, bind a TCP socket here instead of UDS (e.g. :9999) — local dev only")
	indexPath := flag.String("index", "/index.bin", "path to IVF index file")
	nprobeFlag := flag.Int("nprobe", 12, "base IVF nprobe")
	retryFlag := flag.Int("retry-nprobe", 48, "IVF nprobe for ambiguous queries")
	debugAddr := flag.String("debug", "", "if non-empty, expose /debug/timings on this TCP addr (local only)")
	flag.Parse()
	if *debugAddr != "" {
		debugEnabled.Store(true)
		dmux := http.NewServeMux()
		dmux.HandleFunc("/debug/timings", handleDebugTimings)
		dmux.HandleFunc("/debug/reset", handleDebugReset)
		go func() {
			log.Printf("debug timings on %s", *debugAddr)
			if err := http.ListenAndServe(*debugAddr, dmux); err != nil {
				log.Printf("debug server: %v", err)
			}
		}()
	}

	baseNprobe = *nprobeFlag
	retryNprobe = *retryFlag

	loaded, err := index.Load(*indexPath)
	if err != nil {
		log.Fatalf("index load: %v", err)
	}
	idx = loaded
	log.Printf("loaded index: %d vectors, %d clusters", idx.NVectors, idx.NClusters)

	cellBufPool = &sync.Pool{New: func() any {
		buf := make([]index.CentroidDist, idx.NClusters)
		return &buf
	}}

	// Pre-warm pools so the first N requests don't pay allocation cost.
	// Adds maybe 1 MB of resident heap, eliminates fresh-alloc tail spikes.
	for i := 0; i < 64; i++ {
		b := make([]byte, 0, 4096)
		bufPool.Put(&b)
		qFloatPool.Put(new([vector.Dim]float32))
		qInt16Pool.Put(new([vector.Dim]int16))
		top5Pool.Put(new(index.Top5))
		cb := make([]index.CentroidDist, idx.NClusters)
		cellBufPool.Put(&cb)
	}

	var ln net.Listener
	if *listenAddr != "" {
		ln, err = net.Listen("tcp", *listenAddr)
		if err != nil {
			log.Fatalf("listen tcp: %v", err)
		}
		log.Printf("listening on tcp %s (dev mode)", *listenAddr)
	} else {
		_ = os.Remove(*sockPath)
		ln, err = net.Listen("unix", *sockPath)
		if err != nil {
			log.Fatalf("listen uds: %v", err)
		}
		if err := os.Chmod(*sockPath, 0666); err != nil {
			log.Printf("chmod %s: %v", *sockPath, err)
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/fraud-score", handleFraudScore)
	mux.HandleFunc("/ready", handleReady)

	srv := &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 2 * time.Second,
		IdleTimeout:       30 * time.Second,
	}

	runtime.GC()
	log.Printf("serving on %s", *sockPath)
	if err := srv.Serve(ln); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
