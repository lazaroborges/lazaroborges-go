package main

import (
	"flag"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"runtime/debug"
	"sync"
	"syscall"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/index"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/vector"
)

func init() {
	runtime.GOMAXPROCS(1)
	debug.SetGCPercent(-1)
	debug.SetMemoryLimit(150 << 20)
	runtime.GC()
}

var (
	idx *index.Index

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
	distBufPool *sync.Pool // initialised after we know NVectors

	baseNprobe  int
	retryNprobe int
)

// decisive reports whether the top-5 vote is far enough from the 0.6 decision
// threshold that re-running is unlikely to flip the verdict. Retry on counts
// 2 and 3 — these are the boundary cases where extra cells can change the
// majority. Aligned with cmd/accuracy so behavior matches the gate.
func decisive(fraudCount int) bool {
	return fraudCount <= 1 || fraudCount >= 4
}

func main() {
	sockPath := flag.String("socket", "/var/run/api.sock", "UDS path to listen on (when -listen is empty)")
	listenAddr := flag.String("listen", "", "if non-empty, bind a TCP socket here instead of UDS (e.g. :9999) — local dev only")
	indexPath := flag.String("index", "/index.bin", "path to IVF index file")
	nprobeFlag := flag.Int("nprobe", 4, "base IVF nprobe")
	retryFlag := flag.Int("retry-nprobe", 8, "IVF nprobe for ambiguous queries")
	debugAddr := flag.String("debug", "", "if non-empty, expose /debug/timings on this TCP addr (local only)")
	flag.Parse()
	if *debugAddr != "" {
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

	// SIGTERM dump: docker stop sends SIGTERM. We catch it, dump the per-stage
	// timing summary to stdout (which the contest infrastructure captures in
	// container logs), then exit cleanly. Gives us Haswell-ground-truth per-
	// stage latencies after every contest run without needing -debug.
	go func() {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, syscall.SIGTERM, syscall.SIGINT)
		<-ch
		dumpTimings()
		os.Exit(0)
	}()

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

	// Member-scan output buffer needs to fit the largest cluster. Compute
	// once at load and size the pool to that.
	maxCell := 0
	for c := 0; c < idx.NClusters; c++ {
		size := int(idx.ClusterOffsets[c+1] - idx.ClusterOffsets[c])
		if size > maxCell {
			maxCell = size
		}
	}
	log.Printf("largest cluster: %d members", maxCell)
	distBufPool = &sync.Pool{New: func() any {
		buf := make([]int64, maxCell)
		return &buf
	}}

	// Pre-warm pools so the first N requests don't pay allocation cost.
	// Adds maybe 1 MB of resident heap, eliminates fresh-alloc tail spikes.
	for i := 0; i < 64; i++ {
		qFloatPool.Put(new([vector.Dim]float32))
		qInt16Pool.Put(new([vector.Dim]int16))
		top5Pool.Put(new(index.Top5))
		cb := make([]index.CentroidDist, idx.NClusters)
		cellBufPool.Put(&cb)
		db := make([]int64, maxCell)
		distBufPool.Put(&db)
		// Pre-warm the connection-buffer pool too; one per goroutine matches
		// the keep-alive connection count under load.
		connBufPool.Put(&connBuf{data: make([]byte, maxConnBufSize)})
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

	runtime.GC()
	log.Printf("serving on %s", *sockPath)
	if err := serveUDS(ln); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
