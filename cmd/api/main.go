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
)

func init() {
	// GOMAXPROCS=1: parallel Ms run twice as fast through CPU work, but
	// CFS throttles them once the 45ms/100ms quota is spent — at which
	// point the network poller M is *also* throttled, stalling accept
	// (measured: GOMAXPROCS=2 raised k6 p99 from 2.37ms → 2.64ms despite
	// identical server-side total p99). Single M lets async preempt keep
	// the poller alive on the same goroutine.
	runtime.GOMAXPROCS(1)
	debug.SetGCPercent(-1)
	debug.SetMemoryLimit(150 << 20)
	runtime.GC()
}

var (
	idx *index.Index

	// Custom mutex-guarded freeList instead of sync.Pool. sync.Pool's per-P
	// caching + GC-drop semantics add unpredictable behavior; a small mutex
	// over a bounded slice is ~30-50ns uncontended and predictable. Required
	// to be thread-safe because GOMAXPROCS>1 enables parallel request handlers.
	qFloatPool = newFreeList(func() any { return new([16]float32) })
	qInt16Pool = newFreeList(func() any { return new([16]int16) })
	top5Pool   = newFreeList(func() any { return new(index.Top5) })

	cellBufPool *freeList // initialised after we know NClusters
	distBufPool *freeList // initialised after we know NVectors

	// v2 (IVF-HNSW) pools — nil until -search=ivfhnsw is selected.
	qInt8Pool     *freeList
	top5FinalPool *freeList
	scratchPool   *freeList

	baseNprobe  int
	retryNprobe int

	// v2 routing
	useIVFHNSW bool
	baseEf     int
	retryEf    int
)

type freeList struct {
	mu    sync.Mutex
	items []any
	newFn func() any
}

func newFreeList(fn func() any) *freeList {
	return &freeList{items: make([]any, 0, 128), newFn: fn}
}

func (f *freeList) Get() any {
	f.mu.Lock()
	if len(f.items) == 0 {
		f.mu.Unlock()
		return f.newFn()
	}
	item := f.items[len(f.items)-1]
	f.items = f.items[:len(f.items)-1]
	f.mu.Unlock()
	return item
}

func (f *freeList) Put(item any) {
	f.mu.Lock()
	if len(f.items) < cap(f.items) {
		f.items = append(f.items, item)
	}
	f.mu.Unlock()
}

// decisive reports whether the top-5 vote is far enough from the 0.6 decision
// threshold that re-running is unlikely to flip the verdict. Asymmetric: FN
// weighs 3× FP in the scoring rule, so we retry whenever there is any fraud
// signal at all (count ∈ {1,2,3}). Skip retry only on truly unanimous
// "approved" (count == 0) and strong "fraud" (count >= 4). Aligned with
// cmd/accuracy so behavior matches the gate.
func decisive(fraudCount int) bool {
	return fraudCount == 0 || fraudCount >= 4
}

func main() {
	sockPath := flag.String("socket", "/var/run/api.sock", "UDS path to listen on (when -listen is empty)")
	listenAddr := flag.String("listen", "", "if non-empty, bind a TCP socket here instead of UDS (e.g. :9999) — local dev only")
	indexPath := flag.String("index", "/index.bin", "path to IVF index file")
	nprobeFlag := flag.Int("nprobe", 4, "base IVF nprobe")
	retryFlag := flag.Int("retry-nprobe", 8, "IVF nprobe for ambiguous queries")
	searchMode := flag.String("search", "ivf", `search backend: "ivf" (v1, default) or "ivfhnsw" (v2)`)
	efFlag := flag.Int("ef", 32, "base HNSW ef (v2 only)")
	retryEfFlag := flag.Int("retry-ef", 64, "retry HNSW ef (v2 only)")
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
	useIVFHNSW = *searchMode == "ivfhnsw"
	baseEf = *efFlag
	retryEf = *retryEfFlag

	loaded, err := index.Load(*indexPath)
	if err != nil {
		log.Fatalf("index load: %v", err)
	}
	idx = loaded
	log.Printf("loaded index: %d vectors, %d clusters", idx.NVectors, idx.NClusters)

	cellBufPool = newFreeList(func() any {
		buf := make([]index.CentroidDist, idx.NClusters)
		return &buf
	})

	// Member-scan output buffer needs to fit the largest cluster.
	maxCell := 0
	for c := 0; c < idx.NClusters; c++ {
		size := int(idx.ClusterOffsets[c+1] - idx.ClusterOffsets[c])
		if size > maxCell {
			maxCell = size
		}
	}
	log.Printf("largest cluster: %d members", maxCell)
	distBufPool = newFreeList(func() any {
		buf := make([]int64, maxCell)
		return &buf
	})

	// Pre-warm pools so the first N requests don't pay allocation cost.
	for i := 0; i < 128; i++ {
		qFloatPool.Put(new([16]float32))
		qInt16Pool.Put(new([16]int16))
		top5Pool.Put(new(index.Top5))
		cb := make([]index.CentroidDist, idx.NClusters)
		cellBufPool.Put(&cb)
		db := make([]int64, maxCell)
		distBufPool.Put(&db)
		connBufPool.Put(&connBuf{data: make([]byte, maxConnBufSize)})
	}

	if useIVFHNSW {
		visitedSize := (maxCell + 63) / 64
		qInt8Pool = newFreeList(func() any { return new([14]int8) })
		top5FinalPool = newFreeList(func() any { return new(index.Top5Final) })
		scratchPool = newFreeList(func() any {
			return &index.SearchScratch{
				CellBuf: make([]index.CentroidDist, idx.NClusters),
				Visited: make([]uint64, visitedSize),
			}
		})
		for i := 0; i < 128; i++ {
			qInt8Pool.Put(new([14]int8))
			top5FinalPool.Put(new(index.Top5Final))
			scratchPool.Put(&index.SearchScratch{
				CellBuf: make([]index.CentroidDist, idx.NClusters),
				Visited: make([]uint64, visitedSize),
			})
		}
		log.Printf("IVF-HNSW mode: ef=%d retry-ef=%d", baseEf, retryEf)
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
