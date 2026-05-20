package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"

	"lazaroborges-go/internal/handler"
	"lazaroborges-go/internal/index"
	"lazaroborges-go/internal/vectorize"
)

func main() {
	runtime.GOMAXPROCS(1)
	debug.SetGCPercent(-1)           // GOGC=off
	debug.SetMemoryLimit(200 << 20) // 200 MB safety net

	indexPath := envOr("INDEX_PATH", "/app/index.bin")
	normPath := envOr("NORM_PATH", "/app/normalization.json")
	mccPath := envOr("MCC_PATH", "/app/mcc_risk.json")
	addr := envOr("ADDR", ":8080")
	nprobe := envInt("NPROBE", index.NProbe)

	norm, err := loadNorm(normPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load norm:", err)
		os.Exit(1)
	}

	mcc, err := loadMCC(mccPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load mcc:", err)
		os.Exit(1)
	}

	idx, err := index.Open(indexPath, nprobe)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open index:", err)
		os.Exit(1)
	}

	h := handler.New(idx, norm, mcc)
	mux := http.NewServeMux()
	h.Register(mux)

	fmt.Fprintf(os.Stderr, "listening on %s\n", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, "serve:", err)
		os.Exit(1)
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func loadNorm(path string) (vectorize.Normalization, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return vectorize.Normalization{}, err
	}
	var n vectorize.Normalization
	return n, json.Unmarshal(b, &n)
}

func loadMCC(path string) (map[string]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m := make(map[string]float32)
	return m, json.Unmarshal(b, &m)
}
