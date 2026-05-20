package main

import (
	"flag"
	"fmt"
	"os"

	"lazaroborges-go/internal/index"
)

func main() {
	inPath := flag.String("in", "resources/references.json.gz", "input references gzip path")
	outPath := flag.String("out", "index.bin", "output index path")
	flag.Parse()

	fmt.Fprintln(os.Stderr, "loading dataset...")
	buckets, err := index.LoadAndPartition(*inPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load error:", err)
		os.Exit(1)
	}

	for i, b := range buckets {
		fmt.Fprintf(os.Stderr, "bucket %2d: %7d vectors, %2d dims\n", i, b.NVecs(), b.NDims)
	}

	fmt.Fprintln(os.Stderr, "clustering... (next tasks)")
	_ = outPath
}
