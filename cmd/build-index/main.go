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

	if err := index.Build(*inPath, *outPath); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
