// hnsw_build.go is intentionally minimal — the actual v2 writer lives in
// internal/indexwriter/writer.go so it can be called from tests.
// Task 9 will wire -format=v2 in main.go via indexwriter.WriteV2 directly.
package main
