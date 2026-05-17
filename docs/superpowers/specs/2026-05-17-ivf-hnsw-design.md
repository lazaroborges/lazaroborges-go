# IVF-HNSW Hybrid — Design Spec

**Date:** 2026-05-17
**Author:** Lazaro
**Status:** Draft — pending implementation

## Goal

Push the Rinha-2026 score from **5318 → ≥5950** by replacing the IVF flat-member-scan search with an **IVF-HNSW hybrid**: coarse k-means partitioning to localize the search, then a per-cluster HNSW graph to navigate within each selected cell. Re-rank the final candidate set with exact float32 distance to neutralize int8 quantization noise.

## Why this design

Current run: p99 = 2.13ms (score 2671) + E=14 detection errors (score 2647) = **5318**. The IVF flat-scan approach is structurally bound:

- Member scan inside the chosen cells is `O(cluster_size × dim)` work per cell — even with AVX2 VPMADDWD, scanning ~1500 vectors per cell at nprobe=4 takes a sizable fraction of the per-query budget under CFS throttling.
- Raising `nprobe` is the only knob for accuracy, but it linearly scales scan work.
- The 6 detection errors (2 FP + 4 FN) come from boundary cases where k=5 voting flips on small distance perturbations — exactly the kind of error a higher-recall search closes.

HNSW within each cluster gives sub-linear navigation (~`log(cluster_size)` distance evals instead of `cluster_size`) at >99.5% recall@5 for our dimensionality and scale. The IVF outer layer keeps memory locality and lets us avoid building a single 3M-node graph (which would blow our memory budget). The float32 re-rank step at the end fixes the small ranking errors that int8 quantization introduces, recovering most boundary-case FN/FP.

## Hard constraints

- 1 CPU + 350 MB across all docker-compose services (per instance: 0.45 CPU + 160 MB)
- Linux/amd64, Haswell, AVX2 available, no AVX-512
- p99 ≤ 1ms (score cap)
- E (= 1·FP + 3·FN + 5·Err) → near zero (score cap at E=0)
- HAProxy in TCP mode round-robin; LB cannot inspect payloads

## Architecture

Deployment topology unchanged — HAProxy TCP-mode round-robin over two UDS-bound API instances. The change is entirely in the search path inside each API instance.

```
client → HAProxy (TCP :9999) → UDS → api-{1,2}
                                       │
                                       ▼
                                    custom HTTP/1.1 server
                                       │
                                       ▼
                                    SearchIVFHNSW(qInt8, qFloat32, nCells, ef)
                                       │
                          ┌────────────┴────────────┐
                          │ 1. centroid pass (AVX2) │
                          │ 2. pick top-N cells     │
                          │ 3. per-cell HNSW search │
                          │ 4. merge top-5          │
                          │ 5. float32 re-rank      │
                          └────────────┬────────────┘
                                       ▼
                              preallocated response frame
```

## Memory budget (per instance, 160 MB cap)

K coarse cells = **1024** (down from 2048 → fewer/larger clusters = stronger per-cluster HNSW graphs).

| Component | Bytes | Notes |
|---|---|---|
| Member residuals, int8, padded to 16 lanes per node | 48 MB | 3M × 16 (padded so AVX2 VPMOVSXBW loads a full 128-bit chunk safely) |
| HNSW edges, uint16 local indices, M=6 | ~54 MB | layer-0 + upper layers via geometric series; uint16 because per-cluster size < 65536 |
| HNSW node level table (uint8 per node) | 3 MB | one byte = max level for that node |
| Centroids float32 + padded for AVX2 | 0.12 MB | 1024 × 14 + 1024 × 16 |
| Centroids int8 padded (for per-cell query translation) | 16 KB | 1024 × 16 |
| Centroid squared norms float32 | 4 KB | |
| Cluster offsets table uint32 | 4 KB | |
| Per-cluster HNSW headers | ~40 KB | 1024 × 40 bytes; see HnswHeader layout |
| Labels (uint8) | 3 MB | |
| Pools, Go runtime, conn buffers, response frames | ~30 MB | unchanged |
| **Total** | **~138 MB** | **22 MB headroom** |

**Dropped from current index:** member squared norms (24 MB), padded float32 centroids if we accept a slightly slower centroid pass.

## Algorithm details

### Storage layout (Index Format v2)

```
+-------------------------------------------------------------+
| Header (48 bytes)                                           |
|   magic        uint32  = 'RIVH' (new — distinct from RIVF)  |
|   version      uint32  = 2                                  |
|   nVectors     uint32                                       |
|   nClusters    uint32                                       |
|   dim          uint32  = 14                                 |
|   M            uint32                                       |
|   maxLevel     uint32  (max HNSW level across all nodes)    |
|   _pad         uint32  × 5                                  |
+-------------------------------------------------------------+
| Centroids:  nClusters × Dim × float32                       |
+-------------------------------------------------------------+
| CentroidsPadded:  nClusters × 16 × float32                  |
+-------------------------------------------------------------+
| CentroidsInt8:  nClusters × Dim × int8                      |
+-------------------------------------------------------------+
| CentroidNorms:  nClusters × float32                         |
+-------------------------------------------------------------+
| ClusterOffsets:  (nClusters + 1) × uint32                   |
|   offsets[c]..offsets[c+1] = member range in cluster c      |
+-------------------------------------------------------------+
| MemberResiduals:  nVectors × Dim × int8                     |
+-------------------------------------------------------------+
| Labels:  nVectors × uint8                                   |
+-------------------------------------------------------------+
| HnswPerClusterHeaders:  nClusters × HnswHeader (40 bytes)   |
|   entry_point  uint16   (local-id of the top-layer entry)   |
|   maxLevel     uint8    (cap 7; assignments above clamped)  |
|   _pad         uint8                                        |
|   edgeOffset   uint32   (byte offset into HnswEdges)        |
|   levelCount[8] uint32  (nodes present at each level)       |
+-------------------------------------------------------------+
| HnswEdges:  packed per-cluster, then per-level.             |
|   For each cluster c, for each level L 0..maxLevel_c:       |
|     - if L >= 1: nodeIds[levelCount[L]] uint16              |
|         ascending; identifies which local nodes have edges  |
|         at this level. Level 0 omits this list (every node  |
|         is at level 0; local-id IS the index).              |
|     - edges[levelCount[L]] × EdgeSlot                       |
|         EdgeSlot = (degree uint8, _pad uint8,               |
|                    neighbors[M] uint16) = 14 bytes for M=6  |
|         neighbors are local-ids; slots beyond `degree` are  |
|         0xFFFF.                                             |
|                                                             |
|   Lookup at runtime for node n at level L:                  |
|     if L == 0: slot index = n                                |
|     else: slot index = binarySearch(nodeIds_L, n)           |
|       (levelCount[L] is small: ~clusterSize / 6^L)          |
+-------------------------------------------------------------+
```

Why per-cluster HNSW headers in their own section: keeps the hot mmap pages cold-cache-friendly. Centroids + centroid_int8 stay near the front; member residuals + edges are accessed only after a cell is selected.

### Construction (preprocess)

Existing pipeline ends with k-means clustering and int16 quantization of members. Replace the tail:

1. **k-means → 1024 clusters** (was 2048). Re-use existing mini-batch k-means with sampled k-means++ init.
2. **Compute centroids in float32 and int8 representations.** int8 = `round(centroid_float32 * 127)` then clamped.
3. **For each member m in cluster c**:
   - `residual_float = member_vector_float − centroid_c_float`
   - `residual_int8 = clamp(round(residual_float × 127), -128, 127)`
4. **Build per-cluster HNSW**:
   - Parameters: `M = 6`, `efConstruction = 200`, `mL = 1 / ln(M) ≈ 0.558`, random seed fixed.
   - Standard HNSW insertion: random level assignment, greedy descent from entry above target level, beam search at target level with `efConstruction`, M-heuristic neighbor selection (keep diverse neighbors, not just nearest).
   - **Cap max level at 7** to keep level table compact; for 1500-3000 node clusters this is plenty.
   - Distance metric during build: int8 residual squared distance (matches query-time distance).
5. **Parallelize build per cluster** — 1024 independent builds across Mac M-series cores; expected total build time ~30-60s.

### Query path

```go
func (idx *Index) SearchIVFHNSW(
    qFloat *[16]float32,        // float32 query for re-rank
    qInt8 *[14]int8,            // int8-quantized query
    nCells int,                  // base = 2, retry = 4
    ef int,                      // base = 32, retry = 64
    scratch *SearchScratch,
    out *Top5,                   // final top-5 (after float32 re-rank)
) {
    // 1. Distance from qFloat to all 1024 centroids (existing AVX2 kernel).
    centroidPassAvx2(qFloat, idx.CentroidsPadded, idx.CentroidNorms, scratch.CellBuf)

    // 2. Top-N cells by centroid distance.
    selectTopCells(scratch.CellBuf, nCells)

    // 3. Per-cell HNSW search; each returns K=8 candidates tagged with cell.
    var perCell [4]TopK  // max nCells=4, each holds up to 8 candidates
    for k := 0; k < nCells; k++ {
        cellIdx := scratch.CellBuf[k].Cluster
        // qRes[i] = qInt8[i] - centroidInt8[cellIdx][i]
        // Computed as int16 to avoid int8 overflow, then used in the
        // int8-vs-int16 distance kernel.
        translateQueryToResidual(qInt8, idx.CentroidsInt8[cellIdx*16:], &scratch.QRes)
        hnswSearchCluster(idx, cellIdx, &scratch.QRes, ef, 8, &perCell[k], scratch)
    }

    // 4. Merge per-cell K=8 candidates into a single TopK of size 10
    //    by int8 residual distance.
    mergeIntoTopK10(perCell[:nCells], &scratch.Merged)

    // 5. Float32 re-rank: reconstruct each candidate from
    //    centroid_float + residual_int8/127, compute exact L2 to qFloat,
    //    sort by float32 distance, keep top-5 into `out`.
    rerankTop5Float32(&scratch.Merged, qFloat, idx, out)
}
```

**HNSW search inside a cluster** (`hnswSearchCluster`):
- Standard layered greedy descent: start at the cluster's entry_point at maxLevel; greedy-improve neighbor until no neighbor is closer; descend a level; repeat.
- At level 0: beam search with `ef` size, returning the K=5 closest. Visited-set is a small `[]uint64` bitmap sized for the cluster (max ~3000 nodes → 47 uint64s).
- Distance metric: int8 squared distance via AVX2 SIMD kernel — see kernel notes.

**Distance kernel for int8 residuals (AVX2):**
- Pad residuals to 16 lanes per node at storage time.
- `VPMOVSXBW` to sign-extend int8 → int16 in 256-bit register.
- `VPSUBW` (subtract), `VPMADDWD` (multiply-add pairs → int32), `VPADDD` (accumulate to int32 lane).
- Horizontal reduce to int32 scalar at end.
- Target: ~2ns per distance. At ef=32 + ~3-5 hops/level + 3 levels ≈ ~150-200 distances per cell → ~0.4µs per cell SIMD time.

**Float32 re-rank** (`rerankTop5Float32`):
- For each of the up-to-5 candidates in `out`, reconstruct: `member_float[i] = (residual_int8[i] / 127.0) + centroid_float[i]`.
- Compute exact squared distance to `qFloat`.
- Re-sort by float32 distance.
- Vote on the re-ranked top-5.

**Required type change:** the existing `Top5` stores `(int64 dist, uint8 label)`. For re-rank we additionally need each entry's source cluster and local member id so we can reconstruct float32 coordinates. The new search returns a wider top-K pool, then narrows after re-rank:

```go
type Candidate struct {
    Dist    int32   // int8 residual squared distance
    Cluster uint32
    LocalID uint16
    Label   uint8
}

type TopK struct {
    Items [10]Candidate  // wider than 5 to absorb int8 ranking noise
    N     int
}
```

Each per-cell HNSW search returns K=8 candidates tagged with their cluster. Merge across cells into a top-10 by int8 distance. Re-rank those 10 with exact float32, then keep the top-5 by float32 distance for voting. This widens the re-rank candidate pool so a member that int8 ranked 7th but float32 ranks 4th still has a chance to land in the final top-5. The final vote is `approved = fraudCount(top5_after_rerank) < 3`.

### Decisive-retry logic

Current heuristic: retry when `FraudCount ∈ {2, 3}`. Keep it, and add a **distance-margin** trigger:

```go
func decisive(top *Top5) bool {
    fc := top.FraudCount()
    if fc == 0 || fc == 5 { return true }      // unanimous, never retry
    if fc == 1 || fc == 4 { 
        // close to threshold (3 = fraud). Check margin.
        // If d5 is very close to "what would have been d6", a single swap flips the vote.
        // Approximate: if (d5 - d1) / d1 < margin_threshold, retry.
        return marginIsWide(top)
    }
    return false   // fc == 2 or 3 → always retry
}
```

Where `marginIsWide` checks if `top.Dist[4]` is at least `1.3×` `top.Dist[0]` (a calibration constant we tune from the accuracy probe).

### Server hot path

Replaces `processSearch` in `cmd/api/server.go`:

```go
func processSearch(body []byte) []byte {
    qFloat := qFloatPool.Get().(*[16]float32)
    if err := vector.NormalizePayload(body, qFloat); err != nil {
        qFloatPool.Put(qFloat)
        return response.FallbackFrame
    }
    qInt8 := qInt8Pool.Get().(*[14]int8)
    vector.QuantizeInt8(qFloat, qInt8)

    scratch := scratchPool.Get().(*index.SearchScratch)
    top := top5Pool.Get().(*index.Top5)

    idx.SearchIVFHNSW(qFloat, qInt8, 2, 32, scratch, top)
    if !decisive(top) {
        idx.SearchIVFHNSW(qFloat, qInt8, 4, 64, scratch, top)
    }

    frame := response.Frames[top.FraudCount()]

    qFloatPool.Put(qFloat)
    qInt8Pool.Put(qInt8)
    scratchPool.Put(scratch)
    top5Pool.Put(top)
    return frame
}
```

`SearchScratch` bundles per-query buffers (cell buffer, visited bitmap, beam heap, per-cell TopK slots, qRes buffer) into one pool entry so we do one Get/Put per query instead of five.

## Validation strategy

### Correctness gates (must pass before tuning)

1. **Unit test:** HNSW search returns the same top-5 as brute-force for a 1000-vector toy cluster (recall@5 = 1.0).
2. **Quantization round-trip:** int8 residual + centroid reconstructs within ±1/127 per dim of original float32 (per-component absolute).
3. **Vote consistency:** with `ef = clusterSize` (effectively brute force), accuracy probe returns identical labels to current IVF brute-force baseline on a 1000-entry sample.

### Tuning loop (in-process accuracy probe)

Fix `cmd/accuracy` to the new signatures first. Add a **recall@5** metric: for a 200-query sample, compare returned top-5 (by member-id) to exact float32 brute force. Print `recall@5 = X / 1000`.

Tuning order:
1. Baseline run: `M=6, ef=32, nCells=2`. Measure recall@5, FP, FN, per-query avg latency.
2. If `recall@5 < 0.995`: bump `ef → 64`. Re-measure.
3. If recall still short: bump `nCells → 3` (more cells, more candidates).
4. If FP/FN > 4: tune `marginIsWide` threshold (smaller = retry more often).
5. Only as last resort, rebuild with `M=8` (memory cost: +18 MB).

### End-to-end validation

After in-process accuracy is acceptable (E ≤ 2 on full 54100 test set), run `docker-compose up` + `./run.sh` and check the JSON scoreboard. Target: `final_score ≥ 5800` on first pass, ≥5950 after tuning.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| HNSW build time blows up on Haswell contest box | Low | Build happens at Docker image build time on developer Mac; contest box only runs Load(). |
| int8 residual loses too much precision near boundaries | Medium | Float32 re-rank step. If still insufficient, K=8 per cell to widen re-rank pool. |
| Memory accounting is off; container OOMs | Low | SetMemoryLimit(140MB) safety net; measure RSS after Load(). |
| HNSW recall < 99.5% at M=6 | Medium | M=8 fallback (with memory cost). |
| Pointer-chasing in HNSW destroys cache | Medium | Per-cluster layout keeps the working set small (~30KB per cluster fits in L1d); visited bitmap is tiny. |
| Build non-determinism breaks reproducibility | Low | Fixed RNG seed in preprocess. |
| Per-cluster header table grows large for many cells | Low | 1024 cells × ~50 bytes = 50 KB; trivial. |

## Implementation order

1. **Skeleton + format v2 writer/reader.** Define Index struct fields for HNSW, write the new binary format from a stub builder that produces empty graphs. Verify Load() succeeds.
2. **HNSW build (pure Go).** Implement insertion, heuristic neighbor selection, level assignment. Unit test against a small dataset (1000 vectors, brute-force baseline). Add to preprocess pipeline.
3. **HNSW search (pure Go).** Implement greedy descent + beam search. Unit-test recall@5 ≥ 0.99 on 1000-vector cluster.
4. **Int8 residual distance kernel.** Pure Go version first (correctness), then AVX2 assembly. Bench: target ~2ns/distance.
5. **Float32 re-rank step.** Simple loop, no SIMD needed.
6. **SearchScratch pool.** Bundle all per-query buffers.
7. **Replace processSearch.** Wire new search into the custom HTTP server. Keep old SearchIVF in tree, gated behind a build tag, so we can compare side-by-side.
8. **Fix cmd/accuracy.** Update signatures, add recall@5 metric.
9. **Run accuracy gate.** Tune per the tuning loop above. Iterate until E ≤ 2 on full test set.
10. **Docker + k6 validation.** Confirm scoreboard reaches ≥5950.
11. **Cleanup.** Remove the v1-format code path and old SearchIVF after we're confident in v2.

### Rollback plan

Keep the v1 (`RIVF`) format reader present until v2 has shipped one successful run. If v2 misbehaves in docker, revert to v1 index by switching the preprocess output target and re-running `docker-compose build`. The server code change is the irreversible piece — guard it with a `-search=ivf|ivfhnsw` flag during the transition.

## What we are explicitly NOT doing

- **No PQ codebooks.** Plain int8 residual quantization is simpler and good enough at 14 dims; PQ subspaces at this dimensionality (2 dims each) train poorly.
- **No multi-thread per-query.** GOMAXPROCS=1 is set; per-query parallelism would cost more in scheduler than it saves.
- **No persistent visited-set across queries.** Per-query zero-init of a 400-byte bitmap is cheap.
- **No graph-level mutation at runtime.** Index is static; HNSW is built once at preprocess time.
- **No swap to a different HTTP server.** Custom HTTP/1.1 parser stays.

## Performance budget (per request, p50, in-process)

| Phase | Target |
|---|---|
| HTTP parse + JSON normalize + int8 quantize | ~50µs |
| Centroid pass (1024 × AVX2) | ~3µs |
| Top-N cell select (N=2) | <1µs |
| Per-cell query translate (×2) | <1µs |
| HNSW search per cell (ef=32, ~200 dist × 2ns) (×2) | ~10µs |
| Merge + float32 re-rank top-5 | ~2µs |
| Response frame write | ~20µs |
| **Total p50** | **~90µs** |
| **Tail headroom against 1ms cap** | **~910µs** |

Compared to current ~127µs p50 — the search itself drops from ~30-40µs to ~12µs, and we get higher recall to boot.

## Expected scoreboard

| Scenario | p99 | E | `score_p99` | `score_det` | **Final** |
|---|---|---|---|---|---|
| Target | 0.8ms | 0 | ~2970 | 3000 | **~5970** |
| Realistic best | 0.9ms | 1 | ~2950 | ~2870 | **~5820** |
| Realistic median | 1.1ms | 2 | ~2960 | ~2700 | **~5660** |
| Worst-acceptable | 1.5ms | 4 | ~2820 | ~2470 | **~5290** |

The worst-acceptable scenario is roughly today's score, so the design has clear downside protection.
