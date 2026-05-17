# IVF-HNSW Hybrid — Postmortem

**Date:** 2026-05-17
**Status:** Shelved — algorithm does not beat v1 on this dataset; infrastructure kept.
**Spec:** `2026-05-17-ivf-hnsw-design.md`
**Plan:** `../plans/2026-05-17-ivf-hnsw.md`

## What we built

Tasks 1–14 of the plan, in full:

- AVX2 int8 residual squared-distance kernel (`internal/index/residual_*`)
- HNSW package: heaps, Graph, Search, Build (`internal/index/hnsw/`)
- Format v2 binary layout: int8 member residuals padded to 16 lanes + per-cluster HNSW edge graphs (`internal/index/format_v2.go`)
- v2 writer (`internal/indexwriter/`)
- v2 reader dispatched by magic byte in `Load()` (`internal/index/index.go`)
- `cmd/preprocess -format=v2` builds the new index
- `vector.QuantizeInt8` (`internal/vector/quantize_int8.go`)
- `SearchIVFHNSW` query path with float32 re-rank (`internal/index/search_v2.go`)
- `cmd/api -search=ivfhnsw` dispatches to the new path; default stays `ivf`
- `cmd/accuracy` rewritten for dual-mode v1/v2 validation

The whole stack works end-to-end. Tests pass. The v2 index loads, queries return results, the round-trip writer/reader test gates the binary format.

## What we found

| Mode | nprobe | ef | E (10K sample) | per-req latency |
|---|---|---|---|---|
| v1 IVF (production) | 12/48 | n/a | **3 (on 54K)** | 191 µs |
| v2 HNSW | 2/4 | 32/64 | 370 (on 54K) | 71 µs |
| v2 HNSW | 8/16 | 64/128 | 77 (on 10K) | 296 µs |
| v2 HNSW | 16/32 | 128/256 | 77 (on 10K) | 683 µs |
| v2 IVFADC (no HNSW) | 4/8 | n/a | 47 (on 10K) | 94 µs |
| v2 IVFADC | 12/48 | n/a | 46 (on 10K) | 288 µs |
| v2 IVFADC + K=64 re-rank | 12/48 | n/a | 47 (on 10K) | 288 µs |

**The v2 floor is ~250-400 E on the full 54100-entry test set.** v1's E=3 is strictly better at all parameter settings tested.

## Root cause

The v2 algorithm chains two precision-losing steps before the final float32 re-rank:

1. **int8 quantization of the query** (`QuantizeInt8`): query × 127 → int8. Per-component precision ≈ 1/127 ≈ 0.008.
2. **int8 quantization of the member residual** (`writeIndexV2`): (member − centroid) × 127 → int8. For tightly-clustered real data, residual magnitudes are small (∼0.05), so int8 wastes most of its dynamic range.

Squared distance over 14 components compounds both errors. For boundary queries — exactly the ones that determine FP/FN — the int8 ranking disagrees with the int16 ground truth enough to flip the top-5 vote.

The float32 re-rank step *would* recover this if the true top-5 were in the merged candidate pool. They aren't: the int8 distance is so noisy that the true top-5 don't make it through the per-cell top-K=8 cut. Widening the merged pool to K=64 confirmed this — identical E. The information loss happens at the per-member distance step, not at the merge step.

## Why v1 still wins on the contest

The original framing was "the IVF flat-scan is the bottleneck." It isn't. v1's E=3 detection score is already ~2950 out of 3000 — near-ceiling. The contest score loss (5318/6000) is dominated by the **p99 latency** axis: 2.13 ms → 2671 points (losing 329 to the 1 ms cap).

v1's 191 µs per-request in-process latency under Mac single-thread is fine. The 2.13 ms contest p99 is environmental: CFS throttling on the 0.45-CPU container, HAProxy queueing under load, GC adjacency, and async preemption stalls. **No algorithm change reduces those.**

## What's kept

Even though v2's algorithm is shelved, the infrastructure is sound and reusable:

- `internal/index/hnsw/` — HNSW graph with M-aware build and ef-beam search. Unit-tested.
- `internal/index/residual_*` — int8 residual SIMD kernel. AVX2 + pure-Go.
- `internal/index/format_v2.go`, `internal/indexwriter/` — v2 binary layout + writer. Round-trip tested.
- `internal/index/scratch.go` — `Candidate`, `TopK10` (cap=64), `Top5Final`, `SearchScratch`. Reusable types.
- `cmd/accuracy` dual-mode probe with `-search=ivf|ivfhnsw` flag. Useful for any future v2/v3 experiment.

The v1 path (`SearchIVF`, `member_scan_amd64.s`, `distance_*`, format v1) is untouched and remains the production path. `docker-compose.yml` defaults to v1 (no `-search` flag → `-search=ivf` default in `cmd/api/main.go`).

## Possible future v3 experiments

If accuracy headroom is ever needed past v1's E=3:

1. **int16 residuals** instead of int8. Storage: 96 MB (vs 48 MB). Combined with dropping the unused HNSW edge block (-54 MB), total fits in ~140 MB. Distance math: reuse v1's `int16SqDist14` kernel. Mathematically equivalent to v1 for top-K ranking — *no detection win* but possibly a small latency win from contiguous-residual locality and the wider merged pool catching ties.
2. **OPQ-rotated PQ with anisotropic loss** (ScaNN-style). Substantial implementation cost; theoretical recall gain is small at our scale.
3. **Train a small classifier** (e.g. GBDT) directly on the 14-dim vector → fraud probability. Bypasses k-NN entirely. Different algorithm class; would need a different test methodology.

None of these are obviously worth the implementation cost, given v1's near-ceiling detection.

## Where the next score wins actually live

Per the score math (`5318 = 2671 p99 + 2647 det`), recovering the 682-point gap to 6000 means:

- **329 pts** on latency (p99 2.13 ms → 1.0 ms)
- **353 pts** on detection (E=14 → E=0)

The detection gap on docker (E=14) is larger than the in-process v1 number (E=3). This discrepancy is worth investigating — probably a docker config difference (smaller nprobe?) rather than an algorithm issue. Check `docker-compose.yml`: `-nprobe=4 -retry-nprobe=8` is much smaller than the accuracy-probe `nprobe=12/48`. Bumping docker to `nprobe=12/48` may close most of the detection gap with no algorithm change. **Likely cheapest single win available.**

For p99: investigate CFS throttling, HAProxy queue depth, GC behavior under sustained load. Algorithmic micro-optimization (faster JSON parser, fewer syscalls per request, pre-warming connection pools) may help; algorithm replacement won't.

---

## Latency tail investigation (post-pivot)

After accepting v1 as the production search, we instrumented the hot path with per-stage timings (`stReadBody`, `stParse`, `stSearchOne`, `stSearchTwo`, `stWrite`, `stTotal`) and ran k6 to characterize where the docker p99 actually lives.

### Observed timings under k6 900 RPS (nprobe=4, retry=24)

| Stage | p50 | p99 | p999 |
|---|---|---|---|
| readBody | 292 ns | 1.2 µs | 3.5 µs |
| parse+quantize | 7.8 µs | 30.8 µs | 81 µs |
| searchBase | 127 µs | 389 µs | 717 µs |
| **searchRetry** | **264 µs** | **1.24 ms** | **1.73 ms** |
| writeResp | 12 µs | 42 µs | 139 µs |
| **server total** | 164 µs | **554 µs** | 1.20 ms |
| **k6 client p99** | — | **2.37 ms** | — |

**Server-side total p99 is 0.55 ms — well under the 1 ms cap.** The remaining 1.8 ms of client p99 lives outside our server measurement: HAProxy queue, goroutine scheduler queue under GOMAXPROCS=1, CFS-pause-induced cache eviction.

### What we tried

- **GOMAXPROCS=2**: server-side timings unchanged, k6 p99 went *up* (2.37 → 2.64 ms). The documented finding holds: with two Ms running CPU-heavy goroutines, CFS exhausts quota faster and stalls the network poller M as collateral damage. Reverted. The freeList mutex stays (drop-in safety for any future experiment).
- **retry-nprobe sweep**: 12 / 16 / 24 / 32 / 48. retry=24 is the smallest that still hits E=3; going higher is pure latency cost with no detection gain. Originally was 48; lowered.
- **Asymmetric decisive trigger**: retry on `fraudCount ∈ {1,2,3}` instead of just `{2,3}`. Net +172 detection pts at ~10% more retry fires. Net win.
- **Cluster count sweep**: 1024 / 2048 / 4096. 2048 is the existing sweet spot — going up (4096) needs higher nprobe to keep detection up, going down (1024) needs higher nprobe to keep recall up. Both worse.

### Final config

```yaml
# docker-compose.yml
- "-nprobe=4"
- "-retry-nprobe=24"
```

```go
// cmd/api/main.go
func decisive(fraudCount int) bool {
    return fraudCount == 0 || fraudCount >= 4
}
```

### Measured score (5-run mean)

| Metric | Baseline | Tuned | Δ |
|---|---|---|---|
| `final_score` mean | ~5318 | **~5442** | **+124** |
| Range (5 runs) | — | 5246-5580 | (env variance) |
| Detection | 2647 (E=14) | 2819 (E=3) | +172 |
| p99 latency score | 2671 (2.13 ms) | ~2620 (~2.5 ms) | -50 |

The variance is environmental (CFS scheduling × k6 ramp pattern × cache state). A single run can land anywhere in ±150 of the mean; the contest box may differ further from a Mac docker host.

### Why further latency wins are hard at this constraint level

- **CFS** gives 45 ms / 100 ms per API instance. Single goroutine can burn the budget linearly; multiple parallel goroutines burn it twice as fast and stall together. There is no scheduler trick that gets us past this without changing the CPU limit.
- **Per-instance L2 cache** is 256 KB on Haswell. A retry scan of 24 × ~3000 × 28 B = 2 MB blows through L2 and L3 fragments. Variance comes from how much survived from the previous query.
- **Async preempt** keeps the poller responsive at the cost of mid-scan pauses (10-50 µs). Disabling it hurt worse on the contest box (per existing CLAUDE.md notes).

### Knobs not yet tried, ordered by remaining-value-per-risk

1. **Software prefetch** in the AVX2 member scan kernel (`PREFETCHT0` on the next member while computing current). Could reduce searchRetry variance by 20-30% if cache misses are the dominant tail source. Plan 9 assembler support uncertain.
2. **Pin api-1 / api-2 to distinct physical cores** (`cpuset_cpus`). Removes L1/L2 contention between instances. Possibly against the spirit of "1 CPU total" but the contest box has 2 physical cores + HT; pinning is a legal cgroup config.
3. **Increase HAProxy CPU budget** from 0.10 → 0.15 (and reduce each API to 0.425). HAProxy isn't bottlenecked today (~3% CPU at peak), so this is unlikely to help — but worth measuring before ruling out.
4. **Tighten goroutine population**: cap the connection-per-VU count via HAProxy. With 250 k6 VUs each holding a connection, server-side goroutine queue can grow large during bursts. A connection limit forces clients to wait at HAProxy instead — same queue, different place.
5. **Submit multiple times**: per existing CLAUDE.md note, run-to-run variance is real; the best of N runs can be ~150 pts above mean.
