# rinha-de-backend-2026 — Lazaro

Go submission targeting the score ceiling. See [CLAUDE.md](./CLAUDE.md) for the
full architecture writeup.

## Topology

```
client → HAProxy :9999 (TCP mode, round-robin)
          ├── /var/run/sockets/api-1.sock  →  Go API #1
          └── /var/run/sockets/api-2.sock  →  Go API #2
```

Each API instance mmaps the same `index.bin` (IVF + int16 quantization,
~85 MB on disk for 3M reference vectors).

## Resource budget (1 CPU + 350 MB cap)

| Service | CPU | Memory |
| ---     | --- | ---    |
| haproxy | 0.10 | 25 MB |
| api-1   | 0.45 | 160 MB |
| api-2   | 0.45 | 160 MB |
| total   | 1.00 | 345 MB |

## Build & run locally

```bash
# Symlink the dataset that ships with the rinha-de-backend-2026 repo:
ln -sf ../../resources/references.json.gz data/references.json.gz

# Build & start (the preprocess step runs k-means inside the Docker build):
docker compose build
docker compose up

# In another terminal, from the rinha-de-backend-2026 repo root:
./run.sh
```

## Layout

```
cmd/
  api/         HTTP server (stdlib net/http over Unix Domain Socket)
  preprocess/  k-means (mini-batch) + index.bin serializer, runs at build time
internal/
  constants/   Normalization scalars + MCC risk table (compile-time)
  index/       IVF index load (mmap) + adaptive nprobe search + Top-5 voting
  response/    Pre-allocated response bodies (zero-alloc on the response side)
  vector/      JSON→14d normalize, int16 quantize, ISO timestamp helpers
deploy/
  haproxy.cfg  TCP-mode round-robin over the two API UDS sockets
Dockerfile         Multi-stage: build → preprocess → scratch runtime
Dockerfile.haproxy Pulls haproxy:2.9-alpine and bakes in haproxy.cfg
docker-compose.yml
info.json          Required by the Rinha submission format
```

## Tests

```bash
go test ./...
```

Validates: vectorization against both `DETECTION_RULES.md` examples
(legit + fraud), weekday calculations, ISO minute differences, and the
fixed-size top-5 sort.
