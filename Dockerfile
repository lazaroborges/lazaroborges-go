# syntax=docker/dockerfile:1.7

# ---- Stage 1: build the preprocess + api binaries ----
FROM --platform=$BUILDPLATFORM golang:1.23-alpine AS build

WORKDIR /src

# Cache deps separately for fast incremental builds.
COPY rinha-2026/go.mod rinha-2026/go.sum* ./
RUN go mod download

COPY rinha-2026/internal ./internal
COPY rinha-2026/cmd ./cmd

ENV CGO_ENABLED=0 GOOS=linux GOARCH=amd64

RUN go build -trimpath -ldflags="-s -w" -tags=netgo -o /out/preprocess ./cmd/preprocess && \
    go build -trimpath -ldflags="-s -w" -tags=netgo -o /out/api        ./cmd/api

# ---- Stage 2: run the preprocessor against the reference dataset ----
# Produces /out/index.bin from /data/references.json.gz.
FROM build AS index

# Bring in the gzipped 14-d reference set (provided by the build context).
COPY resources/references.json.gz /data/references.json.gz

# k-means tuning: 4096 clusters × ~25 iters of mini-batch is a good speed/
# accuracy point on 3M vectors. Override at build time via --build-arg if
# accuracy validation says otherwise.
ARG CLUSTERS=2048
ARG ITERS=25
ARG BATCH=200000

RUN /out/preprocess \
        -src=/data/references.json.gz \
        -dst=/out/index.bin \
        -clusters=${CLUSTERS} \
        -iters=${ITERS} \
        -batch=${BATCH}

# ---- Stage 3: minimal runtime image ----
# scratch keeps the runtime image at ~api_binary + ~index.bin in size and
# eliminates any chance of stray processes/syscalls inside the container.
FROM scratch AS runtime

COPY --from=build /out/api       /api
COPY --from=index /out/index.bin /index.bin

# The API binds a UDS provided via -socket; HAProxy mounts the same volume.
ENTRYPOINT ["/api"]
