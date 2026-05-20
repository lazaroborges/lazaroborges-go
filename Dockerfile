# Stage 1: Build Go binaries (cross-compile to linux/amd64)
FROM --platform=$BUILDPLATFORM golang:1.26-alpine AS go-builder
WORKDIR /src
COPY go.mod .
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOARCH=amd64 GOOS=linux GOAMD64=v3 \
    go build -ldflags="-s -w" -o /app/server ./
RUN CGO_ENABLED=0 GOARCH=amd64 GOOS=linux \
    go build -ldflags="-s -w" -o /app/build-index ./cmd/build-index/

# Stage 2: Build the IVF index (runs at docker build time, cached after first build)
FROM --platform=linux/amd64 golang:1.26-alpine AS index-builder
WORKDIR /app
COPY --from=go-builder /app/build-index .
COPY resources/references.json.gz ./references.json.gz
RUN ./build-index -in references.json.gz -out index.bin

# Stage 3: Minimal runtime image (Alpine for wget healthcheck)
FROM --platform=linux/amd64 alpine:3.21
RUN adduser -D appuser
WORKDIR /app
COPY --from=go-builder /app/server .
COPY --from=index-builder /app/index.bin .
COPY resources/normalization.json .
COPY resources/mcc_risk.json .

ENV INDEX_PATH=/app/index.bin
ENV NORM_PATH=/app/normalization.json
ENV MCC_PATH=/app/mcc_risk.json
ENV ADDR=:8080
ENV NPROBE=32

USER appuser
EXPOSE 8080
ENTRYPOINT ["/app/server"]
