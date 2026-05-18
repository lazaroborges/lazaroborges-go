package main

import (
	"errors"
	"io"
	"net"
	"time"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/index"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/response"
	"github.com/lazaroborges/rinha-de-backend-2026/internal/vector"
)

// Custom HTTP/1.1 server for the /fraud-score hot path.
//
// Why we don't use net/http here: every net/http request allocates a header
// map, instantiates a response state machine, serialises the response headers
// (including computed Content-Length and Date), and writes through a bufio
// pipeline. On Haswell at 0.45 CPU that overhead is large enough relative to
// the actual search work that it dominated p99 — see plan notes.
//
// This loop talks raw HTTP/1.1 on the UDS socket between HAProxy (TCP mode)
// and the API. HAProxy splices bytes, so what arrives here is exactly what
// k6 sent: `POST /fraud-score HTTP/1.1\r\nHost: …\r\nContent-Length: N\r\n
// Content-Type: application/json\r\n\r\n{…}`. We only need two things: the
// body (between \r\n\r\n and Content-Length bytes later) and Content-Length
// itself.

const (
	maxConnBufSize = 8192 // larger than any expected request; rejects pathological inputs
	maxBodySize    = 4096
)

type connBuf struct {
	data []byte
	used int // bytes valid in data[:used]
	rpos int // next byte to consume in data[rpos:used]
}

var connBufPool = newFreeList(func() any {
	return &connBuf{data: make([]byte, maxConnBufSize)}
})

// serveUDS runs a custom accept loop on `ln`. One goroutine per accepted
// connection, looping for the lifetime of that keep-alive connection.
func serveUDS(ln net.Listener) error {
	for {
		c, err := ln.Accept()
		if err != nil {
			return err
		}
		go handleConn(c)
	}
}

func handleConn(c net.Conn) {
	b := connBufPool.Get().(*connBuf)
	b.used = 0
	b.rpos = 0

	for {
		if !processOne(c, b) {
			break
		}
	}
	c.Close()
	connBufPool.Put(b)
}

// processOne reads one HTTP/1.1 request from `c` into `b`, runs the search
// pipeline, writes the response frame, and returns true if the connection
// should be kept alive for the next request.
//
// Records per-stage timings into the debug rings so we can spot where p99
// tail latency lives. stReadBody covers the network read for body bytes;
// stTotal covers the in-server portion of one request (does not include
// HAProxy queue, TCP RTT to/from client, or scheduler wait time).
func processOne(c net.Conn, b *connBuf) bool {
	headerEnd, err := readUntilHeaderEnd(c, b)
	if err != nil {
		return false
	}
	t0 := time.Now()

	headers := b.data[b.rpos : b.rpos+headerEnd]
	bodyStart := b.rpos + headerEnd + 4

	// Handle /ready health check.
	if len(headers) >= 10 && (headers[0]|0x20 == 'g') && (headers[5]|0x20 == 'r') {
		if _, err := c.Write(response.ReadyFrame); err != nil {
			return false
		}
		b.rpos = bodyStart
		return true
	}

	cl := findContentLength(headers)
	if cl < 0 || cl > maxBodySize {
		_, _ = c.Write(response.FallbackFrame)
		return false
	}

	if err := ensureBody(c, b, bodyStart, cl); err != nil {
		return false
	}
	body := b.data[bodyStart : bodyStart+cl]

	frame := processSearch(body)

	if _, err := c.Write(frame); err != nil {
		return false
	}

	stTotal.record(int64(time.Since(t0)))

	consumed := bodyStart + cl
	if consumed == b.used {
		b.rpos = 0
		b.used = 0
	} else {
		b.rpos = consumed
	}
	return true
}

// processSearch dispatches to the v1 (IVF) or v2 (IVF-HNSW) pipeline
// depending on the -search flag set at startup.
func processSearch(body []byte) []byte {
	if useIVFHNSW {
		return processSearchV2(body)
	}
	return processSearchV1(body)
}

func processSearchV1(body []byte) []byte {
	qFloat := qFloatPool.Get().(*[16]float32)
	if err := vector.NormalizePayload(body, qFloat); err != nil {
		qFloatPool.Put(qFloat)
		return response.FallbackFrame
	}
	qInt := qInt16Pool.Get().(*[16]int16)
	vector.Quantize(qFloat, qInt)

	top := top5Pool.Get().(*index.Top5)
	cellBuf := cellBufPool.Get().(*[]index.CentroidDist)
	distBuf := distBufPool.Get().(*[]int64)

	idx.SearchIVF(qInt, qFloat, baseNprobe, *cellBuf, *distBuf, top)
	if !decisive(top.FraudCount()) {
		idx.SearchIVF(qInt, qFloat, retryNprobe, *cellBuf, *distBuf, top)
	}

	frame := response.Frames[top.FraudCount()]

	qFloatPool.Put(qFloat)
	qInt16Pool.Put(qInt)
	top5Pool.Put(top)
	cellBufPool.Put(cellBuf)
	distBufPool.Put(distBuf)

	return frame
}

// processSearchV2 runs the IVF-HNSW pipeline on `body` and returns one of
// the pre-built response frames. Errors short-circuit to FallbackFrame.
func processSearchV2(body []byte) []byte {
	qFloat := qFloatPool.Get().(*[16]float32)
	if err := vector.NormalizePayload(body, qFloat); err != nil {
		qFloatPool.Put(qFloat)
		return response.FallbackFrame
	}
	qInt8 := qInt8Pool.Get().(*[14]int8)
	vector.QuantizeInt8(qFloat, qInt8)

	scratch := scratchPool.Get().(*index.SearchScratch)
	top := top5FinalPool.Get().(*index.Top5Final)

	idx.SearchIVFHNSW(qFloat, qInt8, baseNprobe, baseEf, scratch, top)
	if !decisive(top.FraudCount()) {
		idx.SearchIVFHNSW(qFloat, qInt8, retryNprobe, retryEf, scratch, top)
	}

	frame := response.Frames[top.FraudCount()]

	qFloatPool.Put(qFloat)
	qInt8Pool.Put(qInt8)
	scratchPool.Put(scratch)
	top5FinalPool.Put(top)

	return frame
}

func readUntilHeaderEnd(c net.Conn, b *connBuf) (int, error) {
	for {
		if idx := findHeaderEnd(b.data[b.rpos:b.used]); idx >= 0 {
			return idx, nil
		}
		// Need more bytes. Compact if we're out of free space at the tail.
		if len(b.data)-b.used == 0 {
			if b.rpos == 0 {
				return -1, errors.New("headers too large")
			}
			n := copy(b.data, b.data[b.rpos:b.used])
			b.rpos = 0
			b.used = n
		}
		n, err := c.Read(b.data[b.used:])
		if err != nil {
			return -1, err
		}
		if n == 0 {
			return -1, io.EOF
		}
		b.used += n
	}
}

// ensureBody blocks until `bodyStart + cl` bytes are present in `b.data`.
func ensureBody(c net.Conn, b *connBuf, bodyStart, cl int) error {
	need := bodyStart + cl
	for b.used < need {
		if len(b.data)-b.used == 0 {
			// Compact: shift unread bytes (which includes the partial body) to
			// the front. bodyStart shifts by the same amount.
			if b.rpos == 0 {
				return errors.New("body too large to buffer")
			}
			shift := b.rpos
			n := copy(b.data, b.data[b.rpos:b.used])
			b.rpos = 0
			b.used = n
			bodyStart -= shift
			need -= shift
		}
		n, err := c.Read(b.data[b.used:])
		if err != nil {
			return err
		}
		if n == 0 {
			return io.EOF
		}
		b.used += n
	}
	return nil
}

// findHeaderEnd returns the offset of the first byte of "\r\n\r\n" in p, or
// -1 if not present. The four delimiter bytes are not included in the result.
func findHeaderEnd(p []byte) int {
	// Linear scan. Headers are <1 KB in practice so vectorising buys nothing.
	if len(p) < 4 {
		return -1
	}
	for i := 0; i <= len(p)-4; i++ {
		if p[i] == '\r' && p[i+1] == '\n' && p[i+2] == '\r' && p[i+3] == '\n' {
			return i
		}
	}
	return -1
}

var contentLengthKey = []byte("\r\ncontent-length:")

// findContentLength returns the request's Content-Length, or -1 if absent.
// Case-insensitive ASCII match against `\r\nContent-Length:`. We require the
// `\r\n` prefix so we don't match the request line; the caller passes only
// the header bytes (request line + headers, no body, no terminating CRLFs).
func findContentLength(headers []byte) int {
	for i := 0; i+len(contentLengthKey) <= len(headers); i++ {
		if matchLowerASCII(headers[i:i+len(contentLengthKey)], contentLengthKey) {
			j := i + len(contentLengthKey)
			for j < len(headers) && (headers[j] == ' ' || headers[j] == '\t') {
				j++
			}
			n := 0
			for j < len(headers) && headers[j] >= '0' && headers[j] <= '9' {
				n = n*10 + int(headers[j]-'0')
				j++
			}
			return n
		}
	}
	return -1
}

// matchLowerASCII compares `a` against `lowerB` treating ASCII letters in `a`
// as case-insensitive. `lowerB` must be all lowercase.
func matchLowerASCII(a, lowerB []byte) bool {
	if len(a) != len(lowerB) {
		return false
	}
	for i := range a {
		ai := a[i]
		if ai >= 'A' && ai <= 'Z' {
			ai += 32
		}
		if ai != lowerB[i] {
			return false
		}
	}
	return true
}
