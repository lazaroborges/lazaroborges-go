package main

import (
	"errors"
	"io"
	"net"
	"sync"

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

var connBufPool = sync.Pool{New: func() any {
	return &connBuf{data: make([]byte, maxConnBufSize)}
}}

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
	defer c.Close()
	b := connBufPool.Get().(*connBuf)
	defer connBufPool.Put(b)
	b.used = 0
	b.rpos = 0

	for {
		if !processOne(c, b) {
			return
		}
	}
}

// processOne reads one HTTP/1.1 request from `c` into `b`, runs the search
// pipeline, writes the response frame, and returns true if the connection
// should be kept alive for the next request.
func processOne(c net.Conn, b *connBuf) bool {
	headerEnd, err := readUntilHeaderEnd(c, b)
	if err != nil {
		// Clean EOF on a keep-alive idle conn is the normal way a connection
		// ends. Don't write a fallback — the peer is gone.
		return false
	}
	headers := b.data[b.rpos : b.rpos+headerEnd]
	bodyStart := b.rpos + headerEnd + 4

	// Handle /ready health check.
	if len(headers) >= 10 && matchLowerASCII(headers[:10], []byte("get /ready")) {
		if _, err := c.Write(response.ReadyFrame); err != nil {
			return false
		}
		// /ready usually isn't pipelined, but if it is, consume the header.
		// Health checks don't have bodies.
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

	// Consume this request from the buffer; any extra bytes belong to the
	// next pipelined request and stay around for the next iteration.
	consumed := bodyStart + cl
	if consumed == b.used {
		b.rpos = 0
		b.used = 0
	} else {
		b.rpos = consumed
	}
	return true
}

// processSearch runs the existing IVF pipeline on `body` and returns one of
// the pre-built response frames. Errors short-circuit to FallbackFrame so we
// never emit a 5xx (per scoring shape, 5xx is weighted 5× in detection error).
func processSearch(body []byte) []byte {
	qFloat := qFloatPool.Get().(*[vector.Dim]float32)
	defer qFloatPool.Put(qFloat)
	if err := vector.NormalizePayload(body, qFloat); err != nil {
		return response.FallbackFrame
	}
	qInt := qInt16Pool.Get().(*[vector.Dim]int16)
	defer qInt16Pool.Put(qInt)
	vector.Quantize(qFloat, qInt)

	top := top5Pool.Get().(*index.Top5)
	defer top5Pool.Put(top)
	cellBuf := cellBufPool.Get().(*[]index.CentroidDist)
	defer cellBufPool.Put(cellBuf)
	distBuf := distBufPool.Get().(*[]int64)
	defer distBufPool.Put(distBuf)

	idx.SearchIVF(qInt, qFloat, baseNprobe, *cellBuf, *distBuf, top)
	if !decisive(top.FraudCount()) {
		idx.SearchIVF(qInt, qFloat, retryNprobe, *cellBuf, *distBuf, top)
	}

	return response.Frames[top.FraudCount()]
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
