package response

import (
	"bytes"
	"strconv"
)

// Per docs/en/DETECTION_RULES.md, fraud_score is always count/5 ∈ {0.0, 0.2,
// 0.4, 0.6, 0.8, 1.0} and approved is fixed by `score < 0.6`. That gives
// exactly 6 valid response shapes, indexed by the integer fraud count [0..5].
//
// We pre-allocate all six bodies as package-level []byte constants. The
// handler does one map-free lookup and one `w.Write`. Zero allocation on the
// response side.

var Bodies = [6][]byte{
	[]byte(`{"approved":true,"fraud_score":0.0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1.0}`),
}

// Fallback is the soft-fail body used on internal errors. Per the scoring
// shape (`Err` is weighted 5× in detection error), returning `approved:true`
// with a default score is strictly better than emitting a 5xx.
var Fallback = Bodies[0]

// Frames are the full HTTP/1.1 200 OK responses (status line + headers + body)
// for each fraud_count, indexed [0..5]. Pre-built at init so the hot path is
// one `conn.Write(Frames[count])` per request: zero allocation, zero header
// serialization, zero state machine. Used by cmd/api's custom HTTP/1.1 server.
var Frames [6][]byte

// ReadyFrame is a simple 200 OK for the /ready health check.
var ReadyFrame = []byte("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}")

// FallbackFrame is the canonical fallback response framed as a full HTTP/1.1
// reply. Soft-fail in the same shape as a normal 200 OK.
var FallbackFrame []byte

func init() {
	for i, body := range Bodies {
		Frames[i] = buildFrame(body)
	}
	FallbackFrame = Frames[0]
}

func buildFrame(body []byte) []byte {
	var b bytes.Buffer
	b.WriteString("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ")
	b.WriteString(strconv.Itoa(len(body)))
	b.WriteString("\r\n\r\n")
	b.Write(body)
	return b.Bytes()
}
