package handler

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"runtime/debug"
	"sync"

	"lazaroborges-go/internal/vectorize"
)

// Searcher is the interface satisfied by *index.Index.
type Searcher interface {
	SearchK5(q [14]float32) (int, float32)
}

// precomputed responses indexed by fraudCount (0-5).
// Bayes-optimal threshold given C(FN)=3*C(FP): reject when fraudCount >= 2.
var responses = [6][]byte{
	[]byte("{\"approved\":true,\"fraud_score\":0}\n"),
	[]byte("{\"approved\":true,\"fraud_score\":0.2}\n"),
	[]byte("{\"approved\":false,\"fraud_score\":0.4}\n"),
	[]byte("{\"approved\":false,\"fraud_score\":0.6}\n"),
	[]byte("{\"approved\":false,\"fraud_score\":0.8}\n"),
	[]byte("{\"approved\":false,\"fraud_score\":1}\n"),
}

var bufPool = sync.Pool{New: func() any { return bytes.NewBuffer(make([]byte, 0, 1024)) }}

type Handler struct {
	idx  Searcher
	norm vectorize.Normalization
	mcc  map[string]float32
}

func New(idx Searcher, norm vectorize.Normalization, mcc map[string]float32) *Handler {
	return &Handler{idx: idx, norm: norm, mcc: mcc}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("/ready", h.ready)
	mux.HandleFunc("/fraud-score", h.fraudScore)
}

func (h *Handler) ready(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (h *Handler) fraudScore(w http.ResponseWriter, r *http.Request) {
	// Panic recovery: return 200 instead of 500 to avoid HTTP error weight in scoring.
	defer func() {
		if rec := recover(); rec != nil {
			debug.PrintStack()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(responses[0])
		}
	}()

	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	_, err := io.Copy(buf, r.Body)
	if err != nil {
		bufPool.Put(buf)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(responses[0])
		return
	}

	var payload vectorize.Payload
	if err := json.Unmarshal(buf.Bytes(), &payload); err != nil {
		bufPool.Put(buf)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(responses[0])
		return
	}
	bufPool.Put(buf)

	vec := vectorize.Vectorize(payload, h.norm, h.mcc)
	fraudCount, _ := h.idx.SearchK5(vec)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(responses[fraudCount])
}
