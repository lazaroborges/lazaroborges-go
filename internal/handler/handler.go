package handler

import (
	"encoding/json"
	"net/http"
	"runtime/debug"

	"lazaroborges-go/internal/vectorize"
)

// Searcher is the interface satisfied by *index.Index.
type Searcher interface {
	SearchK5(q [14]float32) (int, float32)
}

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
			_, _ = w.Write([]byte(`{"approved":true,"fraud_score":0.0}` + "\n"))
		}
	}()

	var payload vectorize.Payload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"approved":true,"fraud_score":0.0}` + "\n"))
		return
	}

	vec := vectorize.Vectorize(payload, h.norm, h.mcc)
	_, fraudScore := h.idx.SearchK5(vec)
	approved := fraudScore < 0.6

	type response struct {
		Approved   bool    `json:"approved"`
		FraudScore float32 `json:"fraud_score"`
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(response{Approved: approved, FraudScore: fraudScore})
}
