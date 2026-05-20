package handler_test

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"lazaroborges-go/internal/handler"
	"lazaroborges-go/internal/vectorize"
)

type mockIndex struct{ fraudCount int }

func (m *mockIndex) SearchK5(_ [14]float32) (int, float32) {
	return m.fraudCount, float32(m.fraudCount) / 5.0
}

func TestReady(t *testing.T) {
	h := handler.New(&mockIndex{}, vectorize.Normalization{}, nil)
	mux := http.NewServeMux()
	h.Register(mux)

	req := httptest.NewRequest("GET", "/ready", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
}

func TestFraudScoreMalformedJSON(t *testing.T) {
	h := handler.New(&mockIndex{}, vectorize.Normalization{}, nil)
	mux := http.NewServeMux()
	h.Register(mux)

	req := httptest.NewRequest("POST", "/fraud-score", bytes.NewBufferString("{bad json"))
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 on bad JSON, got %d", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), "approved") {
		t.Errorf("expected 'approved' in response, got %q", rr.Body.String())
	}
}

func TestFraudScoreApproved(t *testing.T) {
	// 0 fraud neighbors → approved
	h := handler.New(&mockIndex{fraudCount: 0}, vectorize.Normalization{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}, map[string]float32{"5411": 0.15})
	mux := http.NewServeMux()
	h.Register(mux)

	body := `{"id":"tx-1","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}`
	req := httptest.NewRequest("POST", "/fraud-score", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), `"approved":true`) {
		t.Errorf("expected approved:true, got %q", rr.Body.String())
	}
}

func TestFraudScoreDenied(t *testing.T) {
	// 5 fraud neighbors → denied
	h := handler.New(&mockIndex{fraudCount: 5}, vectorize.Normalization{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}, map[string]float32{"7802": 0.75})
	mux := http.NewServeMux()
	h.Register(mux)

	body := `{"id":"tx-2","transaction":{"amount":9505.97,"installments":10,"requested_at":"2026-03-14T05:15:12Z"},"customer":{"avg_amount":81.28,"tx_count_24h":20,"known_merchants":["MERC-008"]},"merchant":{"id":"MERC-068","mcc":"7802","avg_amount":54.86},"terminal":{"is_online":false,"card_present":true,"km_from_home":952.27},"last_transaction":null}`
	req := httptest.NewRequest("POST", "/fraud-score", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), `"approved":false`) {
		t.Errorf("expected approved:false, got %q", rr.Body.String())
	}
}
