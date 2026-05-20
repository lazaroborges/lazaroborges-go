package vectorize_test

import (
	"math"
	"testing"

	"lazaroborges-go/internal/vectorize"
)

const eps = 0.0001

func near(a, b float32) bool { return math.Abs(float64(a-b)) < eps }

func assertVec(t *testing.T, got [14]float32, want [14]float32) {
	t.Helper()
	for i := range want {
		if !near(got[i], want[i]) {
			t.Errorf("dim[%d]: got %.4f, want %.4f", i, got[i], want[i])
		}
	}
}

// tx-1329056812: legit transaction, last_transaction=null
func TestGoldenLegit(t *testing.T) {
	payload := vectorize.Payload{
		Transaction: vectorize.Transaction{
			Amount:       41.12,
			Installments: 2,
			RequestedAt:  "2026-03-11T18:45:53Z",
		},
		Customer: vectorize.Customer{
			AvgAmount:      82.24,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: vectorize.Merchant{
			ID:        "MERC-016",
			MCC:       "5411",
			AvgAmount: 60.25,
		},
		Terminal: vectorize.Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  29.2331036248,
		},
		LastTransaction: nil,
	}

	norm := vectorize.Normalization{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
	mccRisk := map[string]float32{
		"5411": 0.15, "5812": 0.30, "5912": 0.20,
	}

	got := vectorize.Vectorize(payload, norm, mccRisk)
	want := [14]float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	assertVec(t, got, want)
}

// tx-3330991687: fraud transaction, last_transaction=null
func TestGoldenFraud(t *testing.T) {
	payload := vectorize.Payload{
		Transaction: vectorize.Transaction{
			Amount:       9505.97,
			Installments: 10,
			RequestedAt:  "2026-03-14T05:15:12Z",
		},
		Customer: vectorize.Customer{
			AvgAmount:      81.28,
			TxCount24h:     20,
			KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant: vectorize.Merchant{
			ID:        "MERC-068",
			MCC:       "7802",
			AvgAmount: 54.86,
		},
		Terminal: vectorize.Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  952.27,
		},
		LastTransaction: nil,
	}

	norm := vectorize.Normalization{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
	mccRisk := map[string]float32{
		"5411": 0.15, "5812": 0.30, "5912": 0.20,
		"5944": 0.45, "7801": 0.80, "7802": 0.75,
		"7995": 0.85, "4511": 0.35, "5311": 0.25, "5999": 0.50,
	}

	got := vectorize.Vectorize(payload, norm, mccRisk)
	want := [14]float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	assertVec(t, got, want)
}
