package vectorize

import (
	"strings"
	"time"
)

type Normalization struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKm                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

type Transaction struct {
	Amount       float32 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float32  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float32 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float32 `json:"km_from_home"`
}

type LastTransaction struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float32 `json:"km_from_current"`
}

type Payload struct {
	ID              string           `json:"id"`
	Transaction     Transaction      `json:"transaction"`
	Customer        Customer         `json:"customer"`
	Merchant        Merchant         `json:"merchant"`
	Terminal        Terminal         `json:"terminal"`
	LastTransaction *LastTransaction `json:"last_transaction"`
}

func clamp(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// Vectorize transforms a Payload into a 14-dimensional vector.
// mccRisk lookup falls back to 0.5 for unknown MCCs.
func Vectorize(p Payload, n Normalization, mccRisk map[string]float32) [14]float32 {
	var v [14]float32

	// dim 0: amount
	v[0] = clamp(p.Transaction.Amount / n.MaxAmount)

	// dim 1: installments
	v[1] = clamp(float32(p.Transaction.Installments) / n.MaxInstallments)

	// dim 2: amount_vs_avg
	v[2] = clamp((p.Transaction.Amount / p.Customer.AvgAmount) / n.AmountVsAvgRatio)

	// dims 3, 4: hour and weekday from requested_at (UTC)
	t, _ := time.Parse(time.RFC3339, p.Transaction.RequestedAt)
	v[3] = float32(t.UTC().Hour()) / 23.0
	// time.Weekday: Sunday=0, Monday=1 … Saturday=6
	// Challenge: Monday=0 … Sunday=6
	wd := int(t.UTC().Weekday())
	if wd == 0 {
		wd = 6
	} else {
		wd--
	}
	v[4] = float32(wd) / 6.0

	// dims 5, 6: last_transaction
	if p.LastTransaction == nil {
		v[5] = -1
		v[6] = -1
	} else {
		tLast, _ := time.Parse(time.RFC3339, p.LastTransaction.Timestamp)
		tNow, _ := time.Parse(time.RFC3339, p.Transaction.RequestedAt)
		minutes := float32(tNow.Sub(tLast).Minutes())
		v[5] = clamp(minutes / n.MaxMinutes)
		v[6] = clamp(p.LastTransaction.KmFromCurrent / n.MaxKm)
	}

	// dim 7: km_from_home
	v[7] = clamp(p.Terminal.KmFromHome / n.MaxKm)

	// dim 8: tx_count_24h
	v[8] = clamp(float32(p.Customer.TxCount24h) / n.MaxTxCount24h)

	// dim 9: is_online
	if p.Terminal.IsOnline {
		v[9] = 1
	}

	// dim 10: card_present
	if p.Terminal.CardPresent {
		v[10] = 1
	}

	// dim 11: unknown_merchant (1 = not in known_merchants)
	v[11] = 1
	for _, m := range p.Customer.KnownMerchants {
		if strings.EqualFold(m, p.Merchant.ID) {
			v[11] = 0
			break
		}
	}

	// dim 12: mcc_risk
	if risk, ok := mccRisk[p.Merchant.MCC]; ok {
		v[12] = risk
	} else {
		v[12] = 0.5
	}

	// dim 13: merchant_avg_amount
	v[13] = clamp(p.Merchant.AvgAmount / n.MaxMerchantAvgAmount)

	return v
}

// BucketID returns the 4-bit bucket index for a 14-dim vector.
// bit3=is_online, bit2=card_present, bit1=unknown_merchant, bit0=has_last_tx
func BucketID(v [14]float32) int {
	var id int
	if v[9] >= 0.5 {
		id |= 8
	}
	if v[10] >= 0.5 {
		id |= 4
	}
	if v[11] >= 0.5 {
		id |= 2
	}
	if v[5] != -1 {
		id |= 1
	}
	return id
}

// ComparisonDims returns which of the 14 dims are compared within a given bucket.
func ComparisonDims(bucketID int) []int {
	if bucketID&1 == 0 {
		return []int{0, 1, 2, 3, 4, 7, 8, 12, 13}
	}
	return []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13}
}
