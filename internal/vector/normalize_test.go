package vector

import (
	"math"
	"testing"
)

// The legitimate example from docs/en/DETECTION_RULES.md.
// payload → vector [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
func TestNormalizePayload_LegitExample(t *testing.T) {
	body := []byte(`{
      "id": "tx-1329056812",
      "transaction": {"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},
      "customer": {"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},
      "merchant": {"id":"MERC-016","mcc":"5411","avg_amount":60.25},
      "terminal": {"is_online":false,"card_present":true,"km_from_home":29.2331036248},
      "last_transaction": null
    }`)
	var v [Dim]float32
	if err := NormalizePayload(body, &v); err != nil {
		t.Fatalf("normalize: %v", err)
	}
	want := [Dim]float32{
		0.004112, 0.1667, 0.05, 0.7826, 0.3333, -1, -1,
		0.0292, 0.15, 0, 1, 0, 0.15, 0.006025,
	}
	for i := 0; i < Dim; i++ {
		if math.Abs(float64(v[i]-want[i])) > 1e-3 {
			t.Errorf("dim %d = %f, want %f", i, v[i], want[i])
		}
	}
}

// The fraud example from docs/en/DETECTION_RULES.md.
// vector [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
func TestNormalizePayload_FraudExample(t *testing.T) {
	body := []byte(`{
      "id":"tx-3330991687",
      "transaction":{"amount":9505.97,"installments":10,"requested_at":"2026-03-14T05:15:12Z"},
      "customer":{"avg_amount":81.28,"tx_count_24h":20,"known_merchants":["MERC-008","MERC-007","MERC-005"]},
      "merchant":{"id":"MERC-068","mcc":"7802","avg_amount":54.86},
      "terminal":{"is_online":false,"card_present":true,"km_from_home":952.27},
      "last_transaction":null
    }`)
	var v [Dim]float32
	if err := NormalizePayload(body, &v); err != nil {
		t.Fatalf("normalize: %v", err)
	}
	want := [Dim]float32{
		0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1,
		0.9523, 1.0, 0, 1, 1, 0.75, 0.005486,
	}
	for i := 0; i < Dim; i++ {
		if math.Abs(float64(v[i]-want[i])) > 1e-3 {
			t.Errorf("dim %d = %f, want %f", i, v[i], want[i])
		}
	}
}

// Verify Zeller-derived weekday matches Go's stdlib for a span of dates.
func TestParseISOHourDay_Weekday(t *testing.T) {
	cases := []struct {
		ts   string
		hour int
		dow  int
	}{
		// 2026-03-11 is a Wednesday → Mon=0 → dow=2
		{"2026-03-11T18:45:53Z", 18, 2},
		// 2026-03-14 is a Saturday → dow=5
		{"2026-03-14T05:15:12Z", 5, 5},
		// 2026-01-01 is a Thursday → dow=3
		{"2026-01-01T00:00:00Z", 0, 3},
		// 2024-02-29 (leap) is a Thursday → dow=3
		{"2024-02-29T12:34:56Z", 12, 3},
	}
	for _, c := range cases {
		h, d := parseISOHourDay([]byte(c.ts))
		if h != c.hour || d != c.dow {
			t.Errorf("%s: got (%d,%d), want (%d,%d)", c.ts, h, d, c.hour, c.dow)
		}
	}
}

func TestIsoMinutesBetween(t *testing.T) {
	// 2026-03-11T14:58:35Z → 2026-03-11T20:23:35Z = 325 minutes
	got := isoMinutesBetween(
		[]byte("2026-03-11T14:58:35Z"),
		[]byte("2026-03-11T20:23:35Z"),
	)
	if got != 325 {
		t.Errorf("got %d minutes, want 325", got)
	}
}
