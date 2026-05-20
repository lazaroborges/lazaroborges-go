package index

import (
	"os"
	"testing"
)

// TestIntegrationRoundTrip requires index.bin to exist (built by Task 6).
// Verifies the two golden vectors from DETECTION_RULES.md against the real index.
func TestIntegrationRoundTrip(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test — requires index.bin")
	}
	if _, err := os.Stat("../../index.bin"); err != nil {
		t.Skip("index.bin not found — run build-index first")
	}

	idx, err := Open("../../index.bin", NProbe)
	if err != nil {
		t.Fatal("open:", err)
	}
	defer idx.Close()

	// Golden legit: tx-1329056812 → expected approved (fraud_score < 0.6)
	legit := [14]float32{0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	fc, score := idx.SearchK5(legit)
	t.Logf("golden legit: fraudCount=%d score=%.2f", fc, score)
	if score >= 0.6 {
		t.Errorf("expected legit (score < 0.6), got %.2f", score)
	}

	// Golden fraud: tx-3330991687 → expected denied (fraud_score >= 0.6)
	fraud := [14]float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}
	fc2, score2 := idx.SearchK5(fraud)
	t.Logf("golden fraud: fraudCount=%d score=%.2f", fc2, score2)
	if score2 < 0.6 {
		t.Errorf("expected fraud (score >= 0.6), got %.2f", score2)
	}
}
