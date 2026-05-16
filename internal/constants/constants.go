package constants

// Normalization constants from resources/normalization.json — pinned at compile
// time so the runtime image carries no JSON.
const (
	MaxAmount             = 10000.0
	MaxInstallments       = 12.0
	AmountVsAvgRatio      = 10.0
	MaxMinutes            = 1440.0
	MaxKm                 = 1000.0
	MaxTxCount24h         = 20.0
	MaxMerchantAvgAmount  = 10000.0

	DefaultMccRisk = 0.5
)

// MccRisk mirrors resources/mcc_risk.json. Default 0.5 for unknown codes.
var MccRisk = map[string]float32{
	"5411": 0.15,
	"5812": 0.30,
	"5912": 0.20,
	"5944": 0.45,
	"7801": 0.80,
	"7802": 0.75,
	"7995": 0.85,
	"4511": 0.35,
	"5311": 0.25,
	"5999": 0.50,
}
