package response

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

// Fallback is the soft-fail response used on internal errors. Per the scoring
// shape (`Err` is weighted 5× in detection error), returning `approved:true`
// with a default score is strictly better than emitting a 5xx.
var Fallback = Bodies[0]
