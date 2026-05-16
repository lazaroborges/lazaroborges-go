package vector

import (
	"errors"

	"github.com/lazaroborges/rinha-de-backend-2026/internal/constants"
)

// Dim is the fixed dimensionality of the fraud vector.
const Dim = 14

// Quantize maps a 14-d float vector (with `-1` sentinels allowed at indices 5
// and 6) into the int16 layout used by the IVF index.
//
//	x ∈ [-1, 1] → int16(x * 32767)
//
// The sentinel -1 round-trips to -32767, distinguishable from any value
// produced via clamp(0..1) → [0, 32767].
func Quantize(in *[Dim]float32, out *[Dim]int16) {
	for i := 0; i < Dim; i++ {
		v := in[i] * 32767
		if v > 32767 {
			v = 32767
		} else if v < -32767 {
			v = -32767
		}
		out[i] = int16(v)
	}
}

// NormalizePayload parses a /fraud-score JSON body and writes the 14-dim
// normalized float vector into `out`. The parser is order-insensitive: fields
// may appear in any order within their containing object.
func NormalizePayload(body []byte, out *[Dim]float32) error {
	p := parser{data: body}
	p.skipWS()
	if !p.expectByte('{') {
		return errors.New("expected object")
	}

	var (
		amount         float32
		installments   float32
		curTimestamp   []byte
		custAvgAmount  float32
		txCount24h     float32
		merchantID     []byte
		knownArr       [32][]byte
		knownN         int
		mccCode        []byte
		merchAvg       float32
		kmFromHome     float32
		isOnline       bool
		cardPresent   bool
		haveLastTx     bool
		lastTimestamp  []byte
		kmFromCurrent  float32
	)

	for {
		p.skipWS()
		if p.peek() == '}' {
			p.pos++
			break
		}
		if p.peek() != '"' {
			return errors.New("expected key")
		}
		key := p.readString()
		p.skipWS()
		if !p.expectByte(':') {
			return errors.New("expected ':'")
		}
		p.skipWS()

		switch string(key) {
		case "id":
			_ = p.readString()

		case "transaction":
			err := p.forEachObjectField(func(k []byte) error {
				switch string(k) {
				case "amount":
					amount = p.readFloat()
				case "installments":
					installments = p.readFloat()
				case "requested_at":
					curTimestamp = p.readString()
				default:
					p.skipValue()
				}
				return nil
			})
			if err != nil {
				return err
			}

		case "customer":
			err := p.forEachObjectField(func(k []byte) error {
				switch string(k) {
				case "avg_amount":
					custAvgAmount = p.readFloat()
				case "tx_count_24h":
					txCount24h = p.readFloat()
				case "known_merchants":
					if !p.expectByte('[') {
						return errors.New("expected known_merchants array")
					}
					for {
						p.skipWS()
						if p.peek() == ']' {
							p.pos++
							break
						}
						if p.peek() != '"' {
							return errors.New("expected merchant string")
						}
						m := p.readString()
						if knownN < len(knownArr) {
							knownArr[knownN] = m
							knownN++
						}
						p.skipWS()
						if p.peek() == ',' {
							p.pos++
						}
					}
				default:
					p.skipValue()
				}
				return nil
			})
			if err != nil {
				return err
			}

		case "merchant":
			err := p.forEachObjectField(func(k []byte) error {
				switch string(k) {
				case "id":
					merchantID = p.readString()
				case "mcc":
					mccCode = p.readString()
				case "avg_amount":
					merchAvg = p.readFloat()
				default:
					p.skipValue()
				}
				return nil
			})
			if err != nil {
				return err
			}

		case "terminal":
			err := p.forEachObjectField(func(k []byte) error {
				switch string(k) {
				case "is_online":
					isOnline = p.readBool()
				case "card_present":
					cardPresent = p.readBool()
				case "km_from_home":
					kmFromHome = p.readFloat()
				default:
					p.skipValue()
				}
				return nil
			})
			if err != nil {
				return err
			}

		case "last_transaction":
			if p.peek() == 'n' {
				p.pos += 4 // "null"
				haveLastTx = false
			} else {
				haveLastTx = true
				err := p.forEachObjectField(func(k []byte) error {
					switch string(k) {
					case "timestamp":
						lastTimestamp = p.readString()
					case "km_from_current":
						kmFromCurrent = p.readFloat()
					default:
						p.skipValue()
					}
					return nil
				})
				if err != nil {
					return err
				}
			}

		default:
			p.skipValue()
		}

		p.skipWS()
		if p.peek() == ',' {
			p.pos++
		}
	}

	if len(curTimestamp) < 13 {
		return errors.New("missing transaction.requested_at")
	}
	hour, dow := parseISOHourDay(curTimestamp)

	out[0] = clamp01(amount / constants.MaxAmount)
	out[1] = clamp01(installments / constants.MaxInstallments)
	if custAvgAmount > 0 {
		out[2] = clamp01((amount / custAvgAmount) / constants.AmountVsAvgRatio)
	} else {
		out[2] = 1.0
	}
	out[3] = float32(hour) / 23.0
	out[4] = float32(dow) / 6.0
	if haveLastTx {
		mins := isoMinutesBetween(lastTimestamp, curTimestamp)
		out[5] = clamp01(float32(mins) / constants.MaxMinutes)
		out[6] = clamp01(kmFromCurrent / constants.MaxKm)
	} else {
		out[5] = -1
		out[6] = -1
	}
	out[7] = clamp01(kmFromHome / constants.MaxKm)
	out[8] = clamp01(txCount24h / constants.MaxTxCount24h)
	if isOnline {
		out[9] = 1
	} else {
		out[9] = 0
	}
	if cardPresent {
		out[10] = 1
	} else {
		out[10] = 0
	}
	out[11] = 1
	for i := 0; i < knownN; i++ {
		if bytesEq(knownArr[i], merchantID) {
			out[11] = 0
			break
		}
	}
	out[12] = constants.DefaultMccRisk
	if r, ok := constants.MccRisk[string(mccCode)]; ok {
		out[12] = r
	}
	out[13] = clamp01(merchAvg / constants.MaxMerchantAvgAmount)

	return nil
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func bytesEq(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
