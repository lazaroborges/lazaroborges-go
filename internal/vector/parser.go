package vector

import "errors"

// parser is a zero-allocation, single-pass JSON scanner specialized for the
// /fraud-score payload shape. It does NOT validate strict JSON — it scans
// fast and trusts the caller to feed well-formed input. Malformed input
// returns errors but may also produce nonsensical state.
type parser struct {
	data []byte
	pos  int
}

func (p *parser) peek() byte {
	if p.pos >= len(p.data) {
		return 0
	}
	return p.data[p.pos]
}

func (p *parser) skipWS() {
	for p.pos < len(p.data) {
		c := p.data[p.pos]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			p.pos++
			continue
		}
		return
	}
}

func (p *parser) expectByte(b byte) bool {
	if p.pos < len(p.data) && p.data[p.pos] == b {
		p.pos++
		return true
	}
	return false
}

// readString consumes the opening `"`, reads until the closing `"`, and
// returns the bytes between them. Does not unescape — payloads here use
// ASCII for ids/mccs/timestamps so this is safe.
func (p *parser) readString() []byte {
	if p.pos < len(p.data) && p.data[p.pos] == '"' {
		p.pos++
	}
	start := p.pos
	for p.pos < len(p.data) {
		c := p.data[p.pos]
		if c == '"' {
			s := p.data[start:p.pos]
			p.pos++ // closing quote
			return s
		}
		if c == '\\' {
			p.pos += 2 // skip escape
			continue
		}
		p.pos++
	}
	return p.data[start:p.pos]
}

// readFloat parses a JSON number into float32. Supports decimal, leading sign,
// no exponent (the payloads in this competition use plain decimals).
func (p *parser) readFloat() float32 {
	start := p.pos
	if p.pos < len(p.data) && (p.data[p.pos] == '-' || p.data[p.pos] == '+') {
		p.pos++
	}
	for p.pos < len(p.data) {
		c := p.data[p.pos]
		if (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+' {
			p.pos++
			continue
		}
		break
	}
	return atof32(p.data[start:p.pos])
}

func (p *parser) readBool() bool {
	if p.pos >= len(p.data) {
		return false
	}
	c := p.data[p.pos]
	if c == 't' {
		p.pos += 4
		return true
	}
	p.pos += 5
	return false
}

// skipValue advances past a JSON value (object, array, string, number, bool,
// null) at the current position.
func (p *parser) skipValue() {
	p.skipWS()
	if p.pos >= len(p.data) {
		return
	}
	switch p.data[p.pos] {
	case '"':
		p.readString()
	case '{':
		p.pos++
		depth := 1
		for p.pos < len(p.data) && depth > 0 {
			c := p.data[p.pos]
			switch c {
			case '"':
				p.readString()
				continue
			case '{', '[':
				depth++
			case '}', ']':
				depth--
			}
			p.pos++
		}
	case '[':
		p.pos++
		depth := 1
		for p.pos < len(p.data) && depth > 0 {
			c := p.data[p.pos]
			switch c {
			case '"':
				p.readString()
				continue
			case '{', '[':
				depth++
			case '}', ']':
				depth--
			}
			p.pos++
		}
	case 't':
		p.pos += 4
	case 'f':
		p.pos += 5
	case 'n':
		p.pos += 4
	default:
		p.readFloat()
	}
}

// forEachObjectField expects to be positioned at '{' and invokes fn for each
// key in the object, with fn responsible for consuming the value.
func (p *parser) forEachObjectField(fn func(key []byte) error) error {
	if !p.expectByte('{') {
		return errors.New("expected object")
	}
	for {
		p.skipWS()
		if p.peek() == '}' {
			p.pos++
			return nil
		}
		if p.peek() != '"' {
			return errors.New("expected key string")
		}
		k := p.readString()
		p.skipWS()
		if !p.expectByte(':') {
			return errors.New("expected ':'")
		}
		p.skipWS()
		if err := fn(k); err != nil {
			return err
		}
		p.skipWS()
		if p.peek() == ',' {
			p.pos++
		}
	}
}

// atof32 parses an ASCII numeric byte slice into float32 without allocation.
// Supports decimal point and optional exponent.
func atof32(b []byte) float32 {
	if len(b) == 0 {
		return 0
	}
	i := 0
	neg := false
	switch b[0] {
	case '-':
		neg = true
		i++
	case '+':
		i++
	}
	var intPart float64
	for ; i < len(b); i++ {
		c := b[i]
		if c < '0' || c > '9' {
			break
		}
		intPart = intPart*10 + float64(c-'0')
	}
	var fracPart float64
	var fracDiv float64 = 1
	if i < len(b) && b[i] == '.' {
		i++
		for ; i < len(b); i++ {
			c := b[i]
			if c < '0' || c > '9' {
				break
			}
			fracPart = fracPart*10 + float64(c-'0')
			fracDiv *= 10
		}
	}
	value := intPart + fracPart/fracDiv
	if i < len(b) && (b[i] == 'e' || b[i] == 'E') {
		i++
		expNeg := false
		switch b[i] {
		case '-':
			expNeg = true
			i++
		case '+':
			i++
		}
		exp := 0
		for ; i < len(b); i++ {
			c := b[i]
			if c < '0' || c > '9' {
				break
			}
			exp = exp*10 + int(c-'0')
		}
		mult := 1.0
		for k := 0; k < exp; k++ {
			mult *= 10
		}
		if expNeg {
			value /= mult
		} else {
			value *= mult
		}
	}
	if neg {
		value = -value
	}
	return float32(value)
}
