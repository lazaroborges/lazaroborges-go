package vector

// Timestamps in this competition come as fixed-width RFC3339 in UTC:
//   "YYYY-MM-DDTHH:MM:SSZ"   (length 20)
// We bypass time.Parse — it allocates and uses reflection — and decode the
// fields by byte offset.

func atoi2(b []byte, off int) int {
	return int(b[off]-'0')*10 + int(b[off+1]-'0')
}

func atoi4(b []byte, off int) int {
	return int(b[off]-'0')*1000 +
		int(b[off+1]-'0')*100 +
		int(b[off+2]-'0')*10 +
		int(b[off+3]-'0')
}

// parseISOHourDay returns (hour-of-day 0..23, day-of-week with Mon=0..Sun=6)
// from a "YYYY-MM-DDTHH:MM:SSZ" timestamp.
func parseISOHourDay(ts []byte) (hour int, dow int) {
	year := atoi4(ts, 0)
	month := atoi2(ts, 5)
	day := atoi2(ts, 8)
	hour = atoi2(ts, 11)

	// Zeller's congruence (Gregorian). Result: 0=Sat, 1=Sun, 2=Mon, ..., 6=Fri
	m, y := month, year
	if m < 3 {
		m += 12
		y -= 1
	}
	K := y % 100
	J := y / 100
	h := (day + 13*(m+1)/5 + K + K/4 + J/4 + 5*J) % 7
	dow = (h + 5) % 7 // remap to Mon=0..Sun=6
	return
}

// isoMinutesBetween returns floor((a - b) / minute) given two RFC3339 UTC
// timestamps. The result is always non-negative for the competition's
// past-to-present last_transaction ordering.
func isoMinutesBetween(earlier, later []byte) int {
	e := isoToUnixMinutes(earlier)
	l := isoToUnixMinutes(later)
	if l < e {
		return 0
	}
	return l - e
}

// isoToUnixMinutes returns minutes since the Unix epoch for a UTC RFC3339
// timestamp. Accurate for dates 1970–2099.
func isoToUnixMinutes(ts []byte) int {
	year := atoi4(ts, 0)
	month := atoi2(ts, 5)
	day := atoi2(ts, 8)
	hour := atoi2(ts, 11)
	minute := atoi2(ts, 14)

	// Days from Unix epoch (1970-01-01) to the given Y-M-D.
	days := daysFromEpoch(year, month, day)
	return days*24*60 + hour*60 + minute
}

// daysFromEpoch returns the number of days between 1970-01-01 and the given
// Gregorian date. Uses the well-known civil_from_days inverse from Howard
// Hinnant — exact for all Gregorian dates.
func daysFromEpoch(y, m, d int) int {
	if m <= 2 {
		y--
	}
	era := y
	if era < 0 {
		era -= 399
	}
	era /= 400
	yoe := y - era*400
	var mp int
	if m > 2 {
		mp = m - 3
	} else {
		mp = m + 9
	}
	doy := (153*mp+2)/5 + d - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	return era*146097 + doe - 719468
}
