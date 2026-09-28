import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pathsetup  # noqa: F401

from live.duration import format_remaining, parse_duration, parse_until, resolve_end_time


EASTERN = ZoneInfo('America/New_York')


class ParseUntilTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 24, 22, 0, tzinfo=EASTERN)

    def test_us_short_year_with_am(self):
        end = parse_until('8/25/26 12:47 AM', now=self.now)
        self.assertEqual(end, datetime(2026, 8, 25, 0, 47, tzinfo=EASTERN))

    def test_us_four_digit_year_with_pm(self):
        end = parse_until('8/25/2026 1:05 PM', now=self.now)
        self.assertEqual(end, datetime(2026, 8, 25, 13, 5, tzinfo=EASTERN))

    def test_iso_like(self):
        end = parse_until('2026-08-25 00:47', now=self.now)
        self.assertEqual(end, datetime(2026, 8, 25, 0, 47, tzinfo=EASTERN))

    def test_date_only_is_midnight(self):
        end = parse_until('2026-08-25', now=self.now)
        self.assertEqual(end, datetime(2026, 8, 25, 0, 0, tzinfo=EASTERN))

    def test_time_only_rolls_to_tomorrow(self):
        end = parse_until('12:47 AM', now=self.now)
        self.assertEqual(end, datetime(2026, 8, 25, 0, 47, tzinfo=EASTERN))

    def test_offset_aware_input_is_preserved(self):
        end = parse_until('2026-08-25T00:47:00-04:00', now=self.now)
        self.assertEqual(end, datetime(2026, 8, 25, 0, 47, tzinfo=EASTERN))

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            parse_until('next tuesday-ish', now=self.now)


class ParseDurationTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(parse_duration('45s'), timedelta(seconds=45))
        self.assertEqual(parse_duration('90m'), timedelta(minutes=90))
        self.assertEqual(parse_duration('2h'), timedelta(hours=2))
        self.assertEqual(parse_duration('1d'), timedelta(days=1))

    def test_compound(self):
        self.assertEqual(parse_duration('1h30m'), timedelta(minutes=90))

    def test_bare_number_is_minutes(self):
        self.assertEqual(parse_duration('15'), timedelta(minutes=15))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            parse_duration('soon')
        with self.assertRaises(ValueError):
            parse_duration('0m')


class ResolveEndTimeTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 24, 22, 0, tzinfo=EASTERN)

    def test_none_means_run_forever(self):
        self.assertIsNone(resolve_end_time(now=self.now))

    def test_until_wins_over_duration(self):
        end = resolve_end_time(until='8/25/26 12:47 AM', duration='10m', now=self.now)
        self.assertEqual(end, datetime(2026, 8, 25, 0, 47, tzinfo=EASTERN))

    def test_duration_from_now(self):
        end = resolve_end_time(duration='30m', now=self.now)
        self.assertEqual(end, self.now + timedelta(minutes=30))

    def test_past_until_raises(self):
        with self.assertRaises(ValueError):
            resolve_end_time(until='8/24/26 9:00 AM', now=self.now)


class FormatRemainingTests(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(format_remaining(timedelta(seconds=45)), '45s')
        self.assertEqual(format_remaining(timedelta(minutes=2, seconds=5)), '2m 5s')
        self.assertEqual(format_remaining(timedelta(hours=1, minutes=12, seconds=3)), '1h 12m 3s')
        self.assertEqual(format_remaining(timedelta(seconds=-5)), '0s')


if __name__ == '__main__':
    unittest.main()
