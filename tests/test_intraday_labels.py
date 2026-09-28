import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pathsetup  # noqa: F401

from broker.alpaca_data import clamp_sip_end, require_iex_for_latest
from features.session_labeler import regular_close_datetime, session_from_time


EASTERN = ZoneInfo('America/New_York')


class FeedGuardTests(unittest.TestCase):
    def test_latest_rejects_sip(self):
        with self.assertRaises(ValueError):
            require_iex_for_latest('sip')

    def test_latest_allows_iex(self):
        self.assertEqual(require_iex_for_latest('iex'), 'iex')

    def test_sip_end_clamped(self):
        now = datetime(2026, 8, 24, 18, 0, tzinfo=EASTERN).astimezone(ZoneInfo('UTC'))
        end = clamp_sip_end(now, now=now)
        self.assertEqual(end, now - timedelta(minutes=15))


class SessionCloseTests(unittest.TestCase):
    def test_regular_uses_same_day_close(self):
        dt = datetime(2026, 8, 24, 10, 0, tzinfo=EASTERN)
        close = regular_close_datetime(dt)
        self.assertEqual(close, datetime(2026, 8, 24, 16, 0, tzinfo=EASTERN))
        self.assertEqual(session_from_time(dt), 'regular')

    def test_afterhours_uses_next_session_close(self):
        dt = datetime(2026, 8, 24, 17, 0, tzinfo=EASTERN)
        close = regular_close_datetime(dt)
        self.assertEqual(close, datetime(2026, 8, 25, 16, 0, tzinfo=EASTERN))

    def test_weekend_uses_monday_close(self):
        dt = datetime(2026, 8, 22, 12, 0, tzinfo=EASTERN)
        close = regular_close_datetime(dt)
        self.assertEqual(close, datetime(2026, 8, 24, 16, 0, tzinfo=EASTERN))


if __name__ == '__main__':
    unittest.main()
