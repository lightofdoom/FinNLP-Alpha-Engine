import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pathsetup  # noqa: F401

from live.ml_decay import DecayState, apply_floor, decayed_score, minutes_between, update_ml_state


class DecayMathTests(unittest.TestCase):
    def test_time_based_power(self):
        self.assertAlmostEqual(decayed_score(1.0, 1.0, 0.75), 0.75)
        self.assertAlmostEqual(decayed_score(1.0, 2.0, 0.75), 0.5625)
        self.assertAlmostEqual(decayed_score(-0.8, 1.0, 0.75), -0.6)

    def test_zero_dt_keeps_score(self):
        self.assertAlmostEqual(decayed_score(0.5, 0.0, 0.75), 0.5)

    def test_floor_to_zero(self):
        self.assertEqual(apply_floor(0.019, 0.02), 0.0)
        self.assertEqual(apply_floor(-0.019, 0.02), 0.0)
        self.assertAlmostEqual(apply_floor(0.5, 0.02), 0.5)

    def test_fresh_replaces_not_max(self):
        now = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)
        rows = {}
        update_ml_state(rows, 'NVDA', now, new_score=-0.9, fresh=True)
        later = now + timedelta(minutes=1)
        row = update_ml_state(rows, 'NVDA', later, new_score=-0.9, fresh=True)
        self.assertAlmostEqual(row['ml_score'], -0.9)

    def test_quiet_decays_shorts(self):
        now = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)
        rows = {}
        update_ml_state(rows, 'NVDA', now, new_score=-0.8, fresh=True)
        later = now + timedelta(minutes=1)
        row = update_ml_state(rows, 'NVDA', later, fresh=False, lambda_=0.75, floor=0.02)
        self.assertAlmostEqual(row['ml_score'], -0.6)

    def test_max_age_zeros_stale_score(self):
        now = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)
        rows = {}
        update_ml_state(rows, 'NVDA', now, new_score=1.0, fresh=True)
        later = now + timedelta(minutes=16)
        row = update_ml_state(rows, 'NVDA', later, fresh=False, lambda_=0.75, max_age_min=15)
        self.assertEqual(row['ml_score'], 0.0)

    def test_restart_decays_by_timestamp(self):
        now = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)
        rows = {}
        update_ml_state(rows, 'AAPL', now, new_score=0.8, fresh=True)
        later = now + timedelta(minutes=2)
        row = update_ml_state(rows, 'AAPL', later, fresh=False, lambda_=0.75, floor=0.02)
        self.assertAlmostEqual(row['ml_score'], 0.8 * (0.75 ** 2))

    def test_minutes_between_rejects_negative(self):
        a = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)
        b = a - timedelta(minutes=5)
        self.assertEqual(minutes_between(a, b), 0.0)


class DecayStateTests(unittest.TestCase):
    def test_roundtrip_file(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, 'ml_decay_state.json')
        now = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)
        state = DecayState(path=path, lambda_=0.75, floor=0.02, max_age_min=15)
        state.update('MSFT', now, new_score=0.6, fresh=True)
        state.save()
        loaded = DecayState(path=path, lambda_=0.75, floor=0.02, max_age_min=15)
        self.assertAlmostEqual(loaded.score('MSFT'), 0.6)
        tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
