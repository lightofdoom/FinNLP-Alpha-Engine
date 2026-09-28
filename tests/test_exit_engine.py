from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import unittest

import pathsetup  # noqa: F401

from features.price_indicators import PCT5_FLAT
from live.exit_engine import EXIT_PRIORITY, evaluate_exit, update_favorable


EASTERN = ZoneInfo('America/New_York')


def bar(close, high=None, low=None, trade_count=10):
    high = close if high is None else high
    low = close if low is None else low
    return {
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': 100,
        'trade_count': trade_count,
        'vwap': close,
    }


def long_pos(entry=100.0, atr=1.0, qty=10, age_min=10, extra=None, now=None):
    now = now or datetime(2026, 8, 28, 11, 0, tzinfo=EASTERN)
    pos = {
        'side': 'long',
        'entry_price': entry,
        'atr_entry': atr,
        'entry_time': now - timedelta(minutes=age_min),
        'qty': qty,
        'max_favorable': None,
    }
    if extra:
        pos.update(extra)
    return pos


class PriorityTests(unittest.TestCase):
    def test_code_priority_is_source_of_truth(self):
        self.assertEqual(EXIT_PRIORITY[0], 'hard_stop')
        self.assertEqual(EXIT_PRIORITY[1], 'iex_thin')
        self.assertLess(EXIT_PRIORITY.index('hard_stop'), EXIT_PRIORITY.index('iex_thin'))
        self.assertLess(EXIT_PRIORITY.index('iex_thin'), EXIT_PRIORITY.index('take_profit'))

    def test_hard_stop_beats_take_profit_on_wide_bar(self):
        pos = long_pos()
        bars = [bar(100.0)] * 5 + [bar(102.5, high=103.0, low=98.0)]
        out = evaluate_exit(pos, bars, {}, now=datetime(2026, 8, 28, 11, 0, tzinfo=EASTERN))
        self.assertEqual(out['reason'], 'hard_stop')
        self.assertTrue(out['urgent'])

    def test_hard_stop_beats_iex_thin_on_same_bar(self):
        bars = [bar(100.0, trade_count=8) for _ in range(40)]
        bars.extend(bar(100.0, trade_count=0) for _ in range(19))
        bars.append(bar(99.0, high=100.0, low=98.0, trade_count=0))
        out = evaluate_exit(long_pos(), bars, {}, now=datetime(2026, 8, 28, 11, 0, tzinfo=EASTERN))
        self.assertEqual(out['reason'], 'hard_stop')

    def test_iex_thin_fires_when_stop_not_hit(self):
        bars = [bar(100.0, trade_count=8) for _ in range(40)]
        bars.extend(bar(100.0, trade_count=0) for _ in range(20))
        out = evaluate_exit(long_pos(), bars, {}, now=datetime(2026, 8, 28, 11, 0, tzinfo=EASTERN))
        self.assertEqual(out['reason'], 'iex_thin')
        self.assertTrue(out['urgent'])


class RuleTests(unittest.TestCase):
    def test_take_profit(self):
        out = evaluate_exit(long_pos(), [bar(102.0)], {}, now=datetime(2026, 8, 28, 11, 0, tzinfo=EASTERN))
        self.assertEqual(out['reason'], 'take_profit')
        self.assertFalse(out['urgent'])

    def test_slope_reversal(self):
        feats = {'s_by_n': {5: -0.2, 15: -0.1}}
        out = evaluate_exit(long_pos(), [bar(100.0)], feats, now=datetime(2026, 8, 28, 11, 0, tzinfo=EASTERN))
        self.assertEqual(out['reason'], 'slope_reversal')
        self.assertTrue(out['urgent'])

    def test_trailing(self):
        pos = long_pos(extra={'max_favorable': 101.5})
        out = evaluate_exit(pos, [bar(100.4)], {}, now=datetime(2026, 8, 28, 11, 0, tzinfo=EASTERN))
        self.assertEqual(out['reason'], 'trailing')

    def test_rsi_requires_flat_pct5(self):
        now = datetime(2026, 8, 28, 11, 0, tzinfo=EASTERN)
        pos = long_pos(now=now, age_min=10)
        flat = {'rsi7': 80.0, 'pct_5': (PCT5_FLAT * 100.0) * 0.5}
        steep = {'rsi7': 80.0, 'pct_5': 1.0}
        self.assertEqual(evaluate_exit(pos, [bar(100.0)], flat, now=now)['reason'], 'rsi')
        self.assertIsNone(evaluate_exit(pos, [bar(100.0)], steep, now=now))

    def test_time_stop(self):
        now = datetime(2026, 8, 28, 11, 0, tzinfo=EASTERN)
        pos = long_pos(age_min=50, now=now)
        out = evaluate_exit(pos, [bar(100.1)], {}, now=now)
        self.assertEqual(out['reason'], 'time_stop')

    def test_session_flatten(self):
        now = datetime(2026, 8, 28, 15, 50, tzinfo=EASTERN)
        pos = long_pos(now=now, age_min=10)
        out = evaluate_exit(pos, [bar(100.0)], {}, now=now, hold_extended=False)
        self.assertEqual(out['reason'], 'session')
        self.assertIsNone(evaluate_exit(pos, [bar(100.0)], {}, now=now, hold_extended=True))

    def test_update_favorable(self):
        pos = long_pos()
        update_favorable(pos, bar(101.0, high=101.5, low=100.5))
        self.assertEqual(pos['max_favorable'], 101.5)


if __name__ == '__main__':
    unittest.main()
