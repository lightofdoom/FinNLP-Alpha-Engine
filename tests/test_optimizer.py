import unittest

import pathsetup  # noqa: F401

from portfolio.optimizer import (
    allocate_entry_notionals,
    aligned_return_matrix,
    plan_entry_sizes,
    shrunk_covariance,
)


def _bars(closes, start='2026-08-01T14:00:00'):
    rows = []
    for i, px in enumerate(closes):
        minute = 14 * 60 + i
        hh = 14 + (minute // 60 - 14)
        mm = i % 60
        # keep ISO minute unique even past the hour
        total = 14 * 60 + i
        hh = total // 60
        mm = total % 60
        rows.append({
            'timestamp': '2026-08-01T%02d:%02d:00Z' % (hh, mm),
            'close': px,
            'volume': 1000,
            'trade_count': 10,
        })
    return rows


class OptimizerTests(unittest.TestCase):
    def test_aligned_returns_intersect(self):
        a = _bars([100, 101, 102, 103, 104, 105])
        b = _bars([50, 50.5, 51, 51.5, 52, 52.5])
        mat, names = aligned_return_matrix({'AAA': a, 'BBB': b}, ['AAA', 'BBB'])
        self.assertEqual(names, ['AAA', 'BBB'])
        self.assertEqual(mat.shape[1], 2)
        self.assertGreaterEqual(mat.shape[0], 4)

    def test_shrinkage_is_psd(self):
        import numpy as np
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.001, size=(40, 3))
        cov = shrunk_covariance(rets, shrinkage=0.3)
        eig = np.linalg.eigvalsh(cov)
        self.assertTrue((eig > 0).all())

    def test_cap_binds_per_name(self):
        closes = [100.0 + 0.1 * i for i in range(40)]
        bars = {'NVDA': _bars(closes), 'AAPL': _bars([c * 0.5 for c in closes])}
        cands = [
            {'ticker': 'NVDA', 'entry_score': 0.8, 'atr': 1.5, 'price': 140.0, 'direction': 'long',
             'math_score': 0.8, 'sigma_30_ann': 0.25},
            {'ticker': 'AAPL', 'entry_score': 0.7, 'atr': 0.8, 'price': 70.0, 'direction': 'long',
             'math_score': 0.7, 'sigma_30_ann': 0.22},
        ]
        out = allocate_entry_notionals(
            cands,
            existing=[],
            bars_by_ticker=bars,
            equity=100000,
            buying_power=200000,
            notional_cap=1000,
        )
        self.assertIsNotNone(out)
        for ticker, notional in out.items():
            self.assertLessEqual(abs(notional), 1000 + 1e-6)

    def test_plan_respects_too_quiet_gate(self):
        cands = [{
            'ticker': 'NVDA',
            'entry_score': 0.9,
            'atr': 2.0,
            'price': 100.0,
            'direction': 'long',
            'math_score': 1.0,
            'sigma_30_ann': 0.05,
        }]
        sized = plan_entry_sizes(
            cands,
            existing=[],
            bars_by_ticker={'NVDA': _bars([100 + i * 0.01 for i in range(40)])},
            equity=100000,
            buying_power=200000,
            notional_cap=1000,
            enabled=True,
        )
        self.assertFalse(sized['NVDA']['ok'])
        self.assertEqual(sized['NVDA']['reason'], 'too_quiet')

    def test_fallback_when_optimizer_disabled(self):
        cands = [{
            'ticker': 'NVDA',
            'entry_score': 0.9,
            'atr': 2.0,
            'price': 100.0,
            'direction': 'long',
            'math_score': 1.0,
            'sigma_30_ann': 0.25,
        }]
        sized = plan_entry_sizes(
            cands,
            existing=[],
            bars_by_ticker={'NVDA': _bars([100 + i * 0.2 for i in range(40)])},
            equity=100000,
            buying_power=200000,
            notional_cap=1000,
            enabled=False,
        )
        self.assertTrue(sized['NVDA']['ok'])
        self.assertEqual(sized['NVDA']['source'], 'fallback')
        self.assertEqual(sized['NVDA']['qty'], 10)


if __name__ == '__main__':
    unittest.main()
