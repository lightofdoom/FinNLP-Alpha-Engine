import math
import unittest

import pathsetup  # noqa: F401

from features.price_indicators import (
    MINUTES_PER_RTH,
    PCT5_FLAT,
    SESSIONS_PER_YEAR,
    compute_price_features,
    iex_thin_entry,
    iex_thin_exit,
    log_ols_slope,
    realized_vol_ann,
    s_from_slope,
    sigma_n,
    trend_align,
)


def make_bars(n, start=100.0, drift=0.001, range_frac=0.002, trade_count=12, volume=1000.0):
    bars = []
    for i in range(n):
        px = start * math.exp(drift * i)
        rng = px * range_frac
        bars.append({
            'open': px,
            'high': px + rng,
            'low': px - rng,
            'close': px,
            'volume': volume + i,
            'trade_count': trade_count,
            'vwap': px,
            'timestamp': i,
        })
    return bars


class SigmaNTests(unittest.TestCase):
    def test_sigma_n_is_dimensionless_fraction(self):
        # ATR/P / sqrt(n): dollars cancel, result is a 1-min fractional scale.
        self.assertEqual(sigma_n(2.0, 100.0, 4), 0.01)
        self.assertAlmostEqual(sigma_n(1.5, 150.0, 9), 1.5 / (150.0 * 3.0))

    def test_atr_and_price_scale_cancel(self):
        a = sigma_n(2.0, 100.0, 16)
        b = sigma_n(4.0, 200.0, 16)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a, 2.0 / (100.0 * 4.0))

    def test_s_n_numerator_matches_sigma_units(self):
        sig = 0.01
        slope = math.log(1.0 + sig)
        self.assertAlmostEqual(s_from_slope(slope, sig), math.tanh(1.0))

    def test_s_n_invariant_when_price_and_atr_scale(self):
        slope = 0.002
        s_a = s_from_slope(slope, sigma_n(2.0, 100.0, 25))
        s_b = s_from_slope(slope, sigma_n(6.0, 300.0, 25))
        self.assertAlmostEqual(s_a, s_b)


class RealizedVolAnnTests(unittest.TestCase):
    def test_scales_one_minute_stdev_by_rth_and_252(self):
        # Alternating ±8bp log returns: population stdev is exactly 0.0008.
        step = 0.0008
        rets = [step if i % 2 == 0 else -step for i in range(30)]
        closes = [100.0]
        for r in rets:
            closes.append(closes[-1] * math.exp(r))
        expected = step * math.sqrt(MINUTES_PER_RTH * SESSIONS_PER_YEAR)
        self.assertAlmostEqual(realized_vol_ann(closes, 30), expected)
        self.assertGreater(expected, 0.08)
        self.assertLess(expected, 0.80)


class SlopeTests(unittest.TestCase):
    def test_log_ols_recovers_known_slope(self):
        closes = [math.exp(0.001 * i) for i in range(10)]
        self.assertAlmostEqual(log_ols_slope(closes), 0.001, places=12)


class WindowReweightTests(unittest.TestCase):
    def test_twenty_five_bars_omit_s30_and_s60(self):
        feats = compute_price_features(make_bars(25))
        self.assertFalse(feats['math_unavailable'])
        self.assertEqual(feats['available_windows'], [5, 15])
        self.assertNotIn(30, feats['s_by_n'])
        self.assertNotIn(60, feats['s_by_n'])
        self.assertNotIn(60, feats['available_windows'])

    def test_sixty_bars_use_all_windows(self):
        feats = compute_price_features(make_bars(60))
        self.assertEqual(feats['available_windows'], [5, 15, 30, 60])


class TrendAlignTests(unittest.TestCase):
    def test_single_window_is_unconfirmed(self):
        self.assertEqual(trend_align({5: 0.4}), 0.25)
        self.assertEqual(trend_align({5: 0.0}), 0.0)

    def test_two_window_agreement(self):
        self.assertEqual(trend_align({5: 0.4, 15: 0.2}), 0.5)
        self.assertEqual(trend_align({5: -0.3, 15: -0.1}), 0.5)

    def test_three_window_agreement(self):
        self.assertEqual(trend_align({5: 0.4, 15: 0.2, 30: 0.1}), 1.0)

    def test_disagreement_is_zero(self):
        self.assertEqual(trend_align({5: 0.4, 15: -0.2}), 0.0)

    def test_single_window_less_than_two_window_agree(self):
        self.assertLess(trend_align({5: 0.8}), trend_align({5: 0.4, 15: 0.2}))


class ExtAlignedTests(unittest.TestCase):
    def test_ext_aligned_uses_sign_of_raw(self):
        bars = []
        for i in range(70):
            bars.append({
                'open': 132.0, 'high': 132.2, 'low': 131.8, 'close': 132.0,
                'volume': 1_000_000.0, 'trade_count': 20, 'vwap': 100.0, 'timestamp': i,
            })
        for i in range(20):
            px = 132.0 * math.exp(-0.003 * i)
            bars.append({
                'open': px, 'high': px + 0.15, 'low': px - 0.15, 'close': px,
                'volume': 1.0, 'trade_count': 8, 'vwap': px, 'timestamp': 70 + i,
            })
        feats = compute_price_features(bars)
        self.assertFalse(feats['math_unavailable'])
        self.assertGreater(feats['price'] - feats['vwap'], 2.0 * feats['atr'])
        self.assertLess(feats['raw'], 0.0)
        self.assertEqual(feats['ext_aligned'], 0.0)

    def test_chase_turns_on_ext_and_vwap_together(self):
        bars = make_bars(90, start=100.0, drift=0.004, range_frac=0.001, volume=10.0)
        feats = compute_price_features(bars)
        self.assertFalse(feats['math_unavailable'])
        self.assertGreater(feats['raw'], 0.0)
        self.assertLess(feats['vwap_score'], 0.0)
        if abs(feats['price'] - feats['vwap']) > 2.0 * feats['atr']:
            self.assertGreater(feats['ext_aligned'], 0.0)


class ThinTapeTests(unittest.TestCase):
    def test_iex_thin_entry(self):
        bars = make_bars(20, trade_count=10)
        for bar in bars[-5:]:
            bar['trade_count'] = 0
        self.assertTrue(iex_thin_entry(bars))
        feats = compute_price_features(bars)
        self.assertTrue(feats['math_unavailable'])
        self.assertEqual(feats['unavailable_reason'], 'iex_thin')

    def test_iex_thin_exit_needs_prior_activity(self):
        silent = make_bars(20, trade_count=0)
        self.assertFalse(iex_thin_exit(silent))
        active_then_silent = make_bars(40, trade_count=9) + make_bars(20, trade_count=0)
        self.assertTrue(iex_thin_exit(active_then_silent))

    def test_pct_5_is_percent_units(self):
        feats = compute_price_features(make_bars(30, drift=0.001))
        slope = log_ols_slope([b['close'] for b in make_bars(30, drift=0.001)][-5:])
        expected = 100.0 * (math.exp(slope) - 1.0)
        self.assertAlmostEqual(feats['pct_5'], expected, places=8)
        self.assertGreater(PCT5_FLAT * 100.0, 0.0)


class GateTests(unittest.TestCase):
    def test_thin_bars_unavailable(self):
        feats = compute_price_features(make_bars(10))
        self.assertTrue(feats['math_unavailable'])
        self.assertEqual(feats['unavailable_reason'], 'thin_bars')


if __name__ == '__main__':
    unittest.main()
