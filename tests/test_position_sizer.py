import math
import unittest

import pathsetup  # noqa: F401

from live.position_sizer import size_position


class SizerTests(unittest.TestCase):
    def test_small_paper_cap_binds(self):
        sized = size_position(
            equity=100000,
            price=100,
            atr=2,
            math_score=1.0,
            buying_power=200000,
            notional_cap=1000,
            sizing_mode='small_paper',
        )
        self.assertTrue(sized['ok'])
        self.assertEqual(sized['qty'], 10)
        self.assertGreater(sized['intended_notional'], sized['capped_notional'])
        self.assertTrue(sized['cap_bound'])
        self.assertEqual(sized['qty_raw'], math.floor(0.004 * 100000 / (1.5 * 2)))

    def test_too_quiet_skips(self):
        sized = size_position(100000, 100, 2, 1.0, sigma_30_ann=0.05, notional_cap=1000)
        self.assertFalse(sized['ok'])
        self.assertEqual(sized['reason'], 'too_quiet')

    def test_high_vol_halves_before_cap(self):
        quiet = size_position(100000, 100, 2, 1.0, sigma_30_ann=0.20, notional_cap=100000)
        loud = size_position(100000, 100, 2, 1.0, sigma_30_ann=0.90, notional_cap=100000)
        self.assertEqual(loud['qty'], math.floor(quiet['qty'] * 0.5) if quiet['qty'] >= 2 else loud['qty'])
        self.assertLess(loud['qty'], quiet['qty'])

    def test_qty_below_one(self):
        sized = size_position(100000, 2500, 2, 1.0, notional_cap=1000)
        self.assertFalse(sized['ok'])
        self.assertEqual(sized['reason'], 'qty_below_one')


if __name__ == '__main__':
    unittest.main()
