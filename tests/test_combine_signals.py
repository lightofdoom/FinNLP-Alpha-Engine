import unittest

import pathsetup  # noqa: F401

from live.combine_signals import combine_entry, ml_score_from_aggregate


def math_ok(score=0.5):
    return {'math_score': score, 'math_unavailable': False}


def ml_ok(score=0.5, news_count=1):
    return {'ml_score': score, 'news_count': news_count}


class CombineTests(unittest.TestCase):
    def test_long_when_blend_above_entry_t(self):
        decision = combine_entry(math_ok(0.5), ml_ok(0.5), entry_t=0.20)
        self.assertTrue(decision['actionable'])
        self.assertEqual(decision['direction'], 'long')
        self.assertAlmostEqual(decision['entry_score'], 0.5)

    def test_short_when_blend_below_neg_entry_t(self):
        decision = combine_entry(math_ok(-0.5), ml_ok(-0.5), entry_t=0.20)
        self.assertEqual(decision['direction'], 'short')

    def test_below_entry_t_is_flat(self):
        decision = combine_entry(math_ok(0.1), ml_ok(0.1), entry_t=0.20)
        self.assertFalse(decision['actionable'])
        self.assertEqual(decision['reason'], 'below_entry_t')

    def test_disagreeing_signs_block(self):
        decision = combine_entry(math_ok(0.8), ml_ok(-0.8), entry_t=0.20)
        self.assertFalse(decision['actionable'])
        self.assertEqual(decision['reason'], 'disagree')

    def test_math_unavailable_blocks(self):
        decision = combine_entry({'math_score': 0.9, 'math_unavailable': True}, ml_ok(0.9))
        self.assertEqual(decision['reason'], 'math_unavailable')
        self.assertFalse(decision['actionable'])

    def test_no_news_blocks_when_required(self):
        decision = combine_entry(math_ok(0.9), ml_ok(0.0, news_count=0), news_required=True)
        self.assertEqual(decision['reason'], 'no_news')

    def test_decayed_ml_allows_quiet_entry(self):
        decision = combine_entry(
            math_ok(0.9), ml_ok(0.5, news_count=0), news_required=True, fresh_news=False,
        )
        self.assertTrue(decision['actionable'])
        self.assertEqual(decision['direction'], 'long')
        self.assertAlmostEqual(decision['w_math'], 0.85)
        self.assertAlmostEqual(decision['w_ml'], 0.15)

    def test_fresh_news_uses_ml_heavy_weights(self):
        decision = combine_entry(
            math_ok(0.2), ml_ok(0.8, news_count=1), news_required=True, fresh_news=True,
        )
        self.assertTrue(decision['actionable'])
        self.assertAlmostEqual(decision['w_math'], 0.30)
        self.assertAlmostEqual(decision['w_ml'], 0.70)
        self.assertAlmostEqual(decision['entry_score'], 0.30 * 0.2 + 0.70 * 0.8)

    def test_shorts_can_be_disabled(self):
        decision = combine_entry(math_ok(-0.9), ml_ok(-0.9), allow_short=False)
        self.assertEqual(decision['reason'], 'shorts_disabled')


class MlScoreTests(unittest.TestCase):
    def test_combined_is_signed(self):
        agg = {
            'llm_news_count': 1,
            'llm_sentiment_mean': -0.8,
            'llm_interpret_conf_mean': 1.0,
            'llm_actionability_mean': 1.0,
            'llm_is_catalyst_rate': 0.0,
        }
        out = ml_score_from_aggregate(agg)
        self.assertLess(out['ml_score'], 0)

    def test_risk_dampener_logged(self):
        agg = {
            'llm_news_count': 1,
            'llm_sentiment_mean': 1.0,
            'llm_interpret_conf_mean': 1.0,
            'llm_actionability_mean': 1.0,
            'llm_is_catalyst_rate': 0.0,
            'llm_risk_macro_count': 3,
            'llm_cat_earnings_count': 1,
        }
        out = ml_score_from_aggregate(agg)
        self.assertTrue(out['ml_damped'])
        self.assertAlmostEqual(out['ml_score_before_dampener'], 1.0)
        self.assertAlmostEqual(out['ml_score'], 0.8)


if __name__ == '__main__':
    unittest.main()
