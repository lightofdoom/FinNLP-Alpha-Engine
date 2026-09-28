import unittest

import pathsetup  # noqa: F401

from features.news_llm_features import aggregate_signals
from live.signal_engine import decide_from_aggregate, order_side


def make_agg(sentiment, confidence=0.9, actionability=0.9, catalyst=True, count=2):
    signals = []
    for _ in range(count):
        signals.append({
            'sentiment': sentiment,
            'interpret_confidence': confidence,
            'is_catalyst': catalyst,
            'actionability': actionability,
            'time_horizon': 'short',
            'catalyst_types': ['earnings'],
            'risk_flags': ['none'],
            'entities': ['AMZN'],
        })
    return aggregate_signals(signals)


class DecisionTests(unittest.TestCase):
    def test_strong_bullish_is_actionable_long(self):
        decision = decide_from_aggregate(make_agg(0.8))
        self.assertEqual(decision['direction'], 'long')
        self.assertTrue(decision['actionable'])
        self.assertEqual(decision['reason'], 'ok')

    def test_strong_bearish_is_actionable_short(self):
        decision = decide_from_aggregate(make_agg(-0.8))
        self.assertEqual(decision['direction'], 'short')
        self.assertTrue(decision['actionable'])

    def test_shorts_can_be_disabled(self):
        decision = decide_from_aggregate(make_agg(-0.8), allow_short=False)
        self.assertEqual(decision['direction'], 'flat')
        self.assertFalse(decision['actionable'])
        self.assertEqual(decision['reason'], 'shorts_disabled')

    def test_neutral_sentiment_is_flat(self):
        decision = decide_from_aggregate(make_agg(0.02))
        self.assertEqual(decision['direction'], 'flat')
        self.assertEqual(decision['reason'], 'sentiment_below_threshold')

    def test_weak_conviction_below_min_signal(self):
        decision = decide_from_aggregate(make_agg(0.5, confidence=0.2, actionability=0.2))
        self.assertFalse(decision['actionable'])
        self.assertEqual(decision['reason'], 'strength_below_min')

    def test_no_news_is_flat(self):
        decision = decide_from_aggregate(aggregate_signals([]))
        self.assertEqual(decision['reason'], 'no_news')
        self.assertFalse(decision['actionable'])
        self.assertEqual(decision['news_count'], 0)

    def test_strength_capped_at_one(self):
        decision = decide_from_aggregate(make_agg(1.0, confidence=1.0, actionability=1.0))
        self.assertLessEqual(decision['strength'], 1.0)

    def test_catalyst_strategy_requires_catalyst(self):
        no_catalyst = make_agg(0.8, catalyst=False)
        decision = decide_from_aggregate(no_catalyst, strategy='catalyst')
        self.assertEqual(decision['direction'], 'flat')

    def test_sentiment_strategy_ignores_actionability(self):
        decision = decide_from_aggregate(make_agg(0.9, actionability=0.0), strategy='sentiment')
        self.assertTrue(decision['actionable'])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            decide_from_aggregate(make_agg(0.5), strategy='vibes')


class OrderSideTests(unittest.TestCase):
    def test_sides(self):
        self.assertEqual(order_side('long'), 'buy')
        self.assertEqual(order_side('short'), 'sell')
        with self.assertRaises(ValueError):
            order_side('flat')


if __name__ == '__main__':
    unittest.main()
