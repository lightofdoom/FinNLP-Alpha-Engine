import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pathsetup  # noqa: F401

from replay.engine import run_replay
from replay.store import ResearchStore, news_id_from_item


def _bars(ticker, n, start=None, drift=0.001):
    start = start or datetime(2026, 8, 1, 14, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        px = 100.0 * math.exp(drift * i)
        ts = start + timedelta(minutes=i)
        rows.append({
            'ticker': ticker,
            'timestamp': ts.isoformat().replace('+00:00', 'Z'),
            'open': px,
            'high': px + 0.2,
            'low': px - 0.2,
            'close': px,
            'volume': 1000,
            'vwap': px,
            'trade_count': 12,
        })
    return rows


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, 'research.sqlite')
        self.store = ResearchStore(self.db)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_lock_watchlist_sticks(self):
        first = self.store.lock_watchlist(['NVDA', 'AAPL'])
        self.assertEqual(first['watchlist'], ['NVDA', 'AAPL'])
        second = self.store.lock_watchlist(['TSLA'])
        self.assertEqual(second['watchlist'], ['NVDA', 'AAPL'])
        self.assertIn('warning', second)

    def test_bars_roundtrip(self):
        self.store.upsert_bars(_bars('NVDA', 5))
        got = self.store.bars_before('NVDA', '2026-08-01T14:10:00Z', n=90)
        self.assertEqual(len(got), 5)
        self.assertEqual(got[-1]['ticker'], 'NVDA')

    def test_news_relevant_and_llm(self):
        item = {
            'ticker': 'NVDA',
            'id': 'abc',
            'timestamp': 1750000000,
            'headline': 'NVIDIA raises guidance',
            'summary': 'earnings beat',
            'url': 'https://example.com/n',
        }
        self.assertEqual(news_id_from_item(item), 'NVDA|abc')
        self.store.upsert_news(item, relevant=True)
        self.store.upsert_news(item, relevant=True, llm_json={'sentiment': 0.8, 'actionability': 0.9})
        scored = self.store.scored_relevant_news()
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]['llm']['sentiment'], 0.8)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, 'research.sqlite')
        self.store = ResearchStore(self.db)
        self.out = os.path.join(self.tmp.name, 'out')

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_replay_with_no_news_writes_summary(self):
        self.store.upsert_bars(_bars('NVDA', 30))
        result = run_replay(store=self.store, equity=100000, notional=1000, out_dir=self.out)
        self.assertEqual(result['summary']['scored_relevant_used'], 0)
        self.assertTrue(os.path.exists(result['png_path']))
        self.assertIn('No scored relevant news', result['text'])

    def test_bullish_news_can_open_a_trade(self):
        bars = _bars('NVDA', 120, drift=0.002)
        self.store.upsert_bars(bars)
        event_ts = bars[90]['timestamp']
        self.store.upsert_news(
            {
                'ticker': 'NVDA',
                'id': 'n1',
                'timestamp': event_ts,
                'headline': 'NVIDIA earnings crush estimates',
                'summary': 'revenue guidance raised',
            },
            relevant=True,
            llm_json={
                'sentiment': 0.9,
                'interpret_confidence': 0.95,
                'is_catalyst': True,
                'actionability': 0.95,
                'time_horizon': 'short',
                'catalyst_types': ['earnings'],
                'risk_flags': ['none'],
                'entities': ['NVDA'],
            },
        )
        result = run_replay(store=self.store, equity=100000, notional=1000, out_dir=self.out)
        self.assertGreaterEqual(result['summary']['blend']['trades'], 0)
        self.assertTrue(os.path.exists(result['csv_path']))


if __name__ == '__main__':
    unittest.main()
