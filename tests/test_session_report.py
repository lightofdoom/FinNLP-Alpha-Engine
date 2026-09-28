import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pathsetup  # noqa: F401

from live.session_report import SessionStats, build_summary, format_summary


EASTERN = ZoneInfo('America/New_York')


def make_stats():
    stats = SessionStats(
        watchlist=['AMZN'],
        strategy='combined',
        min_signal=0.3,
        notional=1000.0,
        dry_run=False,
        end_time=datetime(2026, 8, 25, 0, 47, tzinfo=EASTERN),
    )
    stats.start_equity = 100000.0
    stats.end_equity = 100250.5
    stats.ticks = 4
    stats.news_seen = 20
    stats.news_new = 5
    stats.news_relevant = 3
    stats.news_scored = 3
    stats.ended_at = stats.started_at + timedelta(minutes=30)
    stats.end_reason = 'end_time_reached'
    return stats


class SummaryTests(unittest.TestCase):
    def test_pnl_computed(self):
        stats = make_stats()
        data = build_summary(stats)
        self.assertAlmostEqual(data['pnl_dollars'], 250.5, places=4)
        self.assertAlmostEqual(data['pnl_percent'], 0.2505, places=4)
        self.assertEqual(data['elapsed_seconds'], 1800)

    def test_counts_actionable_directions(self):
        stats = make_stats()
        stats.record_decision('AMZN', {'direction': 'long', 'strength': 0.5, 'actionable': True, 'reason': 'ok'})
        stats.record_decision('AMZN', {'direction': 'short', 'strength': 0.4, 'actionable': True, 'reason': 'ok'})
        stats.record_decision('AMZN', {'direction': 'flat', 'strength': 0.0, 'actionable': False, 'reason': 'no_news'})
        data = build_summary(stats)
        self.assertEqual(data['decisions'], 3)
        self.assertEqual(data['actionable_signals'], 2)
        self.assertEqual(data['long_signals'], 1)
        self.assertEqual(data['short_signals'], 1)

    def test_missing_equity_gives_no_pnl(self):
        stats = make_stats()
        stats.end_equity = None
        data = build_summary(stats)
        self.assertIsNone(data['pnl_dollars'])
        self.assertIn('P&L:             n/a', format_summary(data))

    def test_text_renders_without_positions(self):
        stats = make_stats()
        stats.record_skip('market_closed')
        stats.record_fill({'event': 'fill', 'symbol': 'AMZN', 'side': 'buy', 'qty': 4, 'price': 233.1})
        text = format_summary(build_summary(stats))
        self.assertIn('ALPACA PAPER SESSION SUMMARY', text)
        self.assertIn('market_closed', text)
        self.assertIn('Open positions at end: none', text)
        self.assertIn('$100,250.50', text)

    def test_shares_filled_totals(self):
        stats = make_stats()
        stats.record_fill({'qty': 4})
        stats.record_fill({'qty': 2})
        stats.record_fill({'qty': None})
        data = build_summary(stats)
        self.assertEqual(data['shares_filled'], 6.0)
        self.assertEqual(data['fill_events'], 3)


if __name__ == '__main__':
    unittest.main()
