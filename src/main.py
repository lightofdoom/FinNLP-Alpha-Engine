"""Entry point.

Default run starts a live Alpaca paper simulation driven by Finnhub news and a local
Ollama model, and keeps going until you stop it (Ctrl-C) or until --until/--duration
elapses. Either way it prints and writes a result summary.

Examples:
    python main.py
    python main.py --fetch-bars --bars-days 45
    python main.py --replay
    python main.py --until "8/25/26 12:47 AM"
    python main.py --duration 90m
    python main.py --dry-run --poll 30
"""

import argparse
import sys

import config
from live.duration import resolve_end_time
from live.live_engine import (
    DEFAULT_DAYS_BACK,
    DEFAULT_MAX_ITEMS_PER_POLL,
    DEFAULT_MODEL,
    DEFAULT_POLL_SEC,
    LiveEngine,
)
from live.signal_engine import DEFAULT_STRATEGY, STRATEGIES


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run the live Finnhub -> Ollama -> Alpaca paper simulation.'
    )

    parser.add_argument('--until', default=None,
                        help='Stop at this time, e.g. "8/25/26 12:47 AM" or "2026-08-25 00:47" (market time).')
    parser.add_argument('--duration', default=None,
                        help='Stop after a span, e.g. 90m, 2h, 1h30m. Ignored if --until is set.')
    parser.add_argument('--poll', type=float, default=DEFAULT_POLL_SEC,
                        help='Seconds between news polls (default %(default)s).')
    parser.add_argument('--tickers', default=None,
                        help='Comma-separated watchlist override (default from config.WATCHLIST).')
    parser.add_argument('--strategy', choices=list(STRATEGIES), default=DEFAULT_STRATEGY,
                        help='Signal strategy (default %(default)s).')
    parser.add_argument('--min-signal', type=float, default=None,
                        help='ENTRY_T for the dynamic math/ML blend (default 0.20).')
    parser.add_argument('--notional', type=float, default=None,
                        help='Dollars per trade (default ALPACA_NOTIONAL).')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Ollama model for signal extraction (default %(default)s).')
    parser.add_argument('--days-back', type=int, default=DEFAULT_DAYS_BACK,
                        help='Finnhub lookback window in days per poll (default %(default)s).')
    parser.add_argument('--max-items', type=int, default=DEFAULT_MAX_ITEMS_PER_POLL,
                        help='Max new headlines scored per poll (default %(default)s).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Score news and log decisions without sending orders.')
    parser.add_argument('--no-short', action='store_true',
                        help='Long-only; ignore bearish signals.')
    parser.add_argument('--flatten-on-exit', action='store_true',
                        help='Close all positions when the session ends.')
    parser.add_argument('--fetch-bars', action='store_true',
                        help='Download 1-min IEX bars from Alpaca into data/research_set/research.sqlite.')
    parser.add_argument('--bars-days', type=int, default=45,
                        help='Calendar days of 1-min bars to fetch (default %(default)s).')
    parser.add_argument('--replay', action='store_true',
                        help='Replay dynamic blend vs ML-only on the research sqlite (bars + scored news).')
    parser.add_argument('--replay-equity', type=float, default=100000.0,
                        help='Starting equity for replay P&L (default %(default)s).')

    return parser


def run_fetch_bars(args):
    config.ensure_dirs()
    from replay.fetch_bars import fetch_history_bars

    watchlist = None
    if args.tickers:
        watchlist = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    result = fetch_history_bars(tickers=watchlist, days=args.bars_days)
    print('Tickers:', ', '.join(result['tickers']))
    print('Bars upserted:', result['bars_upserted'])
    for row in result.get('counts') or []:
        print('  %-6s %s  %s -> %s' % (row.get('ticker'), row.get('n'), row.get('first_ts'), row.get('last_ts')))
    print('Wrote:', config.RESEARCH_DB)
    print('Watchlist lock:', config.RESEARCH_WATCHLIST_JSON)
    return 0


def run_replay_cmd(args):
    config.ensure_dirs()
    from replay.engine import run_replay

    result = run_replay(equity=args.replay_equity, notional=args.notional)
    print()
    print(result['text'])
    print('Wrote:', result['txt_path'])
    print('Wrote:', result['json_path'])
    print('Wrote:', result['csv_path'])
    print('Wrote:', result['png_path'])
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.fetch_bars:
        return run_fetch_bars(args)
    if args.replay:
        return run_replay_cmd(args)

    watchlist = None
    if args.tickers:
        watchlist = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]

    try:
        resolve_end_time(until=args.until, duration=args.duration)
    except ValueError as exc:
        parser.error(str(exc))

    engine = LiveEngine(
        watchlist=watchlist,
        until=args.until,
        duration=args.duration,
        poll_sec=args.poll,
        strategy=args.strategy,
        min_signal=args.min_signal if args.min_signal is not None else float(getattr(config, 'ENTRY_T', 0.20)),
        notional=args.notional,
        dry_run=args.dry_run,
        model=args.model,
        days_back=args.days_back,
        max_items_per_poll=args.max_items,
        allow_short=not args.no_short,
        flatten_on_exit=args.flatten_on_exit,
    )

    engine.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
