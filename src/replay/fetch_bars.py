"""Pull 1-minute IEX bars from Alpaca into the research SQLite store."""

from datetime import datetime, timedelta, timezone

from alpaca.data.timeframe import TimeFrame

import config
from broker.alpaca_auth import assert_paper_or_allowed, require_keys
from broker.alpaca_data import fetch_stock_bars
from replay.store import ResearchStore


DEFAULT_DAYS = 45
CHUNK_DAYS = 7


def fetch_history_bars(tickers=None, days=DEFAULT_DAYS, store=None, end=None, chunk_days=CHUNK_DAYS):
    require_keys()
    assert_paper_or_allowed()
    tickers = [str(t).strip().upper() for t in (tickers or config.WATCHLIST) if str(t).strip()]
    if tickers == []:
        raise ValueError('No tickers to fetch')
    if store is None:
        store = ResearchStore()
    lock = store.lock_watchlist(tickers)
    if lock.get('warning'):
        tickers = list(lock.get('watchlist') or tickers)
        print('Watchlist locked:', ', '.join(tickers))
        print(' ', lock['warning'])

    end = end or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(days=int(days))
    chunk = timedelta(days=int(chunk_days))

    total = 0
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + chunk, end)
        print('Fetching %s  %s -> %s'
              % (', '.join(tickers), cursor.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')))
        rows = fetch_stock_bars(tickers, TimeFrame.Minute, cursor, chunk_end)
        n = store.upsert_bars(rows, source='iex_rest')
        total += n
        print('  wrote %s bars (chunk)' % n)
        cursor = chunk_end

    store.set_meta('bars_fetched_at', datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
    store.set_meta('bars_days', int(days))
    return {
        'tickers': tickers,
        'days': int(days),
        'bars_upserted': total,
        'counts': store.bar_counts(),
        'watchlist': lock,
    }
