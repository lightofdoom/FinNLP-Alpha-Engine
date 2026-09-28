import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame

import config
from broker.alpaca_auth import get_alpaca_credentials, require_keys
from broker.rate_limit import get_rest_limiter


EASTERN = ZoneInfo('America/New_York')
SIP_DELAY = timedelta(minutes=15)
LATEST_FEEDS = {'iex'}


def parse_feed(name=None):
    if name is None:
        creds = get_alpaca_credentials()
        name = creds['feed']
    text = str(name).strip().lower()
    if text == 'sip':
        return DataFeed.SIP
    return DataFeed.IEX


def feed_name(feed):
    if feed is None:
        return 'iex'
    if hasattr(feed, 'value'):
        return str(feed.value).lower()
    return str(feed).strip().lower()


def require_iex_for_latest(feed):
    name = feed_name(feed)
    if name not in LATEST_FEEDS:
        raise ValueError(
            'Latest/snapshot endpoints require feed=iex on the free plan, got %s' % name
        )
    return feed


def clamp_sip_end(end, now=None):
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff = now - SIP_DELAY
    if end is None:
        return cutoff
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    if end > cutoff:
        return cutoff
    return end


def _to_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt):
    if dt is None:
        return None
    return _to_utc(dt).isoformat().replace('+00:00', 'Z')


def build_data_client(creds=None):
    creds = require_keys(creds)
    return StockHistoricalDataClient(creds['api_key'], creds['secret_key'])


def _acquire():
    get_rest_limiter().acquire()


def fetch_stock_bars(symbols, timeframe, start, end=None, feed=None, client=None):
    if client is None:
        client = build_data_client()
    if feed is None:
        feed = parse_feed()
    start = _to_utc(start)
    end = _to_utc(end)
    if feed_name(feed) == 'sip':
        end = clamp_sip_end(end)

    _acquire()
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=timeframe,
        start=start,
        end=end,
        feed=feed,
    )
    barset = client.get_stock_bars(request)
    return _bars_to_rows(barset, feed, timeframe)


def _timeframe_name(timeframe):
    if timeframe is None:
        return 'unknown'
    text = str(timeframe)
    if timeframe == TimeFrame.Day:
        return '1Day'
    if timeframe == TimeFrame.Minute:
        return '1Min'
    return text.replace(' ', '')


def _bars_to_rows(barset, feed, timeframe):
    rows = []
    tf = _timeframe_name(timeframe)
    fname = feed_name(feed)
    data = getattr(barset, 'data', None)
    if data is None:
        if isinstance(barset, dict):
            data = barset
        else:
            return rows

    for symbol, bars in data.items():
        for bar in bars:
            ts = getattr(bar, 'timestamp', None)
            rows.append({
                'ticker': str(symbol),
                'timestamp': _iso(ts),
                'open': _num(getattr(bar, 'open', None)),
                'high': _num(getattr(bar, 'high', None)),
                'low': _num(getattr(bar, 'low', None)),
                'close': _num(getattr(bar, 'close', None)),
                'volume': _num(getattr(bar, 'volume', None)),
                'vwap': _num(getattr(bar, 'vwap', None)),
                'trade_count': _num(getattr(bar, 'trade_count', None)),
                'feed': fname,
                'timeframe': tf,
            })
    return rows


def _num(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_daily_closes(tickers, start_date, end_date, client=None, feed=None):
    start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=EASTERN)
    end = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=EASTERN)
    rows = fetch_stock_bars(
        tickers,
        TimeFrame.Day,
        start.astimezone(timezone.utc),
        end.astimezone(timezone.utc),
        feed=feed,
        client=client,
    )
    closes = []
    for row in rows:
        ts = row.get('timestamp')
        close = row.get('close')
        if ts is None or close is None:
            continue
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(EASTERN)
        closes.append({
            'ticker': row['ticker'],
            'date': dt.date().isoformat(),
            'close': float(close),
        })
    return closes


def fetch_minute_bars(tickers, start, end, client=None, feed=None):
    return fetch_stock_bars(
        tickers,
        TimeFrame.Minute,
        start,
        end,
        feed=feed,
        client=client,
    )


def get_latest_trade(symbols, client=None, feed=None):
    if client is None:
        client = build_data_client()
    if feed is None:
        feed = parse_feed()
    require_iex_for_latest(feed)
    _acquire()
    request = StockLatestTradeRequest(symbol_or_symbols=list(symbols), feed=feed)
    result = client.get_stock_latest_trade(request)
    out = {}
    items = result if isinstance(result, dict) else getattr(result, 'data', result)
    for symbol, trade in items.items():
        out[str(symbol)] = {
            'ticker': str(symbol),
            'price': _num(getattr(trade, 'price', None)),
            'size': _num(getattr(trade, 'size', None)),
            'timestamp': _iso(getattr(trade, 'timestamp', None)),
            'source': 'latest_trade',
        }
    return out


def get_latest_quote(symbols, client=None, feed=None):
    if client is None:
        client = build_data_client()
    if feed is None:
        feed = parse_feed()
    require_iex_for_latest(feed)
    _acquire()
    request = StockLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=feed)
    result = client.get_stock_latest_quote(request)
    out = {}
    items = result if isinstance(result, dict) else getattr(result, 'data', result)
    for symbol, quote in items.items():
        out[str(symbol)] = _quote_dict(symbol, quote, source='latest_quote')
    return out


def get_snapshots(symbols, client=None, feed=None, persist=True):
    if client is None:
        client = build_data_client()
    if feed is None:
        feed = parse_feed()
    require_iex_for_latest(feed)
    _acquire()
    request = StockSnapshotRequest(symbol_or_symbols=list(symbols), feed=feed)
    result = client.get_stock_snapshot(request)
    items = result if isinstance(result, dict) else getattr(result, 'data', result)
    out = {}
    now = datetime.now(timezone.utc)
    for symbol, snap in items.items():
        row = snapshot_to_dict(symbol, snap)
        row['fetched_at'] = _iso(now)
        out[str(symbol)] = row
        if persist:
            append_jsonl(config.SNAPSHOTS_JSONL, row)
    return out


def snapshot_to_dict(symbol, snap):
    trade = getattr(snap, 'latest_trade', None)
    quote = getattr(snap, 'latest_quote', None)
    minute_bar = getattr(snap, 'minute_bar', None)
    daily_bar = getattr(snap, 'daily_bar', None)
    row = {
        'ticker': str(symbol),
        'last': _num(getattr(trade, 'price', None)) if trade is not None else None,
        'last_size': _num(getattr(trade, 'size', None)) if trade is not None else None,
        'trade_ts': _iso(getattr(trade, 'timestamp', None)) if trade is not None else None,
        'bid': _num(getattr(quote, 'bid_price', None)) if quote is not None else None,
        'ask': _num(getattr(quote, 'ask_price', None)) if quote is not None else None,
        'bid_size': _num(getattr(quote, 'bid_size', None)) if quote is not None else None,
        'ask_size': _num(getattr(quote, 'ask_size', None)) if quote is not None else None,
        'quote_ts': _iso(getattr(quote, 'timestamp', None)) if quote is not None else None,
        'bar_close': _num(getattr(minute_bar, 'close', None)) if minute_bar is not None else None,
        'bar_ts': _iso(getattr(minute_bar, 'timestamp', None)) if minute_bar is not None else None,
        'daily_close': _num(getattr(daily_bar, 'close', None)) if daily_bar is not None else None,
        'source': 'snapshot',
        'feed': 'iex',
    }
    return row


def _quote_dict(symbol, quote, source='quote'):
    return {
        'ticker': str(symbol),
        'bid': _num(getattr(quote, 'bid_price', None)),
        'ask': _num(getattr(quote, 'ask_price', None)),
        'bid_size': _num(getattr(quote, 'bid_size', None)),
        'ask_size': _num(getattr(quote, 'ask_size', None)),
        'timestamp': _iso(getattr(quote, 'timestamp', None)),
        'source': source,
    }


def append_jsonl(path, obj):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, default=str) + '\n')
