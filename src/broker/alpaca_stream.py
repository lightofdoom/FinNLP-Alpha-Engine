import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed
from alpaca.data.live.stock import StockDataStream

import config
from broker.alpaca_auth import get_alpaca_credentials, require_keys
from broker.alpaca_data import fetch_minute_bars, get_snapshots, snapshot_to_dict
from features.price_indicators import LOOKBACK, parse_bar_row


class MarketCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.trades = {}
        self.quotes = {}
        self.bars = {}
        self.bar_history = {}
        self.last_stream_at = None
        self.stream_connected = False
        self.last_error = None

    def mark_connected(self, connected):
        with self.lock:
            self.stream_connected = bool(connected)
            if connected:
                self.last_error = None

    def mark_error(self, error):
        with self.lock:
            self.last_error = str(error)
            self.stream_connected = False

    def touch_stream(self):
        with self.lock:
            self.last_stream_at = datetime.now(timezone.utc)
            self.stream_connected = True

    def update_trade(self, trade):
        symbol = str(getattr(trade, 'symbol', ''))
        row = {
            'ticker': symbol,
            'price': _num(getattr(trade, 'price', None)),
            'size': _num(getattr(trade, 'size', None)),
            'timestamp': _iso(getattr(trade, 'timestamp', None)),
            'source': 'stream_trade',
        }
        with self.lock:
            self.trades[symbol] = row
            self.last_stream_at = datetime.now(timezone.utc)
            self.stream_connected = True
        return row

    def update_quote(self, quote):
        symbol = str(getattr(quote, 'symbol', ''))
        row = {
            'ticker': symbol,
            'bid': _num(getattr(quote, 'bid_price', None)),
            'ask': _num(getattr(quote, 'ask_price', None)),
            'bid_size': _num(getattr(quote, 'bid_size', None)),
            'ask_size': _num(getattr(quote, 'ask_size', None)),
            'timestamp': _iso(getattr(quote, 'timestamp', None)),
            'source': 'stream_quote',
        }
        with self.lock:
            self.quotes[symbol] = row
            self.last_stream_at = datetime.now(timezone.utc)
            self.stream_connected = True
        return row

    def update_bar(self, bar):
        symbol = str(getattr(bar, 'symbol', ''))
        row = {
            'ticker': symbol,
            'open': _num(getattr(bar, 'open', None)),
            'high': _num(getattr(bar, 'high', None)),
            'low': _num(getattr(bar, 'low', None)),
            'close': _num(getattr(bar, 'close', None)),
            'volume': _num(getattr(bar, 'volume', None)),
            'vwap': _num(getattr(bar, 'vwap', None)),
            'trade_count': _num(getattr(bar, 'trade_count', None)),
            'timestamp': _iso(getattr(bar, 'timestamp', None)),
            'source': 'stream_bar',
        }
        with self.lock:
            self.bars[symbol] = row
            self.last_stream_at = datetime.now(timezone.utc)
            self.stream_connected = True
        self.upsert_history_bar(symbol, row)
        return row

    def apply_snapshot(self, snapshot):
        symbol = str(snapshot.get('ticker', ''))
        with self.lock:
            if snapshot.get('last') is not None:
                self.trades[symbol] = {
                    'ticker': symbol,
                    'price': snapshot.get('last'),
                    'size': snapshot.get('last_size'),
                    'timestamp': snapshot.get('trade_ts'),
                    'source': 'snapshot',
                }
            if snapshot.get('bid') is not None or snapshot.get('ask') is not None:
                self.quotes[symbol] = {
                    'ticker': symbol,
                    'bid': snapshot.get('bid'),
                    'ask': snapshot.get('ask'),
                    'bid_size': snapshot.get('bid_size'),
                    'ask_size': snapshot.get('ask_size'),
                    'timestamp': snapshot.get('quote_ts'),
                    'source': 'snapshot',
                }
            if snapshot.get('bar_close') is not None:
                self.bars[symbol] = {
                    'ticker': symbol,
                    'close': snapshot.get('bar_close'),
                    'timestamp': snapshot.get('bar_ts'),
                    'source': 'snapshot',
                }

    def quote(self, symbol):
        with self.lock:
            return dict(self.quotes.get(symbol, {}))

    def trade(self, symbol):
        with self.lock:
            return dict(self.trades.get(symbol, {}))

    def bar(self, symbol):
        with self.lock:
            return dict(self.bars.get(symbol, {}))

    def upsert_history_bar(self, symbol, row):
        if not symbol or row is None or row.get('close') is None:
            return
        parsed = parse_bar_row(row)
        parsed['timestamp'] = row.get('timestamp')
        ts = parsed.get('timestamp')
        with self.lock:
            hist = self.bar_history.setdefault(symbol, deque(maxlen=LOOKBACK))
            if hist and ts is not None and hist[-1].get('timestamp') == ts:
                hist[-1] = parsed
            else:
                hist.append(parsed)

    def replace_history(self, symbol, bars):
        parsed = [parse_bar_row(b) for b in (bars or [])]
        for i, row in enumerate(parsed):
            if row.get('timestamp') is None and bars[i].get('timestamp') is not None:
                row['timestamp'] = bars[i].get('timestamp')
        with self.lock:
            self.bar_history[symbol] = deque(parsed[-LOOKBACK:], maxlen=LOOKBACK)
            if parsed:
                self.bars[symbol] = dict(parsed[-1])

    def history_bars(self, symbol):
        with self.lock:
            hist = self.bar_history.get(symbol)
            if not hist:
                return []
            return list(hist)

    def snapshot_view(self, symbol):
        with self.lock:
            trade = self.trades.get(symbol) or {}
            quote = self.quotes.get(symbol) or {}
            bar = self.bars.get(symbol) or {}
        last = trade.get('price')
        if last is None:
            last = bar.get('close')
        return {
            'ticker': symbol,
            'last': last,
            'last_size': trade.get('size'),
            'trade_ts': trade.get('timestamp'),
            'bid': quote.get('bid'),
            'ask': quote.get('ask'),
            'bid_size': quote.get('bid_size'),
            'ask_size': quote.get('ask_size'),
            'quote_ts': quote.get('timestamp'),
            'bar_close': bar.get('close'),
            'bar_ts': bar.get('timestamp'),
            'source': quote.get('source') or trade.get('source') or bar.get('source'),
        }

    def is_stream_down(self, stale_sec=None):
        if stale_sec is None:
            stale_sec = float(getattr(config, 'ALPACA_STREAM_STALE_SEC', 20))
        with self.lock:
            if not self.stream_connected:
                return True
            if self.last_stream_at is None:
                return True
            age = (datetime.now(timezone.utc) - self.last_stream_at).total_seconds()
            return age > stale_sec


class LiveMarketService:
    """IEX WebSocket plus REST snapshot fallback while the stream is down."""

    def __init__(self, symbols, cache=None):
        self.symbols = _unique_symbols(symbols)
        max_symbols = int(getattr(config, 'ALPACA_MAX_SYMBOLS', 30))
        if len(self.symbols) > max_symbols:
            raise ValueError(
                'Watchlist has %s symbols; free IEX stream allows %s'
                % (len(self.symbols), max_symbols)
            )
        self.cache = cache if cache is not None else MarketCache()
        self._stop = threading.Event()
        self._stream = None
        self._stream_thread = None
        self._fallback_thread = None
        self._creds = require_keys()

    def start(self, hydrate=True):
        if hydrate:
            self.hydrate_snapshots()
            try:
                self.hydrate_minute_history()
            except Exception as exc:
                self.cache.mark_error(exc)
        self._stop.clear()
        self._stream_thread = threading.Thread(target=self._stream_loop, name='alpaca-iex-stream', daemon=True)
        self._fallback_thread = threading.Thread(target=self._fallback_loop, name='alpaca-snapshot-fallback', daemon=True)
        self._stream_thread.start()
        self._fallback_thread.start()

    def stop(self):
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
        self.cache.mark_connected(False)

    def hydrate_snapshots(self):
        snaps = get_snapshots(self.symbols, persist=True)
        for symbol, row in snaps.items():
            self.cache.apply_snapshot(row)
        return snaps

    def hydrate_minute_history(self, lookback_days=3):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        rows = fetch_minute_bars(self.symbols, start, end)
        by_symbol = {}
        for row in rows:
            ticker = str(row.get('ticker', '')).upper()
            if ticker == '':
                continue
            by_symbol.setdefault(ticker, []).append(row)
        for symbol, bars in by_symbol.items():
            bars.sort(key=lambda b: b.get('timestamp') or '')
            self.cache.replace_history(symbol, bars)
        return by_symbol

    def bars_for(self, symbol):
        return self.cache.history_bars(symbol)

    def quote_for_order(self, symbol):
        view = self.cache.snapshot_view(symbol)
        if view.get('bid') is not None or view.get('ask') is not None or view.get('last') is not None:
            if not self.cache.is_stream_down():
                return view
        snaps = get_snapshots([symbol], persist=True)
        row = snaps.get(symbol)
        if row is not None:
            self.cache.apply_snapshot(row)
            return self.cache.snapshot_view(symbol)
        return view

    def _stream_loop(self):
        delay = 1.0
        while not self._stop.is_set():
            try:
                creds = self._creds
                stream = StockDataStream(
                    creds['api_key'],
                    creds['secret_key'],
                    feed=DataFeed.IEX,
                )
                self._stream = stream
                cache = self.cache

                async def on_trade(trade):
                    cache.update_trade(trade)

                async def on_quote(quote):
                    cache.update_quote(quote)

                async def on_bar(bar):
                    cache.update_bar(bar)

                stream.subscribe_trades(on_trade, *self.symbols)
                stream.subscribe_quotes(on_quote, *self.symbols)
                stream.subscribe_bars(on_bar, *self.symbols)
                cache.mark_connected(True)
                delay = 1.0
                stream.run()
            except Exception as exc:
                self.cache.mark_error(exc)
            if self._stop.is_set():
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)

    def _fallback_loop(self):
        interval = float(getattr(config, 'ALPACA_SNAPSHOT_FALLBACK_SEC', 15))
        while not self._stop.is_set():
            if self.cache.is_stream_down():
                try:
                    self.hydrate_snapshots()
                except Exception as exc:
                    self.cache.mark_error(exc)
            self._stop.wait(interval)


def _unique_symbols(symbols):
    out = []
    seen = set()
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if symbol == '' or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _num(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(dt):
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    return str(dt)
