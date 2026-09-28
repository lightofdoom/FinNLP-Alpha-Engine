import json
import os
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
from alpaca.trading.stream import TradingStream

import config
from broker.alpaca_auth import assert_paper_or_allowed, get_alpaca_credentials, require_keys
from broker.rate_limit import get_rest_limiter
from features.session_labeler import session_from_time, to_eastern_datetime


EASTERN = ZoneInfo('America/New_York')


def build_trading_client(creds=None):
    creds = assert_paper_or_allowed(require_keys(creds))
    paper = creds['paper']
    return TradingClient(creds['api_key'], creds['secret_key'], paper=paper)


def _acquire():
    get_rest_limiter().acquire()


def get_clock(client=None):
    if client is None:
        client = build_trading_client()
    _acquire()
    return client.get_clock()


def get_account(client=None):
    if client is None:
        client = build_trading_client()
    _acquire()
    return client.get_account()


def get_position(symbol, client=None):
    if client is None:
        client = build_trading_client()
    _acquire()
    try:
        return client.get_open_position(symbol)
    except Exception:
        return None


def get_open_orders(symbol=None, client=None):
    if client is None:
        client = build_trading_client()
    _acquire()
    if symbol is None:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False)
    else:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=False, symbols=[symbol])
    return client.get_orders(request)


def cancel_open_orders(symbol=None, client=None):
    if client is None:
        client = build_trading_client()
    orders = get_open_orders(symbol=symbol, client=client)
    cancelled = []
    for order in orders:
        _acquire()
        client.cancel_order_by_id(order.id)
        cancelled.append(str(order.id))
    return cancelled


def list_positions(client=None):
    if client is None:
        client = build_trading_client()
    _acquire()
    positions = client.get_all_positions()
    out = []
    for pos in positions:
        out.append({
            'symbol': str(getattr(pos, 'symbol', '')),
            'qty': str(getattr(pos, 'qty', '')),
            'side': str(getattr(pos, 'side', '')),
            'avg_entry_price': _first_num(getattr(pos, 'avg_entry_price', None)),
            'current_price': _first_num(getattr(pos, 'current_price', None)),
            'market_value': _first_num(getattr(pos, 'market_value', None)),
            'unrealized_pl': _first_num(getattr(pos, 'unrealized_pl', None)),
        })
    return out


def flatten_all(client=None):
    """Close every open position and cancel working orders. Paper only."""
    if client is None:
        client = build_trading_client()
    _acquire()
    return client.close_all_positions(cancel_orders=True)


def limit_price_from_quote(side, quote, offset_bps=None):
    if offset_bps is None:
        offset_bps = float(getattr(config, 'ALPACA_LIMIT_OFFSET_BPS', 5))
    side_name = _side_name(side)
    bid = quote.get('bid')
    ask = quote.get('ask')
    last = quote.get('last')
    if side_name == 'buy':
        base = ask if ask not in (None, 0) else last
        if base is None:
            raise ValueError('No ask/last to price a buy limit')
        raw = float(base) * (1.0 + float(offset_bps) / 10000.0)
    else:
        base = bid if bid not in (None, 0) else last
        if base is None:
            raise ValueError('No bid/last to price a sell limit')
        raw = float(base) * (1.0 - float(offset_bps) / 10000.0)
    return _round_price(raw)


def qty_from_notional(notional, price):
    if price is None or float(price) <= 0:
        raise ValueError('Cannot size order without a price')
    qty = int(float(notional) // float(price))
    if qty < 1:
        raise ValueError('Notional %s is below one share at %s' % (notional, price))
    return qty


def build_limit_order_request(symbol, side, qty, limit_price, session, time_in_force=TimeInForce.DAY):
    session_name = str(session)
    extended = session_name in ('premarket', 'afterhours')
    if session_name == 'closed':
        raise ValueError('Market is closed; not submitting an order')
    return LimitOrderRequest(
        symbol=str(symbol).upper(),
        qty=qty,
        side=_order_side(side),
        time_in_force=time_in_force,
        limit_price=float(limit_price),
        extended_hours=extended,
    )


def build_market_order_request(symbol, side, qty, session, time_in_force=TimeInForce.DAY):
    session_name = str(session)
    if session_name == 'closed':
        raise ValueError('Market is closed; not submitting an order')
    if session_name != 'regular':
        raise ValueError('Market orders are only submitted in regular hours')
    return MarketOrderRequest(
        symbol=str(symbol).upper(),
        qty=qty,
        side=_order_side(side),
        time_in_force=time_in_force,
        extended_hours=False,
    )


def build_exit_order_request(symbol, side, qty, session, quote, urgent=False, offset_bps=None):
    """Urgent: market in RTH, IOC aggressive limit in extended hours. Else resting limit."""
    session_name = str(session)
    if session_name == 'closed':
        raise ValueError('Market is closed; not submitting an order')
    if urgent and session_name == 'regular':
        return build_market_order_request(symbol, side, qty, session)
    if urgent:
        if offset_bps is None:
            offset_bps = float(getattr(config, 'URGENT_LIMIT_OFFSET_BPS', 20))
        limit_price = limit_price_from_quote(side, quote, offset_bps=offset_bps)
        return build_limit_order_request(
            symbol, side, qty, limit_price, session, time_in_force=TimeInForce.IOC,
        )
    return build_limit_order_request(
        symbol, side, qty, limit_price_from_quote(side, quote, offset_bps=offset_bps), session,
    )


def describe_order_request(request):
    limit_price = getattr(request, 'limit_price', None)
    order_type = getattr(request, 'type', None)
    type_name = str(order_type).split('.')[-1].lower() if order_type is not None else (
        'limit' if limit_price is not None else 'market'
    )
    return {
        'symbol': request.symbol,
        'qty': str(request.qty),
        'side': str(request.side),
        'type': type_name,
        'limit_price': str(limit_price) if limit_price is not None else None,
        'time_in_force': str(request.time_in_force),
        'extended_hours': bool(getattr(request, 'extended_hours', False)),
    }


def submit_order(request, client=None):
    if client is None:
        client = build_trading_client()
    _acquire()
    return client.submit_order(request)


def submit_limit_order(request, client=None):
    return submit_order(request, client=client)


class ClockPoller:
    def __init__(self, client=None, interval_sec=None):
        self.client = client
        self.interval_sec = float(interval_sec if interval_sec is not None else getattr(config, 'ALPACA_CLOCK_POLL_SEC', 60))
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.Lock()
        self.clock = None

    def start(self):
        if self.client is None:
            self.client = build_trading_client()
        self.refresh()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name='alpaca-clock', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def refresh(self):
        clock = get_clock(self.client)
        with self.lock:
            self.clock = clock
        return clock

    def session(self, now=None):
        if now is None:
            with self.lock:
                clock = self.clock
            if clock is not None and getattr(clock, 'timestamp', None) is not None:
                now = clock.timestamp
        if now is None:
            now = datetime.now(EASTERN)
        return session_from_time(to_eastern_datetime(now))

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception:
                pass
            self._stop.wait(self.interval_sec)


class FillRecorder:
    def __init__(self, path=None):
        self.path = path or config.PAPER_FILLS_JSONL
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def record(self, update, signal_id=None):
        order = getattr(update, 'order', None)
        row = {
            'event': str(getattr(update, 'event', '')),
            'order_id': str(getattr(order, 'id', '')) if order is not None else '',
            'symbol': str(getattr(order, 'symbol', '')) if order is not None else '',
            'side': str(getattr(order, 'side', '')) if order is not None else '',
            'qty': _first_num(
                getattr(update, 'qty', None),
                getattr(order, 'filled_qty', None) if order is not None else None,
            ),
            'fill_price': _first_num(
                getattr(update, 'price', None),
                getattr(order, 'filled_avg_price', None) if order is not None else None,
            ),
            'ts': _iso(getattr(update, 'timestamp', None)),
            'signal_id': signal_id,
        }
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, default=str) + '\n')
        return row


class TradeUpdateListener:
    def __init__(self, recorder=None, on_update=None):
        creds = assert_paper_or_allowed(require_keys())
        self.creds = creds
        self.recorder = recorder if recorder is not None else FillRecorder()
        self.on_update = on_update
        self._stream = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='alpaca-trade-updates', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass

    def _run(self):
        stream = TradingStream(self.creds['api_key'], self.creds['secret_key'], paper=self.creds['paper'])
        self._stream = stream
        recorder = self.recorder
        callback = self.on_update

        async def handler(data):
            event = str(getattr(data, 'event', '')).lower()
            if event in ('fill', 'partial_fill', 'partialfill'):
                recorder.record(data)
            if callback is not None:
                callback(data)

        stream.subscribe_trade_updates(handler)
        try:
            stream.run()
        except Exception:
            if not self._stop.is_set():
                raise


def append_order_log(row, path=None):
    if path is None:
        path = config.PAPER_ORDERS_JSONL
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, default=str) + '\n')


def _side_name(side):
    text = str(side).lower()
    if text.endswith('buy') or text == 'b':
        return 'buy'
    if text.endswith('sell') or text == 's':
        return 'sell'
    raise ValueError('Unknown side %s' % side)


def _order_side(side):
    if _side_name(side) == 'buy':
        return OrderSide.BUY
    return OrderSide.SELL


def _round_price(value):
    price = float(value)
    if price >= 1:
        return round(price, 2)
    return round(price, 4)


def _first_num(*values):
    for value in values:
        if value is None or value == '':
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _iso(dt):
    if dt is None:
        return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    return str(dt)
