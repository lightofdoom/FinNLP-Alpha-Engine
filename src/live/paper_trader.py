import argparse
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import config
from broker.alpaca_auth import assert_paper_or_allowed, require_keys
from broker.alpaca_stream import LiveMarketService
from broker.alpaca_trading import (
    ClockPoller,
    FillRecorder,
    TradeUpdateListener,
    append_order_log,
    build_limit_order_request,
    build_trading_client,
    cancel_open_orders,
    describe_order_request,
    get_account,
    get_open_orders,
    get_position,
    limit_price_from_quote,
    qty_from_notional,
    submit_limit_order,
)


EASTERN = ZoneInfo('America/New_York')


def load_kill_state(path=None, today=None, starting_equity=None):
    if path is None:
        path = config.PAPER_KILL_SWITCH_JSON
    if today is None:
        today = datetime.now(EASTERN).date().isoformat()
    state = {
        'date': today,
        'orders_submitted': 0,
        'starting_equity': starting_equity,
    }
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        if loaded.get('date') == today:
            state = loaded
            if state.get('starting_equity') is None and starting_equity is not None:
                state['starting_equity'] = starting_equity
        elif starting_equity is not None:
            state['starting_equity'] = starting_equity
    elif starting_equity is not None:
        state['starting_equity'] = starting_equity
    return state


def save_kill_state(state, path=None):
    if path is None:
        path = config.PAPER_KILL_SWITCH_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def check_kill_switch(state, equity, max_orders=None, max_loss=None):
    if max_orders is None:
        max_orders = int(getattr(config, 'ALPACA_MAX_ORDERS_PER_DAY', 10))
    if max_loss is None:
        max_loss = float(getattr(config, 'ALPACA_MAX_LOSS_DOLLARS', 2000))
    orders = int(state.get('orders_submitted') or 0)
    if orders >= max_orders:
        raise RuntimeError('Kill switch: %s orders already submitted today (max %s)' % (orders, max_orders))
    start_eq = state.get('starting_equity')
    if start_eq is not None and equity is not None:
        loss = float(start_eq) - float(equity)
        if loss >= float(max_loss):
            raise RuntimeError(
                'Kill switch: paper equity loss %s exceeds max %s' % (loss, max_loss)
            )
    return True


def plan_paper_order(symbol, side, quote, session, notional=None, offset_bps=None, flatten=False, has_position=False, open_order_count=0):
    symbol = str(symbol).upper()
    if session == 'closed':
        raise RuntimeError('Market is closed; v1 logs only and does not queue')
    if open_order_count > 0:
        raise RuntimeError('Symbol %s already has an open order' % symbol)
    if has_position and not flatten:
        raise RuntimeError('Symbol %s already has a position; pass --flatten to override' % symbol)
    if notional is None:
        notional = float(getattr(config, 'ALPACA_NOTIONAL', 1000))
    last = quote.get('last')
    if last is None:
        last = quote.get('ask') if str(side).lower() in ('buy', 'orderside.buy') else quote.get('bid')
    qty = qty_from_notional(notional, last)
    limit_price = limit_price_from_quote(side, quote, offset_bps=offset_bps)
    request = build_limit_order_request(symbol, side, qty, limit_price, session)
    return {
        'request': request,
        'qty': qty,
        'limit_price': limit_price,
        'session': session,
        'notional': notional,
        'quote': quote,
    }


def _equity(account):
    value = getattr(account, 'equity', None)
    if value is None:
        return None
    return float(value)


def _buying_power(account):
    value = getattr(account, 'buying_power', None)
    if value is None:
        return None
    return float(value)


def cmd_status(client):
    account = get_account(client)
    clock_poller = ClockPoller(client=client)
    clock = clock_poller.refresh()
    session = clock_poller.session()
    print('paper:', True)
    print('equity:', account.equity)
    print('buying_power:', account.buying_power)
    print('clock_open:', getattr(clock, 'is_open', None))
    print('session:', session)
    print('next_open:', getattr(clock, 'next_open', None))
    print('next_close:', getattr(clock, 'next_close', None))
    state = load_kill_state(starting_equity=_equity(account))
    save_kill_state(state)
    print('orders_today:', state.get('orders_submitted'))
    print('starting_equity:', state.get('starting_equity'))


def cmd_cancel(client, symbol):
    ids = cancel_open_orders(symbol=symbol, client=client)
    print('cancelled:', ids)


def cmd_order(args):
    config.ensure_dirs()
    creds = assert_paper_or_allowed(require_keys())
    client = build_trading_client(creds)
    account = get_account(client)
    equity = _equity(account)
    state = load_kill_state(starting_equity=equity)
    save_kill_state(state)
    check_kill_switch(state, equity)

    symbol = args.symbol.upper()
    position = get_position(symbol, client=client)
    open_orders = get_open_orders(symbol=symbol, client=client)
    clock_poller = ClockPoller(client=client)
    clock_poller.refresh()
    session = clock_poller.session()

    market = LiveMarketService([symbol])
    listener = None
    try:
        market.start(hydrate=True)
        clock_poller.start()
        if not args.preview:
            listener = _listener_with_signal(args.news_id)
            listener.start()
            time.sleep(0.5)

        quote = market.quote_for_order(symbol)
        plan = plan_paper_order(
            symbol,
            args.side,
            quote,
            session,
            notional=args.notional,
            offset_bps=args.offset_bps,
            flatten=args.flatten,
            has_position=position is not None,
            open_order_count=len(open_orders or []),
        )
        request = plan['request']
        bp = _buying_power(account)
        estimated = plan['qty'] * plan['limit_price']
        if bp is not None and estimated > bp and args.side.lower() == 'buy':
            raise RuntimeError('Estimated cost %s exceeds buying power %s' % (estimated, bp))

        payload = describe_order_request(request)
        payload['session'] = session
        payload['quote'] = quote
        payload['news_id'] = args.news_id
        print(json.dumps(payload, indent=2, default=str))

        if args.preview:
            return payload

        order = submit_limit_order(request, client=client)
        state['orders_submitted'] = int(state.get('orders_submitted') or 0) + 1
        save_kill_state(state)
        log_row = {
            'order_id': str(getattr(order, 'id', '')),
            'symbol': symbol,
            'side': args.side,
            'qty': plan['qty'],
            'limit_price': plan['limit_price'],
            'session': session,
            'extended_hours': bool(request.extended_hours),
            'news_id': args.news_id,
            'ts': datetime.now(EASTERN).isoformat(),
            'status': str(getattr(order, 'status', '')),
        }
        append_order_log(log_row)
        print('submitted:', log_row['order_id'], log_row['status'])

        if args.wait_fill > 0:
            deadline = time.time() + args.wait_fill
            while time.time() < deadline:
                time.sleep(0.5)
            print('wait_fill elapsed; check', config.PAPER_FILLS_JSONL)
        return log_row
    finally:
        if listener is not None:
            listener.stop()
        clock_poller.stop()
        market.stop()


def _listener_with_signal(signal_id):
    recorder = FillRecorder()
    original = recorder.record

    def record(update, signal_id_inner=signal_id):
        return original(update, signal_id=signal_id_inner)

    recorder.record = record
    return TradeUpdateListener(recorder=recorder)


def build_parser():
    parser = argparse.ArgumentParser(description='Submit a paper limit order via Alpaca (not wired to the research pipeline).')
    sub = parser.add_subparsers(dest='command', required=True)

    status = sub.add_parser('status', help='Print paper account, clock, and kill-switch state')
    status.set_defaults(command='status')

    cancel = sub.add_parser('cancel', help='Cancel open paper orders for a symbol')
    cancel.add_argument('symbol')

    for name in ('buy', 'sell'):
        order = sub.add_parser(name, help='Submit a paper %s limit order' % name)
        order.add_argument('symbol')
        order.add_argument('--news-id', default=None)
        order.add_argument('--notional', type=float, default=None)
        order.add_argument('--offset-bps', type=float, default=None)
        order.add_argument('--preview', action='store_true')
        order.add_argument('--flatten', action='store_true')
        order.add_argument('--wait-fill', type=float, default=0.0)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == 'status':
        creds = assert_paper_or_allowed(require_keys())
        cmd_status(build_trading_client(creds))
        return 0
    if args.command == 'cancel':
        creds = assert_paper_or_allowed(require_keys())
        cmd_cancel(build_trading_client(creds), args.symbol.upper())
        return 0
    args.side = args.command
    cmd_order(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
