import json
import os
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

import config


EASTERN = ZoneInfo('America/New_York')


class SessionStats:
    def __init__(self, watchlist, strategy, min_signal, notional, dry_run, end_time):
        self.watchlist = list(watchlist)
        self.strategy = strategy
        self.min_signal = min_signal
        self.notional = notional
        self.dry_run = bool(dry_run)
        self.end_time = end_time

        self.started_at = datetime.now(EASTERN)
        self.ended_at = None
        self.end_reason = 'unknown'

        self.ticks = 0
        self.news_seen = 0
        self.news_new = 0
        self.news_relevant = 0
        self.news_scored = 0

        self.decisions = []
        self.orders = []
        self.fills = []
        self.exits = []
        self.skips = Counter()
        self.errors = []

        self.start_equity = None
        self.end_equity = None
        self.start_cash = None
        self.end_cash = None
        self.positions_at_end = []

    def record_decision(self, ticker, decision):
        row = dict(decision)
        row['ticker'] = ticker
        row['ts'] = datetime.now(EASTERN).isoformat()
        self.decisions.append(row)
        return row

    def record_order(self, row):
        self.orders.append(row)
        return row

    def record_fill(self, row):
        self.fills.append(row)
        return row

    def record_exit(self, row):
        self.exits.append(row)
        return row

    def record_skip(self, reason):
        self.skips[str(reason)] += 1

    def record_error(self, where, exc):
        self.errors.append({
            'where': where,
            'error': str(exc),
            'ts': datetime.now(EASTERN).isoformat(),
        })


def _pnl(stats):
    if stats.start_equity is None or stats.end_equity is None:
        return None, None
    delta = float(stats.end_equity) - float(stats.start_equity)
    pct = 0.0
    if float(stats.start_equity) != 0:
        pct = 100.0 * delta / float(stats.start_equity)
    return delta, pct


def build_summary(stats):
    ended = stats.ended_at or datetime.now(EASTERN)
    elapsed = ended - stats.started_at
    delta, pct = _pnl(stats)

    actionable = [d for d in stats.decisions if d.get('actionable')]
    longs = len([d for d in actionable if d.get('direction') == 'long'])
    shorts = len([d for d in actionable if d.get('direction') == 'short'])

    filled_qty = 0.0
    for f in stats.fills:
        try:
            filled_qty += float(f.get('qty') or 0)
        except (TypeError, ValueError):
            pass

    data = {
        'mode': 'dry_run' if stats.dry_run else 'paper',
        'watchlist': stats.watchlist,
        'strategy': stats.strategy,
        'min_signal': stats.min_signal,
        'notional_per_trade': stats.notional,
        'started_at': stats.started_at.isoformat(),
        'ended_at': ended.isoformat(),
        'planned_end': stats.end_time.isoformat() if stats.end_time else None,
        'elapsed_seconds': int(elapsed.total_seconds()),
        'end_reason': stats.end_reason,
        'polls': stats.ticks,
        'news_seen': stats.news_seen,
        'news_new': stats.news_new,
        'news_relevant': stats.news_relevant,
        'news_scored': stats.news_scored,
        'decisions': len(stats.decisions),
        'actionable_signals': len(actionable),
        'long_signals': longs,
        'short_signals': shorts,
        'orders_submitted': len(stats.orders),
        'fill_events': len(stats.fills),
        'exits': len(getattr(stats, 'exits', []) or []),
        'shares_filled': filled_qty,
        'skips': dict(stats.skips),
        'start_equity': stats.start_equity,
        'end_equity': stats.end_equity,
        'start_cash': stats.start_cash,
        'end_cash': stats.end_cash,
        'pnl_dollars': delta,
        'pnl_percent': pct,
        'positions_at_end': stats.positions_at_end,
        'errors': stats.errors,
        'order_log': stats.orders,
        'decision_log': stats.decisions,
        'fill_log': stats.fills,
        'exit_log': list(getattr(stats, 'exits', []) or []),
    }
    return data


def format_summary(data):
    lines = []
    lines.append('=' * 62)
    lines.append('ALPACA PAPER SESSION SUMMARY')
    lines.append('=' * 62)
    lines.append('Mode:            ' + str(data['mode']))
    lines.append('Watchlist:       ' + ', '.join(data['watchlist']))
    lines.append('Strategy:        %s (min signal %s)' % (data['strategy'], data['min_signal']))
    lines.append('Started:         ' + str(data['started_at']))
    lines.append('Ended:           ' + str(data['ended_at']))
    if data['planned_end']:
        lines.append('Planned end:     ' + str(data['planned_end']))
    lines.append('Elapsed:         %ss' % data['elapsed_seconds'])
    lines.append('End reason:      ' + str(data['end_reason']))
    lines.append('')

    lines.append('-- News pipeline --')
    lines.append('Polls:           %s' % data['polls'])
    lines.append('Headlines seen:  %s' % data['news_seen'])
    lines.append('New headlines:   %s' % data['news_new'])
    lines.append('Relevant:        %s' % data['news_relevant'])
    lines.append('Scored by LLM:   %s' % data['news_scored'])
    lines.append('')

    lines.append('-- Signals --')
    lines.append('Decisions:       %s' % data['decisions'])
    lines.append('Actionable:      %s (long %s / short %s)'
                 % (data['actionable_signals'], data['long_signals'], data['short_signals']))
    if data['skips']:
        lines.append('Skipped:')
        for reason, count in sorted(data['skips'].items(), key=lambda kv: -kv[1]):
            lines.append('  %-28s %s' % (reason, count))
    lines.append('')

    lines.append('-- Execution --')
    lines.append('Orders submitted: %s' % data['orders_submitted'])
    lines.append('Fill events:      %s' % data['fill_events'])
    lines.append('Exit events:      %s' % data.get('exits', 0))
    lines.append('Shares filled:    %s' % data['shares_filled'])
    lines.append('')

    lines.append('-- Account --')
    lines.append('Start equity:    %s' % _money(data['start_equity']))
    lines.append('End equity:      %s' % _money(data['end_equity']))
    if data['pnl_dollars'] is not None:
        lines.append('P&L:             %s (%.4f%%)' % (_money(data['pnl_dollars']), data['pnl_percent']))
    else:
        lines.append('P&L:             n/a')

    if data['positions_at_end']:
        lines.append('')
        lines.append('Open positions at end:')
        for pos in data['positions_at_end']:
            lines.append('  %-6s qty %-8s avg %-10s unrealized %s'
                         % (pos.get('symbol'), pos.get('qty'),
                            _money(pos.get('avg_entry_price')), _money(pos.get('unrealized_pl'))))
    else:
        lines.append('')
        lines.append('Open positions at end: none')

    if data['orders_submitted']:
        lines.append('')
        lines.append('Orders:')
        for o in data['order_log']:
            extra = ''
            if o.get('intent') == 'exit':
                extra = ' exit=%s' % o.get('exit_reason', '')
            elif o.get('intended_notional') is not None:
                extra = ' intended=%s capped=%s' % (o.get('intended_notional'), o.get('capped_notional'))
            lines.append('  %s %-4s %-6s qty %-6s limit %-10s [%s]%s'
                         % (o.get('ts', ''), o.get('side', ''), o.get('symbol', ''),
                            o.get('qty', ''), o.get('limit_price', ''), o.get('status', ''), extra))

    if data['errors']:
        lines.append('')
        lines.append('Errors (%s):' % len(data['errors']))
        for e in data['errors'][-10:]:
            lines.append('  [%s] %s: %s' % (e.get('ts', ''), e.get('where', ''), e.get('error', '')))

    lines.append('=' * 62)
    return '\n'.join(lines)


def _money(value):
    if value is None:
        return 'n/a'
    try:
        return '${:,.2f}'.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def write_summary(stats, out_dir=None):
    if out_dir is None:
        out_dir = config.PAPER_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    data = build_summary(stats)
    text = format_summary(data)

    stamp = stats.started_at.strftime('%Y%m%d_%H%M%S')
    txt_path = os.path.join(out_dir, 'session_%s.txt' % stamp)
    json_path = os.path.join(out_dir, 'session_%s.json' % stamp)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

    return {'text': text, 'data': data, 'txt_path': txt_path, 'json_path': json_path}
