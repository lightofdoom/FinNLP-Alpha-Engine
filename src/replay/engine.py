"""Offline replay of the live decision stack on stored IEX bars + scored news."""

import csv
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from features.news_llm_features import aggregate_signals
from features.price_indicators import LOOKBACK, compute_price_features
from features.session_labeler import to_eastern_datetime
from live.combine_signals import combine_entry, ml_score_from_aggregate
from live.exit_engine import evaluate_exit, update_favorable
from live.ml_decay import DecayState
from portfolio.optimizer import plan_entry_sizes
from replay.store import ResearchStore, parse_ts


DEFAULT_EQUITY = 100000.0


def run_replay(
    store=None,
    equity=DEFAULT_EQUITY,
    notional=None,
    strategy='combined',
    allow_short=True,
    out_dir=None,
):
    if store is None:
        store = ResearchStore()
    if notional is None:
        notional = float(getattr(config, 'ALPACA_NOTIONAL', 1000))
    if out_dir is None:
        out_dir = config.REPLAY_RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    news = store.scored_relevant_news()
    bar_counts = store.bar_counts()
    news_stats = store.news_stats()

    blend_trades = _simulate(
        store, news, equity, notional, strategy, allow_short, mode='blend',
    )
    ml_trades = _simulate(
        store, news, equity, notional, strategy, allow_short, mode='ml_only',
    )

    summary = {
        'watchlist': store.watchlist() or list(config.WATCHLIST),
        'news_rows': news_stats.get('n'),
        'news_relevant': news_stats.get('n_relevant'),
        'news_scored': news_stats.get('n_scored'),
        'scored_relevant_used': len(news),
        'bar_counts': bar_counts,
        'equity_start': equity,
        'notional_cap': notional,
        'blend': _metrics(blend_trades, equity),
        'ml_only': _metrics(ml_trades, equity),
    }

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(out_dir, 'replay_%s.json' % stamp)
    csv_path = os.path.join(out_dir, 'replay_trades_%s.csv' % stamp)
    png_path = os.path.join(out_dir, 'replay_equity_%s.png' % stamp)
    txt_path = os.path.join(out_dir, 'replay_%s.txt' % stamp)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    _write_trades_csv(csv_path, blend_trades, ml_trades)
    _plot_equity(png_path, blend_trades, ml_trades, equity)
    text = _format_summary(summary)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')

    store.export_relevant_jsonl(os.path.join(out_dir, 'relevant_news.jsonl'))
    return {
        'summary': summary,
        'text': text,
        'json_path': json_path,
        'csv_path': csv_path,
        'png_path': png_path,
        'txt_path': txt_path,
        'blend_trades': blend_trades,
        'ml_trades': ml_trades,
    }


def _simulate(store, news, equity, notional, strategy, allow_short, mode):
    trades = []
    open_pos = {}
    tickers = list(store.watchlist() or config.WATCHLIST)
    news_by = {}
    for row in news:
        ticker = str(row.get('ticker', '')).upper()
        if ticker == '':
            continue
        news_by.setdefault(ticker, []).append(row)
        if ticker not in tickers:
            tickers.append(ticker)
    for ticker in tickers:
        news_by.setdefault(ticker, [])
        news_by[ticker].sort(key=lambda r: r.get('ts') or '')

    bars_by = {ticker: store.all_bars(ticker) for ticker in tickers}
    events = defaultdict(list)
    for ticker, bars in bars_by.items():
        for bar in bars:
            ts = bar.get('timestamp')
            if ts:
                events[ts].append((ticker, bar))

    windows = {ticker: deque(maxlen=LOOKBACK) for ticker in tickers}
    news_idx = {ticker: 0 for ticker in tickers}
    decay = DecayState(path=None)

    for ts in sorted(events):
        now = parse_ts(ts) or datetime.now(timezone.utc)
        touched = []
        for ticker, bar in events[ts]:
            windows[ticker].append(bar)
            batch = []
            pending = news_by[ticker]
            while news_idx[ticker] < len(pending):
                item = pending[news_idx[ticker]]
                item_ts = item.get('ts') or ''
                if item_ts <= ts:
                    if item.get('llm'):
                        batch.append(item)
                    news_idx[ticker] += 1
                else:
                    break
            fresh = batch != []
            if fresh:
                agg = aggregate_signals([row['llm'] for row in batch])
                ml_raw = ml_score_from_aggregate(agg, strategy=strategy)
                row = decay.update(ticker, now, new_score=ml_raw.get('ml_score'), fresh=True)
                ml_features = dict(ml_raw)
                ml_features['ml_score'] = row['ml_score']
                ml_features['fresh_news'] = True
            else:
                row = decay.update(ticker, now, fresh=False)
                ml_features = {
                    'ml_score': row['ml_score'],
                    'news_count': 0,
                    'fresh_news': False,
                }
            math_features = compute_price_features(list(windows[ticker]))
            touched.append((ticker, bar, math_features, ml_features, fresh, batch))

        exited = set()
        for ticker, bar, math_features, _ml_features, _fresh, _batch in touched:
            pos = open_pos.get(ticker)
            if pos is None:
                continue
            update_favorable(pos, bar)
            decision = evaluate_exit(
                pos, list(windows[ticker]), math_features, now=now, hold_extended=False,
            )
            if decision is None:
                continue
            trades.append(_close_trade(pos, float(bar['close']), bar.get('timestamp'), decision['reason']))
            open_pos.pop(ticker, None)
            exited.add(ticker)

        candidates = []
        for ticker, bar, math_features, ml_features, fresh, batch in touched:
            if ticker in open_pos or ticker in exited:
                continue
            if (not fresh) and abs(float(ml_features.get('ml_score') or 0.0)) == 0.0:
                continue
            if mode == 'ml_only':
                decision = _ml_only_entry(math_features, ml_features, allow_short)
            else:
                decision = combine_entry(
                    math_features,
                    ml_features,
                    news_required=True,
                    allow_short=allow_short,
                    fresh_news=fresh,
                )
            if not decision.get('actionable'):
                continue
            if math_features.get('atr') in (None, 0) or math_features.get('price') in (None, 0):
                continue
            candidates.append({
                'ticker': ticker,
                'decision': decision,
                'math_features': math_features,
                'ml_features': ml_features,
                'entry_score': decision.get('entry_score'),
                'math_score': decision.get('math_score'),
                'atr': math_features.get('atr'),
                'price': math_features.get('price'),
                'sigma_30_ann': math_features.get('sigma_30_ann'),
                'direction': decision.get('direction'),
                'bar': bar,
                'batch': batch,
            })

        if candidates == []:
            continue

        existing = []
        for ticker, pos in open_pos.items():
            existing.append({
                'ticker': ticker,
                'notional': pos.get('notional') or (float(pos['qty']) * float(pos['entry_price'])),
                'side': pos['side'],
            })
        sized_by = plan_entry_sizes(
            candidates,
            existing=existing,
            bars_by_ticker={t: list(windows[t]) for t in tickers},
            equity=equity,
            buying_power=equity,
            notional_cap=notional,
        )
        for cand in candidates:
            ticker = cand['ticker']
            sized = sized_by.get(ticker) or {}
            if not sized.get('ok'):
                continue
            last = cand['bar']
            entry_time = parse_ts(last.get('timestamp')) or parse_ts(ts)
            pos = {
                'ticker': ticker,
                'side': cand['decision']['direction'],
                'entry_price': float(last['close']),
                'atr_entry': float(cand['math_features']['atr']),
                'entry_time': entry_time,
                'qty': sized['qty'],
                'max_favorable': float(last['close']),
                'notional': sized['capped_notional'],
                'mode': mode,
                'entry_score': cand['decision'].get('entry_score'),
                'math_score': cand['decision'].get('math_score'),
                'ml_score': cand['decision'].get('ml_score'),
                'news_ids': [row.get('news_id') for row in cand.get('batch') or []],
                'entry_ts': last.get('timestamp'),
            }
            open_pos[ticker] = pos

    for ticker, pos in list(open_pos.items()):
        bars = bars_by.get(ticker) or []
        if not bars:
            continue
        last = bars[-1]
        trades.append(_close_trade(pos, float(last['close']), last.get('timestamp'), 'end_of_data'))
    return trades


def _ml_only_entry(math_features, ml_features, allow_short):
    """Same ATR requirement as live (exits need it); entry threshold is ML-only."""
    if math_features.get('math_unavailable'):
        return {
            'actionable': False,
            'reason': math_features.get('unavailable_reason') or 'math_unavailable',
            'entry_score': 0.0,
            'math_score': 0.0,
            'ml_score': float(ml_features.get('ml_score') or 0.0),
        }
    return combine_entry(
        {'math_score': float(ml_features.get('ml_score') or 0.0), 'math_unavailable': False},
        ml_features,
        w_math=0.0,
        w_ml=1.0,
        news_required=True,
        allow_short=allow_short,
        fresh_news=bool(ml_features.get('fresh_news')),
    )


def _close_trade(pos, exit_px, exit_ts, reason):
    entry = float(pos['entry_price'])
    qty = float(pos['qty'])
    if pos['side'] == 'long':
        pnl = (exit_px - entry) * qty
        ret = (exit_px / entry) - 1.0 if entry else 0.0
    else:
        pnl = (entry - exit_px) * qty
        ret = (entry / exit_px) - 1.0 if exit_px else 0.0
    return {
        'mode': pos['mode'],
        'ticker': pos['ticker'],
        'side': pos['side'],
        'qty': qty,
        'entry_ts': pos.get('entry_ts'),
        'exit_ts': exit_ts,
        'entry_price': entry,
        'exit_price': exit_px,
        'pnl': round(pnl, 4),
        'return': round(ret, 6),
        'exit_reason': reason,
        'notional': pos.get('notional'),
        'entry_score': pos.get('entry_score'),
        'math_score': pos.get('math_score'),
        'ml_score': pos.get('ml_score'),
        'news_ids': pos.get('news_ids'),
    }


def _metrics(trades, equity):
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    reasons = {}
    for t in trades:
        reasons[t['exit_reason']] = reasons.get(t['exit_reason'], 0) + 1
    total = sum(pnls)
    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': (len(wins) / len(trades)) if trades else None,
        'pnl': round(total, 4),
        'return_pct': round(100.0 * total / equity, 4) if equity else None,
        'exit_reasons': reasons,
    }


def _write_trades_csv(path, blend_trades, ml_trades):
    rows = list(blend_trades) + list(ml_trades)
    if rows == []:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('mode,ticker,side,qty,entry_ts,exit_ts,entry_price,exit_price,pnl,return,exit_reason\n')
        return
    fields = [
        'mode', 'ticker', 'side', 'qty', 'entry_ts', 'exit_ts', 'entry_price',
        'exit_price', 'pnl', 'return', 'exit_reason', 'notional', 'entry_score',
        'math_score', 'ml_score',
    ]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _plot_equity(path, blend_trades, ml_trades, equity):
    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_curve(ax, blend_trades, equity, 'dynamic blend', '#1f77b4')
    _plot_curve(ax, ml_trades, equity, 'ML-only entry', '#ff7f0e')
    ax.set_title('Replay equity (fill at bar close, no slippage)')
    ax.set_ylabel('Equity ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_curve(ax, trades, equity, label, color):
    xs = [None]
    ys = [equity]
    running = equity
    ordered = sorted(trades, key=lambda t: str(t.get('exit_ts') or ''))
    for t in ordered:
        ts = parse_ts(t.get('exit_ts'))
        if ts is None:
            continue
        running += float(t.get('pnl') or 0)
        xs.append(to_eastern_datetime(ts))
        ys.append(running)
    if len(xs) == 1:
        ax.axhline(equity, color=color, linestyle='--', alpha=0.4, label=label + ' (no trades)')
        return
    xs[0] = xs[1]
    ax.plot(xs, ys, color=color, label=label)


def _format_summary(summary):
    lines = []
    lines.append('REPLAY SUMMARY')
    lines.append('Watchlist:     ' + ', '.join(summary.get('watchlist') or []))
    lines.append('News stored:   %s  relevant %s  scored %s  used %s'
                 % (summary.get('news_rows'), summary.get('news_relevant'),
                    summary.get('news_scored'), summary.get('scored_relevant_used')))
    lines.append('Bars:')
    for row in summary.get('bar_counts') or []:
        lines.append('  %-6s %s  %s -> %s' % (row.get('ticker'), row.get('n'), row.get('first_ts'), row.get('last_ts')))
    if not (summary.get('bar_counts') or []):
        lines.append('  (none — run python main.py --fetch-bars)')
    lines.append('')
    for name in ('blend', 'ml_only'):
        m = summary.get(name) or {}
        lines.append('%s: trades %s  win_rate %s  pnl %s  ret %s%%'
                     % (name, m.get('trades'), m.get('win_rate'), m.get('pnl'), m.get('return_pct')))
        reasons = m.get('exit_reasons') or {}
        if reasons:
            bits = ', '.join('%s=%s' % kv for kv in sorted(reasons.items()))
            lines.append('  exits: ' + bits)
    if int(summary.get('scored_relevant_used') or 0) == 0:
        lines.append('')
        lines.append('No scored relevant news yet. Bars are in the DB; run the live loop')
        lines.append('to archive headlines + LLM labels, then re-run --replay.')
    return '\n'.join(lines)
