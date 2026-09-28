"""Entry-only mean-variance allocation for the 8-name watchlist.

Does not resize or flatten existing positions. Those stay with the exit engine.
Existing notionals enter the covariance term so a new name is penalized for
adding to a concentrated book.

μ_i = entry_score_i * (ATR_i / P_i) so the score is in 1-minute return units.
Σ is a shrunk sample covariance of aligned 1-minute log returns.
"""

import math

import numpy as np

import config
from live.position_sizer import BUYING_POWER_FRAC, size_position


def _minute_key(value):
    if value is None:
        return None
    if hasattr(value, 'isoformat'):
        try:
            text = value.isoformat()
        except Exception:
            text = str(value)
    else:
        text = str(value)
    text = text.replace('Z', '+00:00')
    if 'T' in text:
        return text[:16]
    return text[:16]


def log_returns(bars):
    """Return [(minute_key, log_ret), ...] from a bar list."""
    out = []
    prev_close = None
    prev_key = None
    for bar in bars or []:
        try:
            close = float(bar.get('close') or 0)
        except (TypeError, ValueError):
            continue
        key = _minute_key(bar.get('timestamp'))
        if close <= 0 or key is None:
            prev_close = close if close > 0 else prev_close
            prev_key = key
            continue
        if prev_close is not None and prev_close > 0 and prev_key is not None:
            out.append((key, math.log(close / prev_close)))
        prev_close = close
        prev_key = key
    return out


def aligned_return_matrix(bars_by_ticker, tickers):
    """(T, N) log-return matrix on the intersection of minute keys. May be empty."""
    series = {}
    keys_sets = []
    for ticker in tickers:
        pairs = log_returns(bars_by_ticker.get(ticker) or [])
        series[ticker] = {k: r for k, r in pairs}
        if pairs:
            keys_sets.append(set(series[ticker].keys()))
    if keys_sets == []:
        return np.zeros((0, len(tickers))), list(tickers)
    common = set.intersection(*keys_sets)
    ordered = sorted(common)
    if ordered == []:
        return np.zeros((0, len(tickers))), list(tickers)
    mat = np.zeros((len(ordered), len(tickers)), dtype=float)
    for j, ticker in enumerate(tickers):
        col = series[ticker]
        for i, key in enumerate(ordered):
            mat[i, j] = col.get(key, 0.0)
    return mat, list(tickers)


def shrunk_covariance(returns, shrinkage=None):
    """(1-α) sample + α diagonal. Falls back to tiny identity if T is tiny."""
    if shrinkage is None:
        shrinkage = float(getattr(config, 'COV_SHRINKAGE', 0.30))
    shrinkage = min(1.0, max(0.0, float(shrinkage)))
    n = 1 if returns is None or getattr(returns, 'ndim', 0) < 2 else returns.shape[1]
    if returns is None or len(returns) < 5 or n == 0:
        return np.eye(n, dtype=float) * 1e-8
    sample = np.cov(returns, rowvar=False)
    if np.ndim(sample) == 0:
        sample = np.array([[float(sample)]], dtype=float)
    diag = np.diag(np.diag(sample))
    cov = (1.0 - shrinkage) * sample + shrinkage * diag
    cov = cov + np.eye(cov.shape[0]) * 1e-12
    return cov


def _expected_return(entry_score, atr, price):
    try:
        score = float(entry_score or 0.0)
        atr = float(atr or 0.0)
        price = float(price or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if price <= 0 or atr <= 0:
        return 0.0
    return score * (atr / price)


def _signed_existing(existing):
    out = {}
    for row in existing or []:
        ticker = str(row.get('ticker') or '').strip().upper()
        if ticker == '':
            continue
        try:
            notional = abs(float(row.get('notional') or 0.0))
        except (TypeError, ValueError):
            continue
        side = str(row.get('side') or 'long').lower()
        out[ticker] = -notional if side == 'short' else notional
    return out


def allocate_entry_notionals(
    candidates,
    existing=None,
    bars_by_ticker=None,
    equity=None,
    buying_power=None,
    notional_cap=None,
    risk_aversion=None,
    turnover_penalty=None,
    gross_leverage=None,
    shrinkage=None,
):
    """Solve for signed dollar notionals on *new* names only.

    Returns {ticker: signed_notional} or None if the solver cannot run.
    """
    if notional_cap is None:
        notional_cap = float(getattr(config, 'ALPACA_NOTIONAL', 1000))
    if risk_aversion is None:
        risk_aversion = float(getattr(config, 'OPTIMIZER_RISK_AVERSION', 1.0))
    if turnover_penalty is None:
        turnover_penalty = float(getattr(config, 'OPTIMIZER_TURNOVER_PENALTY', 0.0005))
    if gross_leverage is None:
        gross_leverage = float(getattr(config, 'GROSS_LEVERAGE', 0.25))

    cands = []
    for row in candidates or []:
        ticker = str(row.get('ticker') or '').strip().upper()
        if ticker == '':
            continue
        cands.append(row)
    if cands == []:
        return {}

    try:
        from scipy.optimize import minimize
    except Exception:
        return None

    existing_signed = _signed_existing(existing)
    universe = []
    for ticker in existing_signed:
        if ticker not in universe:
            universe.append(ticker)
    for row in cands:
        ticker = str(row['ticker']).upper()
        if ticker not in universe:
            universe.append(ticker)

    bars_by_ticker = bars_by_ticker or {}
    rets, _ = aligned_return_matrix(bars_by_ticker, universe)
    cov = shrunk_covariance(rets, shrinkage=shrinkage)
    if cov.shape[0] != len(universe):
        cov = np.eye(len(universe), dtype=float) * 1e-8

    index = {t: i for i, t in enumerate(universe)}
    n_c = len(cands)
    mu = np.zeros(n_c, dtype=float)
    bounds = []
    for i, row in enumerate(cands):
        mu[i] = _expected_return(row.get('entry_score'), row.get('atr'), row.get('price'))
        direction = str(row.get('direction') or 'long').lower()
        cap = float(notional_cap)
        if direction == 'short':
            bounds.append((-cap, 0.0))
        else:
            bounds.append((0.0, cap))

    existing_vec = np.zeros(len(universe), dtype=float)
    for ticker, notional in existing_signed.items():
        existing_vec[index[ticker]] = notional
    existing_gross = float(np.sum(np.abs(existing_vec)))

    equity = float(equity or 0.0)
    lev_room = max(0.0, float(gross_leverage) * equity - existing_gross)
    if buying_power is not None:
        try:
            lev_room = min(lev_room, float(buying_power) * BUYING_POWER_FRAC)
        except (TypeError, ValueError):
            pass
    if lev_room <= 1e-9:
        return {str(row['ticker']).upper(): 0.0 for row in cands}

    cand_idx = [index[str(row['ticker']).upper()] for row in cands]

    def pack_full(x):
        full = existing_vec.copy()
        for j, idx in enumerate(cand_idx):
            full[idx] = x[j]
        return full

    def objective(x):
        full = pack_full(x)
        var = float(full.dot(cov).dot(full))
        turnover = float(np.sum(np.abs(x))) * float(turnover_penalty)
        return -float(mu.dot(x)) + float(risk_aversion) * var + turnover

    x0 = np.zeros(n_c, dtype=float)
    for i, (lo, hi) in enumerate(bounds):
        # Start at a small same-sign slice of the cap so SLSQP is not stuck at 0.
        if hi > 0:
            x0[i] = min(hi, lev_room / max(n_c, 1)) * 0.25
        elif lo < 0:
            x0[i] = max(lo, -lev_room / max(n_c, 1)) * 0.25

    constraints = [{
        'type': 'ineq',
        'fun': lambda x: lev_room - float(np.sum(np.abs(x))),
    }]

    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False},
        )
    except Exception:
        return None
    if result is None or not bool(getattr(result, 'success', False)):
        return None

    out = {}
    for i, row in enumerate(cands):
        ticker = str(row['ticker']).upper()
        raw = float(result.x[i])
        lo, hi = bounds[i]
        raw = min(hi, max(lo, raw))
        if abs(raw) < 1.0:
            raw = 0.0
        out[ticker] = raw
    return out


def plan_entry_sizes(
    candidates,
    existing=None,
    bars_by_ticker=None,
    equity=None,
    buying_power=None,
    notional_cap=None,
    enabled=None,
):
    """Map candidates to share qty. Optimizer first; isolated ATR/R fallback.

    Each value: size_position-like dict plus 'source' (optimizer|fallback).
    """
    if notional_cap is None:
        notional_cap = float(getattr(config, 'ALPACA_NOTIONAL', 1000))
    if enabled is None:
        enabled = bool(getattr(config, 'OPTIMIZER_ENABLED', True))

    prepared = []
    blocked = {}
    for row in candidates or []:
        ticker = str(row.get('ticker') or '').strip().upper()
        sized = size_position(
            equity,
            row.get('price'),
            row.get('atr'),
            row.get('math_score'),
            buying_power=buying_power,
            sigma_30_ann=row.get('sigma_30_ann'),
            notional_cap=notional_cap,
        )
        if not sized.get('ok'):
            blocked[ticker] = dict(sized)
            blocked[ticker]['source'] = 'gate'
            continue
        payload = dict(row)
        payload['ticker'] = ticker
        payload['_fallback'] = sized
        prepared.append(payload)

    if prepared == []:
        return blocked

    allocated = None
    if enabled:
        allocated = allocate_entry_notionals(
            prepared,
            existing=existing,
            bars_by_ticker=bars_by_ticker,
            equity=equity,
            buying_power=buying_power,
            notional_cap=notional_cap,
        )

    out = dict(blocked)
    if allocated is None:
        remaining = None
        if buying_power is not None:
            remaining = float(buying_power) * BUYING_POWER_FRAC
        ranked = sorted(prepared, key=lambda r: -abs(float(r.get('entry_score') or 0.0)))
        for row in ranked:
            ticker = row['ticker']
            sized = dict(row['_fallback'])
            if remaining is not None:
                cap = min(float(notional_cap), max(0.0, remaining))
                price = float(row.get('price') or 0.0)
                if price > 0 and sized.get('qty', 0) * price > cap:
                    qty = int(math.floor(cap / price))
                    if qty < 1:
                        sized = {
                            'ok': False,
                            'reason': 'qty_below_one',
                            'qty': 0,
                            'qty_raw': sized.get('qty_raw', 0),
                            'intended_notional': sized.get('intended_notional', 0.0),
                            'capped_notional': 0.0,
                            'stop_per_share': sized.get('stop_per_share'),
                            'cap_bound': True,
                            'sizing_mode': sized.get('sizing_mode'),
                        }
                    else:
                        sized['qty'] = qty
                        sized['capped_notional'] = round(qty * price, 4)
                        sized['cap_bound'] = True
                if sized.get('ok'):
                    remaining -= float(sized.get('capped_notional') or 0.0)
            sized['source'] = 'fallback'
            out[ticker] = sized
        return out

    for row in prepared:
        ticker = row['ticker']
        price = float(row.get('price') or 0.0)
        signed = float(allocated.get(ticker) or 0.0)
        target = abs(signed)
        fallback = row['_fallback']
        if price <= 0 or target < price:
            out[ticker] = {
                'ok': False,
                'reason': 'optimizer_zero',
                'qty': 0,
                'qty_raw': fallback.get('qty_raw', 0),
                'intended_notional': fallback.get('intended_notional', 0.0),
                'capped_notional': 0.0,
                'stop_per_share': fallback.get('stop_per_share'),
                'cap_bound': True,
                'sizing_mode': fallback.get('sizing_mode'),
                'source': 'optimizer',
            }
            continue
        qty = int(math.floor(target / price))
        if qty < 1:
            out[ticker] = {
                'ok': False,
                'reason': 'qty_below_one',
                'qty': 0,
                'qty_raw': fallback.get('qty_raw', 0),
                'intended_notional': fallback.get('intended_notional', 0.0),
                'capped_notional': 0.0,
                'stop_per_share': fallback.get('stop_per_share'),
                'cap_bound': True,
                'sizing_mode': fallback.get('sizing_mode'),
                'source': 'optimizer',
            }
            continue
        capped = qty * price
        out[ticker] = {
            'ok': True,
            'reason': 'ok',
            'qty': qty,
            'qty_raw': fallback.get('qty_raw', 0),
            'intended_notional': fallback.get('intended_notional', 0.0),
            'capped_notional': round(capped, 4),
            'stop_per_share': fallback.get('stop_per_share'),
            'cap_bound': capped + 1e-9 < float(fallback.get('intended_notional') or 0.0),
            'sizing_mode': fallback.get('sizing_mode'),
            'source': 'optimizer',
        }
    return out
