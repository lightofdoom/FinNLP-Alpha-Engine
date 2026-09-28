"""ATR/R position sizing, then ALPACA_NOTIONAL cap.

In SIZING_MODE=small_paper (v1 default) the dollar cap binds on a typical $100k
paper account, so the 0.40% equity R-target is computed and logged but does not
drive live size. Raise ALPACA_NOTIONAL or switch to risk_model to exercise R.
"""

import math

import config


R_PCT = 0.004
STOP_ATR_MULT = 1.5
HIGH_VOL_ANN = 0.80
LOW_VOL_ANN = 0.08
BUYING_POWER_FRAC = 0.25
RISK_MODEL_EQUITY_FRAC = 0.08


def size_position(
    equity,
    price,
    atr,
    math_score,
    buying_power=None,
    sigma_30_ann=None,
    notional_cap=None,
    sizing_mode=None,
):
    if notional_cap is None:
        notional_cap = float(getattr(config, 'ALPACA_NOTIONAL', 1000))
    if sizing_mode is None:
        sizing_mode = str(getattr(config, 'SIZING_MODE', 'small_paper')).strip().lower()
    r_pct = float(getattr(config, 'R_PCT', R_PCT))
    stop_mult = float(getattr(config, 'STOP_ATR_MULT', STOP_ATR_MULT))

    if equity is None or price is None or atr is None:
        return _skip('missing_inputs')
    equity = float(equity)
    price = float(price)
    atr = float(atr)
    if equity <= 0 or price <= 0 or atr <= 0:
        return _skip('bad_inputs')

    # sigma_30_ann is annualized (1-min stdev * sqrt(390 * 252)).
    if sigma_30_ann is not None and float(sigma_30_ann) < LOW_VOL_ANN:
        return _skip('too_quiet')

    stop_per_share = stop_mult * atr
    risk_dollars = r_pct * equity
    qty_raw = math.floor(risk_dollars / stop_per_share)
    conv = 0.50 + 0.50 * abs(float(math_score or 0.0))
    qty = math.floor(qty_raw * conv)
    if sigma_30_ann is not None and float(sigma_30_ann) > HIGH_VOL_ANN:
        qty = math.floor(qty * 0.5)

    intended_notional = qty * price if qty > 0 else 0.0

    cap = float(notional_cap)
    if buying_power is not None:
        cap = min(cap, float(buying_power) * BUYING_POWER_FRAC)
    if sizing_mode == 'risk_model':
        cap = min(float(buying_power or 0) * BUYING_POWER_FRAC, equity * RISK_MODEL_EQUITY_FRAC)
        if buying_power is None:
            cap = equity * RISK_MODEL_EQUITY_FRAC

    if qty > 0 and qty * price > cap:
        qty = math.floor(cap / price)

    if qty < 1:
        return _skip('qty_below_one', qty_raw=qty_raw, intended_notional=intended_notional)

    capped_notional = qty * price
    return {
        'ok': True,
        'reason': 'ok',
        'qty': int(qty),
        'qty_raw': int(qty_raw),
        'intended_notional': round(intended_notional, 4),
        'capped_notional': round(capped_notional, 4),
        'stop_per_share': stop_per_share,
        'cap_bound': intended_notional > capped_notional + 1e-9,
        'sizing_mode': sizing_mode,
    }


def _skip(reason, qty_raw=0, intended_notional=0.0):
    return {
        'ok': False,
        'reason': reason,
        'qty': 0,
        'qty_raw': int(qty_raw),
        'intended_notional': round(float(intended_notional), 4),
        'capped_notional': 0.0,
        'stop_per_share': None,
        'cap_bound': False,
        'sizing_mode': None,
    }
