"""Math-only exits. Priority is this module's EXIT_PRIORITY, not any markdown table.

Hard stop always wins on a wide-range bar (stop and target both touched).

Hard stop inspects that bar's high/low even when IEX trade_count is thin: protecting
the position outranks data-quality silence. IEX-thin flatten runs next.

Urgent rules (hard_stop, iex_thin, slope_reversal) must not use resting limits.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from features.price_indicators import PCT5_FLAT, iex_thin_exit
from features.session_labeler import REGULAR_END, to_eastern_datetime


EASTERN = ZoneInfo('America/New_York')

# Source of truth for tie-breaks. Do not infer order from docs tables.
EXIT_PRIORITY = (
    'hard_stop',
    'iex_thin',
    'slope_reversal',
    'trailing',
    'take_profit',
    'rsi',
    'time_stop',
    'session',
)

URGENT = frozenset(('hard_stop', 'iex_thin', 'slope_reversal'))

STOP_ATR = float(getattr(config, 'STOP_ATR_MULT', 1.5))
TARGET_ATR = float(getattr(config, 'TARGET_ATR_MULT', 2.0))
TRAIL_ACTIVATE_ATR = float(getattr(config, 'TRAIL_ACTIVATE_ATR', 1.0))
TRAIL_ATR = float(getattr(config, 'TRAIL_ATR', 1.0))
TIME_STOP_MIN = int(getattr(config, 'TIME_STOP_MIN', 45))
TIME_STOP_ATR = 0.25
SESSION_FLATTEN_MINUTES = 10
RSI_LONG_EXIT = 75.0
RSI_SHORT_EXIT = 25.0


def evaluate_exit(position, bars, features, now=None, hold_extended=False):
    """position: side ('long'|'short'), entry_price, atr_entry, entry_time,
    qty, max_favorable (optional running extreme).
    """
    if position is None or bars is None or len(bars) == 0:
        return None
    side = position.get('side')
    entry = float(position['entry_price'])
    atr = float(position['atr_entry'])
    bar = bars[-1]
    high = float(bar['high'])
    low = float(bar['low'])
    close = float(bar['close'])

    checks = {
        'hard_stop': _hard_stop(side, entry, atr, high, low),
        'iex_thin': iex_thin_exit(bars),
        'slope_reversal': _slope_reversal(side, features),
        'trailing': _trailing(side, position, close, atr),
        'take_profit': _take_profit(side, entry, atr, close),
        'rsi': _rsi_exit(side, features),
        'time_stop': _time_stop(position, close, atr, now),
        'session': _session_flatten(now, hold_extended),
    }

    for reason in EXIT_PRIORITY:
        if checks.get(reason):
            return {
                'reason': reason,
                'urgent': reason in URGENT,
                'side': 'sell' if side == 'long' else 'buy',
                'qty': position.get('qty'),
            }
    return None


def _hard_stop(side, entry, atr, high, low):
    dist = STOP_ATR * atr
    if side == 'long':
        return low <= entry - dist
    return high >= entry + dist


def _take_profit(side, entry, atr, close):
    dist = TARGET_ATR * atr
    if side == 'long':
        return close >= entry + dist
    return close <= entry - dist


def _trailing(side, position, close, atr):
    extreme = position.get('max_favorable')
    if extreme is None:
        return False
    entry = float(position['entry_price'])
    if side == 'long':
        if extreme < entry + TRAIL_ACTIVATE_ATR * atr:
            return False
        return close <= extreme - TRAIL_ATR * atr
    if extreme > entry - TRAIL_ACTIVATE_ATR * atr:
        return False
    return close >= extreme + TRAIL_ATR * atr


def _slope_reversal(side, features):
    s_by_n = (features or {}).get('s_by_n') or {}
    s5 = s_by_n.get(5)
    s15 = s_by_n.get(15)
    if s5 is None or s15 is None:
        return False
    if side == 'long':
        return s5 < 0 and s15 < 0
    return s5 > 0 and s15 > 0


def _rsi_exit(side, features):
    rsi7 = (features or {}).get('rsi7')
    pct5 = (features or {}).get('pct_5')
    if rsi7 is None or pct5 is None:
        return False
    flat = abs(pct5) < (PCT5_FLAT * 100.0)  # pct_5 is stored as percent
    if not flat:
        return False
    if side == 'long':
        return rsi7 > RSI_LONG_EXIT
    return rsi7 < RSI_SHORT_EXIT


def _time_stop(position, close, atr, now):
    entry_time = position.get('entry_time')
    if entry_time is None or now is None:
        return False
    now = to_eastern_datetime(now)
    start = to_eastern_datetime(entry_time)
    if now - start < timedelta(minutes=TIME_STOP_MIN):
        return False
    entry = float(position['entry_price'])
    side = position.get('side')
    if side == 'long':
        unreal = close - entry
    else:
        unreal = entry - close
    return unreal < TIME_STOP_ATR * atr


def _session_flatten(now, hold_extended):
    if hold_extended or now is None:
        return False
    dt = to_eastern_datetime(now)
    close_dt = datetime(
        dt.year, dt.month, dt.day,
        REGULAR_END.hour, REGULAR_END.minute,
        tzinfo=EASTERN,
    )
    return dt >= close_dt - timedelta(minutes=SESSION_FLATTEN_MINUTES)


def update_favorable(position, bar):
    """Track running extreme for trailing stops."""
    if position is None or bar is None:
        return position
    high = float(bar['high'])
    low = float(bar['low'])
    side = position.get('side')
    extreme = position.get('max_favorable')
    if side == 'long':
        position['max_favorable'] = high if extreme is None else max(extreme, high)
    else:
        position['max_favorable'] = low if extreme is None else min(extreme, low)
    return position
