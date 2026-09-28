"""Live 1-minute price features for math_score.

IEX is a small slice of US volume. vol_z and trade_count_z are relative to this
symbol's own recent IEX prints, not the consolidated tape.

VWAP coupling: vwap_score (continuous pullback in the raw blend) and ext_aligned
(chase penalty beyond 2*ATR, applied after raw) both use (close - vwap) / ATR.
They move together when price is far from VWAP; they are not independent features.
Do not drop one without revisiting the other.

ATR is a single value for the bar set. Missing ATR or price blocks all slope
windows via math_unavailable; there is no per-window ATR skip.
"""

import math
from datetime import datetime, timezone


SLOPE_WINDOWS = (5, 15, 30, 60)
SLOPE_WEIGHTS = {5: 0.40, 15: 0.30, 30: 0.20, 60: 0.10}
ALIGN_WINDOWS = (5, 15, 30)
MIN_BARS = 20
LOOKBACK = 90
ATR_PERIOD = 14
RSI_FAST = 7
RSI_SLOW = 14
VOL_Z_WINDOW = 20
IEX_THIN_BARS = 5
PCT5_FLAT = 0.0002  # 0.02% per bar for RSI-exit slope flatten
RSI_CONFLICT_PTS = 20.0


def _clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _closes(bars):
    return [float(b['close']) for b in bars]


def _field(bars, name, default=0.0):
    out = []
    for b in bars:
        v = b.get(name)
        if v is None:
            out.append(default)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(default)
    return out


def true_range(bar, prev_close):
    high = float(bar['high'])
    low = float(bar['low'])
    close_prev = float(prev_close)
    return max(high - low, abs(high - close_prev), abs(low - close_prev))


def wilder_atr(bars, period=ATR_PERIOD):
    if bars is None or len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        trs.append(true_range(bars[i], bars[i - 1]['close']))
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / float(period)
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / float(period)
    if atr <= 0:
        return None
    return atr


def wilder_rsi(closes, period):
    if closes is None or len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / float(period)
    avg_loss = sum(losses[:period]) / float(period)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / float(period)
        avg_loss = (avg_loss * (period - 1) + losses[i]) / float(period)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def log_ols_slope(closes):
    """OLS slope of ln(P) vs bar index. Dimensionless per bar (d ln P / di)."""
    n = len(closes)
    if n < 2:
        return None
    logs = []
    for p in closes:
        if p is None or p <= 0:
            return None
        logs.append(math.log(p))
    mean_i = (n - 1) / 2.0
    mean_y = sum(logs) / float(n)
    cov = 0.0
    var_i = 0.0
    for i, y in enumerate(logs):
        di = i - mean_i
        cov += di * (y - mean_y)
        var_i += di * di
    if var_i == 0:
        return None
    return cov / var_i


def sigma_n(atr, price, n):
    """Fractional 1-min scale: ATR/P / sqrt(n). Same units as (exp(slope)-1)."""
    if atr is None or price is None or atr <= 0 or price <= 0 or n <= 0:
        return None
    return atr / (price * math.sqrt(float(n)))


def s_from_slope(slope, sigma):
    """tanh((exp(slope)-1) / sigma_n): both numerator and denominator are fractions."""
    if slope is None or sigma is None or sigma <= 0:
        return None
    frac = math.exp(slope) - 1.0
    return math.tanh(frac / sigma)


def zscore_last(values, window=VOL_Z_WINDOW):
    if values is None or len(values) < 3:
        return None
    hist = values[-window - 1:-1] if len(values) > window else values[:-1]
    if len(hist) < 2:
        return None
    mean = sum(hist) / float(len(hist))
    var = sum((x - mean) ** 2 for x in hist) / float(len(hist))
    stdev = math.sqrt(var)
    if stdev == 0:
        return None
    return (values[-1] - mean) / stdev


def consecutive_signed_returns(closes):
    if closes is None or len(closes) < 2:
        return 0
    last = 0
    count = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i - 1] <= 0:
            break
        r = math.log(closes[i] / closes[i - 1])
        s = _sign(r)
        if s == 0:
            break
        if last == 0:
            last = s
            count = 1
            continue
        if s != last:
            break
        count += 1
    return last * count


def session_vwap(bars):
    num = 0.0
    den = 0.0
    for b in bars:
        vol = float(b.get('volume') or 0)
        vw = b.get('vwap')
        if vw is not None:
            px = float(vw)
        else:
            px = (float(b['high']) + float(b['low']) + float(b['close'])) / 3.0
        if vol <= 0:
            vol = 1.0
        num += px * vol
        den += vol
    if den <= 0:
        return float(bars[-1]['close'])
    return num / den


MINUTES_PER_RTH = 390.0
SESSIONS_PER_YEAR = 252.0


def realized_vol_ann(closes, window=30):
    """Annualized vol from 1-min log returns: stdev * sqrt(390 * 252)."""
    if closes is None or len(closes) < window + 1:
        return None
    rets = []
    start = len(closes) - window
    for i in range(start, len(closes)):
        if closes[i - 1] <= 0:
            continue
        rets.append(math.log(closes[i] / closes[i - 1]))
    if len(rets) < 5:
        return None
    mean = sum(rets) / float(len(rets))
    var = sum((x - mean) ** 2 for x in rets) / float(len(rets))
    return math.sqrt(var) * math.sqrt(MINUTES_PER_RTH * SESSIONS_PER_YEAR)


def trend_align(s_by_n):
    """s_by_n maps window length -> s_n for available windows only."""
    keys = [k for k in ALIGN_WINDOWS if k in s_by_n]
    if keys == []:
        return 0.0
    if keys == [5]:
        return 0.25 if _sign(s_by_n[5]) != 0 else 0.0
    if 5 in s_by_n and 15 in s_by_n:
        if _sign(s_by_n[5]) != 0 and _sign(s_by_n[5]) == _sign(s_by_n[15]):
            if 30 in s_by_n and _sign(s_by_n[30]) == _sign(s_by_n[5]):
                return 1.0
            return 0.5
        return 0.0
    signs = [_sign(s_by_n[k]) for k in keys]
    if 0 in signs:
        return 0.0
    if len(set(signs)) == 1:
        return 1.0
    return 0.0


def iex_thin_entry(bars, n=IEX_THIN_BARS):
    if bars is None or len(bars) < n:
        return True
    counts = _field(bars[-n:], 'trade_count', default=0.0)
    return all(c == 0 for c in counts)


def iex_thin_exit(bars, silent_minutes=20, baseline=60):
    """True only after a stretch of zero IEX prints following prior activity."""
    if bars is None or len(bars) < silent_minutes:
        return False
    tail = _field(bars[-silent_minutes:], 'trade_count', default=0.0)
    if not all(c == 0 for c in tail):
        return False
    prior = _field(bars[:-silent_minutes][-baseline:], 'trade_count', default=0.0)
    if prior == []:
        return False
    return sum(prior) > 0


def compute_price_features(bars):
    """Return a feature dict. math_unavailable is set instead of inventing scores."""
    empty = {
        'math_unavailable': True,
        'unavailable_reason': 'no_bars',
        'math_score': 0.0,
        'bar_count': 0,
    }
    if bars is None or len(bars) == 0:
        return dict(empty)

    bars = bars[-LOOKBACK:]
    n_bars = len(bars)
    closes = _closes(bars)
    price = closes[-1]
    atr = wilder_atr(bars, ATR_PERIOD)
    rsi7 = wilder_rsi(closes, RSI_FAST)
    rsi14 = wilder_rsi(closes, RSI_SLOW)
    volumes = _field(bars, 'volume', 0.0)
    trade_counts = _field(bars, 'trade_count', 0.0)
    vol_z = zscore_last(volumes, VOL_Z_WINDOW)
    tc_z = zscore_last(trade_counts, VOL_Z_WINDOW)
    vwap = session_vwap(bars)
    mom_15 = None
    if n_bars >= 16 and closes[-16] > 0:
        mom_15 = closes[-1] / closes[-16] - 1.0
    consec = consecutive_signed_returns(closes)
    sigma_30 = realized_vol_ann(closes, 30)
    rsi_conflict = False
    if rsi7 is not None and rsi14 is not None:
        rsi_conflict = abs(rsi7 - rsi14) > RSI_CONFLICT_PTS

    out = {
        'bar_count': n_bars,
        'price': price,
        'atr': atr,
        'rsi7': rsi7,
        'rsi14': rsi14,
        'rsi_conflict': rsi_conflict,
        'vwap': vwap,
        'vol_z': vol_z,
        'trade_count_z': tc_z,
        'mom_15': mom_15,
        'consec': consec,
        'sigma_30_ann': sigma_30,
        'available_windows': [],
        's_by_n': {},
        'pct_5': None,
        'math_unavailable': False,
        'unavailable_reason': None,
        'math_score': 0.0,
    }

    if n_bars < MIN_BARS:
        out['math_unavailable'] = True
        out['unavailable_reason'] = 'thin_bars'
        return out
    if atr is None or price is None or price <= 0:
        out['math_unavailable'] = True
        out['unavailable_reason'] = 'atr_unavailable'
        return out
    if iex_thin_entry(bars):
        out['math_unavailable'] = True
        out['unavailable_reason'] = 'iex_thin'
        return out

    s_by_n = {}
    slopes = {}
    for n in SLOPE_WINDOWS:
        if n_bars < n:
            continue
        slope = log_ols_slope(closes[-n:])
        sig = sigma_n(atr, price, n)
        s_val = s_from_slope(slope, sig)
        if s_val is None:
            continue
        s_by_n[n] = s_val
        slopes[n] = slope

    out['available_windows'] = sorted(s_by_n.keys())
    out['s_by_n'] = dict(s_by_n)
    if 5 in slopes:
        out['pct_5'] = 100.0 * (math.exp(slopes[5]) - 1.0)

    if s_by_n == {}:
        out['math_unavailable'] = True
        out['unavailable_reason'] = 'no_slope_windows'
        return out

    weight_sum = sum(SLOPE_WEIGHTS[n] for n in s_by_n)
    trend = 0.0
    for n, s_val in s_by_n.items():
        trend += (SLOPE_WEIGHTS[n] / weight_sum) * s_val
    align = trend_align(s_by_n)
    trend_score = _clip(trend, -1.0, 1.0) * (0.5 + 0.5 * align)
    out['trend_score'] = trend_score
    out['trend_align'] = align

    rsi_entry = 0.0
    if rsi7 is not None:
        rsi_entry = _clip((rsi7 - 50.0) / 25.0, -1.0, 1.0)
    out['rsi_entry'] = rsi_entry

    vwap_dist = (price - vwap) / atr
    out['vwap_dist'] = vwap_dist
    vwap_score = _clip(-vwap_dist / 2.0, -1.0, 1.0)
    out['vwap_score'] = vwap_score

    consec_term = 0.0
    if trend_score != 0:
        consec_term = math.tanh(abs(consec) / 5.0) * _sign(trend_score)
    out['consec_term'] = consec_term

    raw = 0.45 * trend_score + 0.25 * rsi_entry + 0.15 * vwap_score + 0.15 * consec_term
    out['raw'] = raw

    ext = _clip(abs(price - vwap) / (2.0 * atr) - 1.0, 0.0, 1.0)
    ext_aligned = 0.0
    if _sign(raw) != 0 and _sign(price - vwap) == _sign(raw):
        ext_aligned = ext
    out['ext'] = ext
    out['ext_aligned'] = ext_aligned

    part = 0.0
    if vol_z is not None:
        part = math.tanh(max(vol_z, 0.0) / 2.0)
    out['part'] = part

    math_score = raw * (0.6 + 0.4 * part) * (1.0 - 0.5 * ext_aligned)
    out['math_score'] = _clip(math_score, -1.0, 1.0)
    out['iex_thin_exit'] = iex_thin_exit(bars)
    return out


def parse_bar_row(row):
    """Normalize a stream/REST bar dict."""
    return {
        'open': float(row.get('open') or row.get('o') or 0),
        'high': float(row.get('high') or row.get('h') or 0),
        'low': float(row.get('low') or row.get('l') or 0),
        'close': float(row.get('close') or row.get('c') or row.get('price') or 0),
        'volume': float(row.get('volume') or row.get('v') or 0),
        'vwap': row.get('vwap'),
        'trade_count': float(row.get('trade_count') or row.get('n') or 0),
        'timestamp': row.get('timestamp') or row.get('t'),
    }
