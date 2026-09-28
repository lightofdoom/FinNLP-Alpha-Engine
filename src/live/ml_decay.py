"""Per-ticker exponential ML-score decay across quiet 1-minute cycles.

Replace-or-decay (not algebraic max):
  fresh relevant news -> ml_t = new_score
  quiet cycle        -> ml_t = lambda ** dt_minutes * ml_{t-1}
  |ml_t| < floor     -> 0
  age > NEWS_MAX_AGE_MIN since last fresh print -> 0

dt is wall-clock minutes so a slow LLM poll does not under-decay.
"""

import json
import os
from datetime import datetime, timezone

import config


def parse_dt(value):
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def minutes_between(prev, now):
    if prev is None or now is None:
        return 0.0
    prev = parse_dt(prev) if not isinstance(prev, datetime) else prev
    now = parse_dt(now) if not isinstance(now, datetime) else now
    if prev is None or now is None:
        return 0.0
    if prev.tzinfo is None:
        prev = prev.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    dt = (now.astimezone(timezone.utc) - prev.astimezone(timezone.utc)).total_seconds() / 60.0
    if dt < 0:
        return 0.0
    return dt


def apply_floor(score, floor):
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.0
    try:
        floor = float(floor)
    except (TypeError, ValueError):
        floor = 0.0
    if floor > 0 and abs(score) < floor:
        return 0.0
    return score


def decayed_score(prev_score, dt_minutes, lambda_):
    try:
        prev_score = float(prev_score or 0.0)
    except (TypeError, ValueError):
        prev_score = 0.0
    try:
        lambda_ = float(lambda_)
    except (TypeError, ValueError):
        lambda_ = 0.75
    if lambda_ < 0:
        lambda_ = 0.0
    if dt_minutes is None or dt_minutes <= 0:
        return prev_score
    return prev_score * (lambda_ ** float(dt_minutes))


def update_ml_state(
    rows,
    ticker,
    now,
    new_score=None,
    fresh=False,
    lambda_=None,
    floor=None,
    max_age_min=None,
):
    """Update one ticker in a state dict. Returns the new row."""
    if lambda_ is None:
        lambda_ = float(getattr(config, 'ML_DECAY_LAMBDA', 0.75))
    if floor is None:
        floor = float(getattr(config, 'ML_SCORE_FLOOR', 0.02))
    if max_age_min is None:
        max_age_min = float(getattr(config, 'NEWS_MAX_AGE_MIN', 15))

    ticker = str(ticker).strip().upper()
    now_dt = parse_dt(now) or datetime.now(timezone.utc)
    rows = rows if rows is not None else {}
    prev = dict(rows.get(ticker) or {})
    prev_score = float(prev.get('ml_score') or 0.0)
    prev_ts = parse_dt(prev.get('updated_at'))
    last_fresh = parse_dt(prev.get('last_fresh_at'))

    if fresh:
        try:
            score = float(new_score if new_score is not None else 0.0)
        except (TypeError, ValueError):
            score = 0.0
        last_fresh = now_dt
    else:
        score = decayed_score(prev_score, minutes_between(prev_ts, now_dt), lambda_)

    if last_fresh is not None and max_age_min is not None and float(max_age_min) >= 0:
        if minutes_between(last_fresh, now_dt) > float(max_age_min):
            score = 0.0

    score = apply_floor(score, floor)
    row = {
        'ml_score': round(score, 8),
        'updated_at': to_iso(now_dt),
        'last_fresh_at': to_iso(last_fresh),
        'fresh': bool(fresh),
    }
    rows[ticker] = row
    return row


class DecayState:
    """Tiny on-disk map: ticker -> {ml_score, updated_at, last_fresh_at}."""

    def __init__(
        self,
        path=None,
        lambda_=None,
        floor=None,
        max_age_min=None,
    ):
        self.path = path or getattr(config, 'ML_DECAY_STATE_JSON', None)
        self.lambda_ = float(lambda_ if lambda_ is not None else getattr(config, 'ML_DECAY_LAMBDA', 0.75))
        self.floor = float(floor if floor is not None else getattr(config, 'ML_SCORE_FLOOR', 0.02))
        self.max_age_min = float(
            max_age_min if max_age_min is not None else getattr(config, 'NEWS_MAX_AGE_MIN', 15)
        )
        self.rows = {}
        self.load()

    def load(self):
        self.rows = {}
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.rows = {str(k).upper(): dict(v) for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            self.rows = {}

    def save(self):
        if not self.path:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.rows, f, indent=2)
        os.replace(tmp, self.path)

    def update(self, ticker, now, new_score=None, fresh=False):
        return update_ml_state(
            self.rows,
            ticker,
            now,
            new_score=new_score,
            fresh=fresh,
            lambda_=self.lambda_,
            floor=self.floor,
            max_age_min=self.max_age_min,
        )

    def score(self, ticker):
        row = self.rows.get(str(ticker).strip().upper()) or {}
        try:
            return float(row.get('ml_score') or 0.0)
        except (TypeError, ValueError):
            return 0.0
