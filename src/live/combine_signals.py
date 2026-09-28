"""Dynamic math–ML entry blend. ENTRY_T is the only entry threshold."""

import config


W_MATH = 0.60
W_ML = 0.40
W_MATH_FRESH = 0.30
W_ML_FRESH = 0.70
W_MATH_QUIET = 0.85
W_ML_QUIET = 0.15
ENTRY_T = 0.20
RISK_DAMPENER = 0.8


def _clip(x, lo=-1.0, hi=1.0):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _num(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _risk_count(agg):
    total = 0
    for key, value in (agg or {}).items():
        if not str(key).startswith('llm_risk_'):
            continue
        if str(key).endswith('none_count'):
            continue
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def _catalyst_count(agg):
    total = 0
    for key, value in (agg or {}).items():
        if not str(key).startswith('llm_cat_'):
            continue
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def ml_score_from_aggregate(agg, strategy='combined'):
    """Signed ML score in [-1, 1]. Logs dampener via returned dict."""
    agg = agg or {}
    news_count = int(_num(agg, 'llm_news_count', 0))
    sentiment = _num(agg, 'llm_sentiment_mean', 0.0)
    confidence = _num(agg, 'llm_interpret_conf_mean', 0.0)
    actionability = _num(agg, 'llm_actionability_mean', 0.0)
    catalyst_rate = _num(agg, 'llm_is_catalyst_rate', 0.0)

    if strategy == 'sentiment':
        raw = sentiment * confidence
    elif strategy == 'catalyst':
        raw = sentiment * actionability if catalyst_rate > 0 else 0.0
    else:
        raw = sentiment * confidence * actionability * (1.0 + 0.5 * catalyst_rate)

    before = _clip(raw)
    after = before
    damped = False
    if _risk_count(agg) > _catalyst_count(agg) and before != 0:
        after = _clip(before * RISK_DAMPENER)
        damped = True

    return {
        'ml_score': after,
        'ml_score_before_dampener': before,
        'ml_damped': damped,
        'news_count': news_count,
        'sentiment': sentiment,
        'confidence': confidence,
        'actionability': actionability,
        'catalyst_rate': catalyst_rate,
        'strategy': strategy,
    }


def blend_weights(fresh_news, w_math=None, w_ml=None):
    """Fresh-news cycle is ML-heavy; quiet/decayed cycle is math-heavy."""
    if w_math is not None or w_ml is not None:
        if w_math is None:
            w_math = float(getattr(config, 'W_MATH', W_MATH))
        if w_ml is None:
            w_ml = float(getattr(config, 'W_ML', W_ML))
        return float(w_math), float(w_ml)
    if fresh_news:
        return (
            float(getattr(config, 'W_MATH_FRESH', W_MATH_FRESH)),
            float(getattr(config, 'W_ML_FRESH', W_ML_FRESH)),
        )
    return (
        float(getattr(config, 'W_MATH_QUIET', W_MATH_QUIET)),
        float(getattr(config, 'W_ML_QUIET', W_ML_QUIET)),
    )


def combine_entry(
    math_features,
    ml_features,
    w_math=None,
    w_ml=None,
    entry_t=None,
    news_required=True,
    allow_short=True,
    fresh_news=None,
):
    if entry_t is None:
        entry_t = float(getattr(config, 'ENTRY_T', ENTRY_T))

    math_features = math_features or {}
    ml_features = ml_features or {}

    if fresh_news is None:
        fresh_news = bool(ml_features.get('fresh_news'))

    w_math, w_ml = blend_weights(fresh_news, w_math=w_math, w_ml=w_ml)

    reason = 'ok'
    math_score = float(math_features.get('math_score') or 0.0)
    ml_score = float(ml_features.get('ml_score') or 0.0)
    news_count = int(ml_features.get('news_count') or 0)
    has_ml = abs(ml_score) > 0.0

    if math_features.get('math_unavailable'):
        return _flat(
            'math_unavailable', math_score, ml_score, math_features, ml_features,
            w_math=w_math, w_ml=w_ml, fresh_news=fresh_news,
        )
    if news_required and news_count < 1 and not has_ml:
        return _flat(
            'no_news', math_score, ml_score, math_features, ml_features,
            w_math=w_math, w_ml=w_ml, fresh_news=fresh_news,
        )

    entry_score = w_math * math_score + w_ml * ml_score

    if _sign(math_score) != 0 and _sign(ml_score) != 0 and _sign(math_score) != _sign(ml_score):
        return _flat(
            'disagree', math_score, ml_score, math_features, ml_features,
            entry_score=entry_score, w_math=w_math, w_ml=w_ml, fresh_news=fresh_news,
        )

    if entry_score > entry_t:
        direction = 'long'
    elif entry_score < -entry_t:
        direction = 'short'
    else:
        return _flat(
            'below_entry_t', math_score, ml_score, math_features, ml_features,
            entry_score=entry_score, w_math=w_math, w_ml=w_ml, fresh_news=fresh_news,
        )

    if direction == 'short' and not allow_short:
        return _flat(
            'shorts_disabled', math_score, ml_score, math_features, ml_features,
            entry_score=entry_score, w_math=w_math, w_ml=w_ml, fresh_news=fresh_news,
        )

    return {
        'direction': direction,
        'actionable': True,
        'reason': reason,
        'entry_score': round(entry_score, 6),
        'strength': round(abs(entry_score), 6),
        'math_score': round(math_score, 6),
        'ml_score': round(ml_score, 6),
        'news_count': news_count,
        'fresh_news': bool(fresh_news),
        'w_math': round(w_math, 6),
        'w_ml': round(w_ml, 6),
        'math_features': math_features,
        'ml_features': ml_features,
    }


def _flat(reason, math_score, ml_score, math_features, ml_features, entry_score=None,
          w_math=None, w_ml=None, fresh_news=False):
    if entry_score is None:
        entry_score = 0.0
    return {
        'direction': 'flat',
        'actionable': False,
        'reason': reason,
        'entry_score': round(entry_score, 6),
        'strength': round(abs(entry_score), 6),
        'math_score': round(float(math_score or 0.0), 6),
        'ml_score': round(float(ml_score or 0.0), 6),
        'news_count': int((ml_features or {}).get('news_count') or 0),
        'fresh_news': bool(fresh_news),
        'w_math': None if w_math is None else round(float(w_math), 6),
        'w_ml': None if w_ml is None else round(float(w_ml), 6),
        'math_features': math_features,
        'ml_features': ml_features,
    }
