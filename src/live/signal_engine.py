"""Turn aggregated LLM news features into a single tradable decision per ticker.

The aggregate keys come from features.news_llm_features.aggregate_signals:
  llm_sentiment_mean      -1 bearish .. +1 bullish
  llm_interpret_conf_mean  0..1 how sure the model was of its reading
  llm_actionability_mean   0..1 how tradable the news is
  llm_is_catalyst_rate     0..1 share of articles that look price-moving
"""

STRATEGIES = ('sentiment', 'catalyst', 'combined')

DEFAULT_STRATEGY = 'combined'
DEFAULT_MIN_SIGNAL = 0.30
DEFAULT_SENTIMENT_THRESHOLD = 0.10


def _num(agg, key, default=0.0):
    try:
        return float(agg.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _direction(sentiment, threshold):
    if sentiment > threshold:
        return 'long'
    if sentiment < -threshold:
        return 'short'
    return 'flat'


def decide_from_aggregate(
    agg,
    strategy=DEFAULT_STRATEGY,
    min_signal=DEFAULT_MIN_SIGNAL,
    sentiment_threshold=DEFAULT_SENTIMENT_THRESHOLD,
    allow_short=True,
):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown strategy: %s' % strategy)

    agg = agg or {}
    news_count = int(_num(agg, 'llm_news_count', 0))
    sentiment = _num(agg, 'llm_sentiment_mean', 0.0)
    confidence = _num(agg, 'llm_interpret_conf_mean', 0.0)
    actionability = _num(agg, 'llm_actionability_mean', 0.0)
    catalyst_rate = _num(agg, 'llm_is_catalyst_rate', 0.0)

    direction = _direction(sentiment, sentiment_threshold)

    if strategy == 'sentiment':
        strength = abs(sentiment) * confidence
    elif strategy == 'catalyst':
        if catalyst_rate <= 0.0:
            strength = 0.0
            direction = 'flat'
        else:
            strength = abs(sentiment) * actionability
    else:
        catalyst_boost = 1.0 + 0.5 * catalyst_rate
        strength = abs(sentiment) * confidence * actionability * catalyst_boost

    strength = max(0.0, min(1.0, strength))

    reason = 'ok'
    if news_count == 0:
        direction = 'flat'
        strength = 0.0
        reason = 'no_news'
    elif direction == 'flat':
        reason = 'sentiment_below_threshold'
    elif strength < min_signal:
        reason = 'strength_below_min'
    elif direction == 'short' and not allow_short:
        direction = 'flat'
        reason = 'shorts_disabled'

    actionable = direction in ('long', 'short') and strength >= min_signal and reason == 'ok'

    return {
        'direction': direction,
        'strength': round(strength, 6),
        'actionable': actionable,
        'reason': reason,
        'strategy': strategy,
        'news_count': news_count,
        'sentiment': round(sentiment, 6),
        'confidence': round(confidence, 6),
        'actionability': round(actionability, 6),
        'catalyst_rate': round(catalyst_rate, 6),
    }


def order_side(direction):
    if direction == 'long':
        return 'buy'
    if direction == 'short':
        return 'sell'
    raise ValueError('No order side for direction %s' % direction)
