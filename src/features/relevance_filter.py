import re


IRRELEVANT_KEYWORDS = [
    'etf',
    'fund',
    'portfolio',
    'holdings',
    'rebalanc',
    'index',
    'basket',
    'exposure',
    'leveraged',
    'inverse',
    'distribution',
    'yield',
    'dividend of the fund'
]


RELEVANT_KEYWORDS = [
    'earnings',
    'revenue',
    'profit',
    'guidance',
    'forecast',
    'acquisition',
    'merger',
    'lawsuit',
    'regulator',
    'sec',
    'antitrust',
    'investigation',
    'ceo',
    'cfo',
    'executive',
    'steps down',
    'appoints',
    'product',
    'launch',
    'recall',
    'ai',
    'chip',
    'datacenter'
]


def _normalize(text):
    if text is None:
        return ''
    return text.lower()


def is_relevant(headline, summary):
    h = _normalize(headline)
    s = _normalize(summary)
    text = h + ' ' + s

    for word in IRRELEVANT_KEYWORDS:
        if word in text:
            return False

    for word in RELEVANT_KEYWORDS:
        if word in text:
            return True

    return False


def add_relevance_flag(news):
    flags = []

    for row in news.itertuples(index=False):
        headline = getattr(row, 'headline', '')
        summary = getattr(row, 'summary', '')

        flags.append(is_relevant(headline, summary))

    out = news.copy()
    out['relevant'] = flags
    return out
