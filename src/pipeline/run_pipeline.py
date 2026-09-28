import os
import config

from ingest.finnhub_news_fetcher import fetch_and_write_news
from ingest.news_loader import load_news

from features.relevance_filter import add_relevance_flag
from ai.relevance_classifier import add_ai_relevance_flag

from utils.pretty_news_export import write_news_pretty

from ingest.prices_fetcher import fetch_prices_from_news
from ingest.price_loader import load_prices
from ingest.bars_cache import load_jsonl
from features.price_movement import add_price_movement
from features.intraday_price_movement import add_intraday_price_movement


def fetch_news(days_back):
    return fetch_and_write_news(
        config.NEWS_JSON,
        config.WATCHLIST,
        days_back,
        config.FINNHUB_API_KEY
    )


def load_and_cap_news(max_items):
    news = load_news(config.NEWS_JSON)
    if len(news) > max_items:
        news = news.head(max_items).copy()
    return news


def add_relevance_with_fallback(news):
    cache_path = os.path.join(config.RESULTS_DIR, 'relevance_cache.sqlite')

    try:
        news = add_ai_relevance_flag(news, cache_path, model='mistral:7b', min_confidence=0.60)
        news['relevant_final'] = news['relevant_ai']
        return news, True
    except Exception:
        news = add_relevance_flag(news)
        news['relevant_final'] = news['relevant']
        return news, False


def write_relevance_reports(news):
    pretty_all = os.path.join(config.RESULTS_DIR, 'news_pretty_all.txt')
    pretty_rel = os.path.join(config.RESULTS_DIR, 'news_pretty_relevant.txt')
    pretty_irrel = os.path.join(config.RESULTS_DIR, 'news_pretty_irrelevant.txt')

    write_news_pretty(news, pretty_all, limit=len(news))

    rel = news[news['relevant_final'] == True].copy()
    irrel = news[news['relevant_final'] == False].copy()

    write_news_pretty(rel, pretty_rel, limit=len(rel))
    write_news_pretty(irrel, pretty_irrel, limit=len(irrel))

    return pretty_all, pretty_rel, pretty_irrel, len(rel), len(irrel)


def ensure_prices(news):
    nrows = fetch_prices_from_news(news, config.PRICES_JSON)
    return nrows


def ensure_intraday_bars(news):
    if not config.has_alpaca_keys():
        print('Alpaca keys missing; skipping 1-min bars')
        return 0
    try:
        from ingest.alpaca_price_fetcher import fetch_intraday_bars_for_news
        n = fetch_intraday_bars_for_news(news, config.BARS_1MIN_JSONL)
        print('Intraday 1-min bars fetched:', n)
        return n
    except Exception as exc:
        print('Intraday Alpaca bars skipped:', exc)
        return 0


def compute_price_impact(news):
    prices = load_prices(config.PRICES_JSON)

    rel = news[news['relevant_final'] == True].copy()
    labeled = add_price_movement(rel, prices)

    bars = load_jsonl(config.BARS_1MIN_JSONL)
    if bars:
        labeled = add_intraday_price_movement(labeled, bars)

    out_csv = os.path.join(config.RESULTS_DIR, 'price_impact_relevant.csv')
    labeled.to_csv(out_csv, index=False)

    cols = [
        'ticker', 'timestamp', 'session', 'headline', 'return', 'label',
        'px_at_event', 'ret_1m', 'ret_5m', 'ret_15m', 'ret_60m', 'ret_close',
        'source', 'url',
    ]
    show = []
    for c in cols:
        if c in labeled.columns:
            show.append(c)

    clean_csv = os.path.join(config.RESULTS_DIR, 'price_impact_relevant_clean.csv')
    labeled[show].to_csv(clean_csv, index=False)

    missing = 0
    if 'label' in labeled.columns:
        missing = int((labeled['label'] == 'MISSING').sum())

    return out_csv, clean_csv, len(labeled), missing


def run(max_items=200, days_back=14):
    config.ensure_dirs()

    print('Finnhub key present:', config.FINNHUB_API_KEY != '')

    fetched = fetch_news(days_back)

    news = load_and_cap_news(max_items)

    print('Timestamp type sample:', type(news.iloc[0]['timestamp']))
    print('Timestamp sample value:', news.iloc[0]['timestamp'])

    news, used_ai = add_relevance_with_fallback(news)

    pretty_all, pretty_rel, pretty_irrel, n_rel, n_irrel = write_relevance_reports(news)

    price_rows_written = ensure_prices(news)
    intraday_bars_written = ensure_intraday_bars(news)

    impact_csv, impact_clean_csv, impact_rows, missing_rows = compute_price_impact(news)

    return {
        'news_fetched': fetched,
        'news_processed': len(news),
        'used_ai': used_ai,
        'relevant': n_rel,
        'irrelevant': n_irrel,
        'pretty_all': pretty_all,
        'pretty_relevant': pretty_rel,
        'pretty_irrelevant': pretty_irrel,
        'prices_rows_written': price_rows_written,
        'intraday_bars_written': intraday_bars_written,
        'price_impact_csv': impact_csv,
        'price_impact_clean_csv': impact_clean_csv,
        'price_impact_rows': impact_rows,
        'price_impact_missing_rows': missing_rows
    }
