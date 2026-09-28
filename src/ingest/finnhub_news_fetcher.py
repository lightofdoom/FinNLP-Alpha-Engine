import json
import os
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta


FINNHUB_BASE = 'https://finnhub.io/api/v1'


def _get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode('utf-8'))


def fetch_news_items(tickers, days_back, api_key):
    """Fetch company news without touching disk, for the live polling loop."""
    if api_key == '':
        raise ValueError('FINNHUB_API_KEY is empty')

    end_d = date.today()
    start_d = end_d - timedelta(days=days_back)

    all_items = []

    for t in tickers:
        params = {
            'symbol': t,
            'from': start_d.isoformat(),
            'to': end_d.isoformat(),
            'token': api_key
        }

        url = FINNHUB_BASE + '/company-news?' + urllib.parse.urlencode(params)
        items = _get_json(url)

        for it in items:
            it['ticker'] = t
            if 'datetime' in it:
                it['timestamp'] = it['datetime']
            all_items.append(it)

        time.sleep(0.25)

    return all_items


def fetch_and_write_news(out_path, tickers, days_back, api_key):
    all_items = fetch_news_items(tickers, days_back, api_key)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=2)

    return len(all_items)
