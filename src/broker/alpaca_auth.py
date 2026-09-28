import os

import config


PAPER_API_BASE_URL = 'https://paper-api.alpaca.markets'
LIVE_API_HOST = 'api.alpaca.markets'
PAPER_API_HOST = 'paper-api.alpaca.markets'


def _env(name, fallback=''):
    value = os.environ.get(name)
    if value is not None and str(value).strip() != '':
        return str(value).strip()
    if fallback is None:
        return ''
    return str(fallback).strip()


def is_live_base_url(url):
    """Return True only for the live trading host, not paper-api (substring trap)."""
    if url is None:
        return False
    text = str(url).strip().lower()
    if PAPER_API_HOST in text:
        return False
    if LIVE_API_HOST in text:
        return True
    return False


def get_alpaca_credentials():
    api_key = _env('APCA_API_KEY_ID', getattr(config, 'APCA_API_KEY_ID', ''))
    secret_key = _env('APCA_API_SECRET_KEY', getattr(config, 'APCA_API_SECRET_KEY', ''))
    base_url = _env('APCA_API_BASE_URL', getattr(config, 'APCA_API_BASE_URL', PAPER_API_BASE_URL))
    if base_url == '':
        base_url = PAPER_API_BASE_URL
    feed = _env('ALPACA_DATA_FEED', getattr(config, 'ALPACA_DATA_FEED', 'iex')).lower()
    if feed == '':
        feed = 'iex'
    allow_live = config.allow_live_trading()
    paper = not is_live_base_url(base_url)
    return {
        'api_key': api_key,
        'secret_key': secret_key,
        'base_url': base_url,
        'feed': feed,
        'allow_live': allow_live,
        'paper': paper,
    }


def assert_paper_or_allowed(creds=None):
    if creds is None:
        creds = get_alpaca_credentials()
    if creds['paper']:
        return creds
    if creds['allow_live']:
        return creds
    raise RuntimeError(
        'Refusing live Alpaca trading host %s. Set APCA_API_BASE_URL to %s '
        'or set ALPACA_ALLOW_LIVE=true.' % (creds['base_url'], PAPER_API_BASE_URL)
    )


def require_keys(creds=None):
    if creds is None:
        creds = get_alpaca_credentials()
    if creds['api_key'] == '' or creds['secret_key'] == '':
        raise RuntimeError('Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY')
    return creds
