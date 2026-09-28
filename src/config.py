import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except ImportError:
    pass


DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PAPER_RESULTS_DIR = os.path.join(RESULTS_DIR, 'paper')

SNAPSHOTS_JSONL = os.path.join(RAW_DIR, 'snapshots.jsonl')
PAPER_FILLS_JSONL = os.path.join(PAPER_RESULTS_DIR, 'fills.jsonl')
PAPER_ORDERS_JSONL = os.path.join(PAPER_RESULTS_DIR, 'orders.jsonl')
PAPER_KILL_SWITCH_JSON = os.path.join(PAPER_RESULTS_DIR, 'kill_switch_state.json')
RESEARCH_SET_DIR = os.path.join(DATA_DIR, 'research_set')
RESEARCH_DB = os.path.join(RESEARCH_SET_DIR, 'research.sqlite')
RESEARCH_WATCHLIST_JSON = os.path.join(RESEARCH_SET_DIR, 'tickers.json')
RESEARCH_RELEVANT_JSONL = os.path.join(RESEARCH_SET_DIR, 'relevant_news.jsonl')
REPLAY_RESULTS_DIR = os.path.join(RESULTS_DIR, 'replay')

FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')

# Locked research set: IEX-liquid names with enough Finnhub flow to test the news gate.
# Do not rotate after the 24/7 collection run starts.
WATCHLIST = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'AMD']

PAPER_API_BASE_URL = 'https://paper-api.alpaca.markets'
LIVE_API_BASE_URL = 'https://api.alpaca.markets'

APCA_API_KEY_ID = os.environ.get('APCA_API_KEY_ID', '')
APCA_API_SECRET_KEY = os.environ.get('APCA_API_SECRET_KEY', '')
APCA_API_BASE_URL = os.environ.get('APCA_API_BASE_URL', PAPER_API_BASE_URL)
ALPACA_DATA_FEED = os.environ.get('ALPACA_DATA_FEED', 'iex').strip().lower() or 'iex'
ALPACA_ALLOW_LIVE = os.environ.get('ALPACA_ALLOW_LIVE', 'false')
ALPACA_MAX_SYMBOLS = int(os.environ.get('ALPACA_MAX_SYMBOLS', '30'))
ALPACA_REST_PER_MIN = int(os.environ.get('ALPACA_REST_PER_MIN', '180'))
ALPACA_NOTIONAL = float(os.environ.get('ALPACA_NOTIONAL', '1000'))
ALPACA_LIMIT_OFFSET_BPS = float(os.environ.get('ALPACA_LIMIT_OFFSET_BPS', '5'))
ALPACA_MAX_ORDERS_PER_DAY = int(os.environ.get('ALPACA_MAX_ORDERS_PER_DAY', '10'))
ALPACA_MAX_LOSS_DOLLARS = float(os.environ.get('ALPACA_MAX_LOSS_DOLLARS', '2000'))
ALPACA_SNAPSHOT_FALLBACK_SEC = float(os.environ.get('ALPACA_SNAPSHOT_FALLBACK_SEC', '15'))
ALPACA_CLOCK_POLL_SEC = float(os.environ.get('ALPACA_CLOCK_POLL_SEC', '60'))
ALPACA_STREAM_STALE_SEC = float(os.environ.get('ALPACA_STREAM_STALE_SEC', '20'))


def _truthy(value):
    if value is None:
        return False
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


ENTRY_T = float(os.environ.get('ENTRY_T', '0.20'))
W_MATH = float(os.environ.get('W_MATH', '0.60'))
W_ML = float(os.environ.get('W_ML', '0.40'))
W_MATH_FRESH = float(os.environ.get('W_MATH_FRESH', '0.30'))
W_ML_FRESH = float(os.environ.get('W_ML_FRESH', '0.70'))
W_MATH_QUIET = float(os.environ.get('W_MATH_QUIET', '0.85'))
W_ML_QUIET = float(os.environ.get('W_ML_QUIET', '0.15'))
ML_DECAY_LAMBDA = float(os.environ.get('ML_DECAY_LAMBDA', '0.75'))
ML_SCORE_FLOOR = float(os.environ.get('ML_SCORE_FLOOR', '0.02'))
NEWS_MAX_AGE_MIN = float(os.environ.get('NEWS_MAX_AGE_MIN', '15'))
NEWS_REQUIRED = _truthy(os.environ.get('NEWS_REQUIRED', 'true'))
ML_DECAY_STATE_JSON = os.path.join(PAPER_RESULTS_DIR, 'ml_decay_state.json')
OPTIMIZER_ENABLED = _truthy(os.environ.get('OPTIMIZER_ENABLED', 'true'))
OPTIMIZER_RISK_AVERSION = float(os.environ.get('OPTIMIZER_RISK_AVERSION', '1.0'))
OPTIMIZER_TURNOVER_PENALTY = float(os.environ.get('OPTIMIZER_TURNOVER_PENALTY', '0.0005'))
GROSS_LEVERAGE = float(os.environ.get('GROSS_LEVERAGE', '0.25'))
COV_SHRINKAGE = float(os.environ.get('COV_SHRINKAGE', '0.30'))
R_PCT = float(os.environ.get('R_PCT', '0.004'))
SIZING_MODE = os.environ.get('SIZING_MODE', 'small_paper').strip().lower() or 'small_paper'
STOP_ATR_MULT = float(os.environ.get('STOP_ATR_MULT', '1.5'))
TARGET_ATR_MULT = float(os.environ.get('TARGET_ATR_MULT', '2.0'))
TRAIL_ACTIVATE_ATR = float(os.environ.get('TRAIL_ACTIVATE_ATR', '1.0'))
TRAIL_ATR = float(os.environ.get('TRAIL_ATR', '1.0'))
TIME_STOP_MIN = int(os.environ.get('TIME_STOP_MIN', '45'))
HOLD_EXTENDED = _truthy(os.environ.get('HOLD_EXTENDED', 'false'))
STALE_ENTRY_SEC = float(os.environ.get('STALE_ENTRY_SEC', '60'))
URGENT_LIMIT_OFFSET_BPS = float(os.environ.get('URGENT_LIMIT_OFFSET_BPS', '20'))


def allow_live_trading():
    return _truthy(os.environ.get('ALPACA_ALLOW_LIVE', ALPACA_ALLOW_LIVE))


def has_alpaca_keys():
    key = os.environ.get('APCA_API_KEY_ID', APCA_API_KEY_ID)
    secret = os.environ.get('APCA_API_SECRET_KEY', APCA_API_SECRET_KEY)
    return bool(str(key).strip()) and bool(str(secret).strip())


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PAPER_RESULTS_DIR, exist_ok=True)
    os.makedirs(RESEARCH_SET_DIR, exist_ok=True)
    os.makedirs(REPLAY_RESULTS_DIR, exist_ok=True)
