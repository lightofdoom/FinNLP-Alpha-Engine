import hashlib
import json
import os
import sqlite3
import threading
import typing

import pandas as pd


PROMPT_VERSION = 'v2-ticker'

NEUTRAL_SIGNALS = {
    'sentiment': 0.0,
    'interpret_confidence': 0.0,
    'is_catalyst': False,
    'actionability': 0.0,
    'time_horizon': 'medium',
    'catalyst_types': ['none'],
    'risk_flags': ['none'],
    'entities': [],
}


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class SqliteCache:
    """Cache shared across the live engine's threads, so connections must not be thread-bound."""

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute('CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT NOT NULL)')
        self._conn.commit()

    def get(self, key: str) -> typing.Optional[str]:
        with self._lock:
            cur = self._conn.execute('SELECT v FROM cache WHERE k=?', (key,))
            row = cur.fetchone()
        if row:
            return row[0]
        return None

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute('INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)', (key, value))
            self._conn.commit()


def _coerce_float(x, lo, hi, default=0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def _coerce_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        t = x.strip().lower()
        if t in ['true', 't', 'yes', 'y', '1']:
            return True
        if t in ['false', 'f', 'no', 'n', '0']:
            return False
    return False


def _coerce_list_str(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, str) and v.strip() != '':
                out.append(v.strip())
        return out
    if isinstance(x, str):
        s = x.strip()
        if s == '':
            return []
        return [s]
    return []


def _extract_json_object(raw: str) -> dict:
    raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        chunk = raw[start:end + 1]
        return json.loads(chunk)

    raise ValueError('Could not parse JSON from model response')


def _normalize_signals(obj: dict) -> dict:
    allowed_horizon = {'short', 'medium', 'long'}

    catalyst_allowed = {
        'earnings', 'guidance', 'product', 'mna', 'regulation', 'lawsuit', 'macro',
        'analyst', 'partnership', 'supply_chain', 'management', 'none'
    }

    risk_allowed = {
        'macro', 'regulation', 'valuation', 'competition', 'execution', 'demand',
        'supply_chain', 'geopolitics', 'rates', 'none'
    }

    sentiment = _coerce_float(obj.get('sentiment'), -1.0, 1.0, 0.0)
    conf = _coerce_float(obj.get('interpret_confidence'), 0.0, 1.0, 0.5)
    is_catalyst = _coerce_bool(obj.get('is_catalyst'))
    actionability = _coerce_float(obj.get('actionability'), 0.0, 1.0, 0.3)

    horizon = obj.get('time_horizon')
    if isinstance(horizon, str):
        horizon = horizon.strip().lower()
    if horizon not in allowed_horizon:
        horizon = 'medium'

    cats = [c.strip().lower() for c in _coerce_list_str(obj.get('catalyst_types'))]
    cats = [c for c in cats if c in catalyst_allowed]
    if len(cats) == 0:
        cats = ['none']

    risks = [r.strip().lower() for r in _coerce_list_str(obj.get('risk_flags'))]
    risks = [r for r in risks if r in risk_allowed]
    if len(risks) == 0:
        risks = ['none']

    ents = _coerce_list_str(obj.get('entities'))

    return {
        'sentiment': sentiment,
        'interpret_confidence': conf,
        'is_catalyst': is_catalyst,
        'actionability': actionability,
        'time_horizon': horizon,
        'catalyst_types': cats,
        'risk_flags': risks,
        'entities': ents
    }


class NewsLLMFeatureExtractor:
    def __init__(
        self,
        provider: str = 'ollama',
        model_name: str = 'mistral:7b',
        cache_path: str = 'results/llm_cache.sqlite',
        temperature: float = 0.0,
        ollama_base_url: str = '',
        num_predict: int = 320,
        timeout_sec: float = 90.0
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.cache = SqliteCache(cache_path)
        self.ollama_base_url = ollama_base_url
        self.num_predict = num_predict
        self.timeout_sec = timeout_sec
        self.last_error = None
        self._llm = None

    def _ensure_llm(self) -> None:
        if self._llm is not None:
            return

        if self.provider != 'ollama':
            raise ValueError("Only provider='ollama' is supported in this free setup.")

        try:
            from langchain_ollama import ChatOllama
        except Exception as e:
            raise RuntimeError(
                "Missing dependency. Install:\n"
                "  pip install langchain-ollama\n"
                "and also:\n"
                "  pip install langchain\n"
            ) from e

        kwargs = {'model': self.model_name, 'temperature': self.temperature}
        if self.ollama_base_url.strip() != '':
            kwargs['base_url'] = self.ollama_base_url
        if self.num_predict:
            kwargs['num_predict'] = self.num_predict

        try:
            self._llm = ChatOllama(**kwargs)
        except TypeError:
            self._llm = ChatOllama(model=self.model_name, temperature=self.temperature)

    def _prompt(self, text: str, ticker: str = '') -> list[dict]:
        subject = 'the company in the text'
        if ticker.strip() != '':
            subject = 'ticker ' + ticker.strip().upper()

        return [
            {
                'role': 'system',
                'content': (
                    'You extract structured, trading-relevant signals from short news text.\n'
                    'Use ONLY the provided text.\n'
                    'Return ONLY valid JSON, no extra commentary.\n'
                )
            },
            {
                'role': 'user',
                'content': (
                    'Judge sentiment for the share price of ' + subject + '.\n'
                    'Positive sentiment means the news should push that share price up.\n\n'
                    'Allowed time_horizon: short, medium, long\n'
                    'Allowed catalyst_types: earnings, guidance, product, mna, regulation, lawsuit, macro, '
                    'analyst, partnership, supply_chain, management, none\n'
                    'Allowed risk_flags: macro, regulation, valuation, competition, execution, demand, '
                    'supply_chain, geopolitics, rates, none\n\n'
                    'Return JSON with exactly these keys:\n'
                    '{\n'
                    '  "sentiment": float in [-1, 1],\n'
                    '  "interpret_confidence": float in [0, 1],\n'
                    '  "is_catalyst": boolean,\n'
                    '  "actionability": float in [0, 1],\n'
                    '  "time_horizon": "short"|"medium"|"long",\n'
                    '  "catalyst_types": [string],\n'
                    '  "risk_flags": [string],\n'
                    '  "entities": [string]\n'
                    '}\n\n'
                    'Text:\n'
                    + text
                )
            }
        ]

    def extract_one(self, headline: str, summary: str = '', ticker: str = '') -> dict:
        h = (headline or '').strip()
        s = (summary or '').strip()
        text = h if s == '' else (h + '\n' + s)

        if text == '':
            return dict(NEUTRAL_SIGNALS)

        key = _stable_hash(PROMPT_VERSION + '|' + self.model_name + '|' + str(ticker).upper() + '|' + text)
        cached = self.cache.get(key)
        if cached is not None:
            return json.loads(cached)

        try:
            self._ensure_llm()
            messages = self._prompt(text, ticker=ticker)
            resp = self._llm.invoke(messages)
            raw = resp.content if hasattr(resp, 'content') else str(resp)
            obj = _extract_json_object(raw)
            norm = _normalize_signals(obj)
        except Exception as exc:
            # A live session must not die because one headline confused the model.
            self.last_error = str(exc)
            return dict(NEUTRAL_SIGNALS)

        self.cache.set(key, json.dumps(norm))
        return norm

    def extract_many(self, rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            out.append(
                self.extract_one(
                    r.get('headline', ''),
                    r.get('summary', ''),
                    ticker=r.get('ticker', '')
                )
            )
        return out

# === LLM-DERIVED NEWS FEATURES (EVENT-LEVEL) ===
#
# Sentiment & confidence:
# - llm_sentiment_mean: Average tone of news (-1 bearish → +1 bullish)
# - llm_sentiment_max: Most bullish single article
# - llm_sentiment_min: Most bearish single article
# - llm_interpret_conf_mean: LLM confidence in interpretation (low = vague commentary)
# - llm_actionability_mean: How tradable/actionable the news is (0–1)
#
# Catalyst presence:
# - llm_is_catalyst_rate: Fraction of articles with a clear price-moving catalyst
#
# Time horizon (distribution):
# - llm_time_horizon_short_rate: Immediate impact (days)
# - llm_time_horizon_medium_rate: Medium-term impact (weeks)
# - llm_time_horizon_long_rate: Long-term impact (months+)
#
# Catalyst type counts (what kind of news):
# - llm_cat_earnings_count: Earnings or earnings guidance
# - llm_cat_guidance_count: Forward guidance changes
# - llm_cat_product_count: Product launches or updates
# - llm_cat_mna_count: Mergers & acquisitions
# - llm_cat_regulation_count: Regulatory/government actions
# - llm_cat_lawsuit_count: Legal actions or rulings
# - llm_cat_macro_count: Macro-economic events
# - llm_cat_analyst_count: Analyst ratings or price targets
# - llm_cat_partnership_count: Strategic partnerships
# - llm_cat_supply_chain_count: Manufacturing/logistics issues
# - llm_cat_management_count: Leadership changes
#
# Risk flag counts (what could go wrong):
# - llm_risk_macro_count: Macro-economic risk
# - llm_risk_regulation_count: Regulatory risk
# - llm_risk_valuation_count: Overvaluation concerns
# - llm_risk_competition_count: Competitive threats
# - llm_risk_execution_count: Execution/delivery risk
# - llm_risk_demand_count: Weakening demand
# - llm_risk_supply_chain_count: Supply chain disruption
# - llm_risk_geopolitics_count: Geopolitical risk
# - llm_risk_rates_count: Interest rate sensitivity
#
# These features convert unstructured news into structured signals suitable
# for probabilistic price-move prediction models.

def aggregate_signals(signals: list[dict]) -> dict:
    if signals is None or len(signals) == 0:
        return {
            'llm_news_count': 0,
            'llm_sentiment_mean': 0.0,
            'llm_sentiment_max': 0.0,
            'llm_sentiment_min': 0.0,
            'llm_interpret_conf_mean': 0.0,
            'llm_actionability_mean': 0.0,
            'llm_is_catalyst_rate': 0.0,
            'llm_time_horizon_short_rate': 0.0,
            'llm_time_horizon_medium_rate': 0.0,
            'llm_time_horizon_long_rate': 0.0,
            'llm_cat_earnings_count': 0,
            'llm_cat_guidance_count': 0,
            'llm_cat_product_count': 0,
            'llm_cat_mna_count': 0,
            'llm_cat_regulation_count': 0,
            'llm_cat_lawsuit_count': 0,
            'llm_cat_macro_count': 0,
            'llm_cat_analyst_count': 0,
            'llm_cat_partnership_count': 0,
            'llm_cat_supply_chain_count': 0,
            'llm_cat_management_count': 0,
            'llm_risk_macro_count': 0,
            'llm_risk_regulation_count': 0,
            'llm_risk_valuation_count': 0,
            'llm_risk_competition_count': 0,
            'llm_risk_execution_count': 0,
            'llm_risk_demand_count': 0,
            'llm_risk_supply_chain_count': 0,
            'llm_risk_geopolitics_count': 0,
            'llm_risk_rates_count': 0,

        }

    s = pd.DataFrame(signals)

    out = {
        'llm_news_count': int(len(s)),
        'llm_sentiment_mean': float(s['sentiment'].mean()),
        'llm_sentiment_max': float(s['sentiment'].max()),
        'llm_sentiment_min': float(s['sentiment'].min()),
        'llm_interpret_conf_mean': float(s['interpret_confidence'].mean()),
        'llm_actionability_mean': float(s['actionability'].mean()),
        'llm_is_catalyst_rate': float(s['is_catalyst'].astype(int).mean()),
        'llm_time_horizon_short_rate': float((s['time_horizon'] == 'short').astype(int).mean()),
        'llm_time_horizon_medium_rate': float((s['time_horizon'] == 'medium').astype(int).mean()),
        'llm_time_horizon_long_rate': float((s['time_horizon'] == 'long').astype(int).mean())
    }

    catalyst_keys = [
        'earnings', 'guidance', 'product', 'mna', 'regulation', 'lawsuit', 'macro',
        'analyst', 'partnership', 'supply_chain', 'management'
    ]
    risk_keys = [
        'macro', 'regulation', 'valuation', 'competition', 'execution', 'demand',
        'supply_chain', 'geopolitics', 'rates'
    ]

    def count_list_items(col_name: str, prefix: str, allowed: list[str]):
        flat = []
        for xs in s[col_name].tolist():
            if isinstance(xs, list):
                for x in xs:
                    if isinstance(x, str) and x != 'none':
                        flat.append(x)
        ser = pd.Series(flat)
        for a in allowed:
            out[prefix + a + '_count'] = int((ser == a).sum())

    count_list_items('catalyst_types', 'llm_cat_', catalyst_keys)
    count_list_items('risk_flags', 'llm_risk_', risk_keys)

    return out
