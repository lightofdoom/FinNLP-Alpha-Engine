"""Local research dataset: 1-min IEX bars plus Finnhub news (relevant subset included)."""

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

import config
from features.price_indicators import LOOKBACK, parse_bar_row


def news_id_from_item(item):
    ticker = str((item or {}).get('ticker', '')).strip().upper()
    for field in ('id', 'url'):
        value = (item or {}).get(field)
        if value not in (None, '', 0):
            return ticker + '|' + str(value)
    return '%s|%s|%s' % (ticker, (item or {}).get('timestamp', ''), (item or {}).get('headline', ''))


def ts_to_utc_iso(value):
    dt = parse_ts(value)
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_ts(value):
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts = ts / 1000.0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        text = str(value).strip()
        try:
            if text.isdigit():
                return parse_ts(int(text))
            dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ResearchStore:
    def __init__(self, path=None):
        self.path = path or config.RESEARCH_DB
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA synchronous=NORMAL')
        self._create()

    def close(self):
        with self._lock:
            self._conn.close()

    def _create(self):
        with self._lock:
            self._conn.executescript(
                '''
                CREATE TABLE IF NOT EXISTS meta (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS bars (
                    ticker TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    vwap REAL,
                    trade_count REAL,
                    source TEXT,
                    PRIMARY KEY (ticker, ts)
                );
                CREATE INDEX IF NOT EXISTS bars_ticker_ts ON bars (ticker, ts);
                CREATE TABLE IF NOT EXISTS news (
                    news_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    ts TEXT,
                    headline TEXT,
                    summary TEXT,
                    url TEXT,
                    source TEXT,
                    relevant INTEGER NOT NULL DEFAULT 0,
                    llm_json TEXT,
                    first_seen TEXT,
                    updated_at TEXT
                );
                CREATE INDEX IF NOT EXISTS news_ticker_ts ON news (ticker, ts);
                CREATE INDEX IF NOT EXISTS news_relevant ON news (relevant);
                '''
            )
            self._conn.commit()

    def set_meta(self, key, value):
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        with self._lock:
            self._conn.execute(
                'INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)',
                (str(key), text),
            )
            self._conn.commit()

    def get_meta(self, key, default=None):
        with self._lock:
            cur = self._conn.execute('SELECT v FROM meta WHERE k=?', (str(key),))
            row = cur.fetchone()
        if row is None:
            return default
        text = row['v']
        try:
            return json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return text

    def lock_watchlist(self, tickers, locked_at=None):
        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        existing = self.get_meta('watchlist')
        if isinstance(existing, dict) and existing.get('watchlist'):
            payload = dict(existing)
            if list(existing.get('watchlist')) != tickers:
                payload['warning'] = 'requested %s but dataset is locked to %s' % (
                    tickers, existing.get('watchlist'),
                )
            return payload
        payload = {
            'watchlist': tickers,
            'locked_at': locked_at or datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        }
        self.set_meta('watchlist', payload)
        self._write_tickers_file(payload)
        return payload

    def watchlist(self):
        row = self.get_meta('watchlist') or {}
        return list(row.get('watchlist') or [])

    def upsert_bars(self, rows, source='iex'):
        if not rows:
            return 0
        now_src = source
        payload = []
        for row in rows:
            parsed = parse_bar_row(row)
            ticker = str(row.get('ticker') or row.get('symbol') or '').strip().upper()
            ts = ts_to_utc_iso(parsed.get('timestamp') or row.get('timestamp'))
            if ticker == '' or ts is None or parsed.get('close') in (None, 0):
                continue
            vwap = parsed.get('vwap')
            payload.append((
                ticker,
                ts,
                parsed.get('open'),
                parsed.get('high'),
                parsed.get('low'),
                parsed.get('close'),
                parsed.get('volume'),
                None if vwap is None else float(vwap),
                parsed.get('trade_count'),
                row.get('source') or now_src,
            ))
        if payload == []:
            return 0
        with self._lock:
            self._conn.executemany(
                '''
                INSERT OR REPLACE INTO bars
                (ticker, ts, open, high, low, close, volume, vwap, trade_count, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                payload,
            )
            self._conn.commit()
        return len(payload)

    def bars_before(self, ticker, ts, n=LOOKBACK):
        iso = ts_to_utc_iso(ts)
        if iso is None:
            return []
        with self._lock:
            cur = self._conn.execute(
                '''
                SELECT ticker, ts, open, high, low, close, volume, vwap, trade_count
                FROM bars
                WHERE ticker = ? AND ts <= ?
                ORDER BY ts DESC
                LIMIT ?
                ''',
                (str(ticker).upper(), iso, int(n)),
            )
            rows = cur.fetchall()
        out = [_bar_from_row(r) for r in rows]
        out.reverse()
        return out

    def bars_after(self, ticker, ts, limit=5000):
        iso = ts_to_utc_iso(ts)
        if iso is None:
            return []
        with self._lock:
            cur = self._conn.execute(
                '''
                SELECT ticker, ts, open, high, low, close, volume, vwap, trade_count
                FROM bars
                WHERE ticker = ? AND ts > ?
                ORDER BY ts ASC
                LIMIT ?
                ''',
                (str(ticker).upper(), iso, int(limit)),
            )
            rows = cur.fetchall()
        return [_bar_from_row(r) for r in rows]

    def all_bars(self, ticker=None):
        """All stored bars for one ticker (or every ticker), oldest first."""
        with self._lock:
            if ticker:
                cur = self._conn.execute(
                    '''
                    SELECT ticker, ts, open, high, low, close, volume, vwap, trade_count
                    FROM bars
                    WHERE ticker = ?
                    ORDER BY ts ASC
                    ''',
                    (str(ticker).upper(),),
                )
            else:
                cur = self._conn.execute(
                    '''
                    SELECT ticker, ts, open, high, low, close, volume, vwap, trade_count
                    FROM bars
                    ORDER BY ticker ASC, ts ASC
                    '''
                )
            rows = cur.fetchall()
        return [_bar_from_row(r) for r in rows]

    def bar_counts(self):
        with self._lock:
            cur = self._conn.execute(
                'SELECT ticker, COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM bars GROUP BY ticker'
            )
            return [dict(r) for r in cur.fetchall()]

    def upsert_news(self, item, relevant=None, llm_json=None, source='finnhub'):
        if item is None:
            return None
        news_id = news_id_from_item(item)
        ticker = str(item.get('ticker', '')).strip().upper()
        ts = ts_to_utc_iso(item.get('timestamp') or item.get('datetime'))
        now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        headline = str(item.get('headline', '') or '')
        summary = str(item.get('summary', '') or '')
        url = str(item.get('url', '') or '')
        src = str(item.get('source') or source or '')
        llm_text = None
        if llm_json is not None:
            llm_text = llm_json if isinstance(llm_json, str) else json.dumps(llm_json, default=str)
        rel_val = None if relevant is None else (1 if relevant else 0)

        with self._lock:
            cur = self._conn.execute('SELECT news_id, relevant, llm_json FROM news WHERE news_id=?', (news_id,))
            existing = cur.fetchone()
            if existing is None:
                self._conn.execute(
                    '''
                    INSERT INTO news
                    (news_id, ticker, ts, headline, summary, url, source, relevant, llm_json, first_seen, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        news_id, ticker, ts, headline, summary, url, src,
                        0 if rel_val is None else rel_val, llm_text, now, now,
                    ),
                )
            else:
                keep_rel = existing['relevant'] if rel_val is None else rel_val
                keep_llm = existing['llm_json'] if llm_text is None else llm_text
                self._conn.execute(
                    '''
                    UPDATE news SET
                        ticker=?, ts=COALESCE(?, ts), headline=?, summary=?, url=?,
                        source=?, relevant=?, llm_json=?, updated_at=?
                    WHERE news_id=?
                    ''',
                    (ticker, ts, headline, summary, url, src, keep_rel, keep_llm, now, news_id),
                )
            self._conn.commit()
        return news_id

    def scored_relevant_news(self):
        with self._lock:
            cur = self._conn.execute(
                '''
                SELECT news_id, ticker, ts, headline, summary, url, source, relevant, llm_json
                FROM news
                WHERE relevant = 1 AND llm_json IS NOT NULL AND llm_json != ''
                ORDER BY ts ASC, news_id ASC
                '''
            )
            rows = [dict(r) for r in cur.fetchall()]
        for row in rows:
            try:
                row['llm'] = json.loads(row['llm_json']) if row.get('llm_json') else None
            except (TypeError, ValueError, json.JSONDecodeError):
                row['llm'] = None
        return rows

    def news_stats(self):
        with self._lock:
            cur = self._conn.execute(
                '''
                SELECT
                    COUNT(*) AS n,
                    COALESCE(SUM(CASE WHEN relevant = 1 THEN 1 ELSE 0 END), 0) AS n_relevant,
                    COALESCE(SUM(CASE WHEN llm_json IS NOT NULL AND llm_json != '' THEN 1 ELSE 0 END), 0) AS n_scored
                FROM news
                '''
            )
            row = cur.fetchone()
        return dict(row) if row is not None else {'n': 0, 'n_relevant': 0, 'n_scored': 0}

    def export_relevant_jsonl(self, path=None):
        path = path or config.RESEARCH_RELEVANT_JSONL
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._lock:
            cur = self._conn.execute(
                '''
                SELECT news_id, ticker, ts, headline, summary, url, source, relevant, llm_json, first_seen
                FROM news WHERE relevant = 1 ORDER BY ts ASC
                '''
            )
            rows = [dict(r) for r in cur.fetchall()]
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + '\n')
        return path, len(rows)

    def _write_tickers_file(self, payload):
        if os.path.abspath(self.path) != os.path.abspath(config.RESEARCH_DB):
            return
        path = config.RESEARCH_WATCHLIST_JSON
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)


def _bar_from_row(row):
    return {
        'ticker': row['ticker'],
        'timestamp': row['ts'],
        'open': row['open'],
        'high': row['high'],
        'low': row['low'],
        'close': row['close'],
        'volume': row['volume'],
        'vwap': row['vwap'],
        'trade_count': row['trade_count'] or 0,
    }
