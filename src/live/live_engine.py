"""Live news-to-paper-trade loop.

Cycle: update bars -> cancel stale entries -> math exits -> news/ML with
exponential decay -> dynamic blend -> entry-only portfolio allocation.

Current data only: no historical price labelling happens here. The loop reacts to news that
arrives while it is running, prices come from Alpaca's IEX feed, and fills come from Alpaca's
paper simulator.
"""

import json
import os
import signal
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import config
from broker.alpaca_auth import assert_paper_or_allowed, require_keys
from broker.alpaca_stream import LiveMarketService
from broker.alpaca_trading import (
    ClockPoller,
    FillRecorder,
    TradeUpdateListener,
    append_order_log,
    build_exit_order_request,
    build_limit_order_request,
    build_trading_client,
    cancel_open_orders,
    flatten_all,
    get_account,
    get_open_orders,
    limit_price_from_quote,
    list_positions,
    submit_order,
)
from features.news_llm_features import NewsLLMFeatureExtractor, aggregate_signals
from features.price_indicators import compute_price_features
from features.relevance_filter import is_relevant as rule_relevant
from ingest.finnhub_news_fetcher import fetch_news_items
from live.combine_signals import ENTRY_T, combine_entry, ml_score_from_aggregate
from live.duration import format_remaining, resolve_end_time
from live.exit_engine import evaluate_exit, update_favorable
from live.ml_decay import DecayState
from live.paper_trader import check_kill_switch, load_kill_state, save_kill_state
from live.position_sizer import size_position
from live.session_report import SessionStats, write_summary
from live.signal_engine import DEFAULT_STRATEGY, order_side
from portfolio.optimizer import plan_entry_sizes
from replay.store import ResearchStore, news_id_from_item


EASTERN = ZoneInfo('America/New_York')

DEFAULT_POLL_SEC = 60.0
DEFAULT_MODEL = 'mistral:7b'
DEFAULT_DAYS_BACK = 1
DEFAULT_MAX_ITEMS_PER_POLL = 40


class LiveEngine:
    def __init__(
        self,
        watchlist=None,
        until=None,
        duration=None,
        poll_sec=DEFAULT_POLL_SEC,
        strategy=DEFAULT_STRATEGY,
        min_signal=None,
        notional=None,
        dry_run=False,
        model=DEFAULT_MODEL,
        days_back=DEFAULT_DAYS_BACK,
        max_items_per_poll=DEFAULT_MAX_ITEMS_PER_POLL,
        allow_short=True,
        flatten_on_exit=False,
        seen_path=None,
    ):
        self.watchlist = [str(t).strip().upper() for t in (watchlist or config.WATCHLIST) if str(t).strip()]
        if self.watchlist == []:
            raise ValueError('Watchlist is empty')

        self.poll_sec = float(poll_sec)
        self.strategy = strategy
        self.min_signal = float(min_signal if min_signal is not None else getattr(config, 'ENTRY_T', ENTRY_T))
        self.notional = float(notional) if notional is not None else float(config.ALPACA_NOTIONAL)
        self.dry_run = bool(dry_run)
        self.model = model
        self.days_back = int(days_back)
        self.max_items_per_poll = int(max_items_per_poll)
        self.allow_short = bool(allow_short)
        self.flatten_on_exit = bool(flatten_on_exit)
        self.seen_path = seen_path or os.path.join(config.PAPER_RESULTS_DIR, 'seen_news.json')
        self.hold_extended = bool(getattr(config, 'HOLD_EXTENDED', False))
        self.news_required = bool(getattr(config, 'NEWS_REQUIRED', True))
        self.stale_entry_sec = float(getattr(config, 'STALE_ENTRY_SEC', 60))

        self.end_time = resolve_end_time(until=until, duration=duration)

        self._stop = threading.Event()
        self._seen = set()
        self._client = None
        self._market = None
        self._clock = None
        self._listener = None
        self._extractor = None
        self._kill_state = None
        self._open_meta = {}
        self._store = ResearchStore()
        self._closed_exit_noted = set()
        self._decay = DecayState(
            path=getattr(config, 'ML_DECAY_STATE_JSON', None),
            lambda_=getattr(config, 'ML_DECAY_LAMBDA', 0.75),
            floor=getattr(config, 'ML_SCORE_FLOOR', 0.02),
            max_age_min=getattr(config, 'NEWS_MAX_AGE_MIN', 15),
        )
        lock = self._store.lock_watchlist(self.watchlist)
        if lock.get('warning'):
            print(lock['warning'])
            locked = list(lock.get('watchlist') or [])
            if locked:
                self.watchlist = locked

        self.stats = SessionStats(
            watchlist=self.watchlist,
            strategy=self.strategy,
            min_signal=self.min_signal,
            notional=self.notional,
            dry_run=self.dry_run,
            end_time=self.end_time,
        )

    # ---------- lifecycle ----------

    def request_stop(self, reason='interrupted'):
        if not self._stop.is_set():
            self.stats.end_reason = reason
            self._stop.set()

    def _install_signal_handlers(self):
        def handler(signum, frame):
            print('\nStop requested, finishing current cycle...')
            self.request_stop('interrupted')

        try:
            signal.signal(signal.SIGINT, handler)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, handler)
        except ValueError:
            pass

    def run(self):
        config.ensure_dirs()
        creds = assert_paper_or_allowed(require_keys())
        if config.FINNHUB_API_KEY == '':
            raise RuntimeError('FINNHUB_API_KEY is empty; the live engine needs news')

        self._install_signal_handlers()
        self._load_seen()

        self._client = build_trading_client(creds)
        self._extractor = NewsLLMFeatureExtractor(
            provider='ollama',
            model_name=self.model,
            cache_path=os.path.join(config.RESULTS_DIR, 'llm_cache.sqlite'),
            temperature=0.0,
        )

        account = get_account(self._client)
        self.stats.start_equity = _num(getattr(account, 'equity', None))
        self.stats.start_cash = _num(getattr(account, 'cash', None))
        self._kill_state = load_kill_state(starting_equity=self.stats.start_equity)
        save_kill_state(self._kill_state)

        self._market = LiveMarketService(self.watchlist)
        self._clock = ClockPoller(client=self._client)

        self._print_header()

        try:
            self._market.start(hydrate=True)
            self._clock.start()
            if not self.dry_run:
                self._listener = TradeUpdateListener(
                    recorder=FillRecorder(),
                    on_update=self._on_trade_update,
                )
                self._listener.start()

            self._loop()
        except KeyboardInterrupt:
            self.request_stop('interrupted')
        finally:
            report = self._shutdown()

        return report

    def _loop(self):
        while not self._stop.is_set():
            if self._past_end_time():
                self.request_stop('end_time_reached')
                break

            self.stats.ticks += 1
            try:
                self._tick()
            except Exception as exc:
                self.stats.record_error('tick', exc)
                print('  cycle error:', exc)

            if self._stop.is_set() or self._past_end_time():
                if not self._stop.is_set():
                    self.request_stop('end_time_reached')
                break

            self._stop.wait(self._sleep_seconds())

    def _sleep_seconds(self):
        if self.end_time is None:
            return self.poll_sec
        remaining = (self.end_time - datetime.now(EASTERN)).total_seconds()
        if remaining <= 0:
            return 0.0
        return min(self.poll_sec, remaining)

    def _past_end_time(self):
        if self.end_time is None:
            return False
        return datetime.now(EASTERN) >= self.end_time

    def _shutdown(self):
        if self.flatten_on_exit and not self.dry_run:
            try:
                flatten_all(self._client)
                print('Flattened all positions.')
            except Exception as exc:
                self.stats.record_error('flatten_on_exit', exc)

        for stoppable in (self._listener, self._clock, self._market):
            if stoppable is not None:
                try:
                    stoppable.stop()
                except Exception:
                    pass

        try:
            account = get_account(self._client)
            self.stats.end_equity = _num(getattr(account, 'equity', None))
            self.stats.end_cash = _num(getattr(account, 'cash', None))
        except Exception as exc:
            self.stats.record_error('final_account', exc)

        try:
            self.stats.positions_at_end = list_positions(self._client)
        except Exception as exc:
            self.stats.record_error('final_positions', exc)

        self._save_seen()
        try:
            self._decay.save()
        except Exception as exc:
            self.stats.record_error('save_decay', exc)
        try:
            path, n = self._store.export_relevant_jsonl()
            print('Archived relevant news:', n, '->', path)
        except Exception as exc:
            self.stats.record_error('export_relevant', exc)
        self.stats.ended_at = datetime.now(EASTERN)
        if self.stats.end_reason == 'unknown':
            self.stats.end_reason = 'completed'

        report = write_summary(self.stats)
        print()
        print(report['text'])
        print('Wrote:', report['txt_path'])
        print('Wrote:', report['json_path'])
        return report

    # ---------- one polling cycle ----------

    def _tick(self):
        now = datetime.now(EASTERN)
        session = self._clock.session()
        self._refresh_bars_if_needed()
        self._archive_bars()

        remaining = ''
        if self.end_time is not None:
            remaining = ' | ends in ' + format_remaining(self.end_time - now)

        positions = {}
        orders = []
        account = None
        try:
            if not self.dry_run:
                for pos in list_positions(self._client):
                    positions[str(pos.get('symbol', '')).upper()] = pos
                orders = list(get_open_orders(client=self._client) or [])
                account = get_account(self._client)
        except Exception as exc:
            self.stats.record_error('broker_snapshot', exc)

        orders_by_symbol = {}
        for order in orders:
            symbol = str(getattr(order, 'symbol', '')).upper()
            orders_by_symbol.setdefault(symbol, []).append(order)

        for ticker in self.watchlist:
            self._manage_open_symbol(
                ticker,
                session,
                now,
                positions.get(ticker),
                orders_by_symbol.get(ticker, []),
            )

        items = fetch_news_items(self.watchlist, self.days_back, config.FINNHUB_API_KEY)
        self.stats.news_seen += len(items)

        fresh = []
        for item in items:
            key = _news_key(item)
            if key in self._seen:
                continue
            self._seen.add(key)
            fresh.append(item)

        self._archive_news(fresh)

        fresh.sort(key=lambda it: _item_ts(it), reverse=True)
        to_score = fresh
        if len(to_score) > self.max_items_per_poll:
            to_score = to_score[:self.max_items_per_poll]
        self.stats.news_new += len(fresh)

        print('[%s] poll %s | session %s | new %s%s'
              % (now.strftime('%H:%M:%S'), self.stats.ticks, session, len(fresh), remaining))

        by_ticker = {}
        for item in to_score:
            ticker = str(item.get('ticker', '')).strip().upper()
            if ticker == '':
                continue
            by_ticker.setdefault(ticker, []).append(item)

        equity = _num(getattr(account, 'equity', None)) if account is not None else None
        buying_power = _num(getattr(account, 'buying_power', None)) if account is not None else None

        candidates = []
        for ticker in self.watchlist:
            candidate = self._evaluate_ticker(
                ticker,
                by_ticker.get(ticker, []),
                session,
                now,
                has_position=ticker in positions,
                open_order_count=len(orders_by_symbol.get(ticker, [])),
            )
            if candidate is not None:
                candidates.append(candidate)

        try:
            self._decay.save()
        except Exception as exc:
            self.stats.record_error('save_decay', exc)

        if candidates == []:
            return

        self._submit_candidates(
            candidates,
            session,
            positions,
            equity=equity,
            buying_power=buying_power,
        )

    def _refresh_bars_if_needed(self):
        if self._market is None:
            return
        thin = False
        for ticker in self.watchlist:
            if len(self._market.bars_for(ticker)) < 20:
                thin = True
                break
        if thin or self._market.cache.is_stream_down():
            try:
                self._market.hydrate_minute_history()
            except Exception as exc:
                self.stats.record_error('hydrate_bars', exc)

    def _archive_bars(self):
        if self._market is None:
            return
        rows = []
        for ticker in self.watchlist:
            for bar in self._market.bars_for(ticker):
                row = dict(bar)
                row['ticker'] = ticker
                rows.append(row)
        if rows == []:
            return
        try:
            self._store.upsert_bars(rows, source='live')
        except Exception as exc:
            self.stats.record_error('archive_bars', exc)

    def _archive_news(self, items):
        if not items:
            return
        try:
            for item in items:
                headline = str(item.get('headline', '') or '')
                summary = str(item.get('summary', '') or '')
                self._store.upsert_news(
                    item,
                    relevant=rule_relevant(headline, summary),
                    source='live',
                )
        except Exception as exc:
            self.stats.record_error('archive_news', exc)

    def _manage_open_symbol(self, ticker, session, now, position, open_orders):
        bars = self._market.bars_for(ticker) if self._market is not None else []
        features = compute_price_features(bars) if bars else {'math_unavailable': True, 'atr': None}

        self._cancel_stale_entries(ticker, open_orders, now, has_position=position is not None)

        if position is None:
            if not open_orders:
                self._open_meta.pop(ticker, None)
            return

        meta = self._ensure_meta(ticker, position, features, now)
        last_bar = bars[-1] if bars else None
        if last_bar is not None:
            update_favorable(meta, last_bar)

        if meta.get('atr_entry') in (None, 0):
            return

        decision = evaluate_exit(
            meta,
            bars,
            features,
            now=now,
            hold_extended=self.hold_extended,
        )
        if decision is None:
            self._closed_exit_noted.discard(ticker)
            return

        if session == 'closed':
            if ticker not in self._closed_exit_noted:
                print('  %s: %s queued until the market opens' % (ticker, decision['reason']))
                self._closed_exit_noted.add(ticker)
            return

        self._closed_exit_noted.discard(ticker)
        print('  %s: exit %s%s'
              % (ticker, decision['reason'], ' (urgent)' if decision['urgent'] else ''))
        if self.dry_run:
            self.stats.record_skip('dry_run_exit')
            self.stats.record_exit({
                'symbol': ticker,
                'reason': decision['reason'],
                'urgent': decision['urgent'],
                'dry_run': True,
                'ts': now.isoformat(),
            })
            return

        self._submit_exit(ticker, decision, session, meta)

    def _ensure_meta(self, ticker, position, features, now):
        side = str(position.get('side', '')).lower()
        if side not in ('long', 'short'):
            qty = _num(position.get('qty'))
            side = 'short' if qty is not None and qty < 0 else 'long'
        qty = abs(_num(position.get('qty')) or 0.0)
        entry = _num(position.get('avg_entry_price'))
        meta = self._open_meta.get(ticker)
        if meta is None:
            atr = features.get('atr') if features else None
            meta = {
                'side': side,
                'entry_price': entry,
                'atr_entry': atr,
                'entry_time': now,
                'qty': qty,
                'max_favorable': None,
            }
            self._open_meta[ticker] = meta
        else:
            meta['qty'] = qty
            if meta.get('entry_price') is None:
                meta['entry_price'] = entry
            if meta.get('side') is None:
                meta['side'] = side
        return meta

    def _cancel_stale_entries(self, ticker, open_orders, now, has_position):
        if self.dry_run or not open_orders:
            return
        stale_ids = []
        cutoff = now.astimezone(timezone.utc) - timedelta(seconds=self.stale_entry_sec)
        for order in open_orders:
            submitted = getattr(order, 'submitted_at', None)
            if has_position:
                stale_ids.append(str(order.id))
                continue
            if submitted is None:
                continue
            if submitted.tzinfo is None:
                submitted = submitted.replace(tzinfo=timezone.utc)
            if submitted.astimezone(timezone.utc) <= cutoff:
                stale_ids.append(str(order.id))
        if stale_ids == []:
            return
        try:
            cancel_open_orders(symbol=ticker, client=self._client)
            self.stats.record_skip('stale_entry_cancelled')
            print('  %s: cancelled stale/conflicting entries' % ticker)
        except Exception as exc:
            self.stats.record_error('cancel_stale', exc)

    def _submit_exit(self, ticker, decision, session, meta):
        try:
            cancel_open_orders(symbol=ticker, client=self._client)
        except Exception as exc:
            self.stats.record_error('cancel_before_exit', exc)

        try:
            quote = self._market.quote_for_order(ticker)
            request = build_exit_order_request(
                ticker,
                decision['side'],
                meta.get('qty') or decision.get('qty'),
                session,
                quote,
                urgent=decision['urgent'],
            )
            order = submit_order(request, client=self._client)
        except Exception as exc:
            self.stats.record_error('submit_exit', exc)
            print('    exit submit failed:', exc)
            return

        row = {
            'order_id': str(getattr(order, 'id', '')),
            'symbol': ticker,
            'side': decision['side'],
            'qty': meta.get('qty'),
            'limit_price': getattr(request, 'limit_price', None),
            'order_type': str(getattr(request, 'type', '')),
            'session': session,
            'extended_hours': bool(getattr(request, 'extended_hours', False)),
            'intent': 'exit',
            'exit_reason': decision['reason'],
            'exit_urgency': 'urgent' if decision['urgent'] else 'limit',
            'ts': datetime.now(EASTERN).isoformat(),
            'status': str(getattr(order, 'status', '')),
        }
        append_order_log(row)
        self.stats.record_order(row)
        self.stats.record_exit(row)
        print('    submitted exit %s %s x%s [%s] %s'
              % (decision['side'], ticker, meta.get('qty'), row['status'], decision['reason']))

    def _evaluate_ticker(self, ticker, items, session, now, has_position=False, open_order_count=0):
        relevant = []
        relevant_src = []
        for item in items:
            headline = str(item.get('headline', '') or '')
            summary = str(item.get('summary', '') or '')
            if rule_relevant(headline, summary):
                relevant.append({'headline': headline, 'summary': summary, 'ticker': ticker})
                relevant_src.append(item)

        self.stats.news_relevant += len(relevant)

        bars = self._market.bars_for(ticker) if self._market is not None else []
        math_features = compute_price_features(bars)

        fresh = False
        ml_features = {
            'ml_score': 0.0,
            'news_count': 0,
            'fresh_news': False,
            'ml_damped': False,
        }
        if relevant:
            signals = self._extractor.extract_many(relevant)
            self.stats.news_scored += len(signals)
            try:
                for item, sig in zip(relevant_src, signals):
                    self._store.upsert_news(item, relevant=True, llm_json=sig, source='live')
            except Exception as exc:
                self.stats.record_error('archive_llm', exc)
            agg = aggregate_signals(signals)
            ml_features = ml_score_from_aggregate(agg, strategy=self.strategy)
            row = self._decay.update(ticker, now, new_score=ml_features.get('ml_score'), fresh=True)
            fresh = True
            ml_features['ml_score'] = row['ml_score']
            ml_features['fresh_news'] = True
            ml_features['news_count'] = int(ml_features.get('news_count') or len(relevant))
        else:
            row = self._decay.update(ticker, now, fresh=False)
            ml_features['ml_score'] = row['ml_score']
            ml_features['fresh_news'] = False
            ml_features['news_count'] = 0

        idle = (not fresh) and abs(float(ml_features.get('ml_score') or 0.0)) == 0.0
        if idle:
            if items:
                print('  %s: %s new, none relevant' % (ticker, len(items)))
            return None

        decision = combine_entry(
            math_features,
            ml_features,
            entry_t=self.min_signal,
            news_required=self.news_required,
            allow_short=self.allow_short,
            fresh_news=fresh,
        )
        self.stats.record_decision(ticker, {
            'direction': decision['direction'],
            'strength': decision['strength'],
            'actionable': decision['actionable'],
            'reason': decision['reason'],
            'entry_score': decision.get('entry_score'),
            'math_score': decision.get('math_score'),
            'ml_score': decision.get('ml_score'),
            'news_count': decision.get('news_count'),
            'fresh_news': decision.get('fresh_news'),
            'w_math': decision.get('w_math'),
            'w_ml': decision.get('w_ml'),
            'ml_damped': ml_features.get('ml_damped'),
            'unavailable_reason': math_features.get('unavailable_reason'),
        })

        print('  %s: %s relevant%s | math %.3f ml %.3f w %.2f/%.2f -> %s %.3f (%s)'
              % (ticker, len(relevant), '' if fresh else ' (decay)',
                 decision['math_score'], decision['ml_score'],
                 decision.get('w_math') or 0.0, decision.get('w_ml') or 0.0,
                 decision['direction'], decision['strength'], decision['reason']))

        if has_position:
            self.stats.record_skip('existing_position')
            return None
        if open_order_count > 0:
            self.stats.record_skip('existing_open_order')
            return None

        if not decision['actionable']:
            self.stats.record_skip('signal_' + decision['reason'])
            if not self.dry_run:
                self._cancel_if_gates_failed(ticker)
            return None

        if self.dry_run:
            self.stats.record_skip('dry_run')
            print('    dry run: would %s %s' % (order_side(decision['direction']), ticker))
            return None

        return {
            'ticker': ticker,
            'decision': decision,
            'math_features': math_features,
            'ml_features': ml_features,
            'bars': bars,
            'entry_score': decision.get('entry_score'),
            'math_score': decision.get('math_score'),
            'atr': math_features.get('atr'),
            'price': math_features.get('price'),
            'sigma_30_ann': math_features.get('sigma_30_ann'),
            'direction': decision.get('direction'),
        }

    def _submit_candidates(self, candidates, session, positions, equity=None, buying_power=None):
        if session == 'closed':
            for row in candidates:
                self.stats.record_skip('market_closed')
                print('    skipped %s: market closed' % row['ticker'])
            return

        existing = []
        for ticker, pos in (positions or {}).items():
            mv = _num(pos.get('market_value'))
            qty = _num(pos.get('qty'))
            px = _num(pos.get('current_price')) or _num(pos.get('avg_entry_price'))
            notional = abs(mv) if mv is not None else abs((qty or 0.0) * (px or 0.0))
            side = str(pos.get('side') or 'long').lower()
            if 'short' in side:
                side = 'short'
            else:
                side = 'long'
            existing.append({'ticker': ticker, 'notional': notional, 'side': side})

        bars_by_ticker = {}
        for ticker in self.watchlist:
            if self._market is not None:
                bars_by_ticker[ticker] = self._market.bars_for(ticker)
        for row in candidates:
            bars_by_ticker[row['ticker']] = row.get('bars') or bars_by_ticker.get(row['ticker']) or []

        sized_by = plan_entry_sizes(
            candidates,
            existing=existing,
            bars_by_ticker=bars_by_ticker,
            equity=equity,
            buying_power=buying_power,
            notional_cap=self.notional,
        )

        for row in candidates:
            ticker = row['ticker']
            sized = sized_by.get(ticker) or {'ok': False, 'reason': 'not_sized'}
            if not sized.get('ok'):
                self.stats.record_skip('size_' + str(sized.get('reason') or 'failed'))
                print('    skipped size %s: %s (source %s sigma_30_ann %s)'
                      % (ticker, sized.get('reason'), sized.get('source'),
                         row['math_features'].get('sigma_30_ann')))
                continue
            self._try_order(
                ticker,
                row['decision'],
                session,
                row['math_features'],
                equity=equity,
                buying_power=buying_power,
                sized=sized,
            )

    def _cancel_if_gates_failed(self, ticker):
        try:
            cancel_open_orders(symbol=ticker, client=self._client)
        except Exception as exc:
            self.stats.record_error('cancel_failed_gates', exc)

    def _try_order(self, ticker, decision, session, math_features, equity=None, buying_power=None, open_order_count=0, sized=None):
        if session == 'closed':
            self.stats.record_skip('market_closed')
            print('    skipped: market closed')
            return
        if open_order_count > 0:
            self.stats.record_skip('existing_open_order')
            print('    skipped: open order already working')
            return

        try:
            if equity is None:
                account = get_account(self._client)
                equity = _num(getattr(account, 'equity', None))
                buying_power = _num(getattr(account, 'buying_power', None))
            check_kill_switch(self._kill_state, equity)
        except RuntimeError as exc:
            self.stats.record_skip('kill_switch')
            print('    blocked:', exc)
            self.request_stop('kill_switch')
            return
        except Exception as exc:
            self.stats.record_error('account_check', exc)
            return

        side = order_side(decision['direction'])
        try:
            quote = self._market.quote_for_order(ticker)
            price = math_features.get('price') or quote.get('last') or quote.get('ask') or quote.get('bid')
            if sized is None:
                sized = size_position(
                    equity,
                    price,
                    math_features.get('atr'),
                    decision.get('math_score'),
                    buying_power=buying_power,
                    sigma_30_ann=math_features.get('sigma_30_ann'),
                    notional_cap=self.notional,
                )
            if not sized['ok']:
                self.stats.record_skip('size_' + sized['reason'])
                print('    skipped size: %s (sigma_30_ann %s)'
                      % (sized['reason'], math_features.get('sigma_30_ann')))
                return
            if session == 'closed':
                raise RuntimeError('Market is closed; v1 logs only and does not queue')
            limit_price = limit_price_from_quote(side, quote)
            request = build_limit_order_request(ticker, side, sized['qty'], limit_price, session)
        except RuntimeError as exc:
            self.stats.record_skip(_skip_reason(exc))
            print('    skipped:', exc)
            return
        except ValueError as exc:
            self.stats.record_skip('pricing_' + str(exc)[:40])
            print('    skipped:', exc)
            return
        except Exception as exc:
            self.stats.record_error('order_plan', exc)
            return

        try:
            order = submit_order(request, client=self._client)
        except Exception as exc:
            self.stats.record_error('submit_order', exc)
            print('    submit failed:', exc)
            return

        self._kill_state['orders_submitted'] = int(self._kill_state.get('orders_submitted') or 0) + 1
        save_kill_state(self._kill_state)

        atr = math_features.get('atr')
        self._open_meta[ticker] = {
            'side': decision['direction'],
            'entry_price': float(limit_price),
            'atr_entry': atr,
            'entry_time': datetime.now(EASTERN),
            'qty': sized['qty'],
            'max_favorable': None,
        }

        row = {
            'order_id': str(getattr(order, 'id', '')),
            'symbol': ticker,
            'side': side,
            'qty': sized['qty'],
            'limit_price': limit_price,
            'session': session,
            'extended_hours': bool(request.extended_hours),
            'intent': 'entry',
            'strength': decision['strength'],
            'direction': decision['direction'],
            'entry_score': decision.get('entry_score'),
            'math_score': decision.get('math_score'),
            'ml_score': decision.get('ml_score'),
            'intended_notional': sized.get('intended_notional'),
            'capped_notional': sized.get('capped_notional'),
            'cap_bound': sized.get('cap_bound'),
            'size_source': sized.get('source'),
            'sigma_30_ann': math_features.get('sigma_30_ann'),
            'strategy': self.strategy,
            'ts': datetime.now(EASTERN).isoformat(),
            'status': str(getattr(order, 'status', '')),
        }
        append_order_log(row)
        self.stats.record_order(row)
        print('    submitted %s %s x%s @ %s [%s] intended $%s capped $%s sigma_30_ann %s'
              % (side, ticker, sized['qty'], limit_price, row['status'],
                 sized.get('intended_notional'), sized.get('capped_notional'),
                 math_features.get('sigma_30_ann')))

    def _on_trade_update(self, update):
        event = str(getattr(update, 'event', '')).lower()
        if event not in ('fill', 'partial_fill', 'partialfill'):
            return
        order = getattr(update, 'order', None)
        row = {
            'event': event,
            'symbol': str(getattr(order, 'symbol', '')) if order is not None else '',
            'side': str(getattr(order, 'side', '')) if order is not None else '',
            'qty': _num(getattr(update, 'qty', None)),
            'price': _num(getattr(update, 'price', None)),
            'ts': datetime.now(EASTERN).isoformat(),
        }
        self.stats.record_fill(row)
        print('    fill: %s %s x%s @ %s' % (row['event'], row['symbol'], row['qty'], row['price']))

        ticker = row['symbol'].upper()
        meta = self._open_meta.get(ticker)
        fill_px = row['price']
        if meta is not None and fill_px is not None and event == 'fill':
            meta['entry_price'] = fill_px

    # ---------- seen-news persistence ----------

    def _load_seen(self):
        if not os.path.exists(self.seen_path):
            return
        try:
            with open(self.seen_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                self._seen = set(str(x) for x in data)
        except Exception:
            self._seen = set()

    def _save_seen(self):
        try:
            os.makedirs(os.path.dirname(self.seen_path), exist_ok=True)
            recent = list(self._seen)[-5000:]
            with open(self.seen_path, 'w', encoding='utf-8') as f:
                json.dump(recent, f)
        except Exception as exc:
            self.stats.record_error('save_seen', exc)

    def _print_header(self):
        print('=' * 62)
        print('ALPACA PAPER SIMULATION')
        print('=' * 62)
        print('Watchlist:   ' + ', '.join(self.watchlist))
        print('Model:       ' + self.model)
        print('Blend:       fresh %s/%s  quiet %s/%s  decay λ=%s floor=%s (ENTRY_T %s, strategy %s)'
              % (getattr(config, 'W_MATH_FRESH', 0.30), getattr(config, 'W_ML_FRESH', 0.70),
                 getattr(config, 'W_MATH_QUIET', 0.85), getattr(config, 'W_ML_QUIET', 0.15),
                 getattr(config, 'ML_DECAY_LAMBDA', 0.75), getattr(config, 'ML_SCORE_FLOOR', 0.02),
                 self.min_signal, self.strategy))
        print('Notional cap $%s (%s) optimizer=%s'
              % (self.notional, getattr(config, 'SIZING_MODE', 'small_paper'),
                 getattr(config, 'OPTIMIZER_ENABLED', True)))
        print('Poll:        every %ss' % int(self.poll_sec))
        print('Mode:        ' + ('DRY RUN (no orders)' if self.dry_run else 'PAPER ORDERS'))
        if self.end_time is None:
            print('Ends:        on Ctrl-C')
        else:
            print('Ends:        %s (%s)'
                  % (self.end_time.strftime('%Y-%m-%d %I:%M %p %Z'),
                     format_remaining(self.end_time - datetime.now(EASTERN))))
        print('Equity:      $%s' % self.stats.start_equity)
        print('=' * 62)


def _skip_reason(exc):
    text = str(exc).lower()
    if 'closed' in text:
        return 'market_closed'
    if 'open order' in text:
        return 'existing_open_order'
    if 'position' in text:
        return 'existing_position'
    if 'below one share' in text:
        return 'notional_too_small'
    return 'order_rejected_locally'


def _news_key(item):
    return news_id_from_item(item)


def _item_ts(item):
    value = item.get('timestamp') or item.get('datetime') or 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _num(value):
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
