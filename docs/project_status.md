# Project status

Snapshot for future agents and collaborators. Last updated **2026-09-11** (Pacific).

This is a news-driven **Alpaca paper** trader. It is **not** live-money ready. Do not set `ALPACA_ALLOW_LIVE=true`. The project is **not a git repository**.

Durable decision rules live in [`decision_framework.md`](decision_framework.md). Broker/feed constraints live in [`alpaca_integration_spec.md`](alpaca_integration_spec.md). If those two disagree with code, **code wins** for exits (`EXIT_PRIORITY` in `src/live/exit_engine.py`) and for the live cycle in `src/live/live_engine.py`.

## What it is right now

A working paper loop that:

1. Polls Finnhub for **headline + summary only** (no full article, no URL crawl).
2. Keyword-filters relevance (`src/features/relevance_filter.py`). Live does **not** use an Ollama relevance classifier.
3. Scores remaining items with local Ollama (`mistral:7b` by default) into JSON features.
4. **Decays** the signed ML score across quiet minutes: replace on fresh relevant news, else `ml_t = λ^{dt} * ml_{t-1}` (default `λ=0.75` per wall-clock minute). `|ml| < 0.02` or age `> NEWS_MAX_AGE_MIN` (15) floors to 0. State is `results/paper/ml_decay_state.json`.
5. Blends with 1-minute IEX price math using **dynamic weights**:
   * Fresh relevant news this cycle: **`0.30 * math_score + 0.70 * ml_score`**
   * Quiet / decaying: **`0.85 * math_score + 0.15 * ml_score_decayed`**
6. Enters if a non-zero (fresh or decayed) ML score remains, math and ML signs agree, and `|entry_score| > ENTRY_T` (**0.20** is the only entry threshold). Pure math with `ml=0` is still blocked.
7. Sizes **entry-only** with a shrunk mean-variance allocator (`src/portfolio/optimizer.py`, scipy SLSQP). Per-name cap is still `ALPACA_NOTIONAL` (default $1000). Existing positions are **not** resized; they only affect covariance / gross leverage. Isolated ATR/R sizing is the solver-failure fallback. Vol gates (`too_quiet` / high-vol half) still apply.
8. Exits with **100% math rules**, no LLM. Urgent exits are market in regular hours, IOC aggressive limit in extended hours.

Each live poll also **appends** bars and news into `data/research_set/research.sqlite` so later replay can use the same stack. Replay walks **every 1-minute bar** (not only news minutes) so decay and quiet-cycle entries match live.

The engine is implemented and unit-tested. **Constants are uncalibrated.** Do not retune `ENTRY_T`, weights, decay, or stops until there are enough live + replay trades to look at.

## How to run

Work from `src/` with the project venv (`.venv`). Secrets belong in a gitignored `.env` (Alpaca paper keys + Finnhub). `load_dotenv` does **not** override a `FINNHUB_API_KEY` that is already set in the environment.

```
cd src
python main.py                              # live paper loop (Ctrl-C to stop)
python main.py --dry-run --duration 45s     # score only, no orders
python main.py --fetch-bars --bars-days 45  # backfill 1-min IEX bars into sqlite
python main.py --replay                     # dynamic blend vs ML-only on sqlite
```

Tests:

```
python -m unittest discover -s tests -v
```

Alpaca Basic allows **one IEX websocket**. Two `main.py` processes will fail with `connection limit exceeded`. Data stream + trading stream in one process is expected; a second live process is not.

## Locked watchlist

Do not rotate these names while the collection/replay set is in use:

`NVDA, AAPL, MSFT, AMZN, GOOG, META, TSLA, AMD`

First lock wins in sqlite meta + `data/research_set/tickers.json`. Changing `WATCHLIST` in `config.py` after lock will not silently rewrite the dataset.

Free Alpaca cap is 30 symbols; this set is 8.

## Research sqlite

Path: `data/research_set/research.sqlite` (WAL mode).

| Table | Role |
| --- | --- |
| `bars` | 1-minute IEX OHLCV + vwap + trade_count |
| `news` | headline, summary, url, keyword `relevant`, optional `llm_json` |
| `meta` | watchlist lock |

Historical IEX 1-min bars were already pulled: **103,411** bars across the 8 names, **2026-07-15** through **2026-08-28**. Live appends new bars and headlines while the loop runs.

As of the 2026-08-28 snapshot the sqlite also held **372** news rows (**256** keyword-relevant, **4** with `llm_json`). Replay needs scored relevant news; a replay with almost no scored rows is expected until the live loop scores more items. Pair weekend/after-hours headlines with the **next session’s bars**, not “must have a bar the same minute.”

Outputs go under `results/replay/` (txt, json, trades csv, equity png).

## Live cycle (each poll)

1. Update 1-min bar cache (90-bar lookback on the stream).
2. Cancel stale unfilled **entry** orders.
3. Evaluate exits on open positions (runs even with no news).
4. Fetch news → keyword relevance → LLM JSON (capped by `--max-items`, default 40).
5. For **every** watchlist name: replace or decay `ml_score`, compute price math, dynamic blend. Quiet names with `ml=0` are skipped silently.
6. Collect actionable candidates (no open position / working entry) and run the entry-only optimizer once. Submit sized limit entries.

When the session is `closed`, the loop still archives news, updates decay, and logs queued exits **once**, but it does **not** submit orders (Alpaca rejects them). Weekend overnight running is useful for the news archive, not for trading. Prefer **04:00–20:00 ET weekdays** for actual paper orders; do not drop after-hours news from the archive.

## What is not done / do not do yet

- **No calibration.** Log and revisit fresh/quiet weights, `ML_DECAY_LAMBDA`, `ENTRY_T`, vol 80%/8% gates, risk dampener 0.8, stop 1.5 / target 2.0 ATR, trailing, time stop, optimizer `λ` / turnover / leverage. Slippage is unmodeled except as a turnover penalty in the allocator.
- **No full-book MVO rebalance.** The optimizer does not trim, add to, or flatten existing positions. Exits stay 100% math.
- **No `cvxpy`.** 8 names use `scipy.optimize` SLSQP + diagonal shrinkage. Isolated ATR/R is the fallback.
- **No article scraping** and no summarize-then-extract path. Headline + summary is the product decision.
- **No SIP / live money.** IEX only (~2.5% of US volume). Volume and silence checks are relative to each symbol’s own IEX baseline.
- **Do not enable** `ALPACA_ALLOW_LIVE`.
- After enough data: replay calibration, audit keyword relevance vs false drops, exit-reason mix, Finnhub lag vs IEX, then sizing; walk-forward; stay paper/IEX.

`src/live/signal_engine.py` still has an older ML-only `decide_from_aggregate` used by tests and for `--strategy` naming. **Live entries go through `combine_signals.combine_entry`**, not that helper.

## Ops pitfalls

- One IEX websocket per account. Kill duplicate `main.py` before restarting.
- Restarting the loop does **not** rescore already-seen headlines (`seen_news.json` + sqlite `news_id`). Positions, sqlite, LLM cache, decay JSON, and Alpaca paper state persist. Decay is re-applied from `updated_at` on the next poll.
- Leftover paper positions from earlier sessions can still trip exit rules every poll; closed-session submits are skipped.
- Kill switch: daily max orders (`ALPACA_MAX_ORDERS_PER_DAY`, default 10 in code; `.env` may raise this) and max loss vs starting equity. Quiet-cycle entries plus a multi-name optimizer burst can consume the daily cap faster than the old news-only path.
- Headlines above `--max-items` are marked seen and never scored.

## Current code map

| Area | Path | Role |
| --- | --- | --- |
| Entry point | `src/main.py` | live / `--fetch-bars` / `--replay` |
| Config | `src/config.py` | env, watchlist, paper defaults, decay / optimizer knobs, sqlite paths |
| Live loop | `src/live/live_engine.py` | poll cycle, archive, orders |
| Decay | `src/live/ml_decay.py` | replace-or-decay + floor + max age + JSON persist |
| Blend | `src/live/combine_signals.py` | fresh 30/70 vs quiet 85/15 + disagree veto |
| Size | `src/live/position_sizer.py` | ATR/R then notional cap (optimizer fallback / gates) |
| Allocator | `src/portfolio/optimizer.py` | entry-only shrunk MVO (scipy); does not touch open positions |
| Exits | `src/live/exit_engine.py` | `EXIT_PRIORITY` source of truth |
| Price math | `src/features/price_indicators.py` | slopes, VWAP, ATR, RSI, IEX-thin |
| News LLM | `src/features/news_llm_features.py` | Ollama JSON extract + aggregate |
| Relevance | `src/features/relevance_filter.py` | keyword gate used live |
| Session | `src/features/session_labeler.py` | RTH / extended / closed |
| News ingest | `src/ingest/finnhub_news_fetcher.py` | Finnhub poll |
| Broker | `src/broker/*` | IEX stream, REST, paper trading, auth |
| Store / replay | `src/replay/*` | sqlite, historical bar fetch, clock-driven replay |
| Manual orders | `src/live/paper_trader.py` | kill switch + optional CLI |

## Deleted (2026-08-28)

Removed the old event-study / sklearn path. It was superseded by live + sqlite replay.

- `python main.py --research` and `src/pipeline/run_pipeline.py`
- `src/analysis/summarize_results.py`
- `src/ai/relevance_classifier.py` (Ollama relevance; live never used it)
- `src/simulation/simulate_price_impact.py`
- `src/data_prep/build_training_table.py`, `src/data_prep/build_event_table.py`
- `src/modeling/train_baseline.py`
- `src/ingest/news_loader.py`, `price_loader.py`, `prices_fetcher.py`, `alpaca_price_fetcher.py`, `bars_cache.py`
- `src/features/price_movement.py`, `intraday_price_movement.py`
- `src/utils/pretty_news_export.py`

Leftover **data** from that era may still sit under `data/raw/` (`news.json`, `prices.json`, `bars_1min.jsonl`) and `results/price_impact_*.csv` / pretty-news dumps. Nothing in the current program reads those files. Safe to ignore or delete later; they are not the research set.

## Related docs

- [`decision_framework.md`](decision_framework.md) — entry/size/exit math and uncalibrated knobs
- [`alpaca_integration_spec.md`](alpaca_integration_spec.md) — IEX vs SIP, rate limits, paper fills
