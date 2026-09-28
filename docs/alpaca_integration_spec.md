# Alpaca Market Data and Paper Trading Spec

Project state and how to run: [`project_status.md`](project_status.md). Decision math: [`decision_framework.md`](decision_framework.md).

This document is the implementation spec for Alpaca on the free (Basic) plan: IEX WebSocket for live prices, 1-minute historical bars in the research sqlite, and the paper Trading API for simulated fills.

## Feasibility

Feasible on the free plan, with one caveat: you will not get true consolidated market prices.

| Need | Free Alpaca | Verdict |
| --- | --- | --- |
| Intraday prices for news labels | 1-min bars; SIP allowed only if `end` is ≥15 min old | Yes |
| Live price at decision time | IEX WebSocket (preferred) or REST snapshot | Yes |
| Simulated orders, positions, P&L | Paper trading at `https://paper-api.alpaca.markets` | Yes |
| True NBBO / all-exchange prints | Algo Trader Plus (SIP) | No on free |
| Sub-second HFT / queue position | Not a brokerage-API problem | No |

**IEX:** free real-time data is IEX only (~2.5% of US volume). Train and paper-trade on `iex` so research matches the bot. Delayed SIP (`feed=sip`, `end` ≥ 15 minutes ago) is optional offline research only; never used on `/latest` or snapshot.

**Paper fills** match a simulated NBBO and do not model market impact, queue position, live slippage, dividends, or borrow fees. They may partial-fill randomly and can fill sizes larger than displayed liquidity.

**Latency:** Finnhub news + LLM inference dominate. Do not poll REST for “low latency.”

## Pull frequency

Live / paper:

| Channel | Cadence |
| --- | --- |
| IEX WebSocket `trades` + `quotes` on `WATCHLIST` | Event-driven |
| IEX WebSocket `bars` | Once per minute |
| REST snapshot (all symbols, one call) | On demand at order time; every 15s **only if the stream is down** |
| Trading API `get_clock` | Every 60s |
| Trading stream `trade_updates` | Event-driven (fills/partials) |

REST-only fallback: snapshot the whole watchlist every 5 seconds. Do not poll at 1 Hz.

Historical:

| Job | Timeframe | Feed |
| --- | --- | --- |
| Historical research set (`python main.py --fetch-bars`) | `1Min` | `iex` into `data/research_set/research.sqlite` |
| Live bar cache | `1Min` stream | `iex` |
| Optional true-market research | same windows | `sip` only if `end` ≥ 15m ago, separate files |

REST is rate-limited at 180 requests/min (buffer under the 200/min Basic cap). Watchlist must stay at or under 30 symbols.

## Layout

- [`src/config.py`](../src/config.py) — env, paths, paper defaults
- [`src/broker/alpaca_auth.py`](../src/broker/alpaca_auth.py) — keys, paper URL, live-host guard (`paper-api` is not treated as live)
- [`src/broker/rate_limit.py`](../src/broker/rate_limit.py) — shared token bucket
- [`src/broker/alpaca_data.py`](../src/broker/alpaca_data.py) — bars, latest trade/quote, snapshot; IEX required on latest
- [`src/broker/alpaca_stream.py`](../src/broker/alpaca_stream.py) — IEX `StockDataStream` + in-memory cache + 15s snapshot fallback
- [`src/broker/alpaca_trading.py`](../src/broker/alpaca_trading.py) — paper `TradingClient`, limit/market/IOC, clock poll, fill recorder
- [`src/replay/fetch_bars.py`](../src/replay/fetch_bars.py) — historical 1-min IEX bars into sqlite
- [`src/live/paper_trader.py`](../src/live/paper_trader.py) — kill switch helpers + optional manual paper CLI

1-min bars for research live in sqlite (`data/research_set/research.sqlite`), not JSONL.

Persist UTC; convert with `America/New_York` for decisions ([`src/features/session_labeler.py`](../src/features/session_labeler.py)).

## Config / secrets

Copy [`.env.example`](../.env.example) to `.env`. Required for Alpaca paths: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`. Defaults:

- `APCA_API_BASE_URL=https://paper-api.alpaca.markets`
- `ALPACA_DATA_FEED=iex`
- `ALPACA_ALLOW_LIVE=false` — live host raises unless this is true
- `ALPACA_MAX_SYMBOLS=30`
- `ALPACA_NOTIONAL=1000`
- `ALPACA_LIMIT_OFFSET_BPS=5`
- `ALPACA_MAX_ORDERS_PER_DAY=10`
- `ALPACA_MAX_LOSS_DOLLARS=2000`

## Historical bars

Research bars are 1-minute IEX rows in sqlite (`python main.py --fetch-bars`), not per-headline return labels. Replay walks stored bars + scored news through the same 60/40 stack as live.

## Live simulation (default entry point)

`python main.py` from `src/` runs the live loop: Finnhub news, rule prefilter, Ollama
`mistral:7b` signal extraction, aggregate per ticker, then an Alpaca paper limit order.
Only news that arrives while the loop is running can trigger a trade, so no historical
price data is needed.

```
python main.py                              # until Ctrl-C
python main.py --until "8/25/26 12:47 AM"   # auto-stop at a wall-clock time
python main.py --duration 90m               # auto-stop after a span
python main.py --dry-run                    # score signals, place no orders
python main.py --fetch-bars --bars-days 45  # backfill 1-min IEX bars
python main.py --replay                     # offline 60/40 vs ML-only
```

Entries use the 60/40 blend in [`decision_framework.md`](decision_framework.md), not a standalone min-strength cutoff. `--min-signal` is `ENTRY_T` (default 0.20). `--strategy` only changes how the ML score is built (`combined` default).

End time accepts `8/25/26 12:47 AM`, `2026-08-25 00:47`, a bare `12:47 AM` (next
occurrence), or ISO with offset. Naive input is read as America/New_York. `--until`
wins over `--duration`; with neither, the session runs until interrupted.

Every session writes `results/paper/session_<stamp>.txt` and `.json` with the news
counts, each decision, orders, fills, start/end equity, P&L, and open positions.

Signal strategies (`--strategy`), all driven by `aggregate_signals`:

| Strategy | Strength |
| --- | --- |
| `sentiment` | mean sentiment x confidence |
| `catalyst` | mean sentiment x actionability, requires a catalyst |
| `combined` (default) | sentiment x confidence x actionability x catalyst boost |

`--no-short` makes it long-only.

## Manual paper CLI

For one-off orders, independent of the live loop. From `src/`:

```
python -m live.paper_trader status
python -m live.paper_trader buy AMZN --preview
python -m live.paper_trader sell AMZN
python -m live.paper_trader cancel AMZN
```

Order policy (shared by the live engine and this CLI):

- Regular session: limit at IEX ask (buy) or bid (sell) ± offset, `time_in_force=day`
- Premarket / afterhours: limit + `extended_hours=True`
- Closed: refuse (v1 does not queue)
- Size from `ALPACA_NOTIONAL` / last trade, whole shares, capped by buying power
- Max one open order per symbol; existing position refused unless `--flatten`
- Daily kill switch: max orders and max loss vs starting paper equity
- Fills come only from `trade_updates`, written to `results/paper/fills.jsonl`

## Tests and smoke

```
python -m unittest discover -s tests -v
```

Live smoke without risking an order (needs Finnhub key, Alpaca paper keys, and `ollama` running):

```
python main.py --dry-run --duration 45s --poll 25 --max-items 2
```

That exercises the whole path and writes a session summary. Drop `--dry-run` to let it
place paper orders.

There is no live-money path unless `ALPACA_ALLOW_LIVE=true` and the live host is used.

## Out of scope

- Paid SIP / Algo Trader Plus
- Streaming news (Finnhub polling remains the news path)
- Options, crypto, portfolio optimizer
