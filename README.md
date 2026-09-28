# FinNLP-Alpha-Engine

A news driven **Alpaca paper** trader. It polls Finnhub headlines, scores them with a local Ollama model, blends that signal with 1-minute IEX price math, and submits simulated orders on an eight-name watchlist.

This is a research / paper-trading project. It is **not** live-money ready, **not** financial advice, and **not** a claim that the strategy is profitable. Constants are uncalibrated. Do not set `ALPACA_ALLOW_LIVE=true`.

## How it decides

Each ~60s poll does the following:

1. Refresh the 1-minute IEX bar cache and evaluate **math-only exits** on open positions.
2. Fetch new Finnhub headlines (headline + summary only — no article scrape).
3. Keyword-filter relevance, then extract JSON features with local Ollama (`mistral:7b` by default): sentiment, confidence, actionability, catalysts, risk flags.
4. Maintain a per-ticker ML score. Fresh relevant news **replaces** it; quiet minutes **decay** it exponentially (`λ ≈ 0.75` per wall-clock minute) until it floors to zero.
5. Blend math and ML with dynamic weights: ML-heavy on a fresh-news cycle, math-heavy while the score is decaying. Signs must agree. `|entry_score|` must clear `ENTRY_T` (0.20). Pure price math with `ml=0` does not enter.
6. Size **new entries only** with a small mean-variance allocator (shrunk 1-minute covariance, per-name `$ALPACA_NOTIONAL` cap). Existing positions are not rebalanced; exits stay 100% rules.

The live loop also archives bars and scored news into `data/research_set/research.sqlite` so the same stack can be replayed offline.

Locked watchlist (do not rotate while the research set is in use):

`NVDA, AAPL, MSFT, AMZN, GOOG, META, TSLA, AMD`

## Repo map

| Path | Role |
| --- | --- |
| `src/main.py` | Live paper loop, `--fetch-bars`, `--replay` |
| `src/live/` | Cycle, blend, decay, exits, sizing, session reports |
| `src/features/` | Price math, news LLM extract, relevance, session labels |
| `src/portfolio/optimizer.py` | Entry-only allocator (scipy); ATR/R is the fallback |
| `src/broker/` | Alpaca IEX stream, REST, paper orders, auth |
| `src/replay/` | Sqlite store, historical bar fetch, clock-driven replay |
| `docs/decision_framework.md` | Entry / size / exit math |
| `docs/alpaca_integration_spec.md` | IEX vs SIP, rate limits, paper fills |
| `docs/project_status.md` | Current behavior and ops notes |

If those docs disagree with code, **code wins** for the live cycle and for `EXIT_PRIORITY` in `src/live/exit_engine.py`.

## Requirements

- Python 3.11+ (developed on 3.13)
- An [Alpaca](https://alpaca.markets/) **paper** account
- A [Finnhub](https://finnhub.io/) API key
- [Ollama](https://ollama.com/) running locally with `mistral:7b` (or pass `--model`)

Alpaca Basic allows **one** IEX websocket. Two `main.py` processes on the same account will fail with `connection limit exceeded`.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
ollama pull mistral:7b
```

Copy `.env.example` to `.env` and fill in paper keys plus Finnhub. `.env` is gitignored. `load_dotenv` will not override a `FINNHUB_API_KEY` that is already set in the environment.

```
FINNHUB_API_KEY=...
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
APCA_API_BASE_URL=https://paper-api.alpaca.markets
ALPACA_ALLOW_LIVE=false
```

If Ollama is not running, the extractor swallows the error and writes a zero ML score. The loop stays up, but it will not enter — it still requires a non-zero news signal.

## Run

Work from `src/` with the project venv:

```bash
cd src
python main.py                              # paper loop (Ctrl-C to stop)
python main.py --dry-run --duration 45s     # score only, no orders
python main.py --fetch-bars --bars-days 45  # backfill 1-min IEX bars into sqlite
python main.py --replay                     # dynamic blend vs ML-only on sqlite
```

Tests from the repo root:

```bash
python -m unittest discover -s tests -v
```

Prefer **04:00–20:00 ET weekdays** for paper orders. Overnight and weekend runs are still useful for archiving news; Alpaca will reject submits when the session is `closed`.

## Honest limits

- **Paper only.** Fills are Alpaca’s simulator, not a live book.
- **IEX only** (~2.5% of US volume). Volume and “thin tape” checks are relative to each symbol’s own IEX prints, not the consolidated tape.
- **Headline + summary.** No full-article crawl.
- **Uncalibrated knobs.** Weights, decay, `ENTRY_T`, stops, and optimizer penalties are starting points. Log them; do not retune from a handful of trades.
- **Kill switch.** Daily order cap and max loss vs starting equity (`ALPACA_MAX_ORDERS_PER_DAY`, `ALPACA_MAX_LOSS_DOLLARS`).

## Built with AI, owned by a person

I used AI coding assistants (including Cursor) as a tool on this project: scaffolding modules, drafting tests, project planning, and walking through design tradeoffs. I ran the paper loop myself, wrote and ran the unit tests, diagnosed live failures (Ollama down, session gates, disagree vetoes), and redesigned the system when the earlier versions broke down.

A system of this size is a lot of surface area to cover. Using AI for the mechanical parts is what made it possible for me to spend time on important decision making parts: the core design decisions, what the bot is allowed to do, what it is not allowed to do, and how much of an impact it has in trading decisions.
## License

MIT. See [`LICENSE`](LICENSE).
