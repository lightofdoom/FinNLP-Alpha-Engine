# Decision Framework

Project state and how to run: [`project_status.md`](project_status.md).

How this engine decides **entries** (dynamic math/ML blend with exponential news decay), **size** (entry-only portfolio allocation, ATR/R fallback), and **exits** (100% math). The LLM is never consulted on exits.

ML means the Ollama news stack (sentiment, confidence, actionability, catalysts, risk flags), not a trained classifier.

Uncalibrated constants are listed at the end. IEX is ~2.5% of US volume; volume and silence checks are relative to each symbol’s own IEX baseline.

## Split of responsibility

| Decision | Mix |
| --- | --- |
| Entry | Fresh news: 30% `math_score` + 70% `ml_score`. Quiet/decay: 85% math + 15% decayed ML |
| Size | Entry-only shrunk MVO, then `ALPACA_NOTIONAL` cap. ATR/R is the fallback |
| Exit | 100% rules below; no LLM |

Hard vetoes sit **outside** the blend: market closed, kill switch, existing position, missing quote, fewer than 20 one-minute bars, ATR unavailable, IEX-thin entry.

**News required:** a non-zero ML score — either fresh relevant news this cycle or a decayed residual that has not yet hit the floor / max age. Pure math with `ml=0` does not enter. This stays a news engine with a price overlay; quiet cycles may still enter while sentiment is fading.

## Price math (1-minute IEX bars)

Lookback 90 bars. Refuse entry if `bar_count < 20`. Log-closes for slopes.

### Slopes

OLS of `ln(P)` on bar index. `slope_n = Cov(i, ln P) / Var(i)` (dimensionless per bar).

Windows n = 5, 15, 30, 60. A window is available only if `bar_count >= n`. **Missing windows are omitted and remaining weights renormalized** (not filled with 0).

Base weights `{5: 0.40, 15: 0.30, 30: 0.20, 60: 0.10}`.

**Scale (dimensionless):** `sigma_n = ATR_14 / (P_t * sqrt(n))`. ATR is global; if ATR or price is missing, **all** slope windows are unavailable and entry is blocked (`math_unavailable`). There is no per-window “skip ATR” path.

`s_n = tanh((exp(slope_n) - 1) / sigma_n)`

**Alignment** on available members of {5, 15, 30}:

- All available of those three share a nonzero sign: `1.0`
- 5 and 15 exist and agree: `0.5`
- Only n=5 exists (20–24 bars): `0.25` (unconfirmed; not as confident as two-window agreement)
- Else: `0`

`trend_score = clip(sum w'_n s_n, -1, 1) * (0.5 + 0.5 * trend_align)`

### RSI

Wilder RSI-7 → `rsi_entry = clip((RSI7 - 50) / 25, -1, 1)` (momentum for entries). RSI-14 is logged. `rsi_conflict` (|RSI7 − RSI14| > 20) is **not** a veto.

### VWAP (coupled pair — do not “fix” one without the other)

Both `vwap_score` and `ext_aligned` come from `(close − vwap) / ATR`. They are **not** orthogonal to each other. `vwap_score` is a continuous pullback term; `ext_aligned` is a chase penalty that only turns on beyond 2×ATR **in the same direction as `raw`**. `mom_15` is logged only, not blended.

`vwap_score = clip(-(close - vwap) / (2 * ATR), -1, 1)`

`ext = clip(|close - vwap| / (2 * ATR) - 1, 0, 1)`

`ext_aligned = ext` if `sign(close - vwap) == sign(raw)` and `raw != 0`, else `0`.

### Blend to math_score

`raw = 0.45 * trend_score + 0.25 * rsi_entry + 0.15 * vwap_score + 0.15 * tanh(consec/5) * sign(trend_score)`  
(consec term is 0 if `trend_score == 0`)

`math_score = clip(raw * (0.6 + 0.4 * part) * (1 - 0.5 * ext_aligned), -1, 1)`

`part = tanh(max(vol_z, 0) / 2)` on this symbol’s 20-bar IEX volume.

Entry `iex_thin`: last 5 bars all have `trade_count == 0`.

## ML score

`ml_score = clip(sentiment * confidence * actionability * (1 + 0.5 * catalyst_rate), -1, 1)`  
If risk-flag counts (excluding `none`) exceed catalyst counts, multiply by 0.8 (uncalibrated). Log before and after.

### Exponential decay (per ticker)

Not algebraic `max`. Wall-clock minutes, so a slow LLM poll does not under-decay.

`ml_t = new_score` on a fresh relevant, scored cycle.  
`ml_t = λ^{dt} * ml_{t-1}` on a quiet cycle (`λ` default 0.75).  
`|ml_t| < ML_SCORE_FLOOR` (0.02) → 0. Age since last fresh print `> NEWS_MAX_AGE_MIN` (15) → 0.

Persisted in `results/paper/ml_decay_state.json`. Restart re-decays from `updated_at`.

## Entry

Fresh: `entry_score = 0.30 * math_score + 0.70 * ml_score`  
Quiet: `entry_score = 0.85 * math_score + 0.15 * ml_score_decayed`

Long if `entry_score > ENTRY_T`, short if `< -ENTRY_T`, else flat. **`ENTRY_T = 0.20` is the only entry threshold** (no separate min-strength).

Disagreeing nonzero signs → no trade. Floored `ml=0` does not disagree (sign is 0). Math unavailable → no trade. `ml=0` and no fresh news → no trade.

## Size (small-paper mode)

**Entry-only allocator.** Existing positions are not resized or flattened here; they only enter the covariance / gross-leverage term so a new name is penalized for adding to a concentrated book.

`μ_i = entry_score_i * (ATR_i / P_i)`. `Σ` is a shrunk sample covariance of aligned 1-minute log returns (`COV_SHRINKAGE` default 0.30). Objective: maximize `μ'n − λ n'Σn − γ‖n‖₁` subject to `|n_i| ≤ ALPACA_NOTIONAL` and `∑|n_i| + existing_gross ≤ GROSS_LEVERAGE * equity` (also capped by `0.25 * buying_power`). Solver is scipy SLSQP. On failure, isolated ATR/R then cap.

Always apply ATR/R vol gates first: high vol (>80% ann.) halves the fallback qty; low vol (<8% ann.) skips. `sigma_30_ann = stdev(1-min log ret, 30) * sqrt(390 * 252)`.

`qty = floor(|n_i| / price)` after the allocator. **`ALPACA_NOTIONAL` (default $1000) is the per-name hard cap.** `intended_notional` vs `capped_notional` is logged; `size_source` is `optimizer` or `fallback`.

`SIZING_MODE=risk_model` (equity×8% cap) is specified, not enabled in v1.

## Exits (no LLM)

**Priority is this list, not table order. Hard stop always wins on a wide bar.** Hard stop uses that bar’s high/low even if IEX prints are thin: protecting the position outranks data-quality silence. Silence is checked next.

1. Hard stop (urgent)  
2. IEX-thin flatten (urgent) — 20 consecutive minutes of zero `trade_count` after the name had activity; midday quiet with occasional prints does not flatten  
3. Slope reversal (urgent)  
4. Trailing stop (limit)  
5. Take profit (limit)  
6. RSI-7 extreme and `|pct_5|` flattened (limit)  
7. Time stop 45 min with tiny unrealized (limit)  
8. Session flatten 15:50 ET (limit)

Urgent: **market** in regular hours; **IOC limit through the quote** in extended hours. Non-urgent: limit ± offset.

Stops use ATR **frozen at entry**.

## Live cycle

1. Update 1-min bar cache  
2. Cancel stale unfilled **entry** orders (older than one bar, gates failed, or conflict with an exit)  
3. Exits on open positions  
4. News → ML (fresh replace) or decay  
5. Price features → math on every watchlist name  
6. Dynamic blend and gates  
7. One optimizer call over candidates; submit sized entry limits; log components  

## Uncalibrated (log and revisit)

Fresh/quiet weights, `ML_DECAY_LAMBDA`, `ML_SCORE_FLOOR`, `NEWS_MAX_AGE_MIN`, `ENTRY_T`, vol 80%/8%, risk dampener 0.8, stop 1.5 / target 2.0 ATR (~43% breakeven win rate before costs), `OPTIMIZER_RISK_AVERSION`, `OPTIMIZER_TURNOVER_PENALTY`, `GROSS_LEVERAGE`, `COV_SHRINKAGE`. Slippage unmodeled except as the turnover term.
