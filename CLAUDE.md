# CLAUDE.md — Rollender-Stein

This file is read by Claude Code on every session start. Future sessions should
read it before touching the codebase.

## What this project is

The **Absolute Valuation Engine (AVE)** — a Python toolkit that measures asset
prices in absolute terms by deflating against four numéraires (Time, Liquidity,
Energy, Gold) anchored at a Genesis Timestamp `T0 = 2000-01-03`. The output is
a 3D "phase-space" visualization showing each asset's trajectory through real
purchasing-power dimensions, separating genuine wealth-generation from fiat
illusion.

Repository: `RvL13Capital/Rollender-Stein` on GitHub. Local working tree:
`/Users/vonlinck/Desktop/Sympathy for the Devil/`.

The user originally provided a 7-phase forensic engineering spec (saved in
session memory as `project_AVE_spec.md`). Phase 4's Kalman methodology was
empirically demoted to a diagnostic during the audit-fix campaign — see
"Spec deviations" below.

## Build / test commands

```bash
# Run from repo root.
.venv/bin/pytest -q                      # full test suite (~131 tests)
.venv/bin/mypy --strict src              # type check (must be clean)
.venv/bin/ruff check src tests           # lint (must be clean)
.venv/bin/pytest --cov=src/rollender_stein --cov-report=term  # coverage
```

Three-tool gate before any commit: pytest → mypy → ruff. All three must be
green. CI is not yet wired; verify locally.

## Module layout

```
src/rollender_stein/
├── calendar.py              # T0_DATE constant + master_calendar()
├── bitemporal.py            # DuckDB schema, open_db, insert_*, migrations
├── locf.py                  # forward_fill_to_calendar() — Phase 2
├── config.py                # .env loader (FRED_API_KEY, EIA_API_KEY)
├── valuation.py             # build_division_array — Phase 5 (per-share)
├── marketcap.py             # build_market_cap_division — Phase 5b (whole-company)
├── dashboard.py             # Plotly 3D figure (static + animated) — Phase 6
├── audit.py                 # truncation_hash_audit — Phase 7
├── assets.py                # ingest_yahoo_asset + build_pipeline_for_asset
├── persist.py               # dump_all_artifacts → data/derived/*
├── patterns.py              # z-scores, correlations, residual diagnostics
├── io/
│   ├── fred.py              # FRED + ALFRED clients + PUBLICATION_LAG_BD
│   ├── eia.py               # EIA petroleum spot client
│   ├── yahoo.py             # yfinance wrappers (history + OHLCV)
│   └── sec_edgar.py         # SEC EDGAR companyfacts client (shares outstanding)
└── numeraires/
    ├── time.py              # N_Time = AHETPI / AHETPI(T0) * 100
    ├── liquidity.py         # N_Liq = Global Fiat Ocean (US+EZ+JP)
    ├── energy.py            # N_Energy = Brent → MWh (floor $0.10)
    └── gold.py              # N_Gold = raw GC=F (Kalman is diagnostic only)
```

## Two valuation layers

The AVE provides two parallel views for any individual asset, answering two
distinct questions:

1. **Per-share** (default): `valuation.build_division_array` measures one
   share's purchasing power in T0-deflated USD using yfinance's
   dividend+split-adjusted close. This is the right view for an investor
   holding a fixed number of shares — dividends and splits are baked in
   correctly. Output: `data/derived/divisions/{TICKER}.parquet`.

2. **Market cap** (whole company): `marketcap.build_market_cap_division` uses
   `raw_shares × yfinance_close` (with shares back-adjusted for cumulative
   future splits to match yfinance's price basis). This measures the
   company's TOTAL purchasing power — separates real wealth creation from
   buyback / issuance effects. Requires `ingest_sec_shares(ticker, ...)` to
   pull share history from SEC EDGAR's companyfacts API.

The diagnostic `marketcap.per_share_vs_market_cap_multiplier(con, ticker)`
returns the ratio. Ratio > 1 indicates buybacks (per-share appreciated more
than the company's total value); ratio < 1 indicates dilution. AAPL is ~2.0
(half of its per-share gain came from buybacks); TSLA is ~0.4 (per-share
underperformed market cap by 2.5x due to dilution).

Coverage: market-cap layer applies only to **US-listed individual stocks**
that file 10-K/10-Q with the SEC. Indexes, ETFs, futures, crypto, and foreign
issuers are out of scope (they don't have meaningful "shares outstanding" in
the same sense). EDGAR's XBRL coverage starts ~2008-2009.

## Forensic principles (HARD RULES, not preferences)

These rules are enforced in code; never relax them without explicit user
sign-off:

1. **No look-ahead in LOCF.** Macro data joins onto the daily calendar via
   `pd.merge_asof(... on='release_date', direction='backward')` only. Never
   `.interpolate()`, never `.fillna(method='ffill')` over reference_date.
2. **Filtered, not smoothed.** Kalman code uses `results.filtered_state[...]`
   exclusively. `smoothed_state` peeks at future observations.
3. **Vintage macro data.** FRED queries with revisions go through
   `fetch_alfred_first_release` (output_type=4). Daily series with no
   meaningful revisions use `fetch_fred_observations`.
4. **Total Return for equities.** Individual stocks ingest with
   `use_adjusted_as_close=True` so `adj_close` becomes `close`. Pre-TR
   indexes (^SP500TR) use raw close.
5. **Strict typing.** `mypy --strict` zero-error gate is non-negotiable.
6. **Bitemporal release validation.** `_validate_release_after_reference`
   raises ValueError if `release_date < reference_date` (audit patch 01).
7. **Publication lag awareness.** Per-series `PUBLICATION_LAG_BD` table in
   `io/fred.py` shifts release_date for known FRED series (audit patch 02).
   Migrations are auto-run on `open_db`.

## Spec deviations (deliberate, audit-fix campaign)

Eight commits `dfe487d → 98302c9` applied seven audit patches plus three
self-review followups. The methodologically significant deviations:

1. **Patch 06 Option C — N_Gold uses raw XAU, Kalman demoted.**
   The original spec specified a Kalman state-space model orthogonalizing
   XAU against (TIPS, DXY, VIX) noise. On real data that "orthogonalization"
   was empirically degenerate (`corr(μ_t, raw XAU) = 0.97`, non-orthogonal to
   the regression). The Kalman pipeline (`fit_gold_model`,
   `dump_kalman_outputs`) is preserved as a Phase 4.5 *diagnostic*; N_Gold
   itself is `GC=F / GC=F(2000-08-30) * 100`. **N_Gold(T0) = NaN — the 8-
   month residual gap surfaces via the patch-04 RuntimeWarning on every
   `build_division_array` call.**
2. **Patch 03 — Energy floor lowered to $0.10/MWh** (from $20). The old floor
   was binding at the T0 anchor (Brent $14.67/MWh < $20), biasing the entire
   N_Energy index by +36.3% on every non-floor-binding date. The new floor
   never binds on historical Brent.
3. **Phase 7 truncation hash test — frozen-params variant.** The literal
   spec test (refit MLE on truncated data) is mathematically vacuous because
   MLE re-estimation drifts. The implementation freezes params from the full
   fit and verifies recursion equivalence to ≥8 decimals (deterministic by
   construction).
4. **PBOC deferred from Global Fiat Ocean.** No clean ALFRED-style source.
   N_Liq is `US M2 + EZ M3 + JP M3` in USD. The "Global" naming is a slight
   overstatement; flagged in the audit M-12.
5. **EZ/JP M3 splice with growth rates.** FRED's level series stop at
   2023-11; growth-rate variants extend through Dec 2025. `extend_levels_with_growth`
   compounds forward.

## Data and persistence

- `data/ave.duckdb` — bitemporal DuckDB with `macro_release`, `asset_price`,
  `fx_close` tables. Auto-migrates on every `open_db`.
- `data/derived/` (gitignored) — transformed outputs:
  - `numeraires/{n_time,n_liquidity,n_energy,n_gold}.parquet`
  - `panels/kalman_panel.parquet` — Phase 4 input
  - `kalman/{filtered_state,residuals}.parquet` + `params.json`
  - `divisions/{TICKER}.parquet` — one per ingested asset
  - `patterns/{valuation_z_scores,correlation_matrix}.parquet` +
    `kalman_residual_diagnostics.json`
  - `manifest.json` — inventory with row counts and date ranges
- `data/dashboard_*.html` — static + animated Plotly outputs

`.env` (gitignored) holds `FRED_API_KEY` and `EIA_API_KEY`. yfinance has no
key. Never commit `.env`. Keys pasted in chat should be rotated.

## Common workflows

**Add a new target asset:**
```python
from rollender_stein.bitemporal import open_db
from rollender_stein.assets import ingest_yahoo_asset
with open_db("data/ave.duckdb") as con:
    # Stores BOTH raw close and adj_close. The use_adjusted_as_close
    # parameter is deprecated — choose at read time via
    # get_asset_closes(prefer_adjusted=...).
    ingest_yahoo_asset(con, "TICKER")
```

**Add the market-cap layer for a stock:**
```python
from rollender_stein.marketcap import ingest_sec_shares, build_market_cap
USER_AGENT = "YourOrg you@example.com"  # SEC fair-access requirement
with open_db("data/ave.duckdb") as con:
    ingest_sec_shares(con, "AAPL", USER_AGENT)
    mc = build_market_cap(con, "AAPL")  # daily market cap in current-basis USD
```

**Re-dump everything after data changes:**
```python
from rollender_stein.persist import dump_all_artifacts
from rollender_stein.patterns import dump_pattern_report
with open_db("data/ave.duckdb") as con:
    tickers = [r[0] for r in con.execute(
        "SELECT DISTINCT series_id FROM asset_price"
    ).fetchall()]
    dump_all_artifacts(con, tickers=tickers)
    dump_pattern_report()
```

**Generate a dashboard for one asset:**
```python
from rollender_stein.assets import build_pipeline_for_asset, save_asset_dashboard
with open_db("data/ave.duckdb") as con:
    result = build_pipeline_for_asset(con, "BTC-USD", animate=True)
    path = save_asset_dashboard(result, suffix="_4axis_animated")
```

## Known limitations / gaps

- **N_Gold T0 gap (8 months).** GC=F starts 2000-08-30; no free spot source
  predates that. Patch-04 warning surfaces this.
- **Parameter-level look-ahead in Kalman.** MLE fits on the full panel;
  rolling MLE not implemented. Diagnostic-only since patch 06; if N_Gold
  ever returns to Kalman-driven, this becomes a real backtest concern.
- **PBOC missing.** ~50% of global broad money is excluded.
- **No CI.** Tests run locally only.
- **Heavy-tailed gold returns.** The Kalman is QMLE; std errors from
  statsmodels are wrong by orders of magnitude (we don't report them).
- **Plotly 3D animation** is heavy on the browser at high frame counts —
  default `frame_step=21` (monthly) is the sweet spot.

## When in doubt

- Project memory: `~/.claude/projects/-Users-vonlinck-Desktop-Sympathy-for-the-Devil/memory/` —
  contains `project_AVE_spec.md` (the original 7-phase spec + audit
  resolutions) and `feedback_forensic_hygiene.md` (the hard rules).
- Audit findings: `/Users/vonlinck/Downloads/files/` (if still present) —
  `rollender_stein_audit_pass4_findings.md`,
  `rollender_stein_audit_addendum.md`, `rollender_stein_math_audit.md`.
- Commit history is clean and message-rich; `git log` is canonical for
  understanding why each piece exists.

## What this codebase is NOT

- A trading system. The patterns module emits descriptive statistics, not
  signals. Anyone using these outputs to size positions needs walk-forward
  backtesting, transaction costs, regime testing — none of which is here.
- Fully spec-compliant. Phase 4's Kalman narrative was demoted because
  empirically it had no orthogonalization signal. The spec's intent (measure
  absolute value across multiple gauges) is preserved; the specific mechanism
  is not.
- Production-monitored. There's no alerting, no SLA on FRED API calls, no
  retry logic beyond requests' default. Real production deployment would
  need a layer on top.
