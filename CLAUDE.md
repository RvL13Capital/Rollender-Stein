# CLAUDE.md — Rollender-Stein

This file is read by Claude Code on every session start. Future sessions should
read it before touching the codebase.

## What this project is

This repo implements the **URTRIF v3.0** framework (Unified Real Total
Return Index Framework) and generalizes it to a multi-numéraire
architecture. The canonical spec lives in [`URTRIF.md`](URTRIF.md);
the original PDF is at [`docs/URTRIF_v3.0.pdf`](docs/URTRIF_v3.0.pdf).

URTRIF v3.0 transforms each raw price series into a single
real-total-return index `I_real` against one CPI deflator. The repo's
**Absolute Valuation Engine (AVE)** applies the same mathematical core
to **four parallel deflators** at once — Time (AHETPI hourly wages),
Energy (Brent → MWh), Liquidity (G3 broad money: US M2 + EZ M3·EURUSD +
JP M3/USDJPY), and Gold (raw GC=F) — anchored at the Genesis Timestamp
`T0 = 2000-01-03`. The output is a 3D phase-space visualization where
each asset's trajectory reveals its real performance across multiple
purchasing-power dimensions; the *divergence between axes* is the
forensic information.

Mathematically: `Asset_in_X(t) = nominal_USD(t) / N_X(t) · 100` for
X ∈ {Time, Liquidity, Gold, Energy}. This is URTRIF's `I_real(t)`
written in pointwise rather than cumulative-log form, with N_X playing
the role of URTRIF's CPI deflator. See [`URTRIF.md`](URTRIF.md) §6 for
the full algebraic bridge.

Repository: `RvL13Capital/Rollender-Stein` on GitHub. Local working tree:
`/Users/vonlinck/Desktop/Sympathy for the Devil/`.

The user originally provided a 7-phase forensic engineering spec (saved in
session memory as `project_AVE_spec.md`). Phase 4's Kalman methodology was
empirically demoted to a diagnostic during the audit-fix campaign — see
"Spec deviations" below. URTRIF v3.0 was adopted in May 2026 as the
canonical specification document; the existing AVE architecture was
re-positioned as URTRIF's multi-numéraire generalization rather than
reimplemented.

## Build / test commands

```bash
# Run from repo root.
.venv/bin/pytest -q                      # full test suite (~182 tests)
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
├── patterns.py              # z-scores, correlations, Kalman innovation diagnostics
├── volume.py                # dollar-turnover + rolling vol z-score (dashboard conviction channel)
├── runconfig.py             # YAML config dataclass for the background engine
├── run.py                   # refresh-pipeline orchestrator (RunResult, StepResult)
├── cli.py                   # `ave refresh` argparse CLI entry point
├── __main__.py              # `python -m rollender_stein` forwarder
├── io/
│   ├── fred.py              # FRED + ALFRED clients + PUBLICATION_LAG_BD
│   ├── eia.py               # EIA petroleum spot client
│   ├── yahoo.py             # yfinance wrappers (history + OHLCV)
│   └── sec_edgar.py         # SEC EDGAR companyfacts client (shares outstanding)
└── numeraires/
    ├── time.py              # N_Time = AHETPI / AHETPI(T0) * 100
    ├── liquidity.py         # N_Liq = G3 Systemic Liquidity (US+EZ+JP); excludes PBOC by design
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

## Dashboard conviction channel (volume → marker opacity)

`build_pipeline_for_asset` reads raw close + volume and computes
`dollar_turnover = raw_close * volume`. The raw close is intentional:
yfinance retroactively split-adjusts both Close and Volume in opposite
directions, so the product cancels splits and yields the historical USD
turnover. `adj_close * volume` would distort by the accumulated dividend
yield.

`dashboard.build_phase_space_figure` then computes a 252-day rolling z-score
of `log(dollar_turnover)` (zeros masked to NaN before log to avoid -inf
poisoning the window) and maps it to per-marker opacity in [0.25, 1.0].
Plotly 3D requires baking alpha into RGBA color strings (per-marker
`opacity` is scalar-only); the line trace keeps the time-color gradient.

Fallback to scalar opacity 0.7 when the asset has <50% positive turnover —
catches indexes (`^SP500TR` has volume=0 always), legacy ingests with
sparse coverage, and brand-new tickers below the rolling window's warm-up.

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
4. **Total Return for equities.** Individual stocks ingest both raw
   `close` and dividend+split-adjusted `adj_close`; consumers choose at
   read time via `get_asset_closes(prefer_adjusted=True)` for per-share
   absolute valuation (TR-equivalent), or `prefer_adjusted=False` for
   market-cap (raw shares × raw close). The legacy
   `use_adjusted_as_close` ingest parameter is deprecated.
5. **Strict typing.** `mypy --strict` zero-error gate is non-negotiable.
6. **Bitemporal release validation.** `_validate_release_after_reference`
   raises ValueError if `release_date < reference_date` (audit patch 01).
7. **Publication lag awareness.** Per-series `PUBLICATION_LAG_BD` table in
   `io/fred.py` shifts release_date for known FRED series (audit patch 02).
   Migrations are auto-run on `open_db`.

## Spec deviations (deliberate, audit-fix campaign)

Initial audit-fix campaign: eight commits `dfe487d → 98302c9` applied
seven audit patches plus three self-review followups. A second-pass audit
(`af3956a` → `96340ca`, six commits across 2026-04-27/28) closed out the
step-by-step review with patches P1-P4 (tech hygiene, innovations switch,
G3 rename, better-way docstrings) plus the formal decision log
[`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md). The methodologically
significant deviations:

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
4. **PBOC deliberately excluded — N_Liq is G3, not "Global".** China's M2
   is ~$47T USD-equivalent (comparable in size to the entire G3 ocean), but
   PBOC data is opaque, frequently revised in methodology, subject to state
   intervention, and convertible via a heavily-managed CNY rate. Injecting
   it would contaminate the numéraire with synthetic FX-conversion noise.
   Per audit finding 9.M-12, the docstring + error messages now use "G3
   Systemic Liquidity" honestly. The function symbol `build_n_liq` and
   Series name `N_Liq` are unchanged for API stability.
5. **EZ/JP M3 splice with growth rates.** FRED's level series stop at
   2023-11; growth-rate variants extend through Dec 2025. `extend_levels_with_growth`
   compounds forward.
6. **Kalman innovations, not filtered residuals (P2, `bec7a64`).** The
   prior implementation persisted `XAU - filtered_state - X@beta` and
   called those "residuals". On production data those filtered residuals
   have ~340x lower variance than true one-step-ahead innovations
   (`fit.results.resid`) — they are an over-smoothed look-ahead artefact,
   not a residual. Patch P2 switched the persisted output and downstream
   diagnostics to true innovations. The shift is itself a quantification
   of audit finding 10.M-13: σ²_level/σ²_irregular ≈ 15. Consumers using
   scale-invariant metrics (recent_to_alltime_std_ratio, last_in_sigmas,
   autocorr) are unaffected; absolute std/mean shifted ~18.5x σ-scale.
   See AUDIT_DECISIONS.md "P2 calibration baseline".

## Data and persistence

- `data/ave.duckdb` — bitemporal DuckDB with `macro_release`, `asset_price`,
  `fx_close` tables. Auto-migrates on every `open_db`.
- `data/derived/` (gitignored) — transformed outputs:
  - `numeraires/{n_time,n_liquidity,n_energy,n_gold}.parquet`
  - `panels/kalman_panel.parquet` — Phase 4.5 input
  - `kalman/{filtered_state,innovations}.parquet` + `params.json` (the
    `params.json` includes a self-describing `innovation_summary: {mean,
    std, fit_window}` block — see Spec Deviation #6)
  - `divisions/{TICKER}.parquet` — one per ingested asset
  - `patterns/{valuation_z_scores,correlation_matrix}.parquet` +
    `kalman_innovation_diagnostics.json`
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
    result = build_pipeline_for_asset(con, "BTC-USD", animate=True,
                                      recency_window_days=1825)
    path = save_asset_dashboard(result, suffix="_4axis_animated")
```

## Background-engine CLI

The full refresh pipeline (macro ingest → asset ingest → SEC ingest →
`dump_all_artifacts` → dashboards) is exposed via a single CLI entry-point.
Per-step error tolerance: a failure in one ticker doesn't kill the run;
the result summary itemizes successes and errors.

```bash
# Plan only — no I/O, no DB writes
ave refresh --config config/refresh.yaml --dry

# Real run — needs FRED_API_KEY + EIA_API_KEY in .env
ave refresh --config config/refresh.yaml

# Partial runs (CLI flags OR-compose with YAML skip flags)
ave refresh --config config/refresh.yaml --no-macro --no-sec
```

Module form (no install needed): `python -m rollender_stein refresh ...`.

Config schema: see [`config/refresh.yaml.example`](config/refresh.yaml.example).
Only user-specific items are configurable (Yahoo tickers, SEC filers,
dashboard list). The macro layer (FRED/EIA series IDs) is intentionally
hardcoded in `numeraires/*.py` — exposing it to YAML would invite silent
methodological drift.

**Local scheduling** (recommended for the kind of forensic vintage-data
work AVE does — DB persistence on a stable machine, not ephemeral CI runners):

```cron
# crontab -e — daily refresh at 22:30 local time, log to data/refresh.log
30 22 * * 1-5 cd /Users/you/Desktop/Sympathy\ for\ the\ Devil && \
  .venv/bin/ave refresh --config config/refresh.yaml >> data/refresh.log 2>&1
```

Or via `launchd` (macOS) or `systemd timer` (Linux) for similar effect.

**GitHub Actions** (`.github/workflows/ci.yml`): runs the three-tool gate
(pytest + mypy strict + ruff) on every push to `main` and every PR.
Refresh is intentionally NOT scheduled in Actions because the DuckDB
file needs persistent storage; ephemeral runners would lose vintage
history every run. Run refresh locally via cron and push the code; CI
verifies the code, you keep the data.

## Known limitations / gaps

- **N_Gold T0 gap (8 months).** GC=F starts 2000-08-30; no free spot source
  predates that. Patch-04 warning surfaces this.
- **Parameter-level look-ahead in Kalman (§6 causality, accepted).** MLE
  fits on the full panel; ``θ̂`` is F_T-measurable, so ``μ_t(θ̂)``
  technically peeks at the future via the parameter vector. Status:
  **accepted as a documented limitation, not a defect to be fixed**.
  Rationale: (a) Kalman is Phase-4.5 diagnostic-only since patch 06,
  not on the N_Gold pipeline path; (b) the model is identification-
  degenerate (σ²_level/σ²_irregular ≈ 15.5 — see Spec Deviation #6 and
  AUDIT_DECISIONS.md "§6 Causality"), so rolling-MLE would produce
  unstable degeneracy, not a cleaner estimate; (c) any future user
  who wants Kalman-driven *signals* should fork the diagnostic with
  proper expanding-window MLE + walk-forward backtest, which is its
  own project.
- **PBOC deliberately excluded.** ~50% of global broad money is out of
  scope by design (see Spec Deviation #4 — China's data is opaque, state-
  managed, CNY-conversion-contaminated). N_Liq is honestly G3, not global.
- **CI is code-only, not data.** GitHub Actions runs the three-tool gate
  on every push (`.github/workflows/ci.yml`); the refresh pipeline runs
  locally via cron because the DuckDB needs persistent storage that
  ephemeral runners don't provide.
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
- Audit decisions log: [`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) — per-
  finding resolution status (DONE / REWORKED / WON'T FIX) for the
  step-by-step review at audit baseline `af3956a`. Includes per-finding
  rationale for each rejected suggestion so future audits don't re-open
  closed decisions.
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
