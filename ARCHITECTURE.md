# Architecture

This document describes how the runtime realizes the **URTRIF v3.0**
specification ([`URTRIF.md`](URTRIF.md)) and generalizes it to four
parallel deflators. URTRIF defines `I_real(t) = I_base(t) · C_t0/C_t`
for one CPI-deflator `C`; this implementation produces four such indices
in parallel — one per **numéraire** ("absolute ruler") — all anchored at
the Genesis Timestamp **`T0 = 2000-01-03`** (first NYSE trading day of
the millennium).

The mapping in one line: AVE's `Asset_in_X(t) = nominal_USD(t) / N_X(t) ·
100` is URTRIF's `I_real(t)` written pointwise rather than via
cumulative log-returns, with `N_X(t) = C_X(t) / C_X(t_0) · 100` for each
of the four deflators X ∈ {Time, Liquidity, Energy, Gold}. See
[`URTRIF.md`](URTRIF.md) §6 for the full algebraic bridge.

For methodological depth see [`CLAUDE.md`](CLAUDE.md); for per-finding
decision rationale (DONE / REWORKED / WON'T FIX) see
[`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md); for what the system guarantees
see [`INVARIANTS.md`](INVARIANTS.md); for what it deliberately doesn't see
[`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md).

## Data flow

```
                          EXTERNAL SOURCES
        ┌──────────────┬─────────────┬─────────────┬─────────────┐
        │  FRED        │  EIA v2     │  Yahoo      │  SEC EDGAR  │
        │  + ALFRED    │  petroleum  │  Finance    │  XBRL       │
        └──────┬───────┴──────┬──────┴──────┬──────┴──────┬──────┘
               │              │             │             │
        io/fred.py     io/eia.py    io/yahoo.py    io/sec_edgar.py
        first-release  spot prices  OHLCV +       company facts /
        + live + per-  (release =   splits        shares
        series PUB.    reference)
        LAG_BD
               ▼              ▼             ▼             ▼
        ╔════════════════════════════════════════════════════════╗
        ║  bitemporal store (DuckDB)        bitemporal.py        ║
        ║  - macro_release  PK (series_id, reference, release)   ║
        ║      CHECK release_date >= reference_date              ║
        ║  - asset_price    PK (series_id, trade_date)           ║
        ║  - shares_outstanding  PK (ticker, period_end, filing) ║
        ║      CHECK filing_date >= period_end_date              ║
        ║  - fx_close       PK (pair, trade_date)                ║
        ║  thread-safe inserts via con.cursor() (Patch 07)       ║
        ║  idempotent migrate_publication_lags() on open_db      ║
        ╚════════════════════════╤═══════════════════════════════╝
                                 │
                    latest_release_stream(series_id)
                                 │
                                 ▼
                ╔══════════════════════════════════╗
                ║  locf.py   forward_fill_to_      ║
                ║  calendar(macro, calendar_idx)   ║
                ║  pd.merge_asof(direction="back-  ║
                ║  ward", on=release_date)         ║
                ║  → F_t-measurable daily output   ║
                ╚════════════════╤═════════════════╝
                                 │
        ┌────────────┬───────────┴───────────┬───────────┐
        │            │                       │           │
        ▼            ▼                       ▼           ▼
     N_Time      N_Energy                N_Liq        N_Gold
   (AHETPI)    (Brent/MWh,              (G3 only:    (raw GC=F,
   numeraires/  $0.10 floor)            US M2 +      Kalman demoted
   time.py     numeraires/              EZ M3·EUR +  to diagnostic)
                energy.py                JP M3/JPY)  numeraires/
                                        numeraires/  gold.py
                                        liquidity.py
        │            │                       │           │
        └────────────┴───────────┬───────────┴───────────┘
                                 │
                                 ▼
        ╔════════════════════════════════════════════════════╗
        ║  Two parallel valuation layers                     ║
        ║                                                    ║
        ║  valuation.py    — per-share (TR-equivalent)       ║
        ║    Asset_in_X(t) = nominal_USD(t) / N_X(t) · 100   ║
        ║    zero-guard: N_X(t) = 0 → NaN (not ±∞)           ║
        ║    T0=100 invariant verification with warning      ║
        ║                                                    ║
        ║  marketcap.py    — whole-company (US single stocks)║
        ║    market_cap(t) = adj_shares(t) · raw_close(t)    ║
        ║    cumulative-future-split factor unifies SEC      ║
        ║    point-in-time shares with yfinance basis        ║
        ╚════════════════════════╤═══════════════════════════╝
                                 │
                ┌────────────────┼─────────────────┐
                ▼                ▼                 ▼
          dashboard.py     persist.py        patterns.py
          3D phase-space   parquet artefacts z-scores, log-return
          (Plotly)         + manifest.json   correlations,
          + volume-z-score                    Kalman innovation
          marker opacity                      diagnostics
          + recency-fade
          overlay
                                 │
                                 ▼
                ╔══════════════════════════════════╗
                ║  Phase-4.5 Kalman diagnostic     ║
                ║  (numeraires/gold.py             ║
                ║   + persist.dump_kalman_outputs) ║
                ║  - filtered_state.parquet        ║
                ║  - innovations.parquet (true     ║
                ║    one-step-ahead, ~340x σ²      ║
                ║    of pre-P2 filtered residuals) ║
                ║  - params.json (with self-       ║
                ║    describing innovation_summary)║
                ║  → NOT on the N_Gold pipeline    ║
                ║    path; descriptive only        ║
                ╚══════════════════════════════════╝
```

## Layers and responsibilities

### Layer 1 — Ingest (`io/`)

Pulls external sources into the bitemporal store.

- **[`io/fred.py`](src/rollender_stein/io/fred.py)** — two functions:
  `fetch_alfred_first_release` (vintage-aware, `output_type=4`, used for
  AHETPI) and `fetch_fred_observations` (live endpoint, used for daily
  series). The live-endpoint variant applies a per-series
  `PUBLICATION_LAG_BD` business-day offset to convert reference_date into a
  conservative release_date — see Patch 02.
- **[`io/eia.py`](src/rollender_stein/io/eia.py)** — EIA v2 petroleum spot
  with paginated `length=5000` fetches.
- **[`io/yahoo.py`](src/rollender_stein/io/yahoo.py)** — yfinance wrapper
  exposing `fetch_yahoo_history` (macro_release shape, used for `GC=F`)
  and `fetch_yahoo_ohlcv` (asset_price shape, used for target assets).
- **[`io/sec_edgar.py`](src/rollender_stein/io/sec_edgar.py)** — XBRL
  companyfacts JSON for `CommonStockSharesOutstanding`. Tag-fallback
  hierarchy: `us-gaap` first, `dei` second; `WeightedAverage*` is
  deliberately rejected (period-average, not point-in-time).

### Layer 2 — Bitemporal store (`bitemporal.py`)

DuckDB-backed columnar storage at `data/ave.duckdb`. Four tables:

- `macro_release` — PK `(series_id, reference_date, release_date)`,
  enforces `CHECK release_date >= reference_date` at schema and
  application level (Patch 01).
- `asset_price` — PK `(series_id, trade_date)`, **unitemporal by design**
  (see KNOWN_LIMITATIONS.md L-6 / AUDIT_DECISIONS.md ID 3.3).
- `shares_outstanding` — bitemporal, PK `(ticker, period_end_date,
  filing_date)`, `CHECK filing_date >= period_end_date`.
- `fx_close` — auxiliary table for FX retrieval.

`open_db()` is the only sanctioned entry point; it auto-runs idempotent
migrations (currently `migrate_publication_lags`). Inserts use
`con.cursor()` with unique view names so concurrent writes on the same
connection don't deadlock (Patch 07).

### Layer 3 — LOCF (`locf.py`)

`forward_fill_to_calendar(macro, calendar_idx)` is the **only** sanctioned
synchronisation primitive. It joins macro releases onto the daily NYSE
calendar via `pd.merge_asof(direction="backward", on="release_date")`,
guaranteeing F_t-measurability: the output at calendar day `t` depends only
on rows with `release_date <= t`. Forbidden alternatives that look correct
but leak future data — `interpolate()`, `ffill()` over `reference_date`,
joins keyed on `reference_date` — are excluded by construction.

### Layer 4 — Numéraires (`numeraires/`)

Four daily indices, each anchored so `N_X(T0) = 100` (with one
documented exception — see `gold.py`):

- **N_Time** ([`numeraires/time.py`](src/rollender_stein/numeraires/time.py))
  — `AHETPI(t) / AHETPI(T0) · 100` (BLS hourly earnings of nonsupervisory
  employees, ALFRED first-release).
- **N_Energy** ([`numeraires/energy.py`](src/rollender_stein/numeraires/energy.py))
  — `(Brent/1.699)(t) / (Brent/1.699)(T0) · 100`. EIA `RBRTE` Brent spot
  divided by 1.699 to convert USD/bbl to USD/MWh, with a $0.10/MWh floor
  for numerical safety against negative-energy anomalies (Patch 03; old
  $20 floor was binding at T0 and biased the index by +36%).
- **N_Liq** ([`numeraires/liquidity.py`](src/rollender_stein/numeraires/liquidity.py))
  — **G3 Systemic Liquidity** in USD: `US_M2 + EZ_M3·EURUSD +
  JP_M3/USDJPY`. PBOC M2 is **deliberately excluded** (P3 / ID 9.M-12
  REWORKED — see KNOWN_LIMITATIONS L-3). EZ/JP M3 levels stop in 2023-11
  on FRED; `extend_levels_with_growth` splices them forward via the
  matching growth-rate series.
- **N_Gold** ([`numeraires/gold.py`](src/rollender_stein/numeraires/gold.py))
  — `XAU(t) / XAU(T0_or_first_valid) · 100` from yfinance `GC=F`. Anchors
  at T0 if available; otherwise at the first valid date (in practice
  2000-08-30, the start of GC=F). The Kalman state-space model that
  previously drove N_Gold was demoted to a Phase-4.5 diagnostic in
  Patch 06 Option C — empirically the model is identification-degenerate
  (`σ²_level / σ²_irregular ≈ 15.5`); see KNOWN_LIMITATIONS L-1 / L-11.

### Layer 5 — Per-share valuation (`valuation.py`)

`build_division_array(asset_usd, n_time, n_liq, n_gold, n_energy, ...)`
produces `Asset_in_X(t) = nominal_USD(t) / N_X(t) · 100` for each
numéraire. Output unit is **T0-deflated USD**: an asset's purchasing power
expressed in T0-anchored dollars. The function emits a `RuntimeWarning`
whenever a numéraire deviates from 100 at T0 (Patch 04) — this is how
the documented N_Gold T0 gap surfaces on every build.

### Layer 6 — Market-cap valuation (`marketcap.py`)

Parallel layer for **US single stocks** with SEC XBRL coverage (~2008+).
`build_market_cap` reconciles SEC point-in-time shares with yfinance's
split-adjusted price basis via `_cumulative_future_split_factor`:

```
adjusted_shares(t)          = raw_shares(t) · cumulative_future_split_factor(t)
market_cap(t)               = adjusted_shares(t) · yfinance_close(t)
MarketCap_in_X(t)           = market_cap(t) / N_X(t) · 100
```

The diagnostic
[`per_share_vs_market_cap_multiplier`](src/rollender_stein/marketcap.py)
returns the per-share / market-cap ratio: `> 1` indicates buybacks (AAPL ≈
2.0 — half its per-share gain came from buybacks); `< 1` indicates
dilution (TSLA ≈ 0.4 — per-share underperformed market cap by 2.5×). Out
of scope: indexes, ETFs, futures, crypto, foreign issuers.

### Layer 7 — Patterns + Volume (`patterns.py`, `volume.py`)

- **`volume.py`** — `compute_dollar_turnover(raw_close, volume)`
  (split-cancellation: yfinance retroactively split-adjusts both Close
  and Volume in opposite directions, so their product cancels splits and
  yields true historical USD turnover). `rolling_volume_zscore` computes
  a strictly-trailing 252-day z-score of `log(turnover)` with `min_periods=63`.
- **`patterns.py`** — `compute_valuation_z_scores`,
  `compute_correlation_matrix`, `compute_kalman_innovation_diagnostics`.
  Descriptive only — explicitly NOT predictive signals (see module
  docstring + CLAUDE.md "What this codebase is NOT").

### Layer 8 — Output (`dashboard.py`, `persist.py`, `audit.py`)

- **`dashboard.py`** — interactive 3D Plotly figure encoding **six**
  dimensions: three deflated axes, time as Viridis line color, nominal
  USD as marker size, dollar-turnover z-score as marker opacity (the
  "conviction channel"), plus an opt-in recency-fade overlay
  (`final_alpha = volume_alpha · recency_factor`). RGBA color strings
  bake alpha per marker (Plotly 3D's scalar `marker.opacity` limitation).
- **`persist.py`** — writes parquet artefacts under `data/derived/`:
  `numeraires/`, `panels/`, `kalman/` (with self-describing
  `innovation_summary` block in `params.json` — see Patch P2 calibration
  baseline in AUDIT_DECISIONS.md), `divisions/`, plus a `manifest.json`
  inventory. `dump_all_artifacts` caches a single `fit_gold_model` call
  across the whole run (Patch 05; previously fit 3× per invocation).
- **`audit.py`** — `truncation_hash_audit` is a **software guard** against
  future code drift (e.g. accidental `filtered_state` →
  `smoothed_state` substitution), not a theoretical Kalman test. ID
  12.M-6 REWORKED — see KNOWN_LIMITATIONS L-5.

### Layer 9 — Refresh engine (`run.py`, `cli.py`, `runconfig.py`)

Background-engine MVP for unattended refreshes:

- **`runconfig.py`** — `RefreshConfig` dataclass, YAML-loaded. Only
  user-specific items are configurable (Yahoo tickers, SEC filers,
  dashboard list, skip flags); macro series IDs are hardcoded in
  `numeraires/*.py` to prevent silent methodological drift.
- **`run.py`** — `refresh(config, dry=...)` orchestrates
  `macro_ingest → asset_ingest → sec_ingest → dump_all_artifacts → dashboards`
  with per-step error tolerance. Returns a `RunResult` summary.
- **`cli.py`** — `ave refresh --config <path>` argparse entry point with
  `--dry`, `--no-macro`, `--no-assets`, `--no-sec`, `--no-dashboards`.
  Module form: `python -m rollender_stein refresh ...`.

CI (`.github/workflows/ci.yml`) runs the three-tool gate (pytest +
mypy --strict + ruff) on every push to `main` and every PR. The refresh
pipeline itself is **not** scheduled in Actions — DuckDB needs persistent
storage; ephemeral runners would lose vintage history every run. Run
refresh locally via cron / launchd / systemd timer; CI verifies the
code, the local machine keeps the data.

## Bitemporal semantics

Every macro observation carries two timestamps:

- **`reference_date`** — the period the value describes (e.g. "January
  2024" for monthly AHETPI).
- **`release_date`** — the day the public learned the value (e.g.
  2024-02-02 for the January AHETPI release).

LOCF on `release_date` (not `reference_date`) is what guarantees
F_t-measurability. Per-series `PUBLICATION_LAG_BD` corrects the
common case where the live FRED endpoint returns observations whose
`reference_date` and effective publication date differ by a known offset.

## F_t-measurability table

| Pipeline point                         | F_t-measurable? |
|----------------------------------------|:---:|
| Ingest with `PUBLICATION_LAG_BD`       | ✓ |
| LOCF on `release_date`                 | ✓ |
| Numéraire construction (all four)      | ✓ |
| Per-share `Asset_in_X` division        | ✓ |
| Market-cap `MarketCap_in_X` division   | ✓ |
| Volume conviction channel (252-d strictly trailing) | ✓ |
| Kalman state recursion at frozen params | ✓ |
| **Kalman MLE parameter vector**        | **✗ — full-panel fit** |

The **only** pipeline point that is not F_t-measurable is the Kalman MLE
parameter vector itself (`fit_gold_model` runs MLE on the full panel, so
`θ̂` is F_T-measurable, hence `μ_t(θ̂)` peeks at the future via the
parameters). Since Patch 06 Option C, the Kalman is no longer on the
N_Gold pipeline path — it lives purely as a Phase-4.5 diagnostic. The
diagnostic outputs (`filtered_state.parquet`, `innovations.parquet`) are
explicitly documented as non-causal (see KNOWN_LIMITATIONS L-1 + the
formal `§6 Causality` discussion in AUDIT_DECISIONS.md).

## T0 anchor

All four numéraires (and all derived `Asset_in_X` and `MarketCap_in_X`
values) are gauged on **`T0 = 2000-01-03`**. The choice is conventional —
the first NYSE trading day of the millennium — and affects only the level
of the indices, not the shape of the trajectories. Three numéraires
(`N_Time`, `N_Energy`, `N_Liq`) anchor exactly at 100; **`N_Gold` has a
documented 8-month T0 gap** because GC=F starts only on 2000-08-30.

The gap surfaces via a `RuntimeWarning` on every `build_division_array`
call (Patch 04). See KNOWN_LIMITATIONS L-2.

## How the layers realize URTRIF v3.0

URTRIF v3.0 specifies three sequential operations and their numerical
discipline ([`URTRIF.md`](URTRIF.md) §2 + §3). The mapping onto AVE
layers:

| URTRIF step | Where in AVE | Difference / extension |
|---|---|---|
| **1. Corporate actions** (splits + dividends, day-exact) | `valuation.py` reads `prefer_adjusted=True` (yfinance `adj_close`), which already incorporates splits + dividend-reinvestment retroactively | URTRIF's day-of-split rule is more reproducible across re-ingests; AVE accepts the trade-off and documents it as KNOWN_LIMITATIONS L-6 |
| **2. FX conversion** (into base, exact cross term) | Implicit in `N_Liq` (G3 Ocean USD-equivalent uses EURUSD / USDJPY) | AVE only deflates USD-denominated assets; non-USD listings have no clean explicit FX axis. Candidate future addition |
| **3. Inflation deflation** (CPI, repainting-free) | Layer 4 numéraire division: `Asset_in_X = nominal_USD / N_X · 100` for each X | AVE replaces URTRIF's single CPI deflator with **four** parallel deflators (Time, Energy, Liquidity, Gold), each producing one of the four `Asset_in_X` axes |

URTRIF's engineering disciplines map onto AVE infrastructure:

| URTRIF property | AVE realization |
|---|---|
| **Gap Trap Fix** (`ffill()` on close + FX before shift) | LOCF via `pd.merge_asof(direction='backward', on='release_date')` in `locf.py`; asset-side `closes.reindex(base_idx, method='ffill')` in `valuation.py` |
| **Self-Anchoring** (C_t0 from first valid CPI value) | `build_n_gold` falls back to first valid date when T0 is uncovered; warned via `RuntimeWarning` in Patch 04 |
| **Numerical stability** (log + cumsum + exp instead of cumprod) | n/a — AVE doesn't compound returns; pointwise division at each `t` avoids drift trivially |
| **No look-ahead** (`merge_asof(direction='backward')`) | Bitemporal store enforces `release_date >= reference_date` at schema level (Patch 01); ALFRED `output_type=4` for vintage-aware deflators (Patch 02) |

Where AVE goes **beyond** what URTRIF specifies:

1. **First-release vintage data** via FRED ALFRED `output_type=4` —
   URTRIF's spec is silent on whether the deflator is release-dated or
   reference-dated; AVE makes it a schema invariant.
2. **Per-series publication-lag table** (`PUBLICATION_LAG_BD`) — corrects
   live-endpoint FRED series to their de-facto publication dates (Patch 02).
3. **Splice with growth rates** when level series stop publishing (e.g.
   EZ/JP M3 levels in 2023-11) — `extend_levels_with_growth`.
4. **Multi-numéraire output** — four parallel `Asset_in_X` axes form a
   3D phase-space trajectory (the fourth axis enters the visualization
   via marker color); the divergence between axes is the unit of analysis.

Where AVE **does not match** URTRIF and the gap is documented:

- Day-of-split anchoring → KNOWN_LIMITATIONS L-6 (yfinance retroactive
  adjustment); URTRIF would be more reproducible.
- Explicit FX axis for non-USD assets → no current AVE layer; URTRIF
  treats this as a first-class concern.

## Reference docs

- [`URTRIF.md`](URTRIF.md) — **canonical specification** (URTRIF v3.0)
- [`docs/URTRIF_v3.0.pdf`](docs/URTRIF_v3.0.pdf) — original specification PDF
- [`README.md`](README.md) — quickstart and forensic principles
- [`CLAUDE.md`](CLAUDE.md) — methodological depth and runbooks
- [`INVARIANTS.md`](INVARIANTS.md) — what the system guarantees
- [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) — what it deliberately doesn't
- [`CHANGELOG.md`](CHANGELOG.md) — patch and feature history
- [`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) — per-finding decision log
  (DONE / REWORKED / WON'T FIX) for the step-by-step audit baseline
  `af3956a`
