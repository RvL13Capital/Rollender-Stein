# Changelog

History of methodologically significant changes to the AVE codebase. The
authoritative per-finding decision rationale (DONE / REWORKED / WON'T FIX)
lives in [`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md); this document is the
chronological view, organised by patch / feature batch, with commit hashes
and audit IDs cross-referenced.

`git log --oneline` is the canonical record; this file is curated commentary
on top of it.

---

## Background-engine MVP (`f2d7fc1`, 2026-04)

`feat(engine): MVP background engine — CLI, orchestrator, GitHub Actions CI`

- New modules: `runconfig.py`, `run.py`, `cli.py`, `__main__.py`
- `ave refresh --config <path>` argparse entry point with `--dry`,
  `--no-macro`, `--no-assets`, `--no-sec`, `--no-dashboards`
- `RefreshConfig` YAML loader: only user-specific items configurable
  (Yahoo tickers, SEC filers, dashboard list); macro series IDs stay
  hardcoded in `numeraires/*.py` to prevent silent methodological drift
- `run.refresh` orchestrates `macro → asset → sec → dump → dashboards`
  with per-step error tolerance; returns a `RunResult` aggregating
  `(success, rows, duration, error)` per step
- `.github/workflows/ci.yml` runs the three-tool gate on every push to
  `main` and every PR
- Tests: 188 → 199 (+11 covering CLI, dry-run, skip flags, per-step
  isolation, runconfig parsing — see I-19)

## §6 Causality formalised (`28b9185`, 2026-04-28)

`docs(audit): formalize §6 Kalman parameter look-ahead as accepted limitation`

Documents in `AUDIT_DECISIONS.md` why Kalman MLE parameter look-ahead
(L-1) is an **accepted limitation, not a defect**. Three load-bearing
reasons:

1. Out of pipeline — Patch 06 made N_Gold = `raw GC=F`, Kalman is purely
   diagnostic.
2. Identification-degenerate — `σ²_level / σ²_irregular ≈ 15.5` (P2
   calibration baseline below). Rolling-MLE would produce unstable
   degeneracy, not better estimates.
3. Cost-benefit asymmetry — multiple days of work for a feature whose
   only consumer (N_Gold) doesn't depend on it.

## Recency-fade overlay (`0e7719a`, 2026-04)

`feat(dashboard): recency-fade overlay for time-bounded analysis windows`

- `build_phase_space_figure(recency_window_days=...)` enables a
  multiplicative time-decay overlay on marker opacity
- Composition rule: `final_alpha = volume_alpha · recency_factor`, hard
  floor `0.02` for trajectory continuity
- Piecewise-linear schedule: `[0, window]` → 1.0; `(window, window+fade]`
  → linear fade to `recency_floor`; older → `recency_floor`
- Tests: 182 → 188 (+6 — see I-17)

## P2 σ-baseline documentation (`96340ca`, 2026-04)

`feat(kalman): document sigma-baseline shift post-innovations-switch`

Adds the empirical σ-baseline shift after the P2 innovations switch to
both `params.json` (`innovation_summary` block) and
`AUDIT_DECISIONS.md` "P2 calibration baseline":

| Statistic | OLD filtered residual | NEW innovation | Ratio |
|---|---:|---:|---:|
| std | 1.34 | 24.68 | 18.46× |
| variance | 1.79 | 609.0 | **340.7×** |

The `innovation_summary` block makes every persisted Kalman snapshot
self-describing — comparing two runs with different baselines is then a
documented act, not a silent confusion. See I-13.

## README + CLAUDE + pyproject sync (`c7b3b9f`, 2026-04)

`docs: sync README, CLAUDE.md, pyproject after audit-fix-pass-2 + volume`

Documentation refresh post-Pass 2 + volume conviction channel.

## AUDIT_DECISIONS.md formalised (`59f0e7d`, 2026-04)

`docs(audit): explicit per-finding decision log for the step-by-step review`

Creates `AUDIT_DECISIONS.md` with the per-finding resolution status table
(42 IDs: 19 DONE / 4 REWORKED / 19 WON'T FIX) and per-WON'T-FIX
rationales, so future audits don't re-open closed decisions.

---

## Audit Resolution Pass 2 — P1–P4

Hardening pass post step-by-step review (audit baseline `af3956a`).

### P4 — Better-way docstrings (`24993c2`, 2026-04)

`docs: P4 — better-way framing for audit/yahoo/bitemporal docstrings`

- ID 3.3 (Major) REWORKED — `bitemporal.py` documents `asset_price`'s
  unitemporal scope (yfinance splits/dividends are unit re-denominations,
  not epistemic revisions) → KNOWN_LIMITATIONS L-6
- ID 6.1 (Major) REWORKED — `io/yahoo.py` GC=F docstring acknowledges
  ~5%/year roll bias as a named N_Gold limitation, not a Kalman residual
  artefact → KNOWN_LIMITATIONS L-4
- ID 12.M-6 (Major) REWORKED — `audit.py` reframes `truncation_hash_audit`
  as a software guard against future code drift, not a theoretical
  Kalman test → KNOWN_LIMITATIONS L-5

### P3 — G3 Systemic Liquidity rename (`0de9383`, 2026-04)

`docs(liquidity): P3 — rename "Global Fiat Ocean" → "G3 Systemic Liquidity"`

- ID 9.M-12 (Critical) REWORKED — auditor proposed adding PBOC + renaming
  to G3; we kept the G3 scope (PBOC excluded — opaque, methodology-
  revised, state-managed; CNY conversion injects synthetic FX noise)
  and renamed the labels to honestly describe the scope
- API symbol `build_n_liq` and Series name `N_Liq` unchanged for
  stability; only docstrings, error messages, and logging updated
- → KNOWN_LIMITATIONS L-3

### P2 — Innovations switch (`bec7a64`, 2026-04)

`fix(kalman): P2 — innovations switch (15.M-5 / 16.F-Major)`

- IDs 15.M-5 (Major), 16.F-Major (Major) — `persist.dump_kalman_outputs`
  now writes `fit.results.resid` (statsmodels' true one-step-ahead
  innovations) to `kalman/innovations.parquet`, instead of
  `XAU - filtered_state - X@beta` (filtered residuals)
- File rename: `kalman/residuals.parquet` → `kalman/innovations.parquet`
- `params.json` carries `residual_kind: "one_step_ahead_innovation"`
- `patterns.compute_kalman_innovation_diagnostics` reads the new file
- See "P2 calibration baseline" in `AUDIT_DECISIONS.md` for the
  ~340× variance shift on production data; scale-invariant downstream
  metrics are unaffected
- → INVARIANTS I-13

### P1 — Tech hygiene (`2284fdb`, 2026-04)

`fix(tech-hygiene): P1 — pandas 3.0 compat, log(0) guard, dtype hygiene`

- ID 2.2 (Major) — `pd.Timestamp.utcnow()` → `pd.Timestamp.now(tz="UTC")`
  (deprecation under pandas 3.0)
- ID 4.2 (Major) — `forward_fill_to_calendar` empty-frame handling: dodges
  pandas-3.0 `MergeError` from `<M8[s]>` vs `<M8[us]>` dtype mismatch
- ID 6.2 (Minor) — `pd.NA` → `np.nan` in OHLCV partial columns to keep
  float64 dtype (avoid object-coerce that breaks downstream arithmetic)
- ID 15.1 (Minor) — `np.log(0)` guard via `.where(s > 0)` in
  `patterns.compute_valuation_z_scores`; explicit warning surfaces
  bad-data days
- Tests 177 → 180

---

## Volume conviction channel (`1481092` + `8c066d1`, 2026-04)

`feat(volume): integrate conviction channel via rgba marker encoding`
`fix(volume): four PR-review patches — color-drift, NaN guard, WTI-neg, index API`

- New module `volume.py`: `compute_dollar_turnover` (raw close × volume,
  exploits yfinance split-cancellation), `rolling_volume_zscore`
  (252-day strictly trailing, `min_periods=63`)
- `dashboard.py`: per-marker opacity in `[0.25, 1.0]` derived from
  rolling z-score of `log(turnover)`; falls back to scalar 0.7 when
  positive-turnover coverage < 50% (catches indexes like `^SP500TR` and
  legacy ingests with sparse coverage)
- `_time_alpha_rgba` precomputes RGBA color strings exactly once on the
  full sub-sampled data; per-frame work is pure slicing — eliminates
  O(N²) total work and fixes the prior animation-frame Viridis
  re-normalisation bug (forensic-determinism violation)
- New `get_asset_volume` in `bitemporal.py`; `compute_dollar_turnover`
  inner-joins on indexes, propagates NaN, masks negative turnover
  (WTI April-2020) before `np.log`
- Tests: 14 new tests in `tests/test_volume.py` + 18 in
  `tests/test_dashboard.py` covering RGBA color generation, opacity
  fallback, animation hue stability — see I-15, I-17

## Market-cap valuation layer (`8520096`, 2026-04)

`feat(marketcap): SEC EDGAR shares-outstanding + market-cap absolute valuation`

- New module `marketcap.py`: `build_market_cap`,
  `build_market_cap_division`, `per_share_vs_market_cap_multiplier`
- New `shares_outstanding` table in DuckDB schema, PK `(ticker,
  period_end_date, filing_date)`, `CHECK filing_date >= period_end_date`
- `latest_shares_stream` two-stage dedup (earliest filing per period;
  latest period per filing) — analogous to `latest_release_stream` for
  macro
- `_cumulative_future_split_factor` reconciles SEC point-in-time shares
  with yfinance's split-adjusted close basis; without this, raw shares ×
  split-adjusted close produces market caps off by the cumulative split
  factor (28× for AAPL)
- `io/sec_edgar.py`: SEC fair-access `User-Agent` enforced; `us-gaap`
  `CommonStockSharesOutstanding` with `dei` fallback;
  `WeightedAverage*` deliberately excluded (period-average, not
  point-in-time)
- Coverage: US single stocks with SEC XBRL only (~2008+) — see
  KNOWN_LIMITATIONS L-10
- Tests: 15 new tests in `tests/test_marketcap.py` — see I-18

---

## Audit Resolution Pass 1 — Patches 01–07

Initial audit resolution pass against the empirical-verification audit
findings. Eight commits `dfe487d → 98302c9` over the campaign.

### Patch 07 — Thread-safe bitemporal inserts (`9ad95bd`)

`fix(ave/bitemporal): patch 07 — thread-safe cursor pattern in inserts`

- ID 3.4 (Major) — `con.register` is not thread-safe; concurrent inserts
  on the same connection deadlocked >10s
- Fix: `cur = con.cursor()` per call with unique view names
  (`f"_incoming_*_{id(df):x}"`); each call gets its own statement context
  within the shared connection
- Same pattern applied in `migrate_publication_lags`

### Patch 06 Option C — N_Gold via raw GC=F, Kalman demoted (`dfe487d`)

`fix(ave/gold): patch 06 Option C — N_Gold uses raw XAU, Kalman demoted`

- IDs 10.C-1 (Critical), 10.M-13 (Major) — auditor's "C-1 anchor break"
  finding showed N_Gold was anchoring at ~2006 (DXY's start), not T0;
  10.M-13 showed the Kalman model is identification-degenerate
  (`σ²_level / σ²_irregular ≈ 15.5`)
- **Three options were considered:**
  - A: Fix the Kalman (rolling-MLE, regime-switching) — rejected because
    rolling-MLE on a degenerate model produces unstable degeneracy
  - B: Use TIPS-implied real gold (TIPS only since 2003) — rejected
    because anchor would still miss T0
  - **C: Use raw GC=F, demote Kalman to Phase-4.5 diagnostic** — chosen
    because it's the only option that restores dimensional homogeneity
    on the gold axis without inventing a synthetic anchor
- N_Gold = `raw GC=F / GC=F(2000-08-30) · 100`, leaving an 8-month T0
  gap surfaced via Patch 04's `RuntimeWarning` (KNOWN_LIMITATIONS L-2)
- Kalman pipeline (`fit_gold_model`, `dump_kalman_outputs`) preserved
  for diagnostic use; no longer participates in any numéraire

### Patch 05 — Kalman caching + MLE convergence guard (`5e0cabb`)

`fix(ave/persist+gold): patch 05 — cache Kalman fit + MLE convergence guard`

- ID 16.M-4 (Major) — `dump_all_artifacts` was running `fit_gold_model` 3×
  per call (via `dump_numeraires` → `dump_kalman_outputs` → per-ticker
  refit). New helpers thread a single cached `GoldFit` through to
  `dump_phase4_panel` and `dump_kalman_outputs`
- ID 10.M-7 (Major) — `fit_gold_model` now raises `RuntimeError` if the
  MLE optimisation does not converge:
  `assert results.mle_retvals.get("converged", True)`
- → INVARIANTS I-12

### Patch 04 — Zero-guard, T0 invariant, N_Gold NaN warning (`4bff54a`)

`fix(ave/valuation): patch 04 — defensive guards on T0 invariant + zero denom`

- ID 11.2 (Minor) — `nominal / num.reindex(base_idx)` produced silent ±∞
  when a numéraire was zero. Fix: `denom.where(denom != 0)` so NaN
  propagates instead → INVARIANTS I-9
- ID 11.C-2 (Critical) — `_ratio` did not verify the T0=100 invariant.
  Fix: `_ratio` now emits a `RuntimeWarning` when a numéraire deviates
  from 100 at T0 (NaN / off-by-tolerance / missing observation), with a
  configurable `t0_invariant_tol` (default `1e-6`) → INVARIANTS I-3
- N_Gold's documented 8-month T0 gap surfaces via the same warning path

### Patch 03 — Energy floor lowered to $0.10/MWh (`cf51cee`)

`fix(ave/energy): patch 03 — lower MWh floor to $0.10 (was $20)`

- ID 8.1 (Critical) — old `MWH_PRICE_FLOOR_USD = 20.0` was binding at
  the T0 anchor (Brent T0 = $24.93/bbl → raw MWh = $14.67, well below
  $20). Anchoring on the floor instead of the true cost biased the
  entire N_Energy index by +36% on every non-floor-binding date.
- Fix: floor → $0.10/MWh (numerical safety net only — never binds on
  any historically observed Brent level; Brent's all-time low was
  ~$9/bbl → $5.30/MWh, far above $0.10)
- Forensic guard: `RuntimeWarning` if the floor ever binds at T0 →
  INVARIANTS I-10

### Patch 02 — Per-series PUBLICATION_LAG_BD + idempotent migration (`22206c2`)

`fix(ave/io+bitemporal): patch 02 — publication-lag table + idempotent migration`

- IDs 5.1 (Critical), 5.2 (Critical) — pre-fix `release_date == reference_date`
  for ALL live-endpoint series, embedding a same-day-to-multi-week
  look-ahead in the LOCF stream (e.g. monthly M3 reference_date had
  ~30-day publication lag in reality)
- New `PUBLICATION_LAG_BD` table in `io/fred.py` with conservative
  business-day offsets per series (1 BD for daily yields/FX, 5 BD for
  weekly H.6 WM2NS, 30 BD for monthly OECD MEI broad-money series, 0
  for VIX which publishes at session close)
- New `migrate_publication_lags` runs idempotently on every `open_db`,
  scoped to listed series with `release_date == reference_date`,
  collision-safe (warns + skips on PK clashes with existing
  ALFRED-anchored rows)
- → INVARIANTS I-5, I-8

### Patch 01 — Bitemporal release validation (`4b2ad3f`)

`fix(ave/bitemporal): patch 01 — release_date >= reference_date invariant`

- ID 3.1 (Major) — `macro_release` schema had no constraint on the
  temporal ordering of `(reference_date, release_date)`, allowing
  synthetic look-ahead injection via the application API
- Fix: `CHECK release_date >= reference_date` in `_SCHEMA_DDL` for
  fresh DBs; `_validate_release_after_reference` at the
  `insert_macro_releases` boundary for pre-existing DBs (DuckDB does
  not yet support `ALTER TABLE ADD CONSTRAINT CHECK`)
- → INVARIANTS I-6

### Patch 01–07 follow-ups (`98302c9`, `9ef6c6e`, `f1b00c3`)

- `feat(ave): three followups from the audit-fix self-review` —
  defensive followups discovered during patch self-review
- `fix(ave/bitemporal): surface migration failures via warning instead
  of swallow` — migration `Exception` no longer silently absorbed via
  `contextlib.suppress`; emits `RuntimeWarning` with diagnostic context
  while keeping the DB usable (next `open_db` retries idempotently)
- `chore(ave): cosmetic fixes from audit (B-Minor, m-1, m-13)` —
  ID 13.1 (B-Minor) hover template `$xxxx.xx (T0-USD)` instead of
  multiplier-suggesting `xxxx.xxx`, plus `m-1` T0 timestamp comment and
  `m-13` env.example sync

---

## Pre-audit baseline

The early development phases (Phase 1 bitemporal store, Phase 2 LOCF,
Phase 3 numéraires, Phase 4 Kalman diagnostic, Phase 5 valuation, Phase 6
dashboard, Phase 7 truncation-hash audit) landed in commits `e5f93d7` →
`af3956a`. See `git log --oneline` for the chronological scaffold.

The descriptive pattern extraction (`af3956a`, `feat(patterns)`) marks the
boundary between scaffold and audit campaign — `AUDIT_DECISIONS.md`
treats `af3956a` as the **audit baseline** and pins each finding's
resolution against it.

---

## Test-count progression

| Milestone | Tests | Δ |
|---|---:|---:|
| Pre-audit scaffold (`af3956a`) | ~94 | — |
| Patches 01–07 + follow-ups | ~177 | +83 |
| P1 tech hygiene | 180 | +3 |
| P2 / P3 / P4 + market-cap | 182 | +2 |
| Volume conviction channel | 188 | +6 |
| Background engine MVP | **199** | +11 |

199 tests, all green; mypy --strict and ruff clean; CI gate enforced on
every push.
