# Invariants

What the AVE codebase **guarantees**. Each entry below is a property the
test suite verifies; the cited tests are the contract. A pull request that
modifies code touched by an invariant must either keep the cited tests
green or **explicitly change the invariant** (update this document, the
tests, and link to the rationale in [`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md)
or [`CLAUDE.md`](CLAUDE.md)).

Test counts as of `HEAD`: **199 tests**, all green
([`.github/workflows/ci.yml`](.github/workflows/ci.yml) enforces
`pytest -q && mypy --strict src && ruff check src tests` on every push and PR).

---

## I-1 — Numéraire T0 anchor (with documented N_Gold gap)

Three of the four numéraires anchor exactly at `T0 = 2000-01-03` to within
machine epsilon. `N_Gold` is documented as having an 8-month gap (GC=F
starts 2000-08-30); the gap surfaces via a `RuntimeWarning` rather than a
silent miscalibration.

- `N_Time(T0) = 100` — [`tests/test_numeraire_time.py::test_n_time_is_exactly_100_at_t0`](tests/test_numeraire_time.py)
- `N_Energy(T0) = 100` — [`tests/test_numeraire_energy.py::test_n_energy_is_exactly_100_at_t0`](tests/test_numeraire_energy.py)
- `N_Liq(T0) = 100` — [`tests/test_numeraire_liquidity.py::test_n_liq_is_exactly_100_at_t0`](tests/test_numeraire_liquidity.py)
- `N_Gold` anchors at T0 when XAU is present —
  [`tests/test_numeraire_gold.py::test_build_n_gold_anchors_at_t0_exactly_when_xau_present`](tests/test_numeraire_gold.py)
- `N_Gold` falls back to first valid date when T0 uncovered —
  [`tests/test_numeraire_gold.py::test_build_n_gold_anchors_at_first_valid_when_t0_uncovered`](tests/test_numeraire_gold.py)
- N_Gold T0 NaN surfaces as warning in `valuation` —
  [`tests/test_valuation.py::test_warning_fires_when_numeraire_is_nan_at_t0`](tests/test_valuation.py)

## I-2 — Missing T0 source data raises (no silent NaN)

If a numéraire's source data does not cover T0, the builder raises
`RuntimeError`. This guards against accidentally producing a numéraire
that silently anchors at a different date.

- [`tests/test_numeraire_time.py::test_build_n_time_raises_when_t0_unanchored`](tests/test_numeraire_time.py)
- [`tests/test_numeraire_energy.py::test_n_energy_raises_when_t0_unanchored`](tests/test_numeraire_energy.py)
- [`tests/test_numeraire_liquidity.py::test_n_liq_raises_when_t0_unanchored`](tests/test_numeraire_liquidity.py)
- [`tests/test_numeraire_gold.py::test_build_n_gold_raises_when_xau_not_ingested`](tests/test_numeraire_gold.py)

## I-3 — Dimensional homogeneity of `Asset_in_X`

`Asset_in_X(t) = nominal_USD(t) / N_X(t) · 100` is in T0-deflated USD on
every axis where `N_X(T0) = 100`. Deviations from the T0=100 invariant
trigger an explicit `RuntimeWarning` (Patch 04) — this is how the
documented N_Gold T0 gap surfaces on every `build_division_array` call.

- `Asset_in_X(T0)` equals nominal when N_X(T0)=100 —
  [`tests/test_valuation.py::test_asset_in_x_at_t0_equals_nominal_when_n_x_is_100`](tests/test_valuation.py)
- No warning when all numéraires anchored —
  [`tests/test_valuation.py::test_no_warning_when_all_numeraires_anchored_at_100_at_t0`](tests/test_valuation.py)
- Warning when numéraire ≠ 100 at T0 —
  [`tests/test_valuation.py::test_warning_fires_when_numeraire_not_100_at_t0`](tests/test_valuation.py)
- Warning when numéraire has no T0 observation —
  [`tests/test_valuation.py::test_warning_fires_when_numeraire_has_no_t0_observation`](tests/test_valuation.py)
- `t0_invariant_tol` parameter silences warning when loose / disables when inf —
  [`tests/test_valuation.py::test_t0_invariant_tol_silences_warning_when_loose`](tests/test_valuation.py),
  [`tests/test_valuation.py::test_t0_invariant_tol_inf_disables_warning`](tests/test_valuation.py)

## I-4 — F_t-measurability of LOCF

The LOCF output at calendar day `t` depends only on rows with
`release_date <= t`. Pre-release rows are NaN, post-release rows hold the
most-recent-release value. This is enforced by
`pd.merge_asof(direction="backward", on="release_date")` in
[`locf.py`](src/rollender_stein/locf.py); forbidden alternatives
(`interpolate`, `ffill` over reference_date, joins keyed on
reference_date) are rejected by construction.

- [`tests/test_locf.py::test_value_appears_only_on_or_after_release`](tests/test_locf.py)
- [`tests/test_locf.py::test_january_rows_have_no_january_reference_value`](tests/test_locf.py)

## I-5 — Per-series publication lag

`fetch_fred_observations` applies a per-series `PUBLICATION_LAG_BD`
business-day offset to convert reference_date into a conservative
release_date. Series listed in the table get their declared lag; unlisted
series default to 0 (`release_date == reference_date`).

| Series                  | Lag (BD) | Rationale                                |
|-------------------------|---------|-------------------------------------------|
| `DFII10`, `DTWEXBGS`, `DTWEXM`, `DEXUSEU`, `DEXJPUS` | 1 | Daily H.15 / H.10 yields and FX, next-BD release |
| `VIXCLS`                | 0 | Close-of-day publication, available at session close |
| `WM2NS`                 | 5 | Weekly H.6, ~5 BD after Monday reference |
| `MABMM301EZM189S`, `MABMM301JPM189S`, and growth twins (`...M657S`) | 30 | Monthly OECD MEI broad-money aggregates |

Source: [`src/rollender_stein/io/fred.py`](src/rollender_stein/io/fred.py)
`PUBLICATION_LAG_BD`. Tests:

- [`tests/test_fred_io.py::test_fetch_fred_observations_applies_publication_lag`](tests/test_fred_io.py)
- [`tests/test_fred_io.py::test_fetch_fred_observations_applies_30bd_lag_for_monthly_aggregates`](tests/test_fred_io.py)
- [`tests/test_fred_io.py::test_fetch_fred_observations_sets_release_to_reference_for_unknown_series`](tests/test_fred_io.py)

## I-6 — Schema enforces `release_date >= reference_date`

The `macro_release` and `shares_outstanding` schemas both reject inserts
where the publication date precedes the period being described. The
constraint is enforced both at the schema level (`CHECK` in `_SCHEMA_DDL`
for fresh DBs) and at the application level
(`_validate_release_after_reference` for pre-existing DBs that DuckDB
cannot retro-constrain via `ALTER TABLE ADD CONSTRAINT`).

- [`tests/test_bitemporal.py::test_release_after_reference_invariant_blocks_bad_insert`](tests/test_bitemporal.py)
- [`tests/test_bitemporal.py::test_release_equal_to_reference_is_allowed`](tests/test_bitemporal.py)
- [`tests/test_bitemporal.py::test_release_after_reference_is_allowed`](tests/test_bitemporal.py)
- [`tests/test_bitemporal.py::test_release_after_reference_validation_message_lists_offending_rows`](tests/test_bitemporal.py)
- shares variant —
  [`tests/test_marketcap.py::test_insert_shares_rejects_filing_before_period`](tests/test_marketcap.py)

## I-7 — Idempotent inserts on the bitemporal store

`insert_macro_releases`, `insert_asset_prices`, and
`insert_shares_outstanding` are idempotent: calling the same insert twice
with the same data leaves the store in the same state (`INSERT OR REPLACE`
on the PK).

- [`tests/test_bitemporal.py::test_insert_or_replace_overwrites_same_pk`](tests/test_bitemporal.py)
- [`tests/test_assets.py::test_insert_or_replace_asset_prices`](tests/test_assets.py)
- [`tests/test_marketcap.py::test_insert_shares_outstanding_round_trip`](tests/test_marketcap.py)

## I-8 — `migrate_publication_lags` is idempotent and collision-safe

The schema migration applied on every `open_db` call only touches rows
that have not yet been migrated, skips PK collisions with a warning rather
than aborting the whole series, and does not mutate ALFRED-anchored rows.

- [`tests/test_bitemporal.py::test_migrate_publication_lags_updates_listed_series`](tests/test_bitemporal.py)
- [`tests/test_bitemporal.py::test_migrate_publication_lags_is_idempotent`](tests/test_bitemporal.py)
- [`tests/test_bitemporal.py::test_migrate_publication_lags_does_not_touch_unlisted_series`](tests/test_bitemporal.py)
- [`tests/test_bitemporal.py::test_migrate_publication_lags_warns_on_pk_collision_and_skips`](tests/test_bitemporal.py)
- [`tests/test_bitemporal.py::test_migrate_publication_lags_collision_does_not_delete_other_rows`](tests/test_bitemporal.py)
- [`tests/test_bitemporal.py::test_migrate_publication_lags_preserves_alfred_anchored_rows`](tests/test_bitemporal.py)

## I-9 — `Asset_in_X` boundary behaviour: zero numéraire → NaN, not ±∞

Where `N_X(t) = 0` or NaN, `Asset_in_X(t)` is NaN. Implemented by
`denom.where(denom != 0)` in `valuation._ratio` (Patch 04 — guards against
silent ±∞ propagation downstream).

- [`tests/test_valuation.py::test_zero_in_numeraire_produces_nan_not_inf`](tests/test_valuation.py)
- [`tests/test_valuation.py::test_asset_in_x_is_nan_where_numeraire_is_nan`](tests/test_valuation.py)

## I-10 — Energy floor does not bind on real Brent levels

`MWH_PRICE_FLOOR_USD = 0.10` is constructed to never bind on any
historically observed Brent price (Brent's all-time low was ~$9/bbl →
$5.30/MWh, far above $0.10). If the floor ever binds at the T0 anchor, a
forensic `RuntimeWarning` is emitted (Patch 03 condition).

- Floor passes through real values —
  [`tests/test_numeraire_energy.py::test_n_energy_applies_mwh_floor`](tests/test_numeraire_energy.py)
- Real April-2020 Brent is not clipped —
  [`tests/test_numeraire_energy.py::test_n_energy_does_not_clip_real_april_2020_brent`](tests/test_numeraire_energy.py)
- Floor binding at T0 emits warning —
  [`tests/test_numeraire_energy.py::test_n_energy_warns_when_floor_binds_at_t0`](tests/test_numeraire_energy.py)
- Real Brent does not trigger warning —
  [`tests/test_numeraire_energy.py::test_n_energy_floor_does_not_warn_on_real_brent_levels`](tests/test_numeraire_energy.py)
- Brent → MWh divisor is 1.699 —
  [`tests/test_numeraire_energy.py::test_brent_normalization_uses_correct_divisor`](tests/test_numeraire_energy.py)

## I-11 — N_Liq splice identity (`extend_levels_with_growth`)

When EZ/JP M3 levels stop on FRED (2023-11), the growth-rate series carry
the index forward by compounding `(1 + g/100)` per step. The splice is
mathematically equivalent to the underlying BIS broad-money level had it
been published continuously; growth observations on or before the last
level point are ignored.

- [`tests/test_numeraire_liquidity.py::test_extend_compounds_after_last_level`](tests/test_numeraire_liquidity.py)
- [`tests/test_numeraire_liquidity.py::test_extend_ignores_growth_at_or_before_last_level`](tests/test_numeraire_liquidity.py)
- [`tests/test_numeraire_liquidity.py::test_extend_with_no_forward_growth_returns_levels`](tests/test_numeraire_liquidity.py)
- [`tests/test_numeraire_liquidity.py::test_extend_raises_on_empty_levels`](tests/test_numeraire_liquidity.py)

## I-12 — Single Kalman fit per `dump_all_artifacts` call

`dump_all_artifacts` calls `assemble_panel` and `fit_gold_model` exactly
once and threads the cached results through to `dump_phase4_panel`,
`dump_kalman_outputs`, and the per-ticker `build_division_array` calls
(Patch 05; previously fit 3× per invocation).

- [`tests/test_persist.py::test_dump_all_artifacts_caches_kalman_fit`](tests/test_persist.py)
- [`tests/test_persist.py::test_dump_kalman_outputs_uses_precomputed_fit`](tests/test_persist.py)
- [`tests/test_persist.py::test_dump_phase4_panel_uses_precomputed_panel`](tests/test_persist.py)

## I-13 — Kalman diagnostic emits true innovations + self-describing baseline

`dump_kalman_outputs` writes `fit.results.resid` (statsmodels' true
one-step-ahead innovations) to `innovations.parquet`, **not** filtered
residuals (Patch P2 / IDs 15.M-5 + 16.F-Major). `params.json` carries an
`innovation_summary` block (`{mean, std, fit_window}`) so every persisted
snapshot is self-describing — see
[`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) "P2 calibration baseline" for
the empirical σ-shift (~340× variance ratio).

- [`tests/test_persist.py::test_innovations_file_matches_statsmodels_resid`](tests/test_persist.py)
- [`tests/test_persist.py::test_params_json_carries_innovation_summary`](tests/test_persist.py)

## I-14 — LOCF is robust to empty / pathological inputs

`forward_fill_to_calendar` returns an all-NaN frame when given an empty
macro frame (rather than raising the pandas-3.0 `MergeError` from a
`<M8[s]>` vs `<M8[us]>` dtype mismatch — Patch P1). Duplicate release
dates, missing release column, and non-datetime release dtypes are
explicitly rejected.

- [`tests/test_locf.py::test_forward_fill_handles_empty_macro_frame`](tests/test_locf.py)
- [`tests/test_locf.py::test_duplicate_release_dates_rejected`](tests/test_locf.py)
- [`tests/test_locf.py::test_missing_release_column_rejected`](tests/test_locf.py)
- [`tests/test_locf.py::test_release_column_must_be_datetime`](tests/test_locf.py)

## I-15 — Volume conviction channel is strictly non-look-ahead

`rolling_volume_zscore` uses a strictly trailing 252-day window with
`min_periods=63`. Zeros and negatives in turnover (e.g. WTI April-2020
negatives) are masked to NaN before `np.log` so a single bad-data day
cannot poison the next 252 observations with `-inf` or `NaN+RuntimeWarning`.
Constant-turnover windows yield NaN, not 0/0.

- [`tests/test_volume.py::test_zscore_no_lookahead`](tests/test_volume.py)
- [`tests/test_volume.py::test_zscore_warmup_below_min_periods_is_nan`](tests/test_volume.py)
- [`tests/test_volume.py::test_zscore_log_zero_does_not_poison_window`](tests/test_volume.py)
- [`tests/test_volume.py::test_zscore_handles_negative_turnover_silently`](tests/test_volume.py)
- [`tests/test_volume.py::test_zscore_constant_turnover_yields_nan_not_inf`](tests/test_volume.py)
- [`tests/test_volume.py::test_zscore_post_warmup_centered_around_zero`](tests/test_volume.py)
- [`tests/test_volume.py::test_zscore_rejects_invalid_windows`](tests/test_volume.py)

## I-16 — Truncation-hash software guard (frozen-params variant)

[`audit.truncation_hash_audit`](src/rollender_stein/audit.py) freezes the
MLE parameter vector from a full-panel fit, then re-runs the Kalman filter
at the frozen parameters on a prefix of the data. The filtered state at
the truncation point must match the full-panel run to ≥8 decimal places.
This is a **software guard** against future code drift (e.g. accidental
`filtered_state` → `smoothed_state` substitution), not a theoretical
Kalman test (REWORKED ID 12.M-6 — see KNOWN_LIMITATIONS L-5).

- [`tests/test_audit.py::test_truncation_audit_passes_at_midpoint`](tests/test_audit.py)
- [`tests/test_audit.py::test_truncation_audit_passes_near_end`](tests/test_audit.py)
- [`tests/test_audit.py::test_truncation_audit_passes_early`](tests/test_audit.py)

## I-17 — Animated dashboard does not silently shift historical hues

`build_phase_space_figure(animate=True)` precomputes Viridis time-color +
RGBA opacity strings exactly once on the full sub-sampled data and slices
that array per-frame. Per-frame recomputation would re-normalise the
color scale against each frame's prefix-min/max, silently shifting every
historical marker's hue as the animation extends — a forensic-determinism
violation. The pre-fix implementation had this bug; tests pin the
post-fix behaviour.

- [`tests/test_dashboard.py::test_animation_color_arrays_match_frame_lengths`](tests/test_dashboard.py)
- [`tests/test_dashboard.py::test_animation_marker_colors_stable_across_frames`](tests/test_dashboard.py)
- [`tests/test_dashboard.py::test_subsampling_preserves_full_resolution_zscore`](tests/test_dashboard.py)
- [`tests/test_dashboard.py::test_time_alpha_rgba_raises_on_nan_alpha`](tests/test_dashboard.py)

## I-18 — Market-cap layer reconciles SEC shares with yfinance basis

`build_market_cap` multiplies SEC point-in-time shares by a
cumulative-future-split factor so they live in the same basis as
yfinance's split-adjusted close. Without this correction, mixing raw SEC
shares (e.g. AAPL 888M in 2009) with yfinance close ($5.60 in 2009) would
produce market caps off by the cumulative split factor (28× for AAPL's
7-for-1 + 4-for-1 history).

- [`tests/test_marketcap.py::test_cumulative_future_split_factor_no_splits`](tests/test_marketcap.py)
- [`tests/test_marketcap.py::test_cumulative_future_split_factor_aapl_style`](tests/test_marketcap.py)
- [`tests/test_marketcap.py::test_cumulative_future_split_factor_strips_tz`](tests/test_marketcap.py)
- [`tests/test_marketcap.py::test_build_market_cap_applies_split_adjustment`](tests/test_marketcap.py)
- [`tests/test_marketcap.py::test_build_market_cap_uses_shares_locf`](tests/test_marketcap.py)
- [`tests/test_marketcap.py::test_build_market_cap_no_splits_data_falls_back_to_factor_one`](tests/test_marketcap.py)

## I-19 — Refresh engine isolates per-step failures

`run.refresh` records each step's `(success, rows, duration, error)` into
the returned `RunResult` and continues with downstream steps that don't
depend on the failed one. The `--dry` mode produces a planned step list
without touching FRED/EIA/Yahoo/SEC or the database.

- [`tests/test_run.py::test_dry_run_lists_planned_steps_without_io`](tests/test_run.py)
- [`tests/test_run.py::test_skip_flags_omit_phases`](tests/test_run.py)
- [`tests/test_run.py::test_orchestrator_records_per_step_failures_in_isolation`](tests/test_run.py)
- [`tests/test_run.py::test_sec_step_fails_loudly_on_missing_user_agent`](tests/test_run.py)
- [`tests/test_run.py::test_run_result_aggregates_duration_and_success`](tests/test_run.py)

## I-20 — Three-tool gate (CI verification)

Every push to `main` and every PR is gated by
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) running:

```
pytest -q
mypy --strict src
ruff check src tests
```

All three must succeed for the gate to pass. The same three-tool gate is
the local pre-commit discipline documented in
[`CLAUDE.md`](CLAUDE.md) "Build / test commands".

---

## Test categories

```bash
# Full suite (199 tests, all green)
pytest tests/

# Categorical:
pytest tests/test_numeraire_*.py    # I-1, I-2, I-10, I-11
pytest tests/test_valuation.py      # I-3, I-9
pytest tests/test_locf.py           # I-4, I-14
pytest tests/test_fred_io.py        # I-5
pytest tests/test_bitemporal.py     # I-6, I-7, I-8
pytest tests/test_persist.py        # I-12, I-13
pytest tests/test_volume.py         # I-15
pytest tests/test_audit.py          # I-16
pytest tests/test_dashboard.py      # I-17
pytest tests/test_marketcap.py      # I-18
pytest tests/test_run.py            # I-19
```
