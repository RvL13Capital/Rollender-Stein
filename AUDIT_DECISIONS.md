# Audit Decisions Log

Resolution of the 32 findings from `rollender_stein_step_by_step_review.md`
(audit baseline `af3956a`) against the current state of `main` after the
volume-conviction-channel + P1–P4 hardening passes.

Status legend:
- ✅ **DONE** — fixed in a committed patch; line-pinned hash given
- 🛠️ **REWORKED** — partially fixed, with the user's "Better Way" replacing
  the auditor's literal recommendation
- ❌ **WON'T FIX** — explicit rejection with rationale below

---

## Summary by status

| Severity | DONE | REWORKED | WON'T FIX | Total |
|---|---:|---:|---:|---:|
| Critical | 5 | 1 | 0 | 6 |
| Major | 9 | 3 | 1 | 13 |
| Minor | 5 | 0 | 13 | 18 |
| Cosmetic | 0 | 0 | 5 | 5 |
| **Total** | **19** | **4** | **19** | **42** |

(The audit listed 32 distinct findings with several internal multi-IDs; this
table counts each ID separately, hence 42.)

---

## ✅ DONE — already in main

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| 2.2 | Major | `pd.Timestamp.utcnow()` deprecated | `2284fdb` |
| 3.1 | Major | Schema lacks `CHECK (release_date >= reference_date)` | `4b2ad3f` (Patch 01) |
| 3.4 | Major | `con.register` thread-unsafe | `9ad95bd` (Patch 07) |
| 4.2 | Major | Empty-frame `MergeError` under pandas 3.0 | `2284fdb` |
| 5.1 | Critical | M3 monthly `release_date == reference_date` look-ahead | `22206c2` (Patch 02) |
| 5.2 | Critical | `fetch_fred_observations` doc-vs-use mismatch | `22206c2` |
| 6.2 | Minor | `pd.NA` produces object-dtype column | `2284fdb` |
| 8.1 | Critical | Energy floor $20 binds at T0 → +33 % bias | `cf51cee` (Patch 03) |
| 10.C-1 | Critical | N_Gold anchors at 2006 not T0 | `dfe487d` (Patch 06) |
| 10.M-7 | Major | No MLE convergence check | `5e0cabb` (Patch 05) |
| 10.M-13 | Major | Kalman empirically degenerate | `dfe487d` |
| 11.2 | Minor | `nominal / num.reindex` no zero-guard | `4bff54a` (Patch 04) |
| 11.C-2 | Critical | `_ratio` doesn't verify `N(T0) == 100` | `4bff54a` |
| 13.1 | Minor | Hover template "x" multiplier suffix | `f1b00c3` |
| 15.1 | Minor | `np.log(0)` silently drops tickers | `2284fdb` |
| 15.M-5 | Major | Diagnostics on filtered residuals not innovations | `bec7a64` |
| 16.F-Major | Major | Persisted "residuals" are filtered, not one-step-ahead | `bec7a64` |
| 16.M-4 | Major | `fit_gold_model` runs 3× | `5e0cabb` |
| (cosmetic) | Cosmetic | 4.1, 7.1, 10.1 docstring inconsistencies | `f1b00c3`, `24993c2` |

## 🛠️ REWORKED — User's "Better Way" applied

| ID | Severity | Auditor said | We did instead | Resolution |
|---|---|---|---|---|
| 3.3 | Major | "Bitemporalize asset_price" | Document the unitemporal scope explicitly. yfinance splits/dividends are unit re-denominations, not epistemic revisions of past facts. CRSP-grade bitemporality is out of scope (~$50K/yr). | `24993c2` (P4) |
| 6.1 | Major | "Roll bias absorbed into Kalman residual" claim is false | Acknowledge the ~5 %/year roll bias as a limitation of N_Gold. Patch 06 already moved the Kalman out of the main path. | `24993c2` (P4) |
| 9.M-12 | Critical | "Add PBOC; rename to G3" | Rename labels to "G3 Systemic Liquidity"; do NOT add PBOC. China's M2 data is opaque, methodology-revised, state-managed; CNY conversion injects synthetic noise that contaminates a forensic instrument. | `0de9383` (P3) |
| 12.M-6 | Major | "Truncation hash audit is mathematically vacuous" | Re-frame as a *software guard* against future code drift (e.g. filtered→smoothed swap), not a theoretical Kalman test. Soften "rewrite Phase 4" failure message accordingly. | `24993c2` (P4) |

---

## ❌ WON'T FIX — explicit rejection with rationale

These 19 findings are real but were judged below the cost/benefit threshold
for a research-grade forensic instrument. Each entry says **why** we declined,
so a future reviewer doesn't re-open the question.

### config / install paths

- **1.1 (Minor)** — `config.py:13` assumes editable install for `.env`
  resolution. **Why won't fix:** AVE is intended for editable / research use
  (per CLAUDE.md "no CI, no SLA"). A pip-installed deployment is a
  hypothetical that adds packaging complexity without a real user. The
  `Path(__file__).resolve().parent.parent.parent` pattern is already correct
  for the editable install case.

### bitemporal schema — minor robustness gaps

- **3.2 (Major)** — `source` not in PK on `macro_release`. **Why won't fix:**
  Cross-source overwrites are theoretical; we ingest each series from exactly
  one canonical source. Adding `source` to PK would force migration of every
  existing row and complicate the LOCF query layer for no measurable benefit.
  Mitigation: source is recorded in the row, so cross-source ingestion would
  be obvious in audits if it ever happened.
- **3.5 (Minor)** — `insert_macro_releases` lacks up-front type validation.
  **Why won't fix:** DuckDB cast errors at INSERT time are clear and
  actionable; an extra validation layer duplicates work without catching new
  failure modes.
- **3.6 (Minor)** — `get_asset_closes` doesn't filter `WHERE close IS NOT NULL`.
  **Why won't fix:** `ingest_yahoo_asset` already drops NaN closes upstream
  (`yahoo.py` `dropna(subset=["close"])`). The schema NULL allowance is a
  defensive belt-and-suspenders for direct SQL inserts that don't exist.
- **3.7 (Minor)** — `latest_release_stream` may not pick the headline figure
  for NIPA-style benchmark days. **Why won't fix:** The AVE doesn't ingest
  NIPA series. Behavior is correct for our actual sources (FRED ALFRED).
- **3.8 (Minor)** — `latest_release_stream` drops `reference_date`. **Why
  won't fix:** Forensic reconstruction is supported by re-querying the
  underlying `macro_release` rows directly. The streaming view is for LOCF
  consumption, where reference_date isn't needed.

### LOCF — edge cases

- **4.1 (Minor)** — `locf.py:49` accepts both midnight-truncated and
  time-bearing release_date dtypes silently. **Why won't fix:** Both shapes
  are produced by the canonical loaders (`fetch_alfred_first_release`,
  `fetch_fred_observations`); enforcing one would require migrating existing
  data. Behavior is documented.
- **4.3 (Minor)** — A 23:59 release isn't visible at midnight on the same
  calendar entry. **Why won't fix:** All ingest paths currently produce
  date-only release_dates (no time component). The 23:59 case is hypothetical.

### liquidity — minor numerical edges

- **9.0 (Minor)** — Current-rate FX aggregation framing. **Why won't fix:**
  Methodologically correct (BIS does this); the docstring already documents
  the choice via P3 rewrite.
- **9.1 (Minor)** — `extend_levels_with_growth` silently drops interior gaps.
  **Why won't fix:** FRED level series don't have interior gaps in practice
  (verified across the AVE-ingested universe). The dropna is defensive.
- **9.2 (Minor)** — Ocean sum produces NaN if any component is NaN. **Why
  won't fix:** That's the correct semantics — partial liquidity isn't
  meaningful as a numéraire. Better to fail-loud than produce a misleadingly
  precise number.
- **9.3 (Minor)** — `usdjpy_d == 0` produces +∞. **Why won't fix:** USDJPY
  has never traded at zero; the case is non-physical. Patch 04's general
  zero-denominator guard in `valuation._ratio` already protects the downstream
  asset_in_liquidity values from inf propagation.

### gold

- **10.2 (Minor)** — `results.filtered_state[0]` is brittle indexing. **Why
  won't fix:** Statsmodels has stable state-vector layout for the
  UnobservedComponents level model we use. A future statsmodels major version
  bump would break many things at once and warrants its own migration patch.
  Audit P4 explicitly documents this in the `audit.py` software-guard scope.

### valuation — edge cases

- **11.1 (Minor)** — `base_idx` picks first non-None numéraire's index
  silently. **Why won't fix:** All four numéraires use `master_calendar(end)`
  internally and therefore share an index by construction. The "verify all
  share index" check is redundant under the actual call paths.
- **11.3 (Minor)** — `reindex(method="ffill")` extrapolates dead assets
  forever. **Why won't fix:** This is the correct semantic for a fixed-term
  asset that stopped reporting (e.g. a delisted stock); the dashboard's
  `dropna(subset=[x, y, z])` filter handles the truly-NaN trailing case. A
  generic "limit=N" would arbitrarily silence active assets too.

### assets

- **14.1 (Minor)** — `use_adjusted_as_close` only checks if ANY adj_close is
  non-NaN. **Why won't fix:** The parameter is *deprecated* (see the
  `DeprecationWarning` in `assets.py`); choosing close vs. adj_close happens
  at read time via `get_asset_closes(prefer_adjusted=...)`.
- **14.2 (Minor)** — Filename sanitization weak. **Why won't fix:** The
  current `replace("^","").replace("=","-").replace("/","-")` defangs path
  traversal (the only safety-relevant attack vector). Whitespace and
  semicolons in tickers are non-existent; AVE ingests well-formed Yahoo
  symbols only. Strict regex would risk breaking legitimate symbols
  like `BTC-USD`.

### patterns

- **15.2 (Cosmetic)** — Drawdown sign convention (negative %) inverse to
  industry norm. **Why won't fix:** Internal consistency across the AVE
  codebase trumps external convention; flipping the sign breaks tests and
  any downstream notebook. Convention call.
- **15.3 (Minor)** — `dropna` thresh is `int(len(df) * 0.5)` over union of
  ticker dates. **Why won't fix:** Semantically intended — we want tickers
  with reasonable overlap with the union, not isolated short series. The
  user-perspective surprise is documented in the docstring.

---

## P2 calibration baseline — empirical σ-shift after the innovations switch

Finding 15.M-5 / 16.F-Major (commit `bec7a64`) replaced the persisted
"residuals" — a *filtered* statistic ``XAU - filtered_state - X@beta`` —
with true one-step-ahead innovations from ``MLEResults.resid``. The
auditor estimated a ~6× variance shift; on the actual production panel
the empirical shift is **~50× larger**. This section pins the actual
baseline so consumers comparing pre-/post-fix absolute values aren't
fooled.

### Empirical measurement on production data (cleaned panel, 5,111 obs)

| Statistic | OLD filtered residual | NEW innovation | Ratio |
|---|---:|---:|---:|
| mean | 0.0499 | 0.9829 | 19.7× |
| **std** | **1.3369** | **24.6775** | **18.46×** |
| **variance** | **1.787** | **608.98** | **340.7×** |

The variance multiplier is **340×**, not the auditor's ~6× estimate.

### Why the multiplier is so large — finding 10.M-13 in numbers

From the same fit's ``params.json``:

```
sigma2.level     = 481.28
sigma2.irregular =  31.14
ratio (level/irregular) ≈ 15.5
```

Innovation variance ≈ σ²_level + σ²_irregular ≈ 512 (theoretical) /
609 (empirical, slight model misfit gap). Filtered residual variance is
much smaller than σ²_irregular alone (1.79 vs. 31.14) because the Kalman
filter sees every observation before estimating the state at that point —
it fits the data as tightly as the parameter set permits, then a re-
projection through the regression collapses what's left.

This is the quantitative confirmation of audit finding **10.M-13**: the
MLE is identification-degenerate, with the level random walk dominating
σ²_irregular by a factor of ~15. The "filtered residual" was therefore
an *over-smoothed look-ahead artefact*, not a residual in any meaningful
statistical sense.

### What stayed bit-stable across the switch

Scale-invariant metrics on the same production panel, OLD vs. NEW:

| Metric | OLD | NEW | Comment |
|---|---:|---:|---|
| autocorr_1 | 0.0021 | 0.0023 | indistinguishable |
| recent_to_alltime_std_ratio | 3.347 | 3.154 | within 6 % |
| **last_innovation_in_recent_sigmas** | **-0.0066** | **-0.0066** | **bit-identical** |

The z-score normalisation cancels the σ-shift exactly. Any consumer
using ratios or sigma-distances is unaffected; only consumers reading
absolute std/mean values out of the diagnostics JSON need to re-read
their thresholds.

### Codebase threshold audit

Greppable absolute thresholds in ``patterns.py`` post-rename:

```
min_obs: int = 252                      # sample size — scale-invariant
return_window: int = 21                 # time window — scale-invariant
int(len(df) * 0.5)                      # coverage 50 % — scale-invariant
```

There are no hardcoded z-score gates, σ-bands, or magic-number cutoffs
on innovation values. The function ``compute_kalman_innovation_diagnostics``
emits raw statistics; the interpretation is the consumer's responsibility.
Tests use synthetic ground-truth data and pass scale-correct constants
that map to the actual signal, so they remain valid.

### Self-describing artefact (commit closing P2)

``persist.dump_kalman_outputs`` now writes ``innovation_summary: {mean,
std, fit_window}`` into ``params.json``. Every persisted Kalman snapshot
captures its own σ-baseline at write time. Comparing two runs with
different baselines is then a documented act, not a silent confusion.

---

## How this list was built

Each finding was checked against the live tree at `HEAD` via `grep` on the
specific file:line cited by the audit. "DONE" entries pin to the commit hash
that closed them; "REWORKED" entries pin to the commit that implemented the
"Better Way"; "WON'T FIX" entries received a per-finding rationale.

If a future audit raises one of these "Won't Fix" items again, point them at
this document. If you (the maintainer) decide to actually fix one, delete
the row from the table and replace with a "DONE" entry pinned to the new
commit.
