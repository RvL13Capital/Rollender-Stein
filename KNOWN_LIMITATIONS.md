# Known Limitations

What the AVE codebase **deliberately does not** guarantee — design
decisions, accepted scope boundaries, and auditor findings whose rejection
is documented. These are not bugs; raising any of them as a "fix the
codebase" ticket without first reading the cited rationale wastes effort.

Per-finding decision rationales (DONE / REWORKED / WON'T FIX) live in
[`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md). The entries below summarise
the user-facing consequence and link to the deeper discussion.

If you want to **close** one of these limitations, that is a feature PR —
write your own rationale, update [`INVARIANTS.md`](INVARIANTS.md) and the
test suite, and link the new behaviour back from `AUDIT_DECISIONS.md`.

---

## L-1 — Kalman MLE parameters are F_T-measurable, not F_t

`fit_gold_model` runs `UnobservedComponents.fit()` on the **full panel**.
The resulting parameter vector
`θ̂ = (σ²_level, σ²_irregular, β_TIPS, β_DXY, β_VIX)` is therefore
F_T-measurable (where T = end of the dataset). The filtered state
`μ_t(θ̂)` at any historical date `t < T` is computed from a `θ̂` that
has "seen" future data.

**Why it stays:** since Patch 06 Option C the Kalman is no longer on the
N_Gold pipeline path — it lives in `data/derived/kalman/` purely as a
Phase-4.5 diagnostic. Rolling-MLE on a degenerate model (see L-11) would
produce unstable degeneracy, not a cleaner estimate. Three load-bearing
reasons are enumerated in **§6 Causality** of
[`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md).

**If you want Kalman-driven signals:** fork the diagnostic, implement a
proper expanding-window MLE with explicit refit cadence + walk-forward
backtest + regime-shift detection on the parameter trajectory itself.
That's its own project, not a maintenance fix.

## L-2 — N_Gold has an 8-month gap before its anchor

yfinance `GC=F` (the only free daily gold series with multi-decade depth)
begins on **2000-08-30**. `N_Gold` anchors on the first available date,
leaving an 8-month gap (2000-01-03 → 2000-08-29) where the series is NaN.

**How it surfaces:** every `build_division_array` call emits a
`RuntimeWarning` when N_Gold is NaN at T0 (Patch 04). The warning is the
deliberate forensic signal — silencing it would mask the dimensional
asymmetry of the gold axis vs. the other three.

**Workaround for users:** a paid LBMA or ICE gold spot feed with full
T0-onward history would close the gap. Swap the loader, re-ingest under a
different `series_id`, and the bitemporal store distinguishes sources via
the `source` column.

## L-3 — PBOC is deliberately excluded from N_Liq

`N_Liq` is **G3 Systemic Liquidity** (US M2 + EZ M3·EURUSD + JP M3/USDJPY),
not "global" liquidity. China's broad money is comparable in size to the
entire G3 ocean (~$47T USD-equivalent), but PBOC data is opaque, frequently
methodology-revised, subject to capital controls and shadow-banking
aggregates, and convertible only via a heavily managed CNY rate.

**Why it stays excluded:** injecting PBOC into a forensic measurement
instrument would contaminate the otherwise-pristine numéraire with a
synthetic FX-conversion artefact whose sign and magnitude the analyst
cannot verify (REWORKED ID 9.M-12 — see
[`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) and the docstring of
[`numeraires/liquidity.py`](src/rollender_stein/numeraires/liquidity.py)).
The honest framing is "G3 Systemic", not "Global".

## L-4 — GC=F has an unquantified ~5%/year roll bias

The yfinance front-month-future continuum is not roll-adjusted. At each
contract roll (~6× per year) a discontinuity feeds straight through into
N_Gold; the cumulative drift relative to true spot is conservatively
estimated at ~5%/year.

**Why it stays:** ID 6.1 REWORKED — Patch 06 already moved the Kalman
out of the N_Gold path, so the roll bias is a **named limitation of N_Gold
itself**, not a model-residual artefact. Quantifying it precisely would
require a paid spot feed; in the meantime the limitation is honest.

**Diagnostic option:** `patterns.compute_kalman_innovation_diagnostics`
emits autocorrelation and recent-vs-alltime-σ ratios on the Kalman
innovations, which expose roll-bias as recurring residual structure.

## L-5 — `truncation_hash_audit` is a software guard, not a theoretical proof

The Phase-7 audit — fit MLE once on full panel, freeze parameters, re-run
the Kalman filter on a prefix, compare filtered states at the truncation
point — is **theorem-guaranteed to pass** for any correctly-implemented
deterministic Kalman filter. A passing run says nothing new about the
math.

**What it actually catches:** future code drift — a refactor that
substitutes `smoothed_state` for `filtered_state`, a junior engineer who
passes the wrong slice, a statsmodels upgrade that changes state-vector
layout. In those scenarios the recursion stops being a deterministic
function of `(params, data)` and the hash diverges.

**Why we keep the test:** treating a divergence as "implementation
invariant broke" (not "Kalman math is wrong") is exactly the discipline
ID 12.M-6 REWORKED asks for. The test is cheap, pinned to its scope in
the
[`audit.py`](src/rollender_stein/audit.py) docstring, and protects against
real future regressions even if the failure modes look exotic today.

## L-6 — `asset_price` is unitemporal by design

The schema stores one row per `(series_id, trade_date)` — no
`release_date` axis. yfinance retroactively re-denominates closes for
splits and re-injects dividends, but those are **unit re-denominations**,
not epistemic revisions of past facts. A bitemporal `asset_price` would
imply they are revisions, which is methodologically wrong.

**Why it stays:** properly bitemporal asset data requires a CRSP-style
paid feed (~$50K/yr), which is out of scope for AVE (REWORKED ID 3.3 —
see [`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) and
[`bitemporal.py`](src/rollender_stein/bitemporal.py) module docstring).

**Workaround for reproducible backtests:** read raw `close` (not
`adj_close`) via `get_asset_closes(prefer_adjusted=False)` and apply
your own splits/dividends overlay against the vintage you care about.
The market-cap layer already does this — `marketcap.build_market_cap`
multiplies raw close by split-adjusted shares.

## L-7 — `source` is not in the `macro_release` PK

PK is `(series_id, reference_date, release_date)`; the `source` field
trails as a regular column. If the same `series_id` were ingested from
two distinct sources, the second insert silently overwrites the first.

**Why it stays:** ID 3.2 WON'T FIX. Cross-source overwrites are
theoretical — the codebase ingests each series from exactly one canonical
source, and adding `source` to the PK would force migration of every
existing row plus complicate the LOCF query layer for no measurable
benefit. Mitigation: `source` is recorded on every row, so a future
cross-source ingestion would be obvious in audits.

## L-8 — Kalman QMLE: standard errors from statsmodels are wrong

Real gold returns are heavy-tailed (Kurtosis ≫ 3, JB rejects normality).
The local-level + linear-regression model is therefore a **Quasi-MLE**:
point estimates are consistent under regularity conditions, but the
standard errors / confidence intervals emitted by statsmodels are
mis-scaled — sometimes by orders of magnitude.

**Consequence for AVE consumers:** none, as long as the Kalman is used
only descriptively (the only sanctioned use since Patch 06). Anyone
building hypothesis tests on top of innovation autocorrelations should
use bootstrap or robust standard errors instead — none of which we
report.

## L-9 — Phase-space visualisation is qualitative, not quantitative

`build_phase_space_figure` plots `(asset_in_X, asset_in_Y, asset_in_Z)` on
orthogonal axes. Since Patches 03 / 04 / 06, the three axes are
dimensionally homogeneous (all in T0-deflated USD), but:

- **Scales differ across axes** (one axis may run 100→200, another 100→2000).
- **Distance metrics make sense intra-asset** (the same asset's trajectory
  shape over time), not inter-asset (cross-asset Euclidean distance is
  not commensurable).
- **Correlation claims** belong in `patterns.compute_correlation_matrix`
  on log-returns, not in visual inspection of the 3D plot.

The visualisation is primarily **about trajectory shape** ("does this
asset orbit a stable attractor or drift unboundedly?"). Quantitative
statements need the methods in `patterns.py`.

## L-10 — Market-cap layer scope: US single stocks with SEC XBRL only

`marketcap.build_market_cap` works only for **US-listed individual stocks
that file 10-K/10-Q with the SEC**. EDGAR's XBRL coverage starts
~2008-2009 (when XBRL became mandatory for large filers); pre-2008 share
counts must be sourced elsewhere if needed.

**Out of scope:** indexes, ETFs, futures, crypto, foreign issuers — none
have meaningful "shares outstanding" in the same point-in-time sense.
The function raises `RuntimeError` if no SEC shares data is found rather
than silently producing a misleading market cap.

**Multi-class quirks:** EDGAR returns one consolidated
`CommonStockSharesOutstanding` per CIK, so multi-class tickers (BRK-B,
GOOGL, GOOG) get the consolidated A-equivalent count. Computing
`B_class_price · consolidated_shares` is a misrepresentation; the caller
is responsible for handling such cases (see
[`io/sec_edgar.py`](src/rollender_stein/io/sec_edgar.py) module
docstring).

## L-11 — Kalman model is identification-degenerate

Production fits show `σ²_level / σ²_irregular ≈ 15.5` — the level random
walk absorbs essentially all explanatory power; the regression betas
`β_TIPS, β_DXY, β_VIX` are near-zero. This is the empirical confirmation
of audit finding 10.M-13.

**Consequence:** the "filtered core gold" story has no empirical support
on production data — `μ_t` correlates ~0.97 with raw XAU. Patch 06
(N_Gold = raw GC=F) acknowledges this directly. The Kalman remains as a
diagnostic; rolling-MLE on a degenerate model would produce unstable
parameter trajectories, not better signals (see L-1).

The empirical baseline is captured in
[`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) "P2 calibration baseline":
~340× variance ratio between filtered residuals (pre-P2) and true
innovations (post-P2). Scale-invariant metrics (`autocorr_k`,
`recent_to_alltime_std_ratio`, `last_innovation_in_recent_sigmas`) cancel
the σ-shift exactly; only consumers of absolute std/mean values need to
re-read their thresholds.

## L-12 — CI runs the code gate; data refresh runs locally

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the three-tool
gate (pytest + mypy --strict + ruff) on every push and PR. The refresh
pipeline (`ave refresh`) is intentionally **not** scheduled in GitHub
Actions — DuckDB needs persistent storage, and ephemeral runners would
lose vintage history every run.

**How to run refresh:** local cron / launchd / systemd timer (see
[`CLAUDE.md`](CLAUDE.md) "Background-engine CLI" for example crontab).
You keep the data; CI verifies the code.

## L-13 — No production monitoring, alerting, or SLA

There is no alert if FRED returns zero rows, no retry beyond the
underlying `requests` library defaults, no Slack/PagerDuty wiring on
`RunResult.n_failed > 0`, no SLA on macro-data freshness. The
`run.refresh` orchestrator records per-step status into `RunResult` so a
human (or a wrapper script) can detect failure, but no automation does so
out of the box.

This is consistent with AVE being a **forensic measurement instrument**,
not a production trading system — see CLAUDE.md "What this codebase is
NOT". A real production deployment would need a layer on top.

## L-14 — Filename sanitisation is minimal, not adversarial

`save_asset_dashboard` and the per-ticker parquet writers apply
`replace("^", "").replace("=", "-").replace("/", "-")` to ticker symbols,
which defangs path traversal (the only safety-relevant risk) but lets
whitespace and exotic punctuation pass through. AVE ingests well-formed
Yahoo symbols only, so the case is largely hypothetical. ID 14.2 WON'T
FIX — strict regex would risk breaking legitimate symbols like
`BRK-B` or `BTC-USD`.

## L-15 — N_Time captures wage-earner purchasing power, not labor productivity

`N_Time` deflates by AHETPI (US production / nonsupervisory hourly
earnings). The series captures **nominal hourly wages** for that subset
of workers — it does **not** include adjustments for labor productivity,
automation / robotics displacement, total factor productivity, or shifts
in the capital-labor share.

**Why this stays:** the omission is consistent with the AVE's broader
**consumer-side perspective** — all four numéraires measure what the
modal consumer / wage-earner / investor could buy with the asset's value
(hours of own labor, MWh of energy, share of G3 broad money, ounces of
gold). Robotics, software, and capital substitution belong to the *thing
being valued* (the asset's productive capacity), not to the *valuer's
measure*. When a robot makes the iPhone, the human still pays for it
with their salary; the robot is priced into Apple's market cap, not into
AHETPI.

**The corollary that matters:** the divergence between an asset's
N_Time-deflated trajectory and the same asset's N_Liq- or N_Gold-deflated
trajectory is *precisely* the visualization of the labor-vs-capital
share shift over the last 25 years. Deflating by productivity-adjusted
wages (ULC, output / hour, TFP-adjusted) would normalize that divergence
away — destroying exactly the story AVE is built to surface.

**If you want productivity-adjusted real returns:** N_Time is not the
right deflator — fork the diagnostic with output-per-hour or TFP-adjusted
ULC and document it as a separate layer. Do not modify `N_Time` in place.

See [`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) **§7 Perspective
Commitment** for the full formalization, including why this is treated
as load-bearing methodology rather than a data-source choice.

---

## Classification

| ID | Category | Mitigation in code | Real fix |
|---|---|---|---|
| L-1 | Methodology | Kalman is diagnostic only | Rolling/expanding-window MLE |
| L-2 | Data coverage | RuntimeWarning on every build | Paid LBMA/ICE feed |
| L-3 | Methodology / scope | Honest "G3 Systemic" framing | Out of scope by design |
| L-4 | Data quality | Acknowledged + diagnostic via patterns | Paid roll-adjusted spot feed |
| L-5 | Test scope | Test pinned to software-guard purpose | n/a |
| L-6 | Architecture | Unitemporal accepted; raw close available | CRSP-grade paid feed |
| L-7 | Architecture | Disciplinary single-source-per-series | Schema PK migration |
| L-8 | Statistics | Innovations are descriptive only | Robust / bootstrap SE |
| L-9 | Visualisation | Disciplinary use of `patterns.py` | n/a |
| L-10 | Scope | Raises rather than silently misleads | Out of scope by design |
| L-11 | Methodology | Kalman demoted; baseline pinned in P2 | Different model entirely |
| L-12 | Operations | Local cron + GitHub Actions code gate | Paid persistent CI runner |
| L-13 | Operations | RunResult exposes per-step status | Production monitoring layer |
| L-14 | UX | Cosmetic only — no security risk | Stricter regex |
| L-15 | Methodology / scope | Load-bearing perspective commitment (§7) | Out of scope by design |
