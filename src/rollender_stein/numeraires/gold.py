"""N_Gold — the gold-price standard.

Phase 4 of the AVE spec. **Methodology revised after audit (patch 06, Option C).**

N_Gold is the daily gold spot price (yfinance ``GC=F`` continuous front-month
futures used as a spot proxy), normalized to 100 at the first available date.
The Kalman state-space model that previously drove ``N_Gold`` has been demoted
to a **diagnostic artifact** (Phase 4.5) — its filtered state, residuals, and
MLE parameters are still computed and persisted under ``data/derived/kalman/``,
but they no longer participate in any numéraire calculation.

Why the change: the audit's M-13 finding showed the Kalman "orthogonalization"
of XAU against (TIPS, DXY, VIX) is empirically degenerate — on real GC=F data
the filtered state μ_t correlates 0.97 with raw XAU, and the variance
decomposition gives Var(μ_t) ≈ 1.43·Var(y) with negative covariance against
the regression. The "filtered core gold" was mostly a smoothed XAU plus a
complementary regression-cancelling correction; the structural decomposition
story has no empirical support on this data shape. Replacing it with raw XAU
restores dimensional homogeneity at the price of admitting that Phase 4 is
about gold, not "core gold".

Anchor caveat: the only free daily gold series with multi-decade depth is
yfinance ``GC=F``, which begins 2000-08-30. ``N_Gold`` is anchored on that
first available date = 100.0, leaving an 8-month gap (2000-01-03 → 2000-08-29)
where ``N_Gold`` is NaN. This is a 6x-improvement over the previous
2006-anchored Kalman pipeline but does not fully restore the T0 invariant.
The audit's "C-1 dimensional asymmetry" is therefore reduced, not eliminated.

The diagnostic Kalman model still requires the full panel (XAU/TIPS/DXY/VIX)
and is unchanged from the spec — its outputs are useful for regime-shift
detection (the ``recent_to_alltime_std_ratio`` in ``patterns.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from rollender_stein.bitemporal import insert_macro_releases, latest_release_stream
from rollender_stein.calendar import T0_DATE, master_calendar
from rollender_stein.io.fred import fetch_fred_observations
from rollender_stein.io.yahoo import fetch_yahoo_history
from rollender_stein.locf import forward_fill_to_calendar

# The XAU series id used directly to build N_Gold.
XAU_SERIES_ID = "GC=F"

# Heterogeneous sources for the Kalman *diagnostic*:
#   XAU  — yfinance GC=F (gold front-month futures, also the basis of N_Gold)
#   TIPS — FRED DFII10 (live endpoint; daily, 2003-onward)
#   DXY  — FRED DTWEXBGS (live endpoint; daily, 2006-onward)
#   VIX  — FRED VIXCLS (live endpoint; daily, 1990-onward)
SERIES_IDS: dict[str, str] = {
    "XAU": XAU_SERIES_ID,
    "TIPS": "DFII10",
    "DXY": "DTWEXBGS",
    "VIX": "VIXCLS",
}
EXOG_COLS = ["TIPS", "DXY", "VIX"]
SOURCE_FRED = "FRED"
SOURCE_YAHOO = "YAHOO"


def ingest_gold_inputs(
    con: duckdb.DuckDBPyConnection,
    fred_api_key: str,
) -> dict[str, int]:
    """Pull all four daily inputs into the bitemporal store.

    XAU comes from yfinance and is the **direct basis of N_Gold** post-patch-06.
    TIPS/DXY/VIX come from FRED's live endpoint and feed the Kalman diagnostic
    only — they no longer affect the numéraire.
    """
    counts: dict[str, int] = {}

    for short in ("TIPS", "DXY", "VIX"):
        sid = SERIES_IDS[short]
        rows = fetch_fred_observations(sid, fred_api_key)
        counts[short] = insert_macro_releases(con, sid, rows, source=SOURCE_FRED)

    rows = fetch_yahoo_history(SERIES_IDS["XAU"])
    counts["XAU"] = insert_macro_releases(
        con, SERIES_IDS["XAU"], rows, source=SOURCE_YAHOO
    )
    return counts


def assemble_panel(
    con: duckdb.DuckDBPyConnection,
    *,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build the 4-column daily panel (XAU, TIPS, DXY, VIX) on the master calendar.

    This is the input to the Phase 4.5 Kalman *diagnostic*. It does not feed
    N_Gold directly anymore (post-patch-06).
    """
    cal = master_calendar(end=end)
    panel = pd.DataFrame(index=cal)
    for short, sid in SERIES_IDS.items():
        stream = latest_release_stream(con, sid)
        if stream.empty:
            raise RuntimeError(
                f"no rows in macro_release for {sid}; run ingest_gold_inputs() first",
            )
        stream = stream.rename(columns={"value": short})
        merged = forward_fill_to_calendar(stream, cal, value_cols=[short])
        panel[short] = merged[short]
    return panel


@dataclass(frozen=True)
class GoldFit:
    """Output of the Phase 4.5 Kalman *diagnostic*. Not used in any numéraire."""

    results: sm.tsa.statespace.mlemodel.MLEResults
    panel_clean: pd.DataFrame
    filtered_state: pd.Series  # mu_t — the latent "smoothed-XAU" level (not orthogonal)


def fit_gold_model(
    panel: pd.DataFrame,
    *,
    disp: bool = False,
) -> GoldFit:
    """Fit the local-level + linear-regression model on the clean panel.

    Drops rows with any NaN across {XAU, TIPS, DXY, VIX}. The Kalman recursion
    runs on the surviving rows only (in practice 2006-onward, bottlenecked by
    DXY).

    **Diagnostic-only post-patch-06.** The fit is preserved for regime-shift
    detection via the residual-variance ratio (see ``patterns.py``). The
    filtered state μ_t is NOT a "core gold" component — empirically it tracks
    raw XAU at r ≈ 0.97 and is non-orthogonal to the regression term.
    """
    required = {"XAU", *EXOG_COLS}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"panel missing columns {sorted(missing)}")

    clean = panel.dropna(subset=list(required))
    if clean.empty:
        raise RuntimeError("panel has no rows with all of XAU/TIPS/DXY/VIX present")

    model = sm.tsa.UnobservedComponents(
        endog=clean["XAU"],
        level="local level",
        exog=clean[EXOG_COLS],
        initialization="approximate_diffuse",
    )
    results = model.fit(disp=disp)
    # Audit patch 05: surface MLE convergence failures rather than silently
    # returning a fit at a non-optimum. The 'converged' key is set by
    # statsmodels' wrapper around scipy.optimize; default True to allow
    # statsmodels versions that do not expose it.
    if not bool(results.mle_retvals.get("converged", True)):
        raise RuntimeError(
            f"gold-model MLE did not converge: {results.mle_retvals!r}"
        )
    filtered = pd.Series(
        np.asarray(results.filtered_state[0]),
        index=clean.index,
        name="mu_t",
    )
    return GoldFit(results=results, panel_clean=clean, filtered_state=filtered)


def build_n_gold(
    con: duckdb.DuckDBPyConnection,
    *,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Build the daily N_Gold index from raw XAU (GC=F).

    Pipeline:
      1. Read XAU release stream from the bitemporal store (release_date == reference_date
         for daily yfinance data — see ``io/yahoo.py``).
      2. LOCF onto the master NYSE calendar.
      3. Anchor at T0 if XAU has a value there; otherwise at the first valid date
         (typically 2000-08-30, GC=F's start).

    Returns a Series indexed by the master calendar. Values before XAU's first
    available date are NaN — patch 06 reduces but does not eliminate the
    pre-anchor gap originally identified as audit finding C-1.
    """
    stream = latest_release_stream(con, XAU_SERIES_ID)
    if stream.empty:
        raise RuntimeError(
            f"no rows in macro_release for {XAU_SERIES_ID}; run ingest_gold_inputs() first",
        )

    stream = stream.rename(columns={"value": "xau"})
    cal = master_calendar(end=end)
    daily = forward_fill_to_calendar(stream, cal, value_cols=["xau"])

    # Anchor: prefer T0; fall back to first non-NaN date.
    xau_series = daily["xau"]
    if T0_DATE in xau_series.index and pd.notna(xau_series.loc[T0_DATE]):
        anchor = float(cast(float, xau_series.loc[T0_DATE]))
    else:
        first_valid = xau_series.first_valid_index()
        if first_valid is None:
            raise RuntimeError("XAU series has no non-NaN values")
        anchor = float(cast(float, xau_series.loc[first_valid]))

    if not np.isfinite(anchor) or anchor == 0:
        raise RuntimeError(f"N_Gold anchor invalid: {anchor}")

    n_gold = (daily["xau"] / anchor) * 100.0
    return n_gold.rename("N_Gold")
