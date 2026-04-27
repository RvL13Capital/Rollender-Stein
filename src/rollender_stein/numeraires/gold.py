"""N_Gold — the Filtered Core Gold standard.

Phase 4 of the AVE spec. We orthogonalize the daily XAU/USD level (London PM
fix) against three control variables — the 10Y TIPS real yield, the broad-USD
Major-Currencies index, and the CBOE VIX — by fitting a local-level
state-space model and extracting the FILTERED state. Smoothing is forbidden
(it would leak future observations into past state estimates).

Model:

    y_t = mu_t + beta . x_t + eps_t,    eps_t ~ N(0, sigma_eps^2)
    mu_t = mu_{t-1} + eta_t,            eta_t ~ N(0, sigma_eta^2)

with x_t = (TIPS_t, DXY_t, VIX_t). mu_t is "true core gold" — the latent
monetary signal once we strip out real-yield, currency, and risk-aversion
noise.

Anchoring caveat: DFII10 (TIPS) starts 2003-01-02. Rows with NaN exog are
dropped before fitting, so the Kalman recursion runs on 2003-onward only.
N_Gold is therefore normalized to 100 at the FIRST available filtered date
(≈ 2003-01-02), NOT at T0. Pre-2003 values are NaN. This is the consequence
of the "accept the gap" decision in the spec resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from rollender_stein.bitemporal import insert_macro_releases, latest_release_stream
from rollender_stein.calendar import master_calendar
from rollender_stein.io.fred import fetch_fred_observations
from rollender_stein.io.yahoo import fetch_yahoo_history
from rollender_stein.locf import forward_fill_to_calendar

# Heterogeneous sources by necessity:
#   XAU  — yfinance GC=F (gold front-month futures as spot proxy; LBMA was
#          removed from FRED in 2017 and free LBMA feeds no longer exist).
#   TIPS — FRED DFII10 (live endpoint; daily, 2003-onward).
#   DXY  — FRED DTWEXBGS (live endpoint; daily, 2006-onward; replaces the
#          discontinued DTWEXM).
#   VIX  — FRED VIXCLS (live endpoint; daily, 1990-onward).
#
# Effective Kalman fit window is bottlenecked by DXY at 2006-01-02. N_Gold
# is anchored at the first available date (≈ 2006-01-02), NOT at T0.
SERIES_IDS: dict[str, str] = {
    "XAU": "GC=F",
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

    XAU comes from yfinance; TIPS/DXY/VIX come from FRED's live endpoint.
    The bitemporal store distinguishes sources via the ``source`` column so
    we can swap in a paid LBMA feed later without losing audit history.
    """
    counts: dict[str, int] = {}

    # FRED inputs
    for short in ("TIPS", "DXY", "VIX"):
        sid = SERIES_IDS[short]
        rows = fetch_fred_observations(sid, fred_api_key)
        counts[short] = insert_macro_releases(con, sid, rows, source=SOURCE_FRED)

    # Yahoo input (gold)
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
    """Build the 4-column daily panel (XAU, TIPS, DXY, VIX) on the master calendar."""
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
    """Output of the Kalman gold model, ready for downstream Phase 5 use."""

    results: sm.tsa.statespace.mlemodel.MLEResults
    panel_clean: pd.DataFrame
    filtered_state: pd.Series  # mu_t — the latent "true core gold" level


def fit_gold_model(
    panel: pd.DataFrame,
    *,
    disp: bool = False,
) -> GoldFit:
    """Fit the local-level + linear-regression model on the clean panel.

    Drops rows with any NaN across {XAU, TIPS, DXY, VIX}. The Kalman recursion
    runs on the surviving rows only (in practice 2003-onward). Returns the
    statsmodels results, the cleaned panel actually fed to the model, and the
    filtered state series indexed by the cleaned panel's dates.
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
    """Build the daily N_Gold index. Anchors at the first filtered date = 100.0.

    Reindexed onto the full master calendar; values pre-2003 (where TIPS is
    unobserved and the Kalman recursion has not started) are NaN.
    """
    panel = assemble_panel(con, end=end)
    fit = fit_gold_model(panel)

    anchor = float(fit.filtered_state.iloc[0])
    if not np.isfinite(anchor) or anchor == 0:
        raise RuntimeError(f"N_Gold anchor invalid: {anchor}")

    n_gold_clean = (fit.filtered_state / anchor) * 100.0
    return n_gold_clean.reindex(panel.index).rename("N_Gold")
