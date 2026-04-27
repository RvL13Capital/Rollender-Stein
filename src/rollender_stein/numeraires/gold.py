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
from rollender_stein.locf import forward_fill_to_calendar

SERIES_IDS: dict[str, str] = {
    "XAU": "GOLDPMGBD228NLBM",  # LBMA Gold PM Fix, USD/oz, daily, 1968-on
    "TIPS": "DFII10",            # 10Y TIPS yield, %, daily, 2003-on
    "DXY": "DTWEXM",             # Major Currencies Trade-Weighted Dollar, daily, 1973-on
    "VIX": "VIXCLS",             # CBOE VIX close, daily, 1990-on
}
EXOG_COLS = ["TIPS", "DXY", "VIX"]
SOURCE = "FRED"


def ingest_gold_inputs(
    con: duckdb.DuckDBPyConnection,
    api_key: str,
) -> dict[str, int]:
    """Pull XAU + TIPS + DXY + VIX into the bitemporal store from FRED's live endpoint.

    These are daily series for which (a) gold isn't in ALFRED at all, and
    (b) decades of daily yields/VIX exceed FRED's vintage-per-request cap.
    They aren't materially revised after publication, so the live current
    values are forensically equivalent to original prints.
    """
    counts: dict[str, int] = {}
    for short, sid in SERIES_IDS.items():
        rows = fetch_fred_observations(sid, api_key)
        counts[short] = insert_macro_releases(con, sid, rows, source=SOURCE)
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
