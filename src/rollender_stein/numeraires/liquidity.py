"""N_Liq — the Systemic Liquidity Standard.

Phase 3.3 of the AVE spec. The Global Fiat Ocean is the daily USD-equivalent
sum of the world's principal monetary aggregates:

    Global_Ocean(t) = US_M2(t) + EZ_M3(t) * EURUSD(t) + JP_M3(t) / USDJPY(t)
    N_Liq(t)        = (Global_Ocean(t) / Global_Ocean(T0)) * 100

PBOC M2 is deferred (no clean ALFRED-style vintage source for free).

Sourcing nuance: FRED's level series for EZ/JP M3 (``MABMM301EZM189S``,
``MABMM301JPM189S``) stopped updating at 2023-11-01, but their growth-rate
counterparts (``...M657S``) continue through 2025-12. We splice: use levels
for history, then extend forward by compounding (1 + g/100) per the
growth-rate series. The composite is mathematically equivalent to the
underlying BIS broad-money level had it been published continuously.

All series use FRED's live endpoint. WM2NS in ALFRED only has vintage
history back to ~2002 — using ALFRED first-release would leave US M2
NaN at T0=2000. The live endpoint has WM2NS back to 1981 with current
values, at the cost of M2 revisions not being tracked. Revision drift
on US M2 is small (annual seasonal-factor updates) compared with the
order-of-magnitude differences across the three monetary blocs.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from rollender_stein.bitemporal import insert_macro_releases, latest_release_stream
from rollender_stein.calendar import T0_DATE, master_calendar
from rollender_stein.io.fred import fetch_fred_observations
from rollender_stein.locf import forward_fill_to_calendar

US_M2 = "WM2NS"
EZ_M3_LEVEL = "MABMM301EZM189S"
EZ_M3_GROWTH = "MABMM301EZM657S"
JP_M3_LEVEL = "MABMM301JPM189S"
JP_M3_GROWTH = "MABMM301JPM657S"
EURUSD = "DEXUSEU"
USDJPY = "DEXJPUS"

ALL_SERIES: dict[str, str] = {
    "US_M2": US_M2,
    "EZ_M3_LEVEL": EZ_M3_LEVEL,
    "EZ_M3_GROWTH": EZ_M3_GROWTH,
    "JP_M3_LEVEL": JP_M3_LEVEL,
    "JP_M3_GROWTH": JP_M3_GROWTH,
    "EURUSD": EURUSD,
    "USDJPY": USDJPY,
}


def ingest_liquidity_inputs(
    con: duckdb.DuckDBPyConnection,
    fred_api_key: str,
) -> dict[str, int]:
    """Pull all seven inputs into the bitemporal store via FRED's live endpoint."""
    counts: dict[str, int] = {}
    for short, sid in ALL_SERIES.items():
        rows = fetch_fred_observations(sid, fred_api_key)
        counts[short] = insert_macro_releases(con, sid, rows, source="FRED")
    return counts


def extend_levels_with_growth(
    levels: pd.Series,
    growth_rates_pct: pd.Series,
) -> pd.Series:
    """Extend ``levels`` past its last valid date by compounding growth rates.

    For each date ``d`` in ``growth_rates_pct.index`` that comes strictly after
    ``levels.last_valid_index()``, set ``out[d] = out[prev] * (1 + g_d / 100)``.
    The compounding chain starts at the last known level value; growth rates
    in the index but at or before the last level are ignored (the level series
    is authoritative for that range).
    """
    if levels.empty:
        raise ValueError("levels series is empty")
    last_idx = levels.last_valid_index()
    if last_idx is None:
        raise ValueError("levels series has no non-NaN values")

    forward = growth_rates_pct.loc[growth_rates_pct.index > last_idx].sort_index().dropna()
    if forward.empty:
        return levels.copy()

    last_level = float(levels.loc[last_idx])
    factors = (1.0 + forward.to_numpy() / 100.0).cumprod()
    extended_values = last_level * factors

    extension = pd.Series(extended_values, index=forward.index, name=levels.name)
    return pd.concat([levels.dropna(), extension]).sort_index()


def _stream_to_indexed_series(con, series_id: str, name: str) -> pd.Series:
    df = latest_release_stream(con, series_id)
    if df.empty:
        raise RuntimeError(f"no rows for {series_id} in macro_release; ingest first")
    return (
        df.rename(columns={"value": name, "release_date": "trade_date"})
        .set_index("trade_date")[name]
        .sort_index()
    )


def build_n_liq(
    con: duckdb.DuckDBPyConnection,
    *,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Build the daily N_Liq index (100.0 at T0).

    Steps:
      1. Pull each of the 7 streams from the DB.
      2. Extend EZ_M3 and JP_M3 levels past 2023-11 via growth-rate compounding.
      3. LOCF each component onto the NYSE master calendar via release_date.
      4. Convert to common USD scale: WM2NS is in USD billions (multiply by 1e9);
         EZ M3 is in raw EUR (multiply by EURUSD); JP M3 is in raw JPY (divide
         by USDJPY).
      5. Sum to Global_Fiat_Ocean (USD), normalize at T0 = 100.
    """
    cal = master_calendar(end=end)

    # Raw monthly/weekly streams (release-date indexed, source units)
    us_m2_billions = _stream_to_indexed_series(con, US_M2, "us_m2")
    ez_m3_eur_levels = _stream_to_indexed_series(con, EZ_M3_LEVEL, "ez_m3")
    ez_m3_growth_pct = _stream_to_indexed_series(con, EZ_M3_GROWTH, "ez_m3_g")
    jp_m3_yen_levels = _stream_to_indexed_series(con, JP_M3_LEVEL, "jp_m3")
    jp_m3_growth_pct = _stream_to_indexed_series(con, JP_M3_GROWTH, "jp_m3_g")

    ez_m3_eur_extended = extend_levels_with_growth(ez_m3_eur_levels, ez_m3_growth_pct)
    jp_m3_yen_extended = extend_levels_with_growth(jp_m3_yen_levels, jp_m3_growth_pct)

    # LOCF each stream onto the daily master calendar (release_date semantics).
    def _locf(stream: pd.Series, col_name: str) -> pd.Series:
        frame = pd.DataFrame(
            {"release_date": stream.index, col_name: stream.to_numpy()}
        )
        return forward_fill_to_calendar(frame, cal, value_cols=[col_name])[col_name]

    us_m2_d = _locf(us_m2_billions, "us_m2") * 1e9          # billions → raw USD
    ez_m3_d = _locf(ez_m3_eur_extended, "ez_m3")             # raw EUR
    jp_m3_d = _locf(jp_m3_yen_extended, "jp_m3")             # raw JPY
    eurusd_d = _locf(_stream_to_indexed_series(con, EURUSD, "eurusd"), "eurusd")
    usdjpy_d = _locf(_stream_to_indexed_series(con, USDJPY, "usdjpy"), "usdjpy")

    ocean_usd = us_m2_d + ez_m3_d * eurusd_d + jp_m3_d / usdjpy_d

    if T0_DATE not in ocean_usd.index:
        raise RuntimeError(f"calendar does not contain T0 ({T0_DATE.date()})")
    anchor = ocean_usd.loc[T0_DATE]
    if not np.isfinite(anchor) or anchor == 0:
        raise RuntimeError(
            f"Global_Fiat_Ocean at T0 is {anchor!r}; cannot index. "
            "All four inputs (US M2, EZ M3, JP M3, EURUSD, USDJPY) must have a "
            "release_date <= T0 for the anchor to be defined."
        )

    return (ocean_usd / anchor * 100.0).rename("N_Liq")
