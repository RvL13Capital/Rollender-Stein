"""N_Energy — the Thermodynamic Standard.

Phase 3.2 of the AVE spec:

    MWh_cost(t)   = max( Brent_USD_per_bbl(t) / 1.699,  $20.00 )
    N_Energy(t)   = ( MWh_cost(t) / MWh_cost(T0) ) * 100

Brent spot is sourced from EIA (``RBRTE``). The spec forbids futures because
roll-yield contango/backwardation would compound errors over a 25-year window.

The ``$20.00/MWh`` floor is a zero-bound failsafe. Brent itself never went
below ~$10/bbl historically, but during the April 2020 negative-WTI episode
some derived series went negative or near-zero. The floor preserves
mathematical sanity without distorting any non-anomalous date.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from rollender_stein.bitemporal import insert_macro_releases, latest_release_stream
from rollender_stein.calendar import T0_DATE, master_calendar
from rollender_stein.io.eia import fetch_eia_petroleum_spot
from rollender_stein.locf import forward_fill_to_calendar

BRENT_SERIES = "RBRTE"
SOURCE = "EIA"

BBL_TO_MWH_DIVISOR = 1.699  # spec: USD/bbl ÷ 1.699 = USD/MWh
MWH_PRICE_FLOOR_USD = 20.0  # spec: physical extraction-cost zero-bound failsafe


def ingest_brent_spot(
    con: duckdb.DuckDBPyConnection,
    eia_api_key: str,
) -> int:
    """Pull Brent daily spot from EIA into the bitemporal store."""
    rows = fetch_eia_petroleum_spot(BRENT_SERIES, eia_api_key)
    return insert_macro_releases(con, BRENT_SERIES, rows, source=SOURCE)


def build_n_energy(
    con: duckdb.DuckDBPyConnection,
    *,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Build the daily N_Energy index (100.0 at T0).

    Pipeline:
      1. Brent spot stream (USD/bbl) from the bitemporal store
      2. LOCF onto the NYSE master calendar via release_date
      3. Convert to USD/MWh by dividing by 1.699
      4. Apply $20.00/MWh floor
      5. Normalize so the value at T0 = 100
    """
    stream = latest_release_stream(con, BRENT_SERIES)
    if stream.empty:
        raise RuntimeError(
            f"no rows in macro_release for {BRENT_SERIES}; run ingest_brent_spot() first",
        )
    stream = stream.rename(columns={"value": "brent_usd_per_bbl"})

    cal = master_calendar(end=end)
    daily = forward_fill_to_calendar(stream, cal)

    mwh_cost = (daily["brent_usd_per_bbl"] / BBL_TO_MWH_DIVISOR).clip(lower=MWH_PRICE_FLOOR_USD)

    if T0_DATE not in mwh_cost.index:
        raise RuntimeError(f"calendar does not contain T0 ({T0_DATE.date()})")
    anchor = mwh_cost.loc[T0_DATE]
    if pd.isna(anchor) or anchor == 0:
        raise RuntimeError(
            f"MWh cost at T0 ({T0_DATE.date()}) is {anchor!r}; cannot index. "
            "Ensure the Brent ingest covers a release on or before T0."
        )

    return (mwh_cost / anchor * 100.0).rename("N_Energy")
