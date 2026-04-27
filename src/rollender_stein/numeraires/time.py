"""N_Time — the Time-Price Standard.

Phase 3.1 of the AVE spec:

    N_Time(t) = (AHETPI(t) / AHETPI(T0)) * 100

AHETPI = Average Hourly Earnings of Production / Nonsupervisory employees,
in current US dollars per hour. Source: ALFRED (FRED with realtime params).
First-release values only (no revisions). Monthly reference periods, with a
typical ~5-week lag from reference month-end to release.
"""

from __future__ import annotations

from typing import cast

import duckdb
import pandas as pd

from rollender_stein.bitemporal import insert_macro_releases, latest_release_stream
from rollender_stein.calendar import T0_DATE, master_calendar
from rollender_stein.io.fred import fetch_alfred_first_release
from rollender_stein.locf import forward_fill_to_calendar

SERIES_ID = "AHETPI"
SOURCE = "FRED_ALFRED"


def ingest_ahetpi(con: duckdb.DuckDBPyConnection, api_key: str) -> int:
    """Pull original AHETPI releases from ALFRED into the bitemporal store.

    Idempotent: re-running replaces same-PK rows.
    """
    rows = fetch_alfred_first_release(SERIES_ID, api_key)
    return insert_macro_releases(con, SERIES_ID, rows, source=SOURCE)


def build_n_time(
    con: duckdb.DuckDBPyConnection,
    *,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Build the daily N_Time index (100.00 at T0).

    Reads the AHETPI release stream from the DB, LOCFs onto the NYSE calendar
    starting at T0, and normalizes so the value at T0 = 100. The first AHETPI
    release on or before T0 must exist (true since AHETPI history goes back
    to 1964); otherwise this raises.
    """
    stream = latest_release_stream(con, SERIES_ID)
    if stream.empty:
        raise RuntimeError(
            f"no rows in macro_release for {SERIES_ID}; run ingest_ahetpi() first"
        )

    stream = stream.rename(columns={"value": "ahetpi"})
    stream["release_date"] = pd.to_datetime(stream["release_date"])

    cal = master_calendar(start=T0_DATE, end=end)
    daily = forward_fill_to_calendar(stream, cal)

    if T0_DATE not in daily.index:
        raise RuntimeError(f"calendar does not contain T0 ({T0_DATE.date()})")
    anchor_raw = cast(float, daily.loc[T0_DATE, "ahetpi"])
    if pd.isna(anchor_raw) or anchor_raw == 0:
        raise RuntimeError(
            f"AHETPI at T0 ({T0_DATE.date()}) is {anchor_raw}; cannot index. "
            "Ensure ingest covered realtime_start <= T0 - ~6 weeks."
        )
    n_time: pd.Series = daily["ahetpi"] / anchor_raw * 100.0
    return n_time.rename("N_Time")
