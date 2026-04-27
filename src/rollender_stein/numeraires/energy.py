"""N_Energy — the Thermodynamic Standard.

Phase 3.2 of the AVE spec:

    MWh_cost(t)   = max( Brent_USD_per_bbl(t) / 1.699,  $0.10 )
    N_Energy(t)   = ( MWh_cost(t) / MWh_cost(T0) ) * 100

Brent spot is sourced from EIA (``RBRTE``). The spec forbids futures because
roll-yield contango/backwardation would compound errors over a 25-year window.

The ``$0.10/MWh`` floor is a numerical safety net for divide-by-near-zero
under anomalous events (e.g. April-2020 negative-WTI episode). It is
deliberately set FAR BELOW any plausible Brent-derived MWh value so that it
never binds at a real-world data point.

**Audit fix (patch 03)** — the previous floor of $20/MWh was binding at the
T0 anchor. Brent at T0 = $24.93/bbl → raw MWh cost = $14.67, well below the
$20 floor. Anchoring on the floor instead of the true cost biased the entire
N_Energy index by +36% relative to its non-clipped form on every date where
the floor did not bind, even though the floor binding period itself was only
2000-2003. The fix: drop the floor to $0.10 so the anchor uses the true raw
value (no clip) on real Brent data, while still protecting numeric stability
against any future negative-energy anomaly.
"""

from __future__ import annotations

import warnings
from typing import cast

import duckdb
import pandas as pd

from rollender_stein.bitemporal import insert_macro_releases, latest_release_stream
from rollender_stein.calendar import T0_DATE, master_calendar
from rollender_stein.io.eia import fetch_eia_petroleum_spot
from rollender_stein.locf import forward_fill_to_calendar

BRENT_SERIES = "RBRTE"
SOURCE = "EIA"

BBL_TO_MWH_DIVISOR = 1.699  # spec: USD/bbl ÷ 1.699 = USD/MWh
# Lowered from $20 to $0.10 (audit patch 03) — old floor was binding at the T0
# anchor and biasing the entire index by +36%. The new floor still protects
# against true zero/negative anomalies (April 2020 collapse) while never
# binding on any historically observed Brent level (Brent low was ~$9/bbl →
# $5.30/MWh, far above $0.10).
MWH_PRICE_FLOOR_USD = 0.10


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
      4. Apply $0.10/MWh floor (numerical safety net only — never binds in
         real data; if the floor DOES bind at the T0 anchor, a RuntimeWarning
         is issued because the resulting N_Energy will mis-represent the
         early-window thermodynamic axis)
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

    raw_mwh = daily["brent_usd_per_bbl"] / BBL_TO_MWH_DIVISOR
    mwh_cost = raw_mwh.clip(lower=MWH_PRICE_FLOOR_USD)

    # Forensic guard: if the floor binds at T0, the anchor is the floor rather
    # than the true energy cost, biasing the entire index. Surface loudly.
    if T0_DATE in raw_mwh.index:
        raw_at_t0 = raw_mwh.loc[T0_DATE]
        if pd.notna(raw_at_t0) and float(cast(float, raw_at_t0)) < MWH_PRICE_FLOOR_USD:
            warnings.warn(
                f"MWh floor ${MWH_PRICE_FLOOR_USD:.2f} binds at T0 (raw="
                f"${float(cast(float, raw_at_t0)):.4f}); N_Energy will mis-represent the "
                "early-window thermodynamic axis (audit M-2 / patch 03 condition).",
                RuntimeWarning,
                stacklevel=2,
            )

    if T0_DATE not in mwh_cost.index:
        raise RuntimeError(f"calendar does not contain T0 ({T0_DATE.date()})")
    anchor_raw = mwh_cost.loc[T0_DATE]
    if pd.isna(anchor_raw) or anchor_raw == 0:
        raise RuntimeError(
            f"MWh cost at T0 ({T0_DATE.date()}) is {anchor_raw!r}; cannot index. "
            "Ensure the Brent ingest covers a release on or before T0."
        )
    anchor = float(cast(float, anchor_raw))
    n_energy: pd.Series = mwh_cost / anchor * 100.0
    return n_energy.rename("N_Energy")
