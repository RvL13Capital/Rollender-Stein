"""FRED / ALFRED API client.

The default ``fetch_alfred_first_release`` returns ONLY the original publication
of each reference period (FRED ``output_type=4``). Revisions are dropped.
This is what the AVE Phase 2 LOCF stream consumes — the value the public saw
on the original release date, with ``release_date == realtime_start``.

If revision tracking is needed later, add a ``fetch_alfred_all_vintages`` variant
that uses ``output_type=2`` and stores the full bitemporal grid.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

FRED_BASE = "https://api.stlouisfed.org/fred"
DEFAULT_REALTIME_START = "1990-01-01"

# Publication lag in business days from the reference date to the earliest
# moment the public could USE the value (audit C-3 / patch 02). The previous
# code used `release_date == reference_date` for ALL live-endpoint series,
# which embedded a same-day-to-multi-week look-ahead in the LOCF stream
# depending on each series' real publication cadence.
#
# Conservative — round UP to be on the safe side for backtesting.
#   * H.15 (DFII10):              yields published next BD ~4:15pm ET
#   * H.10 (DTWEXBGS, DEX*):      broad dollar / FX, next BD
#   * VIXCLS:                     close-of-day publication, available
#                                 at session close (lag = 0)
#   * H.6 (WM2NS):                weekly, Tuesday for week ending prior
#                                 Monday → ~5 BD after the Monday reference
#   * MABMM301* (BIS broad money): monthly aggregates, ~30 days after
#                                 month-end via OECD MEI
#
# Series not listed default to 0 BD lag. Adding a series here without
# also running migrate_publication_lags() will produce inconsistent
# release_dates between newly-fetched rows (with lag) and pre-existing
# rows (without lag). The migration is idempotent and scoped to known
# series, so re-running is safe.
PUBLICATION_LAG_BD: dict[str, int] = {
    "DFII10":          1,
    "DTWEXBGS":        1,
    "DTWEXM":          1,   # legacy series, kept for completeness
    "DEXUSEU":         1,
    "DEXJPUS":         1,
    "VIXCLS":          0,
    "WM2NS":           5,
    "MABMM301EZM189S": 30,
    "MABMM301JPM189S": 30,
    "MABMM301EZM657S": 30,
    "MABMM301JPM657S": 30,
}


class _RequestsLike(Protocol):
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] = ...,
        timeout: float = ...,
    ) -> Any: ...


def fetch_alfred_first_release(
    series_id: str,
    api_key: str,
    *,
    realtime_start: str = DEFAULT_REALTIME_START,
    realtime_end: str | None = None,
    timeout: float = 30.0,
    session: _RequestsLike | None = None,
) -> pd.DataFrame:
    """Fetch the original publication of each reference period for ``series_id``.

    Uses FRED ``output_type=4`` (Initial Release Only). For each reference period
    whose initial release fell within the requested realtime range, returns one
    row containing the original value and the original release date.

    Parameters
    ----------
    series_id
        FRED series identifier (e.g. ``"AHETPI"``, ``"WM2NS"``, ``"DFII10"``).
    api_key
        FRED API key.
    realtime_start
        Lower bound on initial-release dates to include. Default ``"1990-01-01"``
        — guaranteed to capture the most recent pre-T0 release for any series
        published since 1990.
    realtime_end
        Upper bound on initial-release dates. Default: ``"9999-12-31"``, FRED's
        sentinel for "include everything available." Avoids 400s from clock-skew
        between the local machine and FRED servers when local "today" runs ahead.
    timeout
        Per-request timeout in seconds.
    session
        Optional injectable requests.Session for testing/mocking.

    Returns
    -------
    DataFrame with columns ``reference_date``, ``release_date``, ``value``,
    sorted by ``reference_date``. Rows where FRED reported missing values
    (``"."``) are dropped.
    """
    if realtime_end is None:
        realtime_end = "9999-12-31"

    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": realtime_start,
        "realtime_end": realtime_end,
        "output_type": 4,
    }

    if session is None:
        import requests

        session = requests.Session()

    resp = session.get(f"{FRED_BASE}/series/observations", params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if "observations" not in payload:
        raise ValueError(
            f"FRED response missing 'observations' field; got keys: {list(payload)}"
        )

    if not payload["observations"]:
        return pd.DataFrame(
            {
                "reference_date": pd.Series([], dtype="datetime64[ns]"),
                "release_date": pd.Series([], dtype="datetime64[ns]"),
                "value": pd.Series([], dtype="float64"),
            }
        )

    df = pd.DataFrame(payload["observations"])
    df["reference_date"] = pd.to_datetime(df["date"])
    df["release_date"] = pd.to_datetime(df["realtime_start"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return (
        df.loc[df["value"].notna(), ["reference_date", "release_date", "value"]]
        .sort_values("reference_date")
        .reset_index(drop=True)
    )


def fetch_fred_observations(
    series_id: str,
    api_key: str,
    *,
    observation_start: str = "1990-01-01",
    observation_end: str | None = None,
    timeout: float = 30.0,
    session: _RequestsLike | None = None,
) -> pd.DataFrame:
    """Fetch CURRENT FRED observations (live endpoint, no vintage tracking).

    Use this for daily series where revisions are rare to nonexistent — Treasury
    yields, market indexes, asset prices like the LBMA gold fix. Two reasons
    to prefer this over ``fetch_alfred_first_release``:

    1. Some series simply aren't in ALFRED (e.g. ``GOLDPMGBD228NLBM``).
    2. Daily series with decades of history exceed FRED's per-request vintage
       limit on ``output_type=4`` (~5000 vintage dates max).

    Returns DataFrame with columns ``reference_date``, ``release_date``,
    ``value``. Since the live endpoint is not bitemporal, ``release_date`` is
    set equal to ``reference_date`` — accurate-enough for series that aren't
    materially revised after publication.
    """
    if observation_end is None:
        observation_end = "9999-12-31"

    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "observation_end": observation_end,
    }

    sess: _RequestsLike
    if session is None:
        import requests

        sess = requests.Session()
    else:
        sess = session

    resp = sess.get(f"{FRED_BASE}/series/observations", params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if "observations" not in payload:
        raise ValueError(
            f"FRED response missing 'observations' field; got keys: {list(payload)}"
        )

    if not payload["observations"]:
        return pd.DataFrame(
            {
                "reference_date": pd.Series([], dtype="datetime64[ns]"),
                "release_date": pd.Series([], dtype="datetime64[ns]"),
                "value": pd.Series([], dtype="float64"),
            }
        )

    df = pd.DataFrame(payload["observations"])
    df["reference_date"] = pd.to_datetime(df["date"])
    lag_bd = PUBLICATION_LAG_BD.get(series_id, 0)
    if lag_bd > 0:
        df["release_date"] = df["reference_date"] + pd.tseries.offsets.BDay(lag_bd)
    else:
        df["release_date"] = df["reference_date"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return (
        df.loc[df["value"].notna(), ["reference_date", "release_date", "value"]]
        .sort_values("reference_date")
        .reset_index(drop=True)
    )
