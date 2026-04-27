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


class _RequestsLike(Protocol):
    def get(self, url: str, params: dict[str, Any], timeout: float) -> Any: ...


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
