"""EIA (U.S. Energy Information Administration) v2 API client.

Used for petroleum spot prices — primarily Brent (``RBRTE``) for Phase 3.2
N_Energy. The v2 API replaced the legacy v1 ``series/`` endpoint in 2023;
results are paginated with a max ``length`` of 5000 rows per response.

Forensic note: EIA spot prices are not materially revised after publication,
so we treat ``release_date == reference_date``. The bitemporal model still
applies — just trivially for daily commodity spot.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

EIA_BASE = "https://api.eia.gov/v2"
PAGE_SIZE = 5000


class _RequestsLike(Protocol):
    def get(self, url: str, params: dict[str, Any], timeout: float) -> Any: ...


def fetch_eia_petroleum_spot(
    series: str,
    api_key: str,
    *,
    start: str = "1990-01-01",
    end: str | None = None,
    timeout: float = 30.0,
    session: _RequestsLike | None = None,
) -> pd.DataFrame:
    """Fetch daily EIA petroleum spot prices for ``series``.

    Examples of series codes:
      - ``"RBRTE"`` — Brent Crude (USD/bbl)
      - ``"RWTC"``  — WTI Cushing (USD/bbl)

    Returns DataFrame with columns ``reference_date``, ``release_date``,
    ``value`` (USD per barrel). ``release_date == reference_date`` since EIA
    spot prices are not materially revised.

    Pagination is handled internally — repeats requests until all rows for the
    requested date range are fetched. EIA's max page size is 5000.
    """
    if end is None:
        end = "2099-12-31"

    if session is None:
        import requests

        session = requests.Session()

    all_rows: list[dict[str, Any]] = []
    offset = 0

    while True:
        params: dict[str, Any] = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": series,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": PAGE_SIZE,
        }
        resp = session.get(
            f"{EIA_BASE}/petroleum/pri/spt/data/",
            params=params,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        response = payload.get("response")
        if response is None:
            raise ValueError(f"EIA response missing 'response' field: keys={list(payload)}")

        page = response.get("data", [])
        all_rows.extend(page)

        total = int(response.get("total", 0))
        if offset + len(page) >= total or not page:
            break
        offset += len(page)

    if not all_rows:
        return pd.DataFrame(
            {
                "reference_date": pd.Series([], dtype="datetime64[ns]"),
                "release_date": pd.Series([], dtype="datetime64[ns]"),
                "value": pd.Series([], dtype="float64"),
            }
        )

    df = pd.DataFrame(all_rows)
    df["reference_date"] = pd.to_datetime(df["period"])
    df["release_date"] = df["reference_date"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return (
        df.loc[df["value"].notna(), ["reference_date", "release_date", "value"]]
        .sort_values("reference_date")
        .reset_index(drop=True)
    )
