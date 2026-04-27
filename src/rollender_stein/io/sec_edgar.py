"""SEC EDGAR JSON API client.

Used for historical shares-outstanding data via the ``companyfacts`` endpoint.
EDGAR provides XBRL-tagged facts as JSON since 2009 (when XBRL became
mandatory for large filers); pre-2009 data is generally unavailable
through this API.

SEC fair-access policy requires a meaningful ``User-Agent`` header on every
request. Format: ``"OrgName contact@example.com"``. Per-second request
ceiling is 10; we don't hit that with ~30 tickers but the caller should
add their own rate-limiting if scaling to thousands.

Free, no API key. Two endpoints used:
- ``https://www.sec.gov/files/company_tickers.json`` — ticker → CIK mapping
- ``https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json`` — facts

Multi-class share considerations: a single CIK reports a single
``CommonStockSharesOutstanding`` series. For tickers like BRK-B (B-class
of Berkshire) and GOOGL/GOOG (Class A vs Class C of Alphabet), this
endpoint returns the consolidated A-equivalent share count, NOT the
class-specific count. Market cap computed by multiplying B-class price by
the consolidated share count is therefore a misrepresentation. The caller
is responsible for handling such cases — this module just returns what
EDGAR reports.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

EDGAR_BASE = "https://data.sec.gov"
SEC_BASE = "https://www.sec.gov"


class _RequestsLike(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] = ...,
        timeout: float = ...,
    ) -> Any: ...


def fetch_ticker_to_cik(
    user_agent: str,
    *,
    timeout: float = 30.0,
    session: _RequestsLike | None = None,
) -> dict[str, int]:
    """Return a mapping of {TICKER: CIK} from the SEC's master ticker file.

    The SEC publishes ~10,000+ tickers; this is a single download.
    """
    if session is None:
        import requests

        session = requests.Session()

    resp = session.get(
        f"{SEC_BASE}/files/company_tickers.json",
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    return {entry["ticker"]: int(entry["cik_str"]) for entry in payload.values()}


def fetch_company_shares_outstanding(
    cik: int,
    user_agent: str,
    *,
    timeout: float = 30.0,
    session: _RequestsLike | None = None,
) -> pd.DataFrame:
    """Fetch ``CommonStockSharesOutstanding`` history for ``cik`` from EDGAR.

    Returns DataFrame with columns:
        period_end_date  : pd.Timestamp — period the value describes
        filing_date      : pd.Timestamp — when the filing was made
        shares           : int — common shares outstanding
        form             : str — '10-K', '10-Q', '10-K/A', etc.

    Sorted by period_end_date then filing_date. Empty DataFrame if the CIK
    has no CommonStockSharesOutstanding facts (e.g. some non-corp filers).

    Forensic note: DOES include amendments (e.g. '10-K/A'). The caller
    should pick the earliest filing per period_end_date to get the
    original print — see ``latest_shares_stream`` in bitemporal.py.
    """
    if session is None:
        import requests

        session = requests.Session()

    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik:010d}.json"
    resp = session.get(
        url, headers={"User-Agent": user_agent}, timeout=timeout
    )
    resp.raise_for_status()
    payload = resp.json()

    facts = payload.get("facts", {})
    us_gaap = facts.get("us-gaap", {})
    dei = facts.get("dei", {})

    # Tag-fallback hierarchy: us-gaap is the canonical place for point-in-time
    # shares for filers who use it. Many filers (JNJ, KO, BRK, etc.) report
    # only the cover-page dei tag instead. We try us-gaap first and fall back
    # to dei. WeightedAverage* is intentionally NOT considered — that's
    # period-average shares, not point-in-time, and would give wrong market caps.
    cso = us_gaap.get("CommonStockSharesOutstanding") or dei.get(
        "EntityCommonStockSharesOutstanding"
    )
    if not cso:
        return pd.DataFrame(
            {
                "period_end_date": pd.Series([], dtype="datetime64[ns]"),
                "filing_date": pd.Series([], dtype="datetime64[ns]"),
                "shares": pd.Series([], dtype="int64"),
                "form": pd.Series([], dtype="object"),
            }
        )

    units = cso.get("units", {}).get("shares", [])
    if not units:
        return pd.DataFrame(
            {
                "period_end_date": pd.Series([], dtype="datetime64[ns]"),
                "filing_date": pd.Series([], dtype="datetime64[ns]"),
                "shares": pd.Series([], dtype="int64"),
                "form": pd.Series([], dtype="object"),
            }
        )

    df = pd.DataFrame(units)
    df["period_end_date"] = pd.to_datetime(df["end"])
    df["filing_date"] = pd.to_datetime(df["filed"])
    df["shares"] = pd.to_numeric(df["val"], errors="coerce").astype("Int64")
    df["form"] = df.get("form", pd.Series([None] * len(df))).astype("object")
    return (
        df.loc[df["shares"].notna(), ["period_end_date", "filing_date", "shares", "form"]]
        .sort_values(["period_end_date", "filing_date"])
        .reset_index(drop=True)
    )
