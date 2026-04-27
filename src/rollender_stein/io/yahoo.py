"""Yahoo Finance loader (via yfinance).

Used for daily series unavailable from FRED — primarily gold (the LBMA series
were removed from FRED in 2017 due to licensing changes).

Forensic note on ``GC=F`` (gold front-month continuous futures): the AVE spec
forbids futures for Phase 3.2 N_Energy because Brent's roll-yield contango
materially distorts a 25-year price series. For gold, the basis carry is much
smaller — gold has near-zero storage cost relative to its value — and the
remaining noise is absorbed into the Kalman observation residual. ``GC=F`` is
therefore an acceptable spot-gold proxy under this spec, and the only free
24/7-available source with full T0-onward history.

If a paid LBMA / ICE feed becomes available later, swap this loader out and
re-ingest under a different ``series_id``; the bitemporal store distinguishes
sources via the ``source`` column.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class _YfinanceLike(Protocol):
    def download(self, *args: Any, **kwargs: Any) -> pd.DataFrame: ...


def fetch_yahoo_history(
    ticker: str,
    *,
    start: str = "1990-01-01",
    end: str | None = None,
    auto_adjust: bool = False,
    yf_module: _YfinanceLike | None = None,
) -> pd.DataFrame:
    """Fetch daily close history from Yahoo Finance for ``ticker``.

    Returns DataFrame with columns ``reference_date``, ``release_date``, ``value``
    (the unadjusted close), in the same shape as ``fetch_fred_observations``.
    ``release_date == reference_date`` since Yahoo prices are not bitemporal.

    Parameters
    ----------
    ticker
        Yahoo Finance ticker (``"GC=F"``, ``"GLD"``, etc.).
    start, end
        Date strings. ``end=None`` → today.
    auto_adjust
        Pass through to yfinance. Default ``False`` to get raw closes (spec
        requires raw price for the Kalman; dividend-adjustment for equities is
        handled separately via TR series).
    yf_module
        Injectable for testing — the module providing ``.download()``. Defaults
        to importing ``yfinance``.
    """
    if yf_module is None:
        import yfinance as yf

        yf_module = yf

    df = yf_module.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame(
            {
                "reference_date": pd.Series([], dtype="datetime64[ns]"),
                "release_date": pd.Series([], dtype="datetime64[ns]"),
                "value": pd.Series([], dtype="float64"),
            }
        )

    # yfinance occasionally returns MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError(
            f"yfinance returned no Close column for {ticker!r}; got {list(df.columns)}"
        )

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_localize(None)

    out = pd.DataFrame(
        {
            "reference_date": idx,
            "release_date": idx,
            "value": pd.to_numeric(df["Close"], errors="coerce").to_numpy(),
        }
    )
    return (
        out.loc[out["value"].notna()]
        .sort_values("reference_date")
        .reset_index(drop=True)
    )


def fetch_yahoo_ohlcv(
    ticker: str,
    *,
    start: str = "1990-01-01",
    end: str | None = None,
    auto_adjust: bool = False,
    yf_module: _YfinanceLike | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV history from Yahoo Finance for ``ticker``.

    Returns DataFrame with columns ``trade_date``, ``open``, ``high``, ``low``,
    ``close``, ``adj_close``, ``volume`` — the schema of ``asset_price``. Use
    this for target assets (e.g. ``^SP500TR``, ``BTC-USD``) that get persisted
    in the bitemporal store via ``bitemporal.insert_asset_prices``.

    For Phase 4 gold (``GC=F``) we use ``fetch_yahoo_history`` instead since
    that one takes the macro_release shape (reference_date / release_date /
    value) the rest of the numéraire pipeline expects.
    """
    if yf_module is None:
        import yfinance as yf

        yf_module = yf

    df = yf_module.download(
        ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False
    )
    if df.empty:
        return pd.DataFrame(
            {
                "trade_date": pd.Series([], dtype="datetime64[ns]"),
                "open": pd.Series([], dtype="float64"),
                "high": pd.Series([], dtype="float64"),
                "low": pd.Series([], dtype="float64"),
                "close": pd.Series([], dtype="float64"),
                "adj_close": pd.Series([], dtype="float64"),
                "volume": pd.Series([], dtype="float64"),
            }
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError(
            f"yfinance returned no Close column for {ticker!r}; got {list(df.columns)}"
        )

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_localize(None)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    out = pd.DataFrame({"trade_date": idx})
    for src, dst in rename_map.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce").to_numpy()
        else:
            out[dst] = pd.NA

    return out.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
