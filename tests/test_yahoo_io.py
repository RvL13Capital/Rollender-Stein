from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from rollender_stein.io.yahoo import fetch_yahoo_history


def _fake_yf(df: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(download=lambda *a, **kw: df)


def test_parses_simple_close_series() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], name="Date")
    df = pd.DataFrame({"Open": [100, 101, 99], "Close": [101, 100, 102]}, index=idx)

    out = fetch_yahoo_history("GC=F", yf_module=_fake_yf(df))
    assert list(out.columns) == ["reference_date", "release_date", "value"]
    assert out["value"].tolist() == [101.0, 100.0, 102.0]
    assert (out["reference_date"] == out["release_date"]).all()


def test_drops_nan_close() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], name="Date")
    df = pd.DataFrame({"Close": [101.0, float("nan"), 102.0]}, index=idx)

    out = fetch_yahoo_history("GC=F", yf_module=_fake_yf(df))
    assert len(out) == 2
    assert out["value"].tolist() == [101.0, 102.0]


def test_handles_multiindex_columns() -> None:
    """yfinance sometimes returns MultiIndex columns even for a single ticker."""
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    cols = pd.MultiIndex.from_tuples([("Open", "GC=F"), ("Close", "GC=F")])
    df = pd.DataFrame([[100, 101], [101, 100]], index=idx, columns=cols)

    out = fetch_yahoo_history("GC=F", yf_module=_fake_yf(df))
    assert out["value"].tolist() == [101.0, 100.0]


def test_strips_timezone_from_index() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC", name="Date")
    df = pd.DataFrame({"Close": [101.0, 102.0]}, index=idx)

    out = fetch_yahoo_history("GC=F", yf_module=_fake_yf(df))
    assert out["reference_date"].dt.tz is None


def test_empty_download_returns_typed_empty_frame() -> None:
    df = pd.DataFrame()
    out = fetch_yahoo_history("BOGUS", yf_module=_fake_yf(df))
    assert out.empty
    assert list(out.columns) == ["reference_date", "release_date", "value"]
    assert out["reference_date"].dtype == "datetime64[ns]"


def test_ohlcv_partial_columns_use_float64_nan_not_pd_na() -> None:
    """Pre-fix, missing OHLCV columns (e.g. yfinance returns Close+Volume
    only for crypto-like tickers) were filled with ``pd.NA``. That promoted
    the entire column to ``object`` dtype, breaking downstream numeric
    arithmetic in ``marketcap`` and the DuckDB persistence layer (which
    expects ``DOUBLE``). Use ``np.nan`` so the column stays ``float64``."""
    from rollender_stein.io.yahoo import fetch_yahoo_ohlcv

    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    # Only Close present — Open/High/Low/Adj Close/Volume missing.
    df = pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)
    out = fetch_yahoo_ohlcv("BTC-USD", yf_module=_fake_yf(df))

    for col in ("open", "high", "low", "adj_close", "volume"):
        assert col in out.columns
        assert out[col].dtype == "float64", (
            f"column {col!r} should be float64, got {out[col].dtype}"
        )
        assert out[col].isna().all()
