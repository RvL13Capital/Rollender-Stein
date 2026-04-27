from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from rollender_stein.assets import save_asset_dashboard
from rollender_stein.bitemporal import (
    get_asset_closes,
    insert_asset_prices,
    open_db,
)
from rollender_stein.io.yahoo import fetch_yahoo_ohlcv


def _fake_yf(df: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(download=lambda *a, **kw: df)


@pytest.fixture
def con():
    with open_db(":memory:") as c:
        yield c


# ----- bitemporal asset_price round-trip ------------------------------------


def test_insert_and_read_asset_closes(con) -> None:
    rows = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [100.0, 101.0, 99.0],
            "high": [102.0, 102.5, 101.0],
            "low": [99.5, 100.0, 98.5],
            "close": [101.0, 100.0, 100.5],
            "adj_close": [101.0, 100.0, 100.5],
            "volume": [1_000_000, 950_000, 1_100_000],
        }
    )
    n = insert_asset_prices(con, "TEST", rows, source="UNITTEST")
    assert n == 3

    closes = get_asset_closes(con, "TEST")
    assert list(closes.values) == [101.0, 100.0, 100.5]
    assert closes.name == "TEST"
    assert closes.index[0] == pd.Timestamp("2024-01-02")


def test_insert_asset_prices_accepts_partial_columns(con) -> None:
    """For Yahoo fetches that return only close (no OHLV), insert should still work."""
    rows = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [42.0, 43.0],
        }
    )
    n = insert_asset_prices(con, "MIN", rows, source="UNITTEST")
    assert n == 2
    closes = get_asset_closes(con, "MIN")
    assert closes.tolist() == [42.0, 43.0]


def test_insert_or_replace_asset_prices(con) -> None:
    rows1 = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2024-01-02"]), "close": [100.0]}
    )
    insert_asset_prices(con, "TEST", rows1, source="UNITTEST")
    rows2 = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2024-01-02"]), "close": [101.0]}
    )
    insert_asset_prices(con, "TEST", rows2, source="UNITTEST")
    closes = get_asset_closes(con, "TEST")
    assert closes.tolist() == [101.0], "INSERT OR REPLACE should overwrite same PK"


def test_get_asset_closes_respects_date_range(con) -> None:
    dates = pd.date_range("2024-01-02", "2024-01-10", freq="D")
    rows = pd.DataFrame(
        {"trade_date": dates, "close": [100.0 + i for i in range(len(dates))]}
    )
    insert_asset_prices(con, "TEST", rows, source="UNITTEST")

    sub = get_asset_closes(
        con, "TEST",
        start=pd.Timestamp("2024-01-04"),
        end=pd.Timestamp("2024-01-07"),
    )
    assert sub.index.min() == pd.Timestamp("2024-01-04")
    assert sub.index.max() == pd.Timestamp("2024-01-07")
    assert len(sub) == 4


def test_get_asset_closes_unknown_series_returns_empty(con) -> None:
    closes = get_asset_closes(con, "DOES_NOT_EXIST")
    assert closes.empty
    assert closes.dtype == "float64"


def test_insert_asset_prices_rejects_missing_trade_date(con) -> None:
    bad = pd.DataFrame({"close": [100.0]})
    with pytest.raises(KeyError, match="trade_date"):
        insert_asset_prices(con, "BAD", bad, source="UNITTEST")


# ----- fetch_yahoo_ohlcv shape ------------------------------------------------


def test_fetch_yahoo_ohlcv_returns_asset_price_shape() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 102.5],
            "Low": [99.5, 100.0],
            "Close": [101.0, 100.0],
            "Adj Close": [101.0, 100.0],
            "Volume": [1_000_000, 950_000],
        },
        index=idx,
    )
    out = fetch_yahoo_ohlcv("TEST", yf_module=_fake_yf(df))
    assert list(out.columns) == [
        "trade_date", "open", "high", "low", "close", "adj_close", "volume"
    ]
    assert len(out) == 2
    assert out["close"].tolist() == [101.0, 100.0]


def test_fetch_yahoo_ohlcv_handles_missing_columns() -> None:
    """If yfinance returns just Close (some tickers do), other OHLV are NA."""
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    df = pd.DataFrame({"Close": [101.0, 100.0]}, index=idx)
    out = fetch_yahoo_ohlcv("TEST", yf_module=_fake_yf(df))
    assert out["close"].tolist() == [101.0, 100.0]
    assert out["open"].isna().all()


# ----- save_asset_dashboard naming -------------------------------------------


def test_save_asset_dashboard_sanitizes_ticker(tmp_path) -> None:
    from rollender_stein.assets import AssetPipelineResult
    from rollender_stein.valuation import DivisionArray

    # Build a minimal stub result; we only test path construction.
    da = DivisionArray(
        nominal_usd=pd.Series([1.0], name="nominal_usd"),
        asset_indexed=pd.Series([1.0]),
        asset_in_time=None, asset_in_liquidity=None,
        asset_in_gold=None, asset_in_energy=None,
    )

    class _StubFig:
        @staticmethod
        def write_html(path, **kwargs): tmp_path.joinpath("touch").write_text(path)
    stub_result = AssetPipelineResult(ticker="^SP500TR", division=da, figure=_StubFig())  # type: ignore[arg-type]

    path = save_asset_dashboard(stub_result, out_dir=str(tmp_path))
    assert path.endswith("dashboard_SP500TR.html")
    # `=` becomes `-`
    stub_btc = AssetPipelineResult(ticker="BTC-USD", division=da, figure=_StubFig())  # type: ignore[arg-type]
    p2 = save_asset_dashboard(stub_btc, out_dir=str(tmp_path))
    assert p2.endswith("dashboard_BTC-USD.html")
