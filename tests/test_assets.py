from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rollender_stein.assets import save_asset_dashboard
from rollender_stein.bitemporal import (
    get_asset_closes,
    get_asset_volume,
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


def test_get_asset_volume_round_trip(con) -> None:
    rows = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "close": [101.0, 100.0, 100.5],
            "volume": [1_000_000, 950_000, 1_100_000],
        }
    )
    insert_asset_prices(con, "TEST", rows, source="UNITTEST")
    vol = get_asset_volume(con, "TEST")
    assert list(vol.values) == [1_000_000.0, 950_000.0, 1_100_000.0]
    assert vol.index[0] == pd.Timestamp("2024-01-02")


def test_get_asset_volume_returns_empty_for_index_with_no_volume(con) -> None:
    """Indexes (^SP500TR etc.) typically have NULL volume — must round-trip
    cleanly to an empty Series so the dashboard's coverage check fires."""
    rows = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [4500.0, 4510.0],
            # volume column omitted → stored as NULL
        }
    )
    insert_asset_prices(con, "INDEX", rows, source="UNITTEST")
    vol = get_asset_volume(con, "INDEX")
    assert vol.empty
    assert vol.dtype == "float64"


def test_get_asset_volume_filters_out_null_rows(con) -> None:
    """Mixed NULL / present rows: only the populated ones come back."""
    rows = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "close": [100.0, 101.0, 102.0],
            "volume": [1_000_000.0, np.nan, 2_000_000.0],
        }
    )
    insert_asset_prices(con, "MIX", rows, source="UNITTEST")
    vol = get_asset_volume(con, "MIX")
    assert len(vol) == 2
    assert vol.index[0] == pd.Timestamp("2024-01-02")
    assert vol.index[1] == pd.Timestamp("2024-01-04")


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


# ----- ingest_yahoo_asset: TR-via-adj-close + idempotency ---------------------


def test_ingest_yahoo_asset_stores_both_close_and_adj_close(con, monkeypatch) -> None:
    """Post-marketcap-refactor: ingest stores BOTH raw close and adj_close
    in their proper columns. Consumers select via get_asset_closes(prefer_adjusted=...).
    """
    import rollender_stein.assets as assets_mod
    from rollender_stein.assets import ingest_yahoo_asset

    fake = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 101.0],
            "close": [101.0, 100.0],       # raw
            "adj_close": [99.0, 98.0],     # dividend+split-adjusted
        }
    )
    monkeypatch.setattr(assets_mod, "fetch_yahoo_ohlcv", lambda *a, **kw: fake)

    n = ingest_yahoo_asset(con, "FAKE_TICK")
    assert n == 2

    # prefer_adjusted=True (default): TR-style adjusted close
    adj = get_asset_closes(con, "FAKE_TICK", prefer_adjusted=True)
    assert adj.tolist() == [99.0, 98.0]

    # prefer_adjusted=False: raw close (for market-cap math)
    raw = get_asset_closes(con, "FAKE_TICK", prefer_adjusted=False)
    assert raw.tolist() == [101.0, 100.0]


def test_ingest_yahoo_asset_use_adjusted_param_is_deprecated(con, monkeypatch) -> None:
    """The old use_adjusted_as_close param is now ignored; passing it emits
    a DeprecationWarning. Both columns are always stored."""
    import rollender_stein.assets as assets_mod
    from rollender_stein.assets import ingest_yahoo_asset

    fake = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "close": [101.0],
            "adj_close": [99.0],
        }
    )
    monkeypatch.setattr(assets_mod, "fetch_yahoo_ohlcv", lambda *a, **kw: fake)

    with pytest.warns(DeprecationWarning, match=r"use_adjusted_as_close"):
        ingest_yahoo_asset(con, "DEPRECATED_TICK", use_adjusted_as_close=True)

    # Both columns present and distinct.
    adj = get_asset_closes(con, "DEPRECATED_TICK", prefer_adjusted=True)
    raw = get_asset_closes(con, "DEPRECATED_TICK", prefer_adjusted=False)
    assert adj.iloc[0] == 99.0
    assert raw.iloc[0] == 101.0


def test_get_asset_closes_falls_back_to_close_when_adj_all_nan(con, monkeypatch) -> None:
    """For futures / crypto where Yahoo doesn't return adj_close, the
    prefer_adjusted=True path falls back to raw close per row."""
    import rollender_stein.assets as assets_mod
    from rollender_stein.assets import ingest_yahoo_asset

    fake = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02"]),
            "close": [101.0],
            "adj_close": [float("nan")],
        }
    )
    monkeypatch.setattr(assets_mod, "fetch_yahoo_ohlcv", lambda *a, **kw: fake)
    ingest_yahoo_asset(con, "WEIRD_TICK")

    adj = get_asset_closes(con, "WEIRD_TICK", prefer_adjusted=True)
    # adj_close NaN → fall back to raw close
    assert adj.tolist() == [101.0]


# ----- build_pipeline_for_asset: end-to-end orchestration ---------------------


@pytest.fixture
def fully_seeded_con():
    """In-memory DB seeded with enough realistic data to build all numéraires
    AND a target asset's price history. Reused for the orchestration tests."""
    from rollender_stein.bitemporal import (
        insert_macro_releases as _insert_mr,
    )
    from rollender_stein.numeraires.gold import (
        SERIES_IDS as GOLD_SIDS,
    )
    from rollender_stein.numeraires.gold import (
        SOURCE_FRED,
        SOURCE_YAHOO,
    )
    from rollender_stein.numeraires.liquidity import ALL_SERIES as LIQ_SIDS
    from rollender_stein.numeraires.time import SERIES_ID as TIME_SID

    with open_db(":memory:") as c:
        full = pd.bdate_range("1995-01-02", "2010-12-31")
        n = len(full)
        monthly = pd.date_range("1995-01-01", "2010-12-01", freq="MS")
        m = len(monthly)

        # AHETPI (N_Time)
        _insert_mr(c, TIME_SID, pd.DataFrame({
            "reference_date": monthly,
            "release_date":   monthly + pd.Timedelta(days=35),
            "value": np.linspace(11.0, 19.0, m),
        }), source="FRED_ALFRED")

        # WM2NS (US M2)
        weekly = pd.date_range("1995-01-04", "2010-12-31", freq="W-WED")
        _insert_mr(c, LIQ_SIDS["US_M2"], pd.DataFrame({
            "reference_date": weekly,
            "release_date":   weekly + pd.Timedelta(days=10),
            "value": np.linspace(3000.0, 9000.0, len(weekly)),
        }), source="FRED")

        for sid_key, lo, hi in [
            ("EZ_M3_LEVEL", 4_500_000_000_000.0, 9_000_000_000_000.0),
            ("JP_M3_LEVEL", 600_000_000_000_000.0, 1_200_000_000_000_000.0),
        ]:
            _insert_mr(c, LIQ_SIDS[sid_key], pd.DataFrame({
                "reference_date": monthly,
                "release_date":   monthly + pd.Timedelta(days=35),
                "value": np.linspace(lo, hi, m),
            }), source="FRED")
        for sid_key, val in [("EZ_M3_GROWTH", 0.3), ("JP_M3_GROWTH", 0.1)]:
            _insert_mr(c, LIQ_SIDS[sid_key], pd.DataFrame({
                "reference_date": monthly, "release_date": monthly,
                "value": [val] * m,
            }), source="FRED")
        for sid_key, val in [("EURUSD", 1.10), ("USDJPY", 110.0)]:
            _insert_mr(c, LIQ_SIDS[sid_key], pd.DataFrame({
                "reference_date": full, "release_date": full,
                "value": [val] * n,
            }), source="FRED")

        _insert_mr(c, "RBRTE", pd.DataFrame({
            "reference_date": full, "release_date": full,
            "value": np.linspace(20.0, 70.0, n),
        }), source="EIA")

        rng = np.random.default_rng(42)
        _insert_mr(c, GOLD_SIDS["XAU"], pd.DataFrame({
            "reference_date": full, "release_date": full,
            "value": 280.0 + np.cumsum(rng.normal(0, 1.0, n)),
        }), source=SOURCE_YAHOO)
        tips_dates = full[full >= "2003-01-02"]
        _insert_mr(c, GOLD_SIDS["TIPS"], pd.DataFrame({
            "reference_date": tips_dates, "release_date": tips_dates,
            "value": [2.0] * len(tips_dates),
        }), source=SOURCE_FRED)
        _insert_mr(c, GOLD_SIDS["DXY"], pd.DataFrame({
            "reference_date": full, "release_date": full,
            "value": [100.0] * n,
        }), source=SOURCE_FRED)
        _insert_mr(c, GOLD_SIDS["VIX"], pd.DataFrame({
            "reference_date": full, "release_date": full,
            "value": [20.0] * n,
        }), source=SOURCE_FRED)

        # Target asset
        asset_dates = pd.bdate_range("2000-01-03", "2010-12-31")
        insert_asset_prices(c, "TGT", pd.DataFrame({
            "trade_date": asset_dates,
            "close": np.linspace(100.0, 1000.0, len(asset_dates)),
        }), source="UNITTEST")

        yield c


def test_build_pipeline_for_asset_returns_division_and_figure(fully_seeded_con) -> None:
    """End-to-end: build_pipeline_for_asset reads asset closes, builds all 4
    numéraires, computes the division array, and produces a plotly figure."""
    import warnings as _w

    from rollender_stein.assets import build_pipeline_for_asset

    with _w.catch_warnings():
        _w.simplefilter("ignore")  # patch-04 warnings on synthetic numéraires
        result = build_pipeline_for_asset(
            fully_seeded_con, "TGT", end=pd.Timestamp("2010-12-31"),
        )
    assert result.ticker == "TGT"
    df = result.division.to_frame()
    assert "nominal_usd" in df.columns
    assert "asset_in_time" in df.columns
    assert "asset_in_gold" in df.columns
    # Figure has at least one trace.
    assert len(result.figure.data) >= 1


def test_build_pipeline_for_asset_raises_on_unknown_ticker(fully_seeded_con) -> None:
    from rollender_stein.assets import build_pipeline_for_asset

    with pytest.raises(RuntimeError, match=r"no rows in asset_price"):
        build_pipeline_for_asset(
            fully_seeded_con, "DOES_NOT_EXIST", end=pd.Timestamp("2010-12-31"),
        )


def test_build_pipeline_for_asset_animate_flag_produces_animated_figure(
    fully_seeded_con,
) -> None:
    """When animate=True, the resulting figure has frames (otherwise empty)."""
    import warnings as _w

    from rollender_stein.assets import build_pipeline_for_asset

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        static_result = build_pipeline_for_asset(
            fully_seeded_con, "TGT", end=pd.Timestamp("2010-12-31"), animate=False,
        )
        animated_result = build_pipeline_for_asset(
            fully_seeded_con, "TGT", end=pd.Timestamp("2010-12-31"), animate=True,
        )
    assert len(static_result.figure.frames) == 0
    assert len(animated_result.figure.frames) > 0
