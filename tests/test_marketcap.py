from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from rollender_stein.bitemporal import (
    insert_asset_prices,
    insert_shares_outstanding,
    latest_shares_stream,
    open_db,
)
from rollender_stein.io.sec_edgar import (
    fetch_company_shares_outstanding,
    fetch_ticker_to_cik,
)
from rollender_stein.marketcap import (
    build_market_cap,
    build_market_cap_division,
)


@pytest.fixture
def con():
    with open_db(":memory:") as c:
        yield c


# ----- bitemporal helpers ----------------------------------------------------


def test_insert_shares_outstanding_round_trip(con) -> None:
    rows = pd.DataFrame(
        {
            "period_end_date": pd.to_datetime(["2023-09-30", "2023-12-31"]),
            "filing_date": pd.to_datetime(["2023-11-02", "2024-01-25"]),
            "shares": [15_500_000_000, 15_300_000_000],
            "form": ["10-Q", "10-K"],
        }
    )
    n = insert_shares_outstanding(con, "AAPL", rows, source="SEC_EDGAR")
    assert n == 2

    stream = latest_shares_stream(con, "AAPL")
    assert len(stream) == 2
    assert stream["shares"].tolist() == [15_500_000_000, 15_300_000_000]


def test_insert_shares_rejects_filing_before_period(con) -> None:
    """The CHECK / application-level guard rejects filing_date < period_end_date."""
    bad = pd.DataFrame(
        {
            "period_end_date": pd.to_datetime(["2024-01-31"]),
            "filing_date":     pd.to_datetime(["2024-01-15"]),  # before period end!
            "shares": [1_000_000],
        }
    )
    with pytest.raises(ValueError, match=r"filing_date.*period_end_date"):
        insert_shares_outstanding(con, "TST", bad, source="UNITTEST")


def test_latest_shares_stream_picks_earliest_filing_per_period(con) -> None:
    """If a period_end has multiple filings (10-Q + 10-K + 10-K/A), the
    stream picks the EARLIEST — the original print, not the amendment."""
    rows = pd.DataFrame(
        {
            "period_end_date": pd.to_datetime(
                ["2023-09-30", "2023-09-30", "2023-09-30"]
            ),
            "filing_date": pd.to_datetime(
                ["2023-11-02", "2024-01-25", "2024-03-15"]
            ),
            "shares": [15_500_000_000, 15_500_000_000, 15_510_000_000],
            "form": ["10-Q", "10-K", "10-K/A"],
        }
    )
    insert_shares_outstanding(con, "AAPL", rows, source="SEC_EDGAR")
    stream = latest_shares_stream(con, "AAPL")
    # Only the earliest 10-Q row (filed 2023-11-02) should appear
    assert len(stream) == 1
    # DuckDB returns DATE columns as datetime.date objects via fetchdf
    assert pd.Timestamp(stream.iloc[0]["filing_date"]) == pd.Timestamp("2023-11-02")


# ----- io/sec_edgar mocked ---------------------------------------------------


def _mock_get(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    s = MagicMock()
    s.get.return_value = resp
    return s


def test_fetch_ticker_to_cik_parses_response() -> None:
    payload = {
        "0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."},
        "1": {"ticker": "MSFT", "cik_str": 789019, "title": "Microsoft Corp."},
    }
    sess = _mock_get(payload)
    out = fetch_ticker_to_cik("UnitTest test@example.com", session=sess)
    assert out == {"AAPL": 320193, "MSFT": 789019}


def test_fetch_company_shares_parses_xbrl_facts() -> None:
    payload = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "end": "2023-09-30", "filed": "2023-11-02",
                                "val": 15_500_000_000, "form": "10-Q",
                            },
                            {
                                "end": "2023-12-31", "filed": "2024-01-25",
                                "val": 15_300_000_000, "form": "10-K",
                            },
                        ]
                    }
                }
            }
        },
    }
    sess = _mock_get(payload)
    df = fetch_company_shares_outstanding(320193, "UnitTest x@y.com", session=sess)
    assert len(df) == 2
    assert df.loc[0, "period_end_date"] == pd.Timestamp("2023-09-30")
    assert df.loc[0, "shares"] == 15_500_000_000


def test_fetch_company_shares_handles_missing_concept() -> None:
    """Some CIKs have no CommonStockSharesOutstanding entry (e.g. partnerships,
    trusts). The function returns an empty typed DataFrame, not raises."""
    payload = {"cik": 0, "entityName": "Foo", "facts": {"us-gaap": {}}}
    sess = _mock_get(payload)
    df = fetch_company_shares_outstanding(0, "UnitTest x@y.com", session=sess)
    assert df.empty
    assert df["period_end_date"].dtype == "datetime64[ns]"


# ----- build_market_cap end-to-end ------------------------------------------


def _seed_aapl_minimal(con) -> None:
    """Seed enough data to build market_cap for AAPL on a small calendar.
    Shares jump from 16B to 15B mid-window — simulates buyback effect."""
    insert_shares_outstanding(
        con, "AAPL",
        pd.DataFrame(
            {
                "period_end_date": pd.to_datetime(["2023-12-31", "2024-06-30"]),
                "filing_date":     pd.to_datetime(["2024-01-25", "2024-08-01"]),
                "shares": [16_000_000_000, 15_500_000_000],  # buyback
                "form": ["10-K", "10-Q"],
            }
        ),
        source="SEC_EDGAR",
    )

    asset_dates = pd.bdate_range("2024-01-26", "2024-12-31")
    insert_asset_prices(
        con, "AAPL",
        pd.DataFrame(
            {
                "trade_date": asset_dates,
                "close": [200.0] * len(asset_dates),  # constant price for clarity
            }
        ),
        source="UNITTEST",
    )


def test_build_market_cap_uses_shares_locf(con) -> None:
    """Market cap reflects the LOCF'd shares — when a new filing lands the
    daily market_cap steps."""
    _seed_aapl_minimal(con)
    mc = build_market_cap(con, "AAPL", end=pd.Timestamp("2024-12-31"))
    # Pre-2024-08-01 (first 10-Q with reduced shares filed): 200 * 16B
    # Post-2024-08-01: 200 * 15.5B
    pre = mc.loc[pd.Timestamp("2024-07-31")]
    post = mc.loc[pd.Timestamp("2024-08-01")]
    assert pre == pytest.approx(200.0 * 16_000_000_000)
    assert post == pytest.approx(200.0 * 15_500_000_000)
    # Buyback registered: market cap fell on the filing date despite flat price.
    assert post < pre


def test_build_market_cap_pre_first_filing_is_nan(con) -> None:
    """Before the first SEC filing, shares are unknown → market cap NaN."""
    _seed_aapl_minimal(con)
    mc = build_market_cap(con, "AAPL", end=pd.Timestamp("2024-12-31"))
    # 2024-01-24 is BEFORE the 2024-01-25 filing → no shares known yet
    assert pd.isna(mc.loc[pd.Timestamp("2024-01-24")])


def test_build_market_cap_raises_when_no_shares_data(con) -> None:
    """Tickers without SEC filings (ETFs, futures) raise loudly."""
    insert_asset_prices(
        con, "GC=F",
        pd.DataFrame(
            {"trade_date": pd.to_datetime(["2024-01-02"]), "close": [2000.0]}
        ),
        source="YAHOO",
    )
    with pytest.raises(RuntimeError, match=r"no SEC shares data"):
        build_market_cap(con, "GC=F", end=pd.Timestamp("2024-12-31"))


def test_cumulative_future_split_factor_no_splits(con) -> None:
    """If a ticker has never split, the factor is 1.0 everywhere."""
    from rollender_stein.marketcap import _cumulative_future_split_factor
    cal = pd.bdate_range("2010-01-04", "2020-12-31")
    factor = _cumulative_future_split_factor(pd.Series([], dtype="float64"), cal)
    assert (factor == 1.0).all()


def test_cumulative_future_split_factor_aapl_style() -> None:
    """AAPL's 7-for-1 (2014-06-09) and 4-for-1 (2020-08-31) splits.

    Historical date BEFORE both splits → factor 7*4 = 28
    Historical date BETWEEN the splits → factor 4
    Historical date AFTER both → factor 1
    """
    from rollender_stein.marketcap import _cumulative_future_split_factor
    splits = pd.Series(
        [7.0, 4.0],
        index=pd.to_datetime(["2014-06-09", "2020-08-31"]),
    )
    cal = pd.to_datetime(["2010-06-30", "2017-06-30", "2024-06-30"])
    factor = _cumulative_future_split_factor(splits, cal)
    assert factor.loc[pd.Timestamp("2010-06-30")] == 28.0
    assert factor.loc[pd.Timestamp("2017-06-30")] == 4.0
    assert factor.loc[pd.Timestamp("2024-06-30")] == 1.0


def test_cumulative_future_split_factor_strips_tz() -> None:
    """yfinance returns tz-aware split timestamps; the function must
    strip them to compare cleanly against a naive calendar."""
    from rollender_stein.marketcap import _cumulative_future_split_factor
    splits = pd.Series([2.0], index=pd.to_datetime(["2020-01-15"]).tz_localize("UTC"))
    cal = pd.to_datetime(["2019-12-31", "2020-06-30"])
    factor = _cumulative_future_split_factor(splits, cal)
    assert factor.loc[pd.Timestamp("2019-12-31")] == 2.0
    assert factor.loc[pd.Timestamp("2020-06-30")] == 1.0


def test_build_market_cap_applies_split_adjustment(con) -> None:
    """The full pipeline: SEC raw shares + yfinance split-adjusted close.

    Synthetic AAPL-like scenario: company had a 10-for-1 split mid-period.
    Pre-split: 100M shares at $100 = $10B market cap.
    Yfinance (modern view): pre-split close shows as $10 (split-adjusted).
    Without our split-adjustment to shares, $10 * 100M would give $1B —
    off by 10x. With the fix, shares are scaled to 1B, giving correct $10B.
    """
    insert_shares_outstanding(
        con, "TEST",
        pd.DataFrame(
            {
                "period_end_date": pd.to_datetime(["2020-12-31"]),
                "filing_date":     pd.to_datetime(["2021-01-15"]),
                "shares": [100_000_000],   # raw count BEFORE the future split
                "form": ["10-K"],
            }
        ),
        source="SEC_EDGAR",
    )
    # Asset price: yfinance has retro-split-adjusted Close ($10 instead of $100)
    insert_asset_prices(
        con, "TEST",
        pd.DataFrame(
            {
                "trade_date": pd.bdate_range("2021-01-15", "2025-12-31"),
                "close": [10.0] * len(pd.bdate_range("2021-01-15", "2025-12-31")),
            }
        ),
        source="UNITTEST",
    )
    # Pass an explicit splits Series simulating a 2023 10-for-1 split.
    splits = pd.Series([10.0], index=pd.to_datetime(["2023-06-01"]))
    mc = build_market_cap(con, "TEST", end=pd.Timestamp("2025-12-31"), splits=splits)
    # Pre-2023-06-01: shares 100M scaled by 10 -> 1B; price $10; mcap = $10B
    assert mc.loc[pd.Timestamp("2022-06-01")] == pytest.approx(10_000_000_000.0)
    # Post-2023-06-01: shares 100M (no future splits); price $10; mcap = $1B
    # That's wrong because we only seeded the 2020-12-31 filing; in real life a
    # post-split 10-Q would report 1B shares. With only the old filing LOCF'd:
    assert mc.loc[pd.Timestamp("2024-06-03")] == pytest.approx(1_000_000_000.0)


def test_build_market_cap_no_splits_data_falls_back_to_factor_one(con) -> None:
    """If yfinance returns no splits and we don't pass any, the factor is 1.0
    everywhere — multiplying raw shares by raw prices works correctly for
    tickers that never split (e.g. JNJ, KO, JPM)."""
    insert_shares_outstanding(
        con, "NEVERSPLIT",
        pd.DataFrame(
            {
                "period_end_date": pd.to_datetime(["2020-12-31"]),
                "filing_date":     pd.to_datetime(["2021-01-15"]),
                "shares": [1_000_000_000],
                "form": ["10-K"],
            }
        ),
        source="SEC_EDGAR",
    )
    insert_asset_prices(
        con, "NEVERSPLIT",
        pd.DataFrame(
            {
                "trade_date": pd.bdate_range("2021-01-15", "2022-12-31"),
                "close": [50.0] * len(pd.bdate_range("2021-01-15", "2022-12-31")),
            }
        ),
        source="UNITTEST",
    )
    mc = build_market_cap(
        con, "NEVERSPLIT", end=pd.Timestamp("2022-12-31"),
        splits=pd.Series([], dtype="float64"),
    )
    # 1B shares x $50 = $50B
    assert mc.loc[pd.Timestamp("2022-06-01")] == pytest.approx(50_000_000_000.0)


def test_build_market_cap_division_returns_four_axes(con) -> None:
    """The division frame has columns for all four numéraires plus the
    raw market_cap_usd."""
    _seed_aapl_minimal(con)
    n_time = pd.Series(
        [120.0] * 252, index=pd.bdate_range("2024-01-02", periods=252), name="N_Time"
    )
    n_liq = n_time.copy().rename("N_Liq") * 2
    n_e = n_time.copy().rename("N_Energy") * 1.5
    n_g = n_time.copy().rename("N_Gold") * 1.8

    df = build_market_cap_division(
        con, "AAPL",
        n_time=n_time, n_liquidity=n_liq, n_gold=n_g, n_energy=n_e,
        end=pd.Timestamp("2024-12-31"),
    )
    expected_cols = {
        "market_cap_usd",
        "market_cap_in_time", "market_cap_in_liquidity",
        "market_cap_in_gold", "market_cap_in_energy",
    }
    assert expected_cols <= set(df.columns)
    # On a date where shares x price = 200 x 16B = $3.2T, and N_Time = 120:
    # market_cap_in_time = 3.2T / 120 * 100 = 2.667T
    sample = df.loc[pd.Timestamp("2024-02-01")]
    assert sample["market_cap_in_time"] == pytest.approx(
        (200.0 * 16_000_000_000) / 120.0 * 100.0, rel=1e-9
    )
