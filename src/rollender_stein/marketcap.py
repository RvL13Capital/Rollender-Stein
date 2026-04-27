"""Market-cap absolute valuation — parallel layer to the per-share AVE.

Per-share Asset_in_X (the existing ``valuation.build_division_array``)
measures one share's purchasing power in T0-deflated USD. That's the right
view for an investor holding shares: dividends and splits are already in
via Yahoo's adjusted close.

Market-cap Asset_in_X measures the **whole company's** purchasing power in
T0-deflated USD by multiplying price by historical shares outstanding from
SEC EDGAR. The two views differ by buyback / issuance effects:

    market_cap(t)        = price(t) * shares(t)                  [USD]
    MarketCap_in_X(t)    = market_cap(t) / N_X(t) * 100          [T0-USD]

For buyback-heavy tickers (AAPL, MSFT, ORCL), per-share appreciates faster
than market_cap because the share count is shrinking — so per-share Asset_in_X
overstates the company's wealth creation. The market-cap view neutralizes this.

For issuance-heavy tickers (TSLA in capital raises, META in M&A), per-share
underperforms market_cap because share count grows.

Coverage: only US-listed individual companies that file 10-K/10-Q with the
SEC. Indexes, ETFs, futures, crypto, and foreign issuers are out of scope.
EDGAR's XBRL coverage starts ~2008-2009. Pre-2008 share counts must be
sourced elsewhere if needed.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from rollender_stein.bitemporal import (
    get_asset_closes,
    insert_shares_outstanding,
    latest_shares_stream,
)
from rollender_stein.calendar import master_calendar
from rollender_stein.io.sec_edgar import (
    fetch_company_shares_outstanding,
    fetch_ticker_to_cik,
)
from rollender_stein.locf import forward_fill_to_calendar

SOURCE = "SEC_EDGAR"


def ingest_sec_shares(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    user_agent: str,
    *,
    cik: int | None = None,
    ticker_to_cik: dict[str, int] | None = None,
) -> int:
    """Pull ``ticker``'s historical shares outstanding from SEC EDGAR and
    persist into ``shares_outstanding``.

    Pass ``cik`` or ``ticker_to_cik`` to skip the per-call ticker mapping
    fetch (useful when ingesting a batch of tickers — fetch the mapping
    once and reuse).

    SEC fair-access policy requires a meaningful User-Agent string of the
    form ``"OrgName contact@example.com"``. The caller is responsible for
    rate-limiting (10 req/sec ceiling) when batching.
    """
    if cik is None:
        if ticker_to_cik is None:
            ticker_to_cik = fetch_ticker_to_cik(user_agent)
        if ticker not in ticker_to_cik:
            raise RuntimeError(
                f"ticker {ticker!r} not found in SEC ticker mapping; "
                "may be foreign-listed, an ETF, or recently delisted."
            )
        cik = ticker_to_cik[ticker]

    rows = fetch_company_shares_outstanding(cik, user_agent)
    if rows.empty:
        return 0
    return insert_shares_outstanding(con, ticker, rows, source=SOURCE)


def _cumulative_future_split_factor(
    splits: pd.Series, calendar: pd.DatetimeIndex
) -> pd.Series:
    """For each date in ``calendar``, return the product of split ratios for
    all splits whose date is STRICTLY AFTER that date.

    yfinance retro-applies all future splits to historical prices: yfinance's
    Close on day d = raw_close(d) / cumulative_future_split_factor(d). To make
    SEC raw shares (point-in-time, NOT split-adjusted) compatible with
    yfinance's split-adjusted prices, we multiply shares by the same factor:
        adjusted_shares(d) = raw_shares(d) * cumulative_future_split_factor(d)
        adjusted_shares(d) * yfinance_close(d) = true_market_cap(d)

    Returns a Series indexed by ``calendar`` with float values (1.0 if no
    future splits).
    """
    if splits is None or splits.empty:
        return pd.Series(1.0, index=calendar, dtype="float64")
    splits_sorted = splits.sort_index()
    # Strip tz from split timestamps (yfinance returns tz-aware) for clean
    # comparison against the naive calendar.
    if isinstance(splits_sorted.index, pd.DatetimeIndex) and splits_sorted.index.tz is not None:
        splits_sorted.index = splits_sorted.index.tz_localize(None)
    factor = pd.Series(1.0, index=calendar, dtype="float64")
    # For each calendar date, cumulative product of splits with date strictly
    # greater than that calendar date. Iterate over splits in reverse-time
    # order so factors compound correctly (most-recent split applies last).
    for split_date, ratio in splits_sorted.items():
        # All calendar dates STRICTLY BEFORE split_date pick up this ratio.
        mask = factor.index < split_date
        factor.loc[mask] *= float(ratio)
    return factor


def _fetch_split_history(ticker: str) -> pd.Series:
    """Fetch the ticker's split history from yfinance. Returns an empty
    Series if no splits or if yfinance can't reach Yahoo. Errors are
    swallowed because a missing splits feed is a degraded-mode condition,
    not a fatal one.
    """
    try:
        import yfinance as yf

        s = yf.Ticker(ticker).splits
        if s is None or len(s) == 0:
            return pd.Series([], dtype="float64")
        out: pd.Series = s.astype("float64")
        return out
    except Exception:
        return pd.Series([], dtype="float64")


def build_market_cap(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    end: pd.Timestamp | None = None,
    splits: pd.Series | None = None,
) -> pd.Series:
    """Daily market cap for ``ticker`` in current-basis USD, indexed on the
    master NYSE calendar.

    Pipeline:
      1. Read shares stream (LOCF input) from the bitemporal store.
      2. LOCF raw SEC shares onto NYSE master calendar via filing_date.
      3. Apply cumulative-future-split factor to shares so they live in
         the same basis as yfinance's split-adjusted close column. Without
         this, mixing point-in-time SEC shares (e.g. 888M for AAPL in 2009)
         with yfinance's already-split-adjusted close ($5.60 for AAPL
         in 2009) produces market caps that are off by the cumulative
         split factor (28x for AAPL's 7-for-1 + 4-for-1 history).
      4. Multiply: market_cap(t) = adjusted_shares(t) * yfinance_close(t).

    The result is the company's true market cap on day t, expressed in
    current-share-basis dollars (which is what yfinance's prices live in).

    ``splits`` may be passed pre-fetched (testing / batching). If None, the
    function calls yfinance for the ticker's split history; failure is
    silent (returns 1.0 factors → still correct for split-free tickers).

    Returns NaN for dates before the asset's first close OR before the first
    SEC filing. Assets with no SEC filings (foreign, ETFs, futures) raise.
    """
    shares_stream = latest_shares_stream(con, ticker)
    if shares_stream.empty:
        raise RuntimeError(
            f"no SEC shares data for {ticker!r}; ingest with ingest_sec_shares() first "
            "or confirm the ticker is a US-listed individual company"
        )

    cal = master_calendar(end=end)
    # LOCF raw shares (point-in-time SEC counts) onto the daily calendar.
    shares_for_locf = shares_stream.rename(
        columns={"filing_date": "release_date", "shares": "shares_outstanding"}
    )[["release_date", "shares_outstanding"]]
    daily_raw_shares = forward_fill_to_calendar(
        shares_for_locf, cal, value_cols=["shares_outstanding"]
    )["shares_outstanding"].astype("float64")

    # Split-adjust shares to match yfinance's price basis.
    if splits is None:
        splits = _fetch_split_history(ticker)
    split_factor = _cumulative_future_split_factor(splits, cal)
    daily_adjusted_shares = daily_raw_shares * split_factor

    # Use yfinance's split-adjusted close (the "close" column post-refactor).
    # Mixing this with raw shares WAS the bug; with adjusted shares it works.
    closes = get_asset_closes(con, ticker, end=end, prefer_adjusted=False)
    if closes.empty:
        raise RuntimeError(
            f"no asset_price rows for {ticker!r}; ingest_yahoo_asset() first"
        )
    daily_prices = closes.reindex(cal, method="ffill")

    market_cap = daily_prices * daily_adjusted_shares
    return market_cap.rename(f"{ticker}_market_cap_usd")


def build_market_cap_division(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    n_time: pd.Series | None = None,
    n_liquidity: pd.Series | None = None,
    n_gold: pd.Series | None = None,
    n_energy: pd.Series | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute MarketCap_in_X for the four numéraires and return a DataFrame.

    Output columns:
        market_cap_usd       : nominal market cap in USD
        market_cap_in_time   : T0-deflated USD per N_Time
        market_cap_in_liquidity, market_cap_in_gold, market_cap_in_energy

    Numéraire arguments default to None; pass already-built series to avoid
    recomputation when batching multiple tickers (same caching trick as
    ``persist.dump_all_artifacts``).
    """
    if n_time is None or n_liquidity is None or n_energy is None or n_gold is None:
        from rollender_stein.numeraires.energy import build_n_energy
        from rollender_stein.numeraires.gold import build_n_gold
        from rollender_stein.numeraires.liquidity import build_n_liq
        from rollender_stein.numeraires.time import build_n_time

        if n_time is None:
            n_time = build_n_time(con, end=end)
        if n_liquidity is None:
            n_liquidity = build_n_liq(con, end=end)
        if n_energy is None:
            n_energy = build_n_energy(con, end=end)
        if n_gold is None:
            n_gold = build_n_gold(con, end=end)

    market_cap = build_market_cap(con, ticker, end=end)
    base_idx = market_cap.index

    out = pd.DataFrame({"market_cap_usd": market_cap})

    def _ratio(num: pd.Series, label: str) -> pd.Series:
        denom = num.reindex(base_idx)
        denom = denom.where(denom != 0)
        ratio: pd.Series = (market_cap / denom) * 100.0
        return ratio.rename(label)

    out["market_cap_in_time"] = _ratio(n_time, "market_cap_in_time")
    out["market_cap_in_liquidity"] = _ratio(n_liquidity, "market_cap_in_liquidity")
    out["market_cap_in_gold"] = _ratio(n_gold, "market_cap_in_gold")
    out["market_cap_in_energy"] = _ratio(n_energy, "market_cap_in_energy")

    return out


def per_share_vs_market_cap_multiplier(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    numeraire: str = "asset_in_gold",
    *,
    end: pd.Timestamp | None = None,
) -> dict[str, str | float]:
    """Quick diagnostic: how much does the per-share Asset_in_X overstate /
    understate the market-cap-equivalent? The ratio is a direct measure of
    the buyback (>1) or dilution (<1) effect.

    Reads the previously-dumped per-share division array from
    ``data/derived/divisions/{TICKER}.parquet``. If the user hasn't run
    dump_all_artifacts, this raises.

    The user's "T0_DATE" anchor for per-share is whatever build_division_array
    produced; for market_cap it's the first day shares x price are both known.
    Returned ratio uses each series' first-and-last values.
    """
    from pathlib import Path

    safe = ticker.replace("^", "").replace("=", "-").replace("/", "-")
    per_share_path = Path("data/derived/divisions") / f"{safe}.parquet"
    if not per_share_path.exists():
        raise RuntimeError(
            f"{per_share_path} missing; run dump_all_artifacts() first"
        )
    per_share = pd.read_parquet(per_share_path)
    if numeraire not in per_share.columns:
        raise KeyError(f"numeraire {numeraire!r} not in per-share frame")
    ps_clean = per_share[numeraire].dropna()
    if ps_clean.empty:
        raise RuntimeError(f"per-share {numeraire} has no values for {ticker}")

    mc_div = build_market_cap_division(con, ticker, end=end)
    mc_col = numeraire.replace("asset_in_", "market_cap_in_")
    mc_clean = mc_div[mc_col].dropna()
    if mc_clean.empty:
        raise RuntimeError(f"market_cap {mc_col} has no values for {ticker}")

    # Compare first / last in the OVERLAPPING window where both are defined.
    common = ps_clean.index.intersection(mc_clean.index)
    if len(common) == 0:
        raise RuntimeError("no overlap between per-share and market-cap windows")
    common = common.sort_values()
    first, last = common[0], common[-1]
    ps_mult = float(ps_clean.loc[last] / ps_clean.loc[first])
    mc_mult = float(mc_clean.loc[last] / mc_clean.loc[first])
    return {
        "ticker": ticker,
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "per_share_multiplier": ps_mult,
        "market_cap_multiplier": mc_mult,
        "ratio": ps_mult / mc_mult if mc_mult else float("nan"),
    }
