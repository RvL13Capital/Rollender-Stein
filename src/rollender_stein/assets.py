"""Target-asset ingest and orchestration.

Phase 5 sits on top of the four numéraires; the AVE divides the target
asset's nominal USD price by each numéraire to produce the absolute-value
arrays. This module handles the *target side* — fetching the asset, persisting
it to the bitemporal store's ``asset_price`` table, and orchestrating the
end-to-end pipeline (numéraires + division + figure) for any ticker.

For equity targets the spec requires Total Return indices (``^SP500TR``);
for crypto, BTC-USD via yfinance is the canonical reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd
import plotly.graph_objects as go

from rollender_stein.bitemporal import (
    get_asset_closes,
    insert_asset_prices,
)
from rollender_stein.dashboard import build_phase_space_figure, save_dashboard_html
from rollender_stein.io.yahoo import fetch_yahoo_ohlcv
from rollender_stein.numeraires.energy import build_n_energy
from rollender_stein.numeraires.gold import build_n_gold
from rollender_stein.numeraires.liquidity import build_n_liq
from rollender_stein.numeraires.time import build_n_time
from rollender_stein.valuation import DivisionArray, build_division_array


def ingest_yahoo_asset(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    start: str = "1990-01-01",
    use_adjusted_as_close: bool = False,
) -> int:
    """Pull ``ticker`` OHLCV from Yahoo and persist into ``asset_price``.

    Idempotent: re-running replaces rows on (series_id, trade_date).

    ``use_adjusted_as_close``: when True, the dividend+split-adjusted close
    (Yahoo's "Adj Close") is stored in the ``close`` column. The rest of the
    pipeline (``get_asset_closes`` → ``build_division_array``) reads ``close``,
    so this is how individual stocks get total-return treatment per the AVE
    spec's "must use TR for equities" rule. For pre-TR indexes like ``^SP500TR``
    or for non-dividend assets like ``BTC-USD`` / futures, leave False.
    """
    rows = fetch_yahoo_ohlcv(ticker, start=start)
    if use_adjusted_as_close and "adj_close" in rows.columns:
        adj = rows["adj_close"]
        if adj.notna().any():
            rows = rows.copy()
            rows["close"] = adj
    return insert_asset_prices(con, ticker, rows, source="YAHOO")


@dataclass(frozen=True)
class AssetPipelineResult:
    ticker: str
    division: DivisionArray
    figure: go.Figure


def build_pipeline_for_asset(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    end: pd.Timestamp | None = None,
    animate: bool = False,
    title: str | None = None,
) -> AssetPipelineResult:
    """Run the full Phase 3-6 pipeline for ``ticker``.

    Assumes all numéraire inputs are already ingested in the DB. Reads the
    asset's close series from ``asset_price`` (must have been ingested via
    ``ingest_yahoo_asset`` first), builds the four numéraires, computes the
    division array, and returns the dashboard figure ready to save.
    """
    closes = get_asset_closes(con, ticker, end=end)
    if closes.empty:
        raise RuntimeError(
            f"no rows in asset_price for {ticker!r}; call ingest_yahoo_asset() first"
        )

    n_time = build_n_time(con, end=end)
    n_liq = build_n_liq(con, end=end)
    n_energy = build_n_energy(con, end=end)
    n_gold = build_n_gold(con, end=end)

    division = build_division_array(
        closes,
        n_time=n_time,
        n_liquidity=n_liq,
        n_gold=n_gold,
        n_energy=n_energy,
    )
    figure = build_phase_space_figure(
        division,
        x="asset_in_energy",
        y="asset_in_liquidity",
        z="asset_in_gold",
        title=title or f"AVE: {ticker} Phase-Space Attractor",
        animate=animate,
    )
    return AssetPipelineResult(ticker=ticker, division=division, figure=figure)


def save_asset_dashboard(
    result: AssetPipelineResult,
    out_dir: str = "data",
    *,
    suffix: str = "",
) -> str:
    """Save ``result.figure`` to ``out_dir/dashboard_{ticker}{suffix}.html``.

    Returns the path written. ``ticker`` is sanitized — `^` and `=` become `-`
    so the filename is shell-friendly.
    """
    safe = result.ticker.replace("^", "").replace("=", "-").replace("/", "-")
    path = f"{out_dir}/dashboard_{safe}{suffix}.html"
    save_dashboard_html(result.figure, path)
    return path
