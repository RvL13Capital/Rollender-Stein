"""Bitemporal storage for AVE — DuckDB-backed.

Phase 1.2 of the AVE spec. Every macroeconomic row carries BOTH:

    reference_date  — the period the value describes (e.g. 2024-01-31 for January AHETPI)
    release_date    — the day the public learned the value (e.g. 2024-02-02)

The PK on ``macro_release`` is (series_id, reference_date, release_date), supporting
ALFRED-style vintage histories: the same reference_period can have multiple release_dates
as agencies revise. Phase 2 LOCF consumes the ``latest_release_stream`` view, which
collapses to one row per release_date by taking the value tied to the latest
reference_date released on that day (the headline figure).

Storage: a single DuckDB file at ``data/ave.duckdb``. No server. Columnar. Schema is
portable to Postgres+Timescale if we ever need horizontal scale.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path("data") / "ave.duckdb"

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS macro_release (
    series_id      VARCHAR  NOT NULL,
    reference_date DATE     NOT NULL,
    release_date   DATE     NOT NULL,
    value          DOUBLE,
    source         VARCHAR  NOT NULL,
    vintage        VARCHAR,
    ingested_at    TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (series_id, reference_date, release_date),
    CHECK (release_date >= reference_date)
);

-- shares_outstanding: bitemporal historical share counts from SEC EDGAR.
-- One row per (ticker, period_end_date, filing_date) so amendments and
-- restatements are preserved. The latest_shares_stream query picks the
-- earliest filing per period_end (the original print, not amendments).
CREATE TABLE IF NOT EXISTS shares_outstanding (
    ticker          VARCHAR  NOT NULL,
    period_end_date DATE     NOT NULL,
    filing_date     DATE     NOT NULL,
    shares          BIGINT   NOT NULL,
    form            VARCHAR,
    source          VARCHAR  NOT NULL,
    ingested_at     TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (ticker, period_end_date, filing_date),
    CHECK (filing_date >= period_end_date)
);

CREATE TABLE IF NOT EXISTS asset_price (
    series_id    VARCHAR  NOT NULL,
    trade_date   DATE     NOT NULL,
    open         DOUBLE,
    high         DOUBLE,
    low          DOUBLE,
    close        DOUBLE,
    adj_close    DOUBLE,
    volume       DOUBLE,
    source       VARCHAR  NOT NULL,
    ingested_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (series_id, trade_date)
);

CREATE TABLE IF NOT EXISTS fx_close (
    pair         VARCHAR  NOT NULL,
    trade_date   DATE     NOT NULL,
    close        DOUBLE   NOT NULL,
    source       VARCHAR  NOT NULL,
    ingested_at  TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (pair, trade_date)
);
"""


@contextmanager
def open_db(path: Path | str = DEFAULT_DB_PATH) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the AVE DuckDB at ``path`` and ensure the schema + idempotent
    migrations have been applied.

    Pass ``":memory:"`` for an ephemeral in-memory DB (useful for tests).
    """
    if str(path) == ":memory:":
        con = duckdb.connect(":memory:")
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(str(p))
    try:
        con.execute(_SCHEMA_DDL)
        # Idempotent migrations — safe to re-run on every open. We emit a
        # warning instead of failing the open, because the migrations are
        # idempotent and the next open will retry; opening the DB is more
        # valuable than perfect schema parity. The warning preserves
        # diagnostic visibility — earlier `contextlib.suppress(Exception)`
        # silently swallowed migration failures (e.g. the pathological PK
        # collision when a shifted release_date coincides with an existing
        # revision row).
        try:
            migrate_publication_lags(con)
        except Exception as e:
            warnings.warn(
                f"migrate_publication_lags failed during open_db: "
                f"{type(e).__name__}: {e}. The DB is still usable; "
                "the migration will retry on next open.",
                RuntimeWarning,
                stacklevel=2,
            )
        yield con
    finally:
        con.close()


def _validate_release_after_reference(rows: pd.DataFrame) -> None:
    """Application-level enforcement of the CHECK constraint.

    DuckDB does not currently support ``ALTER TABLE ADD CONSTRAINT CHECK``
    on an existing table (NotImplementedException), so we cannot retro-apply
    the constraint to a pre-existing DB created before audit patch 01. The
    inline CHECK in ``_SCHEMA_DDL`` enforces it for new DBs.

    For pre-existing DBs we enforce the same invariant in software at the
    only documented write path (``insert_macro_releases``). Manual SQL
    INSERTs bypass this, but the audit's threat model is synthetic
    look-ahead injection via the application API, which is closed.
    """
    bad_mask = rows["release_date"] < rows["reference_date"]
    if bad_mask.any():
        bad = rows.loc[bad_mask, ["reference_date", "release_date"]].head(3)
        raise ValueError(
            f"release_date < reference_date violates audit patch 01 invariant; "
            f"first offending rows:\n{bad.to_string(index=False)}"
        )


def migrate_publication_lags(con: duckdb.DuckDBPyConnection) -> int:
    """Apply ``PUBLICATION_LAG_BD`` to existing rows that still have
    ``release_date == reference_date`` (audit patch 02).

    Idempotent and scoped: only updates rows for series_ids listed in
    ``PUBLICATION_LAG_BD`` with non-zero lag, AND only those rows where the
    lag has not already been applied (i.e. ``release_date == reference_date``).

    Returns the total number of rows updated across all series.
    """
    # Local import to avoid a circular dependency: io.fred → bitemporal would
    # be a cycle if PUBLICATION_LAG_BD lived there and we imported it at
    # module top.
    from rollender_stein.io.fred import PUBLICATION_LAG_BD

    total_updated = 0
    for series_id, lag_bd in PUBLICATION_LAG_BD.items():
        if lag_bd <= 0:
            continue

        # Read rows still at release == reference (i.e. not yet migrated).
        existing = con.execute(
            """
            SELECT reference_date, release_date, value, source, vintage
            FROM macro_release
            WHERE series_id = ? AND release_date = reference_date
            """,
            [series_id],
        ).fetchdf()
        if existing.empty:
            continue

        # Compute new release_dates with the BD offset.
        new_release = pd.to_datetime(existing["reference_date"]) + pd.tseries.offsets.BDay(
            lag_bd
        )

        # Detect potential PK collisions BEFORE attempting the INSERT.
        # The pathological case is a series with mixed history (e.g. WM2NS
        # has both ALFRED rows with release > reference AND live-endpoint
        # rows with release == reference). When the live-endpoint row's
        # shifted release_date happens to land on an existing ALFRED
        # row's release_date, the INSERT would PK-violate. Detect those
        # rows up front, warn per-collision, and skip them — preserving
        # the rest of the migration rather than rolling back the whole
        # series.
        existing_pks_df = con.execute(
            """
            SELECT reference_date, release_date FROM macro_release
            WHERE series_id = ? AND release_date != reference_date
            """,
            [series_id],
        ).fetchdf()
        existing_pks: set[tuple[object, object]] = set(
            zip(
                pd.to_datetime(existing_pks_df["reference_date"]).dt.date,
                pd.to_datetime(existing_pks_df["release_date"]).dt.date,
                strict=True,
            )
        )
        ref_dates = pd.to_datetime(existing["reference_date"]).dt.date
        new_rel_dates = new_release.dt.date
        collision_mask = pd.Series(
            [(r, n) in existing_pks for r, n in zip(ref_dates, new_rel_dates, strict=True)],
            index=existing.index,
        )
        if collision_mask.any():
            for ref_d, new_d in zip(
                ref_dates[collision_mask], new_rel_dates[collision_mask], strict=True
            ):
                warnings.warn(
                    f"migrate_publication_lags: PK collision on {series_id} "
                    f"reference={ref_d}, would-shift-release={new_d} (already "
                    "exists as a non-migrated row); leaving the un-migrated row "
                    "in place — re-ingest the series for cleaner state.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            existing = existing.loc[~collision_mask].reset_index(drop=True)
            new_release = new_release.loc[~collision_mask].reset_index(drop=True)
            if existing.empty:
                continue

        # Atomic transaction: delete the un-lagged rows (only the non-colliding
        # ones, by reference_date) then insert the shifted ones.
        new_rows = pd.DataFrame(
            {
                "series_id": series_id,
                "reference_date": existing["reference_date"].reset_index(drop=True),
                "release_date": new_release.dt.date.reset_index(drop=True),
                "value": existing["value"].reset_index(drop=True),
                "source": existing["source"].reset_index(drop=True),
                "vintage": existing["vintage"].reset_index(drop=True),
            }
        )

        cur = con.cursor()
        view_name = f"_migrate_{series_id}_{id(new_rows):x}".replace("-", "_").replace(".", "_")
        cur.register(view_name, new_rows)
        try:
            cur.execute("BEGIN TRANSACTION")
            # Tightened DELETE: scope to ONLY the reference_dates we are
            # about to re-insert. This preserves collision-excluded rows
            # (those whose shifted release_date would clash with an existing
            # revision row) — leaving them un-migrated rather than orphaning
            # them. Without this scoping, the broad
            # ``WHERE release_date = reference_date`` would also delete the
            # collision rows that we then DON'T re-insert.
            cur.execute(
                f"""
                DELETE FROM macro_release
                WHERE series_id = ?
                  AND release_date = reference_date
                  AND reference_date IN (
                      SELECT reference_date FROM {view_name}
                  )
                """,
                [series_id],
            )
            cur.execute(
                f"""
                INSERT INTO macro_release
                    (series_id, reference_date, release_date, value, source, vintage)
                SELECT series_id, reference_date, release_date, value, source, vintage
                FROM {view_name}
                """,
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        finally:
            cur.unregister(view_name)
        total_updated += len(existing)

    return total_updated


def insert_macro_releases(
    con: duckdb.DuckDBPyConnection,
    series_id: str,
    rows: pd.DataFrame,
    *,
    source: str,
    vintage: str | None = None,
) -> int:
    """Idempotently insert macro release rows. Existing rows on the PK are replaced.

    ``rows`` must have at minimum: ``reference_date``, ``release_date``, ``value``.
    Both date columns must be datetime-like; the DuckDB DATE column will store dates.
    Returns the number of rows written.
    """
    required = {"reference_date", "release_date", "value"}
    missing = required - set(rows.columns)
    if missing:
        raise KeyError(f"insert_macro_releases: missing columns {sorted(missing)}")

    df = rows[["reference_date", "release_date", "value"]].copy()
    # Audit patch 01: enforce release_date >= reference_date at the application
    # boundary. The inline CHECK in _SCHEMA_DDL covers freshly-created DBs;
    # this guard covers pre-existing DBs (DuckDB does not yet support
    # ALTER TABLE ADD CONSTRAINT CHECK).
    _validate_release_after_reference(df)
    df["series_id"] = series_id
    df["source"] = source
    df["vintage"] = vintage

    # Audit patch 07: per-call cursor + unique view name so concurrent inserts
    # on the same con do not collide. DuckDB connections are not thread-safe
    # by design; using cur = con.cursor() gives each call its own statement
    # context within the shared connection.
    cur = con.cursor()
    view_name = f"_incoming_macro_{id(df):x}"
    cur.register(view_name, df)
    try:
        cur.execute(
            f"""
            INSERT OR REPLACE INTO macro_release
                (series_id, reference_date, release_date, value, source, vintage)
            SELECT series_id, reference_date, release_date, value, source, vintage
            FROM {view_name}
            """,
        )
    finally:
        cur.unregister(view_name)
    return len(df)


def insert_asset_prices(
    con: duckdb.DuckDBPyConnection,
    series_id: str,
    rows: pd.DataFrame,
    *,
    source: str,
) -> int:
    """Idempotently insert daily OHLCV-style rows into ``asset_price``.

    ``rows`` must have ``trade_date``; any of ``open``/``high``/``low``/``close``/
    ``adj_close``/``volume`` may be omitted (stored as NULL). Existing rows on
    the (series_id, trade_date) PK are replaced.
    """
    if "trade_date" not in rows.columns:
        raise KeyError("insert_asset_prices: missing required column 'trade_date'")

    optional = ["open", "high", "low", "close", "adj_close", "volume"]
    df = rows[["trade_date", *[c for c in optional if c in rows.columns]]].copy()
    for col in optional:
        if col not in df.columns:
            df[col] = None
    df["series_id"] = series_id
    df["source"] = source

    # Audit patch 07: per-call cursor + unique view name (see insert_macro_releases).
    cur = con.cursor()
    view_name = f"_incoming_asset_{id(df):x}"
    cur.register(view_name, df)
    try:
        cur.execute(
            f"""
            INSERT OR REPLACE INTO asset_price
                (series_id, trade_date, open, high, low, close, adj_close, volume, source)
            SELECT series_id, trade_date, open, high, low, close, adj_close, volume, source
            FROM {view_name}
            """,
        )
    finally:
        cur.unregister(view_name)
    return len(df)


def insert_shares_outstanding(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    rows: pd.DataFrame,
    *,
    source: str,
) -> int:
    """Idempotently insert quarterly shares-outstanding history into
    ``shares_outstanding``. Existing rows on the (ticker, period_end_date,
    filing_date) PK are replaced.

    ``rows`` must have at minimum: ``period_end_date``, ``filing_date``,
    ``shares``. Optional: ``form`` (10-K, 10-Q, etc.).
    """
    required = {"period_end_date", "filing_date", "shares"}
    missing = required - set(rows.columns)
    if missing:
        raise KeyError(f"insert_shares_outstanding: missing columns {sorted(missing)}")

    # Application-level CHECK (mirrors the schema CHECK; kept for
    # pre-existing DBs since DuckDB doesn't yet support ALTER TABLE ADD CONSTRAINT).
    bad_mask = rows["filing_date"] < rows["period_end_date"]
    if bad_mask.any():
        bad = rows.loc[bad_mask].head(3)
        raise ValueError(
            f"filing_date < period_end_date violates bitemporal invariant; "
            f"first offending rows:\n{bad.to_string(index=False)}"
        )

    df = rows[["period_end_date", "filing_date", "shares"]].copy()
    df["form"] = rows["form"] if "form" in rows.columns else None
    df["ticker"] = ticker
    df["source"] = source

    cur = con.cursor()
    view_name = f"_incoming_shares_{id(df):x}"
    cur.register(view_name, df)
    try:
        cur.execute(
            f"""
            INSERT OR REPLACE INTO shares_outstanding
                (ticker, period_end_date, filing_date, shares, form, source)
            SELECT ticker, period_end_date, filing_date, shares, form, source
            FROM {view_name}
            """,
        )
    finally:
        cur.unregister(view_name)
    return len(df)


def latest_shares_stream(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
) -> pd.DataFrame:
    """Return one row per FILING_DATE for ``ticker``, after two collapse stages:
    (1) per period_end, keep the EARLIEST filing_date (original print, not
        amendments — analogous to first-release vintaging);
    (2) per filing_date (post stage 1), keep the LATEST period_end (the
        "headline" disclosure for that filing — same logic
        ``latest_release_stream`` uses on macro_release).

    The LOCF helper requires unique release_dates; this two-stage dedup
    guarantees that. Output is sorted by filing_date — the canonical input
    to the LOCF-onto-daily-calendar step in market-cap construction.
    """
    return con.execute(
        """
        SELECT period_end_date, filing_date, shares
        FROM (
            SELECT
                period_end_date, filing_date, shares,
                ROW_NUMBER() OVER (
                    PARTITION BY filing_date
                    ORDER BY period_end_date DESC
                ) AS rn2
            FROM (
                SELECT
                    period_end_date, filing_date, shares,
                    ROW_NUMBER() OVER (
                        PARTITION BY period_end_date
                        ORDER BY filing_date ASC
                    ) AS rn1
                FROM shares_outstanding
                WHERE ticker = ?
            )
            WHERE rn1 = 1
        )
        WHERE rn2 = 1
        ORDER BY filing_date ASC
        """,
        [ticker],
    ).fetchdf()


def get_asset_closes(
    con: duckdb.DuckDBPyConnection,
    series_id: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    prefer_adjusted: bool = True,
) -> pd.Series:
    """Read the daily close series for ``series_id`` as a Series indexed by trade_date.

    ``prefer_adjusted=True`` (default) returns the dividend+split-adjusted close
    (TR-equivalent for equities). Falls back to raw close where adj_close is NaN
    (futures, crypto). For market-cap absolute valuation, pass
    ``prefer_adjusted=False`` to get raw close — adj_close has dividend
    reinvestment baked in, which makes ``raw_shares x adj_close`` an incorrect
    proxy for market cap.

    Empty Series if the series has not been ingested.
    """
    where = ["series_id = ?"]
    params: list[object] = [series_id]
    if start is not None:
        where.append("trade_date >= ?")
        params.append(start.date() if hasattr(start, "date") else start)
    if end is not None:
        where.append("trade_date <= ?")
        params.append(end.date() if hasattr(end, "date") else end)
    sql = f"""
        SELECT trade_date, close, adj_close
        FROM asset_price
        WHERE {" AND ".join(where)}
        ORDER BY trade_date ASC
    """
    df = con.execute(sql, params).fetchdf()
    if df.empty:
        return pd.Series(name=series_id, dtype="float64")

    if prefer_adjusted:
        # Use adj_close where present; fall back to close (futures, crypto,
        # any pre-deprecation rows where adj_close column may not have data)
        out = df["adj_close"].where(df["adj_close"].notna(), df["close"])
    else:
        out = df["close"]
    s: pd.Series = pd.Series(
        out.to_numpy(),
        index=pd.DatetimeIndex(df["trade_date"]),
        name=series_id,
    )
    return s


def get_asset_volume(
    con: duckdb.DuckDBPyConnection,
    series_id: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Read the daily volume series for ``series_id`` indexed by trade_date.

    Returned values are the yfinance ``Volume`` column as stored — share count
    for stocks/ETFs, contract count for futures (``=F``), coin count for
    crypto. Rows where ``volume`` is NULL in the DB are dropped from the
    output (they are not represented as NaN, since "volume not recorded" and
    "volume = 0" are semantically different and we conservatively assume the
    former for NULLs in legacy ingests).

    Empty Series if the series has not been ingested or has no recorded
    volume (e.g. ``^SP500TR`` — the index is not a tradeable instrument).
    """
    where = ["series_id = ?"]
    params: list[object] = [series_id]
    if start is not None:
        where.append("trade_date >= ?")
        params.append(start.date() if hasattr(start, "date") else start)
    if end is not None:
        where.append("trade_date <= ?")
        params.append(end.date() if hasattr(end, "date") else end)
    sql = f"""
        SELECT trade_date, volume
        FROM asset_price
        WHERE {" AND ".join(where)} AND volume IS NOT NULL
        ORDER BY trade_date ASC
    """
    df = con.execute(sql, params).fetchdf()
    if df.empty:
        # Explicit DatetimeIndex on the empty Series so callers that .align()
        # against a populated DatetimeIndex series don't blow up on
        # incompatible index types. Pandas' default RangeIndex would silently
        # break that assumption for downstream code.
        return pd.Series(
            name=series_id,
            dtype="float64",
            index=pd.DatetimeIndex([]),
        )
    s: pd.Series = pd.Series(
        df["volume"].to_numpy(),
        index=pd.DatetimeIndex(df["trade_date"]),
        name=series_id,
    )
    return s


def latest_release_stream(
    con: duckdb.DuckDBPyConnection,
    series_id: str,
) -> pd.DataFrame:
    """Collapse to one row per release_date, taking the value of the latest reference_date.

    For each ``release_date`` that touched ``series_id``, the returned row holds the
    value associated with the most recent ``reference_date`` released on that day.
    Same-day revisions of older reference periods are ignored in favor of the new
    headline period. The result is sorted by release_date and is the canonical input
    to the Phase 2 LOCF utility.
    """
    return con.execute(
        """
        SELECT release_date, value
        FROM (
            SELECT
                release_date,
                value,
                ROW_NUMBER() OVER (
                    PARTITION BY release_date
                    ORDER BY reference_date DESC
                ) AS rn
            FROM macro_release
            WHERE series_id = ?
        )
        WHERE rn = 1
        ORDER BY release_date ASC
        """,
        [series_id],
    ).fetchdf()
