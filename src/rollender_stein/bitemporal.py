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
    PRIMARY KEY (series_id, reference_date, release_date)
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
    """Open the AVE DuckDB at ``path`` and ensure the schema exists.

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
        yield con
    finally:
        con.close()


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
    df["series_id"] = series_id
    df["source"] = source
    df["vintage"] = vintage

    con.register("_incoming", df)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO macro_release
                (series_id, reference_date, release_date, value, source, vintage)
            SELECT series_id, reference_date, release_date, value, source, vintage
            FROM _incoming
            """,
        )
    finally:
        con.unregister("_incoming")
    return len(df)


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
