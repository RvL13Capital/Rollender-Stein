"""Refresh-pipeline orchestrator — the heart of the background engine.

Executes the full AVE refresh in dependency order:

    macro_ingest  →  asset_ingest  →  sec_ingest  →  dump_all_artifacts  →  dashboards

Per-step error tolerance: a failure in one step is recorded into the
``RunResult`` but downstream steps that don't depend on it can still
proceed. The macro layer is the strict prerequisite for ``dump_all_artifacts``;
asset and SEC are independent of each other.

Forensic continuity: each step's row count and duration is recorded so
the consumer (CLI, future cron-watcher, GitHub Actions log) can detect
silent regressions — e.g. yfinance returning zero rows where it should
return a daily increment.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any

import duckdb

from rollender_stein.assets import (
    build_pipeline_for_asset,
    ingest_yahoo_asset,
    save_asset_dashboard,
)
from rollender_stein.bitemporal import open_db
from rollender_stein.config import eia_api_key, fred_api_key
from rollender_stein.marketcap import ingest_sec_shares
from rollender_stein.numeraires.energy import ingest_brent_spot
from rollender_stein.numeraires.gold import ingest_gold_inputs
from rollender_stein.numeraires.liquidity import ingest_liquidity_inputs
from rollender_stein.numeraires.time import ingest_ahetpi
from rollender_stein.persist import dump_all_artifacts
from rollender_stein.runconfig import RefreshConfig


@dataclass(frozen=True)
class StepResult:
    """One step of the refresh pipeline. ``rows`` semantics: for ingest
    steps it's number of rows inserted/replaced; for dump it's a coarse
    sum across artefacts; for dashboard it's 1 (binary success)."""

    step: str
    success: bool
    rows: int = 0
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class RunResult:
    started_at: datetime
    finished_at: datetime
    steps: tuple[StepResult, ...]
    dashboards_written: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return all(s.success for s in self.steps)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if not s.success)


def _run_step(name: str, fn: Callable[[], int]) -> StepResult:
    """Execute ``fn``, capturing any exception into a failed StepResult.

    Per-step isolation is deliberate: one bad ticker shouldn't kill the
    whole refresh. The caller decides whether downstream steps can
    proceed despite an upstream failure.
    """
    start = time.monotonic()
    try:
        rows = fn()
        return StepResult(
            step=name,
            success=True,
            rows=int(rows or 0),
            duration_seconds=time.monotonic() - start,
        )
    except Exception as exc:
        return StepResult(
            step=name,
            success=False,
            rows=0,
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=time.monotonic() - start,
        )


def _dump_row_count(manifest: dict[str, Any]) -> int:
    """Sum row counts across the manifest's leaf ArtifactInfo entries.

    Coarse — meant only for run-summary, not for accounting. Numeraires
    and divisions contribute; kalman/panel are excluded as they share rows
    with the inputs.
    """
    total = 0
    for section in ("numeraires", "divisions"):
        for entry in manifest.get(section, {}).values():
            if isinstance(entry, dict) and "rows" in entry:
                total += int(entry["rows"])
    return total


def _build_dashboard(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    config: RefreshConfig,
) -> tuple[StepResult, str | None]:
    start = time.monotonic()
    try:
        result = build_pipeline_for_asset(
            con,
            ticker,
            animate=config.dashboard_animate,
            recency_window_days=config.dashboard_recency_window_days,
            recency_floor=config.dashboard_recency_floor,
        )
        path = save_asset_dashboard(
            result,
            out_dir=str(config.db_path.parent),
            suffix="_refresh",
        )
        return (
            StepResult(
                step=f"dashboard:{ticker}",
                success=True,
                rows=1,
                duration_seconds=time.monotonic() - start,
            ),
            path,
        )
    except Exception as exc:
        return (
            StepResult(
                step=f"dashboard:{ticker}",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.monotonic() - start,
            ),
            None,
        )


def _dry_run(config: RefreshConfig, started: datetime) -> RunResult:
    """Build a synthetic RunResult listing the steps that *would* execute,
    without touching FRED/EIA/Yahoo/SEC or the database. Useful for
    config validation before a real run."""
    plan: list[StepResult] = []
    if not config.skip_macro:
        for s in ("ingest_ahetpi", "ingest_liquidity", "ingest_brent", "ingest_gold"):
            plan.append(StepResult(step=s, success=True, rows=0, duration_seconds=0.0))
    if not config.skip_assets:
        for tk in config.yahoo_tickers:
            plan.append(StepResult(step=f"yahoo:{tk}", success=True, rows=0, duration_seconds=0.0))
    if not config.skip_sec:
        for tk in config.sec_filers:
            plan.append(StepResult(step=f"sec:{tk}", success=True, rows=0, duration_seconds=0.0))
    if not config.skip_dump:
        plan.append(StepResult(
            step="dump_all_artifacts", success=True, rows=0, duration_seconds=0.0,
        ))
    if not config.skip_dashboards:
        for tk in config.dashboard_tickers:
            plan.append(StepResult(
                step=f"dashboard:{tk}", success=True, rows=0, duration_seconds=0.0,
            ))
    return RunResult(
        started_at=started,
        finished_at=datetime.now(UTC),
        steps=tuple(plan),
        dashboards_written=(),
    )


def refresh(config: RefreshConfig, *, dry: bool = False) -> RunResult:
    """Execute the full refresh pipeline, returning a RunResult summary.

    ``dry=True`` skips all I/O and returns the plan only. Useful for
    config-validation in CI or before unattended runs.

    API keys are read from ``.env`` via ``rollender_stein.config``. Both
    FRED and EIA keys are required for the macro layer; the SEC layer
    needs ``sec_user_agent`` set in the config (no API key, but SEC fair-
    access requires a contact identifier in the User-Agent header).
    """
    started = datetime.now(UTC)
    if dry:
        return _dry_run(config, started)

    steps: list[StepResult] = []
    dashboards: list[str] = []

    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Open DB once and run all steps within the same connection — DuckDB
    # holds an exclusive file lock; sharing the connection avoids lock
    # contention with parallel ingest tasks.
    with open_db(config.db_path) as con:
        if not config.skip_macro:
            try:
                fred_key = fred_api_key()
                eia_key = eia_api_key()
            except Exception as exc:
                steps.append(
                    StepResult(
                        step="macro:keys",
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                steps.append(_run_step(
                    "ingest_ahetpi",
                    lambda: ingest_ahetpi(con, fred_key),
                ))
                steps.append(_run_step(
                    "ingest_liquidity",
                    lambda: sum(ingest_liquidity_inputs(con, fred_key).values()),
                ))
                steps.append(_run_step(
                    "ingest_brent",
                    lambda: ingest_brent_spot(con, eia_key),
                ))
                steps.append(_run_step(
                    "ingest_gold",
                    lambda: sum(ingest_gold_inputs(con, fred_key).values()),
                ))

        if not config.skip_assets:
            for ticker in config.yahoo_tickers:
                steps.append(_run_step(
                    f"yahoo:{ticker}",
                    partial(ingest_yahoo_asset, con, ticker),
                ))

        if not config.skip_sec and config.sec_filers:
            if not config.sec_user_agent:
                steps.append(StepResult(
                    step="sec",
                    success=False,
                    error="sec.user_agent missing — SEC fair-access requires "
                    "a contact identifier (e.g. 'YourOrg you@example.com')",
                ))
            else:
                for ticker in config.sec_filers:
                    steps.append(_run_step(
                        f"sec:{ticker}",
                        partial(ingest_sec_shares, con, ticker, config.sec_user_agent),
                    ))

        if not config.skip_dump:
            tickers_for_dump = list(config.yahoo_tickers)

            def _do_dump() -> int:
                manifest = dump_all_artifacts(
                    con, tickers=tickers_for_dump, root=config.output_dir,
                )
                return _dump_row_count(manifest)

            steps.append(_run_step("dump_all_artifacts", _do_dump))

        if not config.skip_dashboards:
            for ticker in config.dashboard_tickers:
                step_result, path = _build_dashboard(con, ticker, config)
                steps.append(step_result)
                if path is not None:
                    dashboards.append(path)

    return RunResult(
        started_at=started,
        finished_at=datetime.now(UTC),
        steps=tuple(steps),
        dashboards_written=tuple(dashboards),
    )
