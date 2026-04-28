"""Tests for the refresh orchestrator.

We don't hit the network here. Each ingest function is monkey-patched to
a stub that returns a known row count or raises a known error; the test
asserts that the orchestrator records the right step results, applies
skip flags correctly, and isolates failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rollender_stein import run as run_mod
from rollender_stein.run import RunResult, StepResult, refresh
from rollender_stein.runconfig import RefreshConfig


@pytest.fixture
def isolated_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Use temp dirs so the orchestrator doesn't touch real data/."""
    return tmp_path / "ave.duckdb", tmp_path / "derived"


def _stub_count(n: int):
    """Returns a callable suitable for monkeypatching ingest functions."""

    def _stub(*args: Any, **kwargs: Any) -> int:
        return n

    return _stub


def _stub_dict(d: dict[str, int]):
    def _stub(*args: Any, **kwargs: Any) -> dict[str, int]:
        return d

    return _stub


def _stub_dump(
    _con: Any, *, tickers: Any = None, end: Any = None, root: Any = None,
) -> dict[str, Any]:
    artefact = {
        "path": "x",
        "rows": 100,
        "first_date": "2000-01-03",
        "last_date": "2026-04-28",
    }
    return {
        "numeraires": {"n_time": artefact},
        "divisions": {
            (tk if isinstance(tk, str) else "x"): {**artefact, "rows": 50}
            for tk in (tickers or [])
        },
    }


def test_dry_run_lists_planned_steps_without_io(
    isolated_paths: tuple[Path, Path],
) -> None:
    db, derived = isolated_paths
    cfg = RefreshConfig(
        db_path=db,
        output_dir=derived,
        yahoo_tickers=("AAPL", "BTC-USD"),
        sec_filers=("AAPL",),
        sec_user_agent="X y@z.com",
        dashboard_tickers=("AAPL",),
    )
    result = refresh(cfg, dry=True)
    step_names = [s.step for s in result.steps]
    # Macro layer planned
    assert "ingest_ahetpi" in step_names
    assert "ingest_liquidity" in step_names
    assert "ingest_brent" in step_names
    assert "ingest_gold" in step_names
    # Per-ticker steps planned
    assert "yahoo:AAPL" in step_names
    assert "yahoo:BTC-USD" in step_names
    assert "sec:AAPL" in step_names
    assert "dump_all_artifacts" in step_names
    assert "dashboard:AAPL" in step_names
    # Dry run files weren't created
    assert not db.exists()
    assert result.success is True


def test_skip_flags_omit_phases(isolated_paths: tuple[Path, Path]) -> None:
    db, derived = isolated_paths
    cfg = RefreshConfig(
        db_path=db,
        output_dir=derived,
        yahoo_tickers=("AAPL",),
        skip_macro=True,
        skip_dashboards=True,
        skip_dump=True,
        skip_sec=True,
    )
    result = refresh(cfg, dry=True)
    step_names = {s.step for s in result.steps}
    assert "ingest_ahetpi" not in step_names
    assert "dump_all_artifacts" not in step_names
    assert not any(s.startswith("dashboard:") for s in step_names)
    assert not any(s.startswith("sec:") for s in step_names)
    # Yahoo still in
    assert "yahoo:AAPL" in step_names


def test_orchestrator_records_per_step_failures_in_isolation(
    monkeypatch: pytest.MonkeyPatch, isolated_paths: tuple[Path, Path],
) -> None:
    """One bad ticker mustn't kill the whole run — downstream steps still
    execute and record their own results."""
    db, derived = isolated_paths

    monkeypatch.setattr(run_mod, "fred_api_key", lambda: "fake-fred")
    monkeypatch.setattr(run_mod, "eia_api_key", lambda: "fake-eia")
    monkeypatch.setattr(run_mod, "ingest_ahetpi", _stub_count(120))
    monkeypatch.setattr(run_mod, "ingest_liquidity_inputs", _stub_dict({"a": 50, "b": 70}))
    monkeypatch.setattr(run_mod, "ingest_brent_spot", _stub_count(6500))
    monkeypatch.setattr(run_mod, "ingest_gold_inputs", _stub_dict({"x": 1000, "y": 1500}))

    def _bad_yahoo(_con: Any, ticker: str, **_kw: Any) -> int:
        if ticker == "BAD":
            raise RuntimeError("yfinance returned 0 rows")
        return 100

    monkeypatch.setattr(run_mod, "ingest_yahoo_asset", _bad_yahoo)
    monkeypatch.setattr(run_mod, "dump_all_artifacts", _stub_dump)

    cfg = RefreshConfig(
        db_path=db,
        output_dir=derived,
        yahoo_tickers=("AAPL", "BAD", "MSFT"),
        skip_dashboards=True,  # don't render real plotly figures
        skip_sec=True,
    )
    result = refresh(cfg)

    step_by_name = {s.step: s for s in result.steps}
    assert step_by_name["yahoo:AAPL"].success is True
    assert step_by_name["yahoo:AAPL"].rows == 100
    assert step_by_name["yahoo:BAD"].success is False
    assert "yfinance returned 0 rows" in (step_by_name["yahoo:BAD"].error or "")
    assert step_by_name["yahoo:MSFT"].success is True  # downstream of BAD still ran
    # Dump still ran (independent of asset failures)
    assert step_by_name["dump_all_artifacts"].success is True
    # Overall: failed (because BAD)
    assert result.success is False
    assert result.n_failed == 1


def test_sec_step_fails_loudly_on_missing_user_agent(
    monkeypatch: pytest.MonkeyPatch, isolated_paths: tuple[Path, Path],
) -> None:
    db, derived = isolated_paths
    monkeypatch.setattr(run_mod, "fred_api_key", lambda: "k")
    monkeypatch.setattr(run_mod, "eia_api_key", lambda: "k")
    monkeypatch.setattr(run_mod, "ingest_ahetpi", _stub_count(0))
    monkeypatch.setattr(run_mod, "ingest_liquidity_inputs", _stub_dict({}))
    monkeypatch.setattr(run_mod, "ingest_brent_spot", _stub_count(0))
    monkeypatch.setattr(run_mod, "ingest_gold_inputs", _stub_dict({}))
    monkeypatch.setattr(run_mod, "dump_all_artifacts", _stub_dump)

    cfg = RefreshConfig(
        db_path=db, output_dir=derived,
        sec_filers=("AAPL",),
        sec_user_agent="",  # missing
        skip_dashboards=True,
    )
    result = refresh(cfg)
    sec_step = next(s for s in result.steps if s.step == "sec")
    assert sec_step.success is False
    assert "user_agent" in (sec_step.error or "")


def test_run_result_aggregates_duration_and_success() -> None:
    from datetime import UTC, datetime, timedelta
    started = datetime(2026, 1, 1, tzinfo=UTC)
    finished = started + timedelta(seconds=12)
    result = RunResult(
        started_at=started,
        finished_at=finished,
        steps=(
            StepResult(step="a", success=True, rows=10, duration_seconds=1.0),
            StepResult(step="b", success=False, rows=0, duration_seconds=2.0,
                       error="boom"),
        ),
    )
    assert result.duration_seconds == pytest.approx(12.0)
    assert result.success is False
    assert result.n_failed == 1
