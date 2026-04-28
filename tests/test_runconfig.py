"""Tests for the YAML-config loader of the AVE refresh engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from rollender_stein.runconfig import RefreshConfig


def test_default_config_has_safe_paths_and_empty_lists() -> None:
    cfg = RefreshConfig()
    assert cfg.db_path == Path("data/ave.duckdb")
    assert cfg.output_dir == Path("data/derived")
    assert cfg.yahoo_tickers == ()
    assert cfg.sec_filers == ()
    assert cfg.dashboard_tickers == ()
    # All skip flags default off
    assert not cfg.skip_macro
    assert not cfg.skip_assets
    assert not cfg.skip_sec
    assert not cfg.skip_dump
    assert not cfg.skip_dashboards


def test_from_dict_parses_full_yaml(tmp_path: Path) -> None:
    data = {
        "db_path": "/tmp/test.duckdb",
        "output_dir": "/tmp/derived",
        "yahoo": {"tickers": ["AAPL", "MSFT"]},
        "sec": {"user_agent": "Org foo@bar.com", "filers": ["AAPL"]},
        "dashboards": {
            "tickers": ["AAPL"],
            "recency_window_days": 1825,
            "recency_floor": 0.05,
            "animate": True,
        },
        "skip": {"sec": True},
    }
    cfg = RefreshConfig.from_dict(data)
    assert cfg.db_path == Path("/tmp/test.duckdb")
    assert cfg.output_dir == Path("/tmp/derived")
    assert cfg.yahoo_tickers == ("AAPL", "MSFT")
    assert cfg.sec_filers == ("AAPL",)
    assert cfg.sec_user_agent == "Org foo@bar.com"
    assert cfg.dashboard_tickers == ("AAPL",)
    assert cfg.dashboard_recency_window_days == 1825
    assert cfg.dashboard_recency_floor == pytest.approx(0.05)
    assert cfg.dashboard_animate is True
    assert cfg.skip_sec is True
    # Other skip flags default off
    assert cfg.skip_macro is False


def test_from_dict_handles_empty_input() -> None:
    """Empty YAML should produce a fully-default config — useful when a
    user only wants to override one or two fields."""
    cfg = RefreshConfig.from_dict({})
    default = RefreshConfig()
    assert cfg == default


def test_from_dict_handles_partial_sections() -> None:
    cfg = RefreshConfig.from_dict({"yahoo": {"tickers": ["BTC-USD"]}})
    assert cfg.yahoo_tickers == ("BTC-USD",)
    assert cfg.sec_filers == ()  # untouched, default empty


def test_from_yaml_roundtrip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "test_config.yaml"
    yaml_path.write_text(
        """
db_path: /tmp/x.duckdb
yahoo:
  tickers:
    - GLD
    - SLV
""".strip()
    )
    cfg = RefreshConfig.from_yaml(yaml_path)
    assert cfg.db_path == Path("/tmp/x.duckdb")
    assert cfg.yahoo_tickers == ("GLD", "SLV")


def test_from_yaml_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RefreshConfig.from_yaml(tmp_path / "does_not_exist.yaml")
