"""Configuration loader for the AVE refresh engine.

Reads a YAML file into typed dataclasses; provides sensible defaults so a
user-config can omit anything it doesn't need. The macro-data layer
(FRED/EIA series, gold inputs) is **not** user-configurable — those are
the methodologically-load-bearing inputs defined in ``numeraires/*.py``;
exposing them as config would make the system susceptible to silent
methodological drift. Users only configure what's *user*-specific:
which Yahoo tickers to track, which SEC filers, which dashboards to
render, and skip-flags for partial runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RefreshConfig:
    """Parsed YAML config for ``ave refresh``.

    All fields have safe defaults; an empty YAML produces a config that
    runs only the macro layer (no asset tickers, no dashboards) and writes
    to ``data/ave.duckdb`` + ``data/derived/``.
    """

    db_path: Path = Path("data/ave.duckdb")
    output_dir: Path = Path("data/derived")
    yahoo_tickers: tuple[str, ...] = ()
    sec_filers: tuple[str, ...] = ()
    sec_user_agent: str = ""
    dashboard_tickers: tuple[str, ...] = ()
    dashboard_recency_window_days: int | None = None
    dashboard_recency_floor: float = 0.10
    dashboard_animate: bool = False
    skip_macro: bool = False
    skip_assets: bool = False
    skip_sec: bool = False
    skip_dump: bool = False
    skip_dashboards: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> RefreshConfig:
        """Load config from a YAML file. Missing keys fall back to defaults."""
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RefreshConfig:
        yahoo = data.get("yahoo") or {}
        sec = data.get("sec") or {}
        dash = data.get("dashboards") or {}
        skip = data.get("skip") or {}
        return cls(
            db_path=Path(data.get("db_path", "data/ave.duckdb")),
            output_dir=Path(data.get("output_dir", "data/derived")),
            yahoo_tickers=tuple(yahoo.get("tickers") or ()),
            sec_filers=tuple(sec.get("filers") or ()),
            sec_user_agent=str(sec.get("user_agent") or ""),
            dashboard_tickers=tuple(dash.get("tickers") or ()),
            dashboard_recency_window_days=dash.get("recency_window_days"),
            dashboard_recency_floor=float(dash.get("recency_floor", 0.10)),
            dashboard_animate=bool(dash.get("animate", False)),
            skip_macro=bool(skip.get("macro", False)),
            skip_assets=bool(skip.get("assets", False)),
            skip_sec=bool(skip.get("sec", False)),
            skip_dump=bool(skip.get("dump", False)),
            skip_dashboards=bool(skip.get("dashboards", False)),
        )
