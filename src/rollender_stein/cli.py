"""Command-line interface for the AVE refresh engine.

Usage:

    ave refresh --config config/refresh.yaml
    ave refresh --config config/refresh.yaml --dry      # plan only
    ave refresh --config config/refresh.yaml --no-dashboards

Or via module form (no install required):

    python -m rollender_stein refresh --config config/refresh.yaml

Exit codes:
    0  — all steps succeeded
    1  — at least one step failed (run summary still printed)
    2  — usage / config error (e.g. missing file)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rollender_stein.run import RunResult, refresh
from rollender_stein.runconfig import RefreshConfig


def _format_run_result(result: RunResult) -> str:
    lines: list[str] = []
    status = "OK" if result.success else f"FAILED ({result.n_failed} step(s))"
    lines.append(f"AVE refresh — {status}  total {result.duration_seconds:.1f}s")
    lines.append(
        f"  started:  {result.started_at.isoformat(timespec='seconds')}"
    )
    lines.append(
        f"  finished: {result.finished_at.isoformat(timespec='seconds')}"
    )
    lines.append("")
    lines.append("Steps:")
    for s in result.steps:
        flag = "OK " if s.success else "ERR"
        line = (
            f"  [{flag}] {s.step:<32}"
            f"  rows={s.rows:>7,}"
            f"  {s.duration_seconds:>6.1f}s"
        )
        if s.error:
            line += f"  ! {s.error}"
        lines.append(line)
    if result.dashboards_written:
        lines.append("")
        lines.append("Dashboards written:")
        for p in result.dashboards_written:
            lines.append(f"  - {p}")
    return "\n".join(lines)


def cmd_refresh(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    try:
        config = RefreshConfig.from_yaml(config_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2

    # CLI flags can override config skip_* fields without editing the YAML.
    # OR-semantics: a CLI flag can ENABLE a skip, but cannot un-skip what the
    # YAML already disabled (that would invite confusion).
    from dataclasses import replace
    config = replace(
        config,
        skip_macro=config.skip_macro or args.no_macro,
        skip_assets=config.skip_assets or args.no_assets,
        skip_sec=config.skip_sec or args.no_sec,
        skip_dashboards=config.skip_dashboards or args.no_dashboards,
    )

    result = refresh(config, dry=args.dry)
    print(_format_run_result(result), flush=True)
    return 0 if result.success else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ave",
        description="AVE — Absolute Valuation Engine refresh CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("refresh", help="run the full refresh pipeline")
    pr.add_argument(
        "--config", required=True, type=Path,
        help="path to a YAML config (see config/refresh.yaml)",
    )
    pr.add_argument(
        "--dry", action="store_true",
        help="show planned steps without executing",
    )
    pr.add_argument("--no-dashboards", action="store_true",
                    help="skip the dashboard-render step")
    pr.add_argument("--no-macro", action="store_true",
                    help="skip the macro-data ingest step")
    pr.add_argument("--no-assets", action="store_true",
                    help="skip the Yahoo asset ingest step")
    pr.add_argument("--no-sec", action="store_true",
                    help="skip the SEC shares-outstanding ingest step")
    pr.set_defaults(fn=cmd_refresh)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    import sys
    sys.exit(main())
