"""Persist transformed AVE outputs to disk for downstream analysis.

Raw ingested data already lives in ``data/ave.duckdb``. This module dumps the
*transformed* outputs — numéraires, the Phase 4 Kalman panel + filtered state +
parameters, and per-asset division arrays — to parquet under
``data/derived/`` plus a manifest.json listing what was written.

Parquet over CSV: smaller, typed (datetimes round-trip cleanly), pandas-native
via pyarrow. CSV equivalents can be added if a non-pandas consumer needs them.

Folder layout:
    data/derived/
        numeraires/       # one parquet per numéraire (single-column Series)
        panels/           # the cleaned XAU/TIPS/DXY/VIX panel feeding Phase 4
        kalman/           # filtered_state, params (JSON), innovations
        divisions/        # one parquet per ingested asset
        manifest.json     # generated_at, T0, file inventory with row counts
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from rollender_stein.bitemporal import get_asset_closes
from rollender_stein.calendar import T0_DATE
from rollender_stein.numeraires.energy import build_n_energy
from rollender_stein.numeraires.gold import (
    SERIES_IDS as GOLD_SERIES_IDS,
)
from rollender_stein.numeraires.gold import (
    GoldFit,
    assemble_panel,
    build_n_gold,
    fit_gold_model,
)
from rollender_stein.numeraires.liquidity import build_n_liq
from rollender_stein.numeraires.time import build_n_time
from rollender_stein.valuation import build_division_array

DEFAULT_DERIVED_ROOT = Path("data") / "derived"


@dataclass(frozen=True)
class ArtifactInfo:
    path: str
    rows: int
    first_date: str | None
    last_date: str | None


def _ensure_dirs(root: Path) -> None:
    for sub in ("numeraires", "panels", "kalman", "divisions"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def _series_summary(s: pd.Series) -> tuple[int, str | None, str | None]:
    valid = s.dropna()
    if valid.empty:
        return len(s), None, None
    return len(s), str(valid.index.min().date()), str(valid.index.max().date())


def _write_series(s: pd.Series, path: Path, value_name: str) -> ArtifactInfo:
    df = s.to_frame(name=value_name)
    df.index.name = "trade_date"
    df.to_parquet(path)
    rows, first, last = _series_summary(s)
    return ArtifactInfo(path=str(path), rows=rows, first_date=first, last_date=last)


def _write_frame(df: pd.DataFrame, path: Path) -> ArtifactInfo:
    out = df.copy()
    if out.index.name is None:
        out.index.name = "trade_date"
    out.to_parquet(path)
    first: str | None = None
    last: str | None = None
    if isinstance(out.index, pd.DatetimeIndex) and not out.empty:
        first = str(out.index.min().date())
        last = str(out.index.max().date())
    return ArtifactInfo(path=str(path), rows=len(out), first_date=first, last_date=last)


def dump_numeraires(
    con: duckdb.DuckDBPyConnection,
    *,
    end: pd.Timestamp | None = None,
    root: Path = DEFAULT_DERIVED_ROOT,
) -> dict[str, ArtifactInfo]:
    """Build all four numéraires and write each to ``data/derived/numeraires/``."""
    _ensure_dirs(root)
    out: dict[str, ArtifactInfo] = {}
    builders = {
        "n_time": (build_n_time, "N_Time"),
        "n_liquidity": (build_n_liq, "N_Liq"),
        "n_energy": (build_n_energy, "N_Energy"),
        "n_gold": (build_n_gold, "N_Gold"),
    }
    for slug, (builder, value_name) in builders.items():
        s = builder(con, end=end)
        info = _write_series(
            s, root / "numeraires" / f"{slug}.parquet", value_name=value_name
        )
        out[slug] = info
    return out


def dump_phase4_panel(
    con: duckdb.DuckDBPyConnection,
    *,
    end: pd.Timestamp | None = None,
    root: Path = DEFAULT_DERIVED_ROOT,
    panel: pd.DataFrame | None = None,
) -> ArtifactInfo:
    """Write the assembled XAU/TIPS/DXY/VIX panel — the input to the Kalman fit.

    ``panel`` may be passed precomputed (audit patch 05 caching) to avoid
    a redundant ``assemble_panel`` call when used inside ``dump_all_artifacts``.
    """
    _ensure_dirs(root)
    if panel is None:
        panel = assemble_panel(con, end=end)
    return _write_frame(panel, root / "panels" / "kalman_panel.parquet")


def dump_kalman_outputs(
    con: duckdb.DuckDBPyConnection,
    *,
    end: pd.Timestamp | None = None,
    root: Path = DEFAULT_DERIVED_ROOT,
    fit: GoldFit | None = None,
) -> dict[str, ArtifactInfo | str]:
    """Write the Kalman filtered state, innovations, and MLE parameters.

    Outputs:
      - kalman/filtered_state.parquet : mu_t (latent "true core gold")
      - kalman/innovations.parquet    : y_t - E[y_t | F_{t-1}] (true
                                        one-step-ahead innovations from
                                        ``MLEResults.resid``)
      - kalman/params.json            : MLE param vector + log-likelihood

    ``fit`` may be passed precomputed (audit patch 05 caching) to avoid the
    full assemble_panel + fit_gold_model cycle when ``dump_all_artifacts``
    has already done it. If None, computes from scratch.

    Innovation note (audit findings 15.M-5 / 16.F-Major): the prior version
    of this function computed ``XAU - filtered_state - X@beta`` and called
    those "residuals". Mathematically those are *filtered* residuals, which
    use data through time t to estimate state at t — their variance is
    artificially compressed (~6x lower than true innovations). The fix is
    to use ``fit.results.resid`` directly, which statsmodels defines as
    ``y_t - E[y_t | data through t-1]``, i.e. the genuine one-step-ahead
    forecast errors. Diagnostics (autocorrelation, variance regime shifts)
    are valid only on innovations.
    """
    _ensure_dirs(root)
    if fit is None:
        panel = assemble_panel(con, end=end)
        fit = fit_gold_model(panel)

    filtered = fit.filtered_state
    state_info = _write_series(
        filtered.rename("mu_t"),
        root / "kalman" / "filtered_state.parquet",
        value_name="mu_t",
    )

    # True one-step-ahead innovations from statsmodels MLEResults. Indexed
    # against the cleaned panel (the model dropped any rows with NaN exog).
    cleaned = fit.panel_clean
    innovations = pd.Series(
        np.asarray(fit.results.resid),
        index=cleaned.index,
        name="innovation",
    )
    inn_info = _write_series(
        innovations,
        root / "kalman" / "innovations.parquet",
        value_name="innovation",
    )

    params: dict[str, Any] = {
        "log_likelihood": float(fit.results.llf),
        "aic": float(fit.results.aic),
        "bic": float(fit.results.bic),
        "n_obs": len(cleaned),
        "fit_window": {
            "first": str(cleaned.index.min().date()),
            "last": str(cleaned.index.max().date()),
        },
        "series_ids": GOLD_SERIES_IDS,
        "params": {
            name: float(val)
            for name, val in zip(fit.results.param_names, fit.results.params, strict=True)
        },
    }
    params_path = root / "kalman" / "params.json"
    params_path.write_text(json.dumps(params, indent=2, sort_keys=True))

    return {
        "filtered_state": state_info,
        "innovations": inn_info,
        "params_path": str(params_path),
    }


def dump_division_array(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    end: pd.Timestamp | None = None,
    root: Path = DEFAULT_DERIVED_ROOT,
) -> ArtifactInfo:
    """Build the division array for a previously-ingested asset and write to disk."""
    _ensure_dirs(root)
    closes = get_asset_closes(con, ticker, end=end)
    if closes.empty:
        raise RuntimeError(
            f"no rows in asset_price for {ticker!r}; run ingest_yahoo_asset() first"
        )
    n_time = build_n_time(con, end=end)
    n_liq = build_n_liq(con, end=end)
    n_energy = build_n_energy(con, end=end)
    n_gold = build_n_gold(con, end=end)
    da = build_division_array(
        closes, n_time=n_time, n_liquidity=n_liq, n_gold=n_gold, n_energy=n_energy,
    )
    safe = ticker.replace("^", "").replace("=", "-").replace("/", "-")
    return _write_frame(da.to_frame(), root / "divisions" / f"{safe}.parquet")


def dump_all_artifacts(
    con: duckdb.DuckDBPyConnection,
    *,
    tickers: list[str] | None = None,
    end: pd.Timestamp | None = None,
    root: Path = DEFAULT_DERIVED_ROOT,
) -> dict[str, Any]:
    """Run the full transform and persist everything under ``root``.

    Numéraires are built once and reused across all ``tickers`` — for N tickers
    this saves O(N) redundant Kalman fits and LOCF passes.

    Writes a manifest.json at ``root/manifest.json`` listing every file produced
    with row counts and date ranges, plus the time of generation.
    """
    _ensure_dirs(root)
    # Audit patch 05: build the assembled panel and fit the Kalman model
    # ONCE, then thread them through to the dump_* helpers that need them.
    # Previously these were recomputed inside each helper, costing two
    # full assemble_panel passes plus one MLE fit per dump_all_artifacts call.
    cached_panel = assemble_panel(con, end=end)
    cached_fit = fit_gold_model(cached_panel)

    numeraires = dump_numeraires(con, end=end, root=root)
    panel = dump_phase4_panel(con, end=end, root=root, panel=cached_panel)
    kalman = dump_kalman_outputs(con, end=end, root=root, fit=cached_fit)

    divisions: dict[str, ArtifactInfo] = {}
    if tickers:
        # Build numéraires once and reuse — avoids the per-ticker Kalman refit.
        n_time = build_n_time(con, end=end)
        n_liq = build_n_liq(con, end=end)
        n_energy = build_n_energy(con, end=end)
        n_gold = build_n_gold(con, end=end)
        for ticker in tickers:
            closes = get_asset_closes(con, ticker, end=end)
            if closes.empty:
                raise RuntimeError(
                    f"no rows in asset_price for {ticker!r}; "
                    "run ingest_yahoo_asset() first",
                )
            da = build_division_array(
                closes,
                n_time=n_time,
                n_liquidity=n_liq,
                n_gold=n_gold,
                n_energy=n_energy,
            )
            safe = ticker.replace("^", "").replace("=", "-").replace("/", "-")
            info = _write_frame(
                da.to_frame(), root / "divisions" / f"{safe}.parquet"
            )
            divisions[ticker] = info

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "t0_date": str(T0_DATE.date()),
        "end_cutoff": str(end.date()) if end is not None else None,
        "numeraires": {k: asdict(v) for k, v in numeraires.items()},
        "panel": asdict(panel),
        "kalman": {
            "filtered_state": asdict(kalman["filtered_state"]),  # type: ignore[arg-type]
            "innovations": asdict(kalman["innovations"]),  # type: ignore[arg-type]
            "params_path": kalman["params_path"],
        },
        "divisions": {k: asdict(v) for k, v in divisions.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
