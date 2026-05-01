# Rollender-Stein

This repository implements the **URTRIF v3.0** framework
([Unified Real Total Return Index Framework](URTRIF.md);
PDF: [`docs/URTRIF_v3.0.pdf`](docs/URTRIF_v3.0.pdf)) and generalizes it
to a multi-numéraire architecture under the name **Absolute Valuation
Engine (AVE)**.

URTRIF v3.0 transforms each raw price series into a single real-total-
return index `I_real` against one deflator (CPI). AVE applies the same
mathematical core to **four orthogonal-ish deflators** at once — Time
(hourly wages), Energy (Brent → MWh), Liquidity (G3 broad money), and
Gold — anchored at `T0 = 2000-01-03`. The four parallel indices form a
3D phase-space trajectory whose **divergence between axes** is the unit
of analysis: where labor outpaces money, where energy costs swamp wage
growth, where monetary expansion shows up before gold reacts.

This is a **forensic measurement instrument**, not a trading system. See
[`URTRIF.md`](URTRIF.md) for the canonical spec and the bridge to AVE,
[`CLAUDE.md`](CLAUDE.md) for methodological depth, and
[`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) for the per-finding resolution
log.

## What you get

For any ingested asset, the AVE produces:

- **Per-share absolute valuation** — `Asset_in_X(t) = nominal_USD(t) / N_X(t) * 100` for X ∈ {Time, Liquidity, Energy, Gold}, in T0-deflated USD.
- **Whole-company market cap** (US single stocks) — `raw_shares(t) × raw_close(t)` with cumulative-future-split correction; separates real wealth creation from buyback / dilution effects.
- **3D Phase-Space dashboard** — interactive Plotly figure encoding six dimensions: three deflated axes, time as Viridis line color, nominal price as marker size, and dollar-turnover z-score as marker opacity (a "conviction" channel).
- **Statistical diagnostics** — z-scores, log-return correlations, Kalman one-step-ahead innovation diagnostics, Phase-7 truncation-hash software guard against future code drift.

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[research,dev]"
.venv/bin/pytest -q
```

Or with [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[research,dev]"
uv run pytest
```

`.env` (gitignored) needs `FRED_API_KEY` and `EIA_API_KEY` — see
[`.env.example`](.env.example) for the full list of required series and
where to register. yfinance has no key.

## Quickstart

```python
from rollender_stein.bitemporal import open_db
from rollender_stein.assets import (
    build_pipeline_for_asset,
    ingest_yahoo_asset,
    save_asset_dashboard,
)

with open_db("data/ave.duckdb") as con:
    ingest_yahoo_asset(con, "AAPL")  # one-time, OHLCV from yfinance
    result = build_pipeline_for_asset(con, "AAPL", animate=True)
    path = save_asset_dashboard(result, suffix="_animated")
    print(f"open {path} in a browser")
```

Numéraire ingestion (FRED / EIA / SEC EDGAR for shares-outstanding) happens
via the loaders in `rollender_stein.io.*`. CLAUDE.md has the runbooks.

## Forensic principles (hard rules)

1. **No look-ahead in LOCF** — `pd.merge_asof(direction='backward', on='release_date')` is the only sanctioned synchronisation primitive.
2. **Vintage macro data** — FRED queries with revisions go through `fetch_alfred_first_release` (output_type=4); daily series with no meaningful revisions go via the live endpoint with explicit `PUBLICATION_LAG_BD` business-day offsets.
3. **Filtered state, not smoothed** — Kalman code uses `results.filtered_state` exclusively; smoothed state runs the recursion both directions.
4. **Total Return for equities** — TR series (e.g. `^SP500TR`) over price-only when comparing against yieldless numéraires.
5. **`mypy --strict` zero-error gate** — non-negotiable.
6. **Bitemporal release validation** — `release_date >= reference_date` enforced at the insert API.
7. **Three-tool gate** before any commit: `pytest` → `mypy` → `ruff`.

## Layout

- [`src/rollender_stein/`](src/rollender_stein/) — package source (~22 modules)
- [`tests/`](tests/) — pytest tests (199 tests, all green)
- `notebooks/` — research notebooks (gitignored output, structure tracked)
- `data/` — local data cache (gitignored: `ave.duckdb`, `derived/`, `dashboard_*.html`)

## Maintainer documentation

The codebase carries seven first-class documents, each with a distinct role:

- [`URTRIF.md`](URTRIF.md) — **canonical specification** (URTRIF v3.0)
  with markdown transcript, mathematical core, and bridge to AVE
- [`docs/URTRIF_v3.0.pdf`](docs/URTRIF_v3.0.pdf) — original specification PDF
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — pipeline layers, data flow, F_t-measurability table
- [`INVARIANTS.md`](INVARIANTS.md) — what the system guarantees, with test-name pinning
- [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) — what it deliberately doesn't, with audit-ID rationale
- [`CHANGELOG.md`](CHANGELOG.md) — chronological patch / feature history
- [`CLAUDE.md`](CLAUDE.md) — methodological depth and runbooks
- [`AUDIT_DECISIONS.md`](AUDIT_DECISIONS.md) — per-finding decision log
  (DONE / REWORKED / WON'T FIX) for the audit baseline `af3956a`

A pull request that changes an invariant must update `INVARIANTS.md`,
the cited tests, and link to the rationale in `AUDIT_DECISIONS.md` or
`CLAUDE.md`. A PR that closes a known limitation is a **feature PR**
with its own justification, not a drive-by fix.

## Status

- **Production-deployed:** no. CI runs the three-tool gate (pytest + mypy
  --strict + ruff) on every push to `main` and every PR via
  [`.github/workflows/ci.yml`](.github/workflows/ci.yml); there is no
  SLA, no monitoring, no scheduled refresh — see
  [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) L-12 / L-13.
- **Research-ready:** yes. 199 tests, mypy strict, ruff clean.
- **Spec-compliant:** mostly — Phase 4's Kalman narrative was empirically demoted to a Phase 4.5 diagnostic during the audit-fix campaign. CLAUDE.md "Spec deviations" enumerates the deliberate departures.

## License

Proprietary. See `pyproject.toml`.
