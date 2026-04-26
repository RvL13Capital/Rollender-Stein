# Rollender-Stein

Stealth-mode market analysis, trading signal generation, and strategy backtesting in Python.

> Status: Early scaffolding. No analysis or backtest logic implemented yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[research,dev]"
pytest
```

Or with [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[research,dev]"
uv run pytest
```

## Layout

- `src/rollender_stein/` — package source
- `tests/` — pytest tests
- `notebooks/` — research notebooks (gitignored output, kept structure)
- `data/` — local data cache (gitignored)
# Rollender-Stein
