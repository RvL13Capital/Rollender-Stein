"""Environment-driven configuration. Reads `.env` from the project root if present.

Never commit a `.env` — it is gitignored. Use `.env.example` as the template.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class MissingEnvError(RuntimeError):
    """Raised when a required environment variable is unset or empty."""


def get_required_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise MissingEnvError(
            f"environment variable {key!r} is not set. "
            f"Add it to .env in the project root (template: .env.example).",
        )
    return val


def fred_api_key() -> str:
    return get_required_env("FRED_API_KEY")


def eia_api_key() -> str:
    return get_required_env("EIA_API_KEY")
