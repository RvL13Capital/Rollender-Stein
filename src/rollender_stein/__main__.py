"""Entry point for ``python -m rollender_stein``.

Forwards to ``cli.main()`` so users can run the engine without an
installed entry-point script:

    python -m rollender_stein refresh --config config/refresh.yaml
"""

from __future__ import annotations

import sys

from rollender_stein.cli import main

if __name__ == "__main__":
    sys.exit(main())
