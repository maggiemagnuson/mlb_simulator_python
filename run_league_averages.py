#!/usr/bin/env python3
"""Convenience launcher for generating league averages without installing the package.

Usage:
    python run_league_averages.py --year 2026 --format json
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_simulator.league_averages import main


if __name__ == "__main__":
    raise SystemExit(main())
