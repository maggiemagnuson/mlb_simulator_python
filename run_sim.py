#!/usr/bin/env python3
"""Convenience launcher for running the simulator without installing the package.

Usage:
    python run_sim.py --date 2026-04-23 --games 100 --output-csv projections.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlb_simulator.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
