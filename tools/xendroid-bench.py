#!/usr/bin/env python3
"""Unified CLI wrapper for Xendroid Bench (5-layer QA & performance workbench)."""

from __future__ import annotations

import pathlib
import sys

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from xendroid_bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
