#!/usr/bin/env python3
"""Zero-install launcher for pcbench.

Keeps ``python3 benchmark.py`` working straight from a checkout — no
installation, no PYTHONPATH setup — so the tool can still be copied onto a
machine and run immediately.

Installed users can equivalently run ``pcbench`` or ``python -m pcbench``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pcbench.cli import entry  # noqa: E402

if __name__ == "__main__":
    entry()
