"""Access to the pinned Music-Source-Separation-Training submodule.

MSST is not an installable package at the pinned commit (and the PyPI `msst` wheel
pins torch<2.12), so its root is put on sys.path. Override with KITSPLIT_MSST_DIR.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from kitsplit.errors import KitsplitError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def msst_root() -> Path:
    env = os.environ.get("KITSPLIT_MSST_DIR")
    return Path(env) if env else PROJECT_ROOT / "third_party" / "Music-Source-Separation-Training"


def ensure_msst_on_path() -> Path:
    root = msst_root()
    if not (root / "utils" / "settings.py").is_file():
        raise KitsplitError(
            f"MSST not found at {root}; run `git submodule update --init` or set KITSPLIT_MSST_DIR"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
