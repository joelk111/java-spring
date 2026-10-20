"""Download model weights into ~/.cache/kitsplit (or KITSPLIT_CACHE), sha256-pinned.

    python scripts/fetch_weights.py B1 B2          # or --all
Prints each model's license first. Eval-only weights are downloaded for evaluation
ONLY: never ship them, never fine-tune from them (docs/LICENSES.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kitsplit.errors import KitsplitError  # noqa: E402
from kitsplit.models import registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true", help="every entry that has weights")
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()
    entries = [e for e in registry.load_registry() if e.weights] if args.all else [registry.resolve(n) for n in args.names]
    if not entries:
        ap.error("give model names or --all")
    status = 0
    for e in entries:
        print(f"== {e.name}\n   license: {e.license}\n   shippable: {e.shippable}")
        try:
            for p in registry.fetch(e, args.cache_dir):
                print(f"   ok (sha256 verified) {p}")
        except KitsplitError as exc:
            print(f"   FAILED: {exc}", file=sys.stderr)
            status = exc.exit_code
    return status


if __name__ == "__main__":
    sys.exit(main())
