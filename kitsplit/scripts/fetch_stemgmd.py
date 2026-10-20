"""Fetch StemGMD from Zenodo — license gate first (plan A2, §13 task 7).

    python scripts/fetch_stemgmd.py --license-only
    python scripts/fetch_stemgmd.py --out data/raw --files StemGMD_eval_session.zip
    python scripts/fetch_stemgmd.py --out data/raw --all        # ~1.13 TB unzipped!

Prints the license id read from the Zenodo API for every record and HALTS (exit 6)
if any is non-commercial ("nc" in the id) unless --allow-nc, or if no license is
stated at all. Downloads are md5-checked against Zenodo's checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

RECORDS = {"part1": "7860223", "part2": "7882857"}
API = "https://zenodo.org/api/records/{}"
EXIT_LICENSE = 6


def record(rec_id: str) -> dict:
    with urllib.request.urlopen(API.format(rec_id)) as r:
        return json.load(r)


def license_gate(meta: dict, allow_nc: bool) -> tuple[bool, str]:
    lic = (meta.get("metadata", {}).get("license") or {}).get("id")
    if not lic:
        return False, "NONE FOUND"
    if "nc" in lic.lower().split("-") and not allow_nc:
        return False, lic
    return True, lic


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv: list[str] | None = None, fetch_record=record) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--files", nargs="*", default=[], help="file names to fetch")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--license-only", action="store_true")
    ap.add_argument("--allow-nc", action="store_true", help="accept a non-commercial license")
    args = ap.parse_args(argv)

    metas = {}
    for part, rec_id in RECORDS.items():
        meta = fetch_record(rec_id)
        ok, lic = license_gate(meta, args.allow_nc)
        print(f"StemGMD {part} (zenodo {rec_id}): license = {lic}")
        if not ok:
            print(f"HALT: license {lic!r} does not allow the intended use (pass --allow-nc only "
                  "if the use really is non-commercial)", file=sys.stderr)
            return EXIT_LICENSE
        metas[part] = meta
    if args.license_only:
        return 0

    wanted = [(f, part) for part, meta in metas.items() for f in meta["files"]
              if args.all or f["key"] in args.files]
    if not wanted:
        ap.error("nothing selected: use --files NAME... or --all")
    args.out.mkdir(parents=True, exist_ok=True)
    for f, part in wanted:
        dest = args.out / f["key"]
        expected = f["checksum"].split(":", 1)[1]
        if dest.is_file() and md5(dest) == expected:
            print(f"ok (cached) {dest}")
            continue
        print(f"downloading {f['key']} ({f['size'] / 1e9:.1f} GB)")
        with urllib.request.urlopen(f["links"]["self"]) as r, open(dest, "wb") as out:
            shutil.copyfileobj(r, out, 1 << 22)
        got = md5(dest)
        if got != expected:
            print(f"md5 mismatch for {dest}: {got} != {expected}", file=sys.stderr)
            return 1
        print(f"ok {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
