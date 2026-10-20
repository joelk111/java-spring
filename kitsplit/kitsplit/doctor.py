"""`kitsplit doctor`: environment checks."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
from pathlib import Path

from kitsplit._msst import PROJECT_ROOT, msst_root
from kitsplit.models.registry import default_cache_dir

REQUIRED = ["numpy", "scipy", "soundfile", "torch", "typer", "yaml"]
OPTIONAL = ["onnxruntime", "onnx", "librosa", "ml_collections", "omegaconf", "torchaudio", "demucs"]
FIXTURE_MANIFEST = PROJECT_ROOT / "tests" / "fixtures" / "MANIFEST.sha256"


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def _submodule_pins() -> list[dict]:
    """Compare checked-out submodule commits with the commits pinned in the superproject."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        out = subprocess.run(["git", "submodule", "status"], cwd=top,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return [_check("submodules", "WARN", f"git not available: {exc}")]
    checks = []
    for line in out.splitlines():
        flag, sha, path = line[0], line[1:41], line[42:].split(" ")[0]
        if flag == "-":
            checks.append(_check(f"submodule {path}", "WARN", f"not initialised (pin {sha[:12]})"))
        elif flag == "+":
            checks.append(_check(f"submodule {path}", "FAIL", f"checked-out commit {sha[:12]} != pinned commit"))
        else:
            checks.append(_check(f"submodule {path}", "OK", f"pinned at {sha[:12]}"))
    return checks


def _fixture_hashes() -> dict:
    if not FIXTURE_MANIFEST.is_file():
        return _check("fixtures", "WARN", "MANIFEST.sha256 missing")
    root = FIXTURE_MANIFEST.parent
    bad, missing = [], []
    lines = [l.split("  ", 1) for l in FIXTURE_MANIFEST.read_text().splitlines() if l.strip()]
    for digest, rel in lines:
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    if bad:
        return _check("fixtures", "FAIL", f"hash mismatch: {bad[:5]}")
    if missing:
        return _check("fixtures", "WARN", f"{len(missing)} not generated yet (run scripts/make_fixtures.py)")
    return _check("fixtures", "OK", f"{len(lines)} files match MANIFEST.sha256")


def run(cache_dir: Path | None = None) -> dict:
    checks = []
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            checks.append(_check(f"dep {mod}", "OK", getattr(m, "__version__", "?")))
        except ImportError as exc:
            checks.append(_check(f"dep {mod}", "FAIL", str(exc)))
    for mod in OPTIONAL:
        try:
            m = importlib.import_module(mod)
            checks.append(_check(f"dep {mod}", "OK", getattr(m, "__version__", "?")))
        except ImportError:
            checks.append(_check(f"dep {mod}", "WARN", "not installed (optional)"))

    try:
        import torch

        cuda = torch.cuda.is_available()
        checks.append(_check("device cuda", "OK" if cuda else "INFO", torch.cuda.get_device_name(0) if cuda else "not available"))
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        checks.append(_check("onnxruntime providers", "INFO", ", ".join(ort.get_available_providers())))
    except ImportError:
        pass

    cache = cache_dir or default_cache_dir()
    try:
        cache.mkdir(parents=True, exist_ok=True)
        ok = os.access(cache, os.W_OK)
        checks.append(_check("cache dir", "OK" if ok else "FAIL", str(cache)))
    except OSError as exc:
        checks.append(_check("cache dir", "FAIL", f"{cache}: {exc}"))

    checks.append(_check("msst", "OK" if (msst_root() / "train.py").is_file() else "WARN", str(msst_root())))
    checks.extend(_submodule_pins())
    checks.append(_fixture_hashes())
    return {"schema": "kitsplit.doctor.v1", "ok": all(c["status"] != "FAIL" for c in checks), "checks": checks}
