"""Model registry: name -> (kind, arch, config, weights url, sha256, license).

Built-in entries live in `registry.yaml` next to this file. Extra registry files can be
added with the KITSPLIT_REGISTRY env var (os.pathsep-separated), and `--model PATH`
accepts a single-entry YAML file whose relative `path:` fields resolve next to it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kitsplit.errors import KitsplitModelError

BUILTIN_REGISTRY = Path(__file__).with_name("registry.yaml")
KINDS = {"oracle", "kitsplit", "msst", "demucs"}


def default_cache_dir() -> Path:
    env = os.environ.get("KITSPLIT_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "kitsplit"


@dataclass
class ModelEntry:
    name: str
    kind: str
    arch: str
    license: str
    shippable: bool
    sha256: str | None
    raw: dict[str, Any] = field(repr=False)
    base_dir: Path | None = None  # set for --model PATH entries

    @property
    def aliases(self) -> list[str]:
        return list(self.raw.get("aliases") or [])

    @property
    def weights(self) -> dict | None:
        return self.raw.get("weights")

    @property
    def config(self) -> dict | None:
        return self.raw.get("config")

    @property
    def stem_map(self) -> dict[str, str] | None:
        return self.raw.get("stem_map")

    @property
    def sample_rate(self) -> int | None:
        return self.raw.get("sample_rate")

    @property
    def native_stems(self) -> int:
        return int(self.raw.get("native_stems", 7))

    @property
    def description(self) -> str:
        return str(self.raw.get("description", ""))


def _parse_entry(raw: dict, base_dir: Path | None) -> ModelEntry:
    missing = [k for k in ("name", "kind", "arch", "license", "shippable", "sha256") if k not in raw]
    if missing:
        raise KitsplitModelError(f"registry entry {raw.get('name', '?')!r} lacks {missing}")
    if raw["kind"] not in KINDS:
        raise KitsplitModelError(f"registry entry {raw['name']!r}: unknown kind {raw['kind']!r}")
    return ModelEntry(
        name=raw["name"],
        kind=raw["kind"],
        arch=raw["arch"],
        license=str(raw["license"]),
        shippable=bool(raw["shippable"]),
        sha256=raw["sha256"],
        raw=raw,
        base_dir=base_dir,
    )


def _load_file(path: Path) -> list[ModelEntry]:
    data = yaml.safe_load(path.read_text())
    entries = data["models"] if isinstance(data, dict) and "models" in data else [data]
    base = None if path == BUILTIN_REGISTRY else path.parent
    return [_parse_entry(e, base) for e in entries]


def load_registry() -> list[ModelEntry]:
    entries = _load_file(BUILTIN_REGISTRY)
    for extra in filter(None, os.environ.get("KITSPLIT_REGISTRY", "").split(os.pathsep)):
        entries.extend(_load_file(Path(extra)))
    return entries


def resolve(name_or_path: str) -> ModelEntry:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        entries = _load_file(p)
        if len(entries) != 1:
            raise KitsplitModelError(f"{p}: a --model file must contain exactly one entry")
        return entries[0]
    for e in load_registry():
        if name_or_path == e.name or name_or_path in e.aliases:
            return e
    names = ", ".join(e.name for e in load_registry())
    raise KitsplitModelError(f"unknown model {name_or_path!r}; known: {names}")


# ------------------------------------------------------------------ files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _file_location(entry: ModelEntry, spec: dict, cache_dir: Path) -> Path:
    if "path" in spec:
        p = Path(spec["path"])
        return p if p.is_absolute() or entry.base_dir is None else entry.base_dir / p
    return cache_dir / entry.name / spec["filename"]


def _check(path: Path, expected: str | None, what: str) -> None:
    if not path.is_file():
        raise KitsplitModelError(f"{what} missing: {path} (run `kitsplit models fetch`)")
    if expected is None:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise KitsplitModelError(f"{what} sha256 mismatch for {path}: expected {expected}, got {actual}")


def weight_path(entry: ModelEntry, cache_dir: Path | None = None) -> Path:
    if not entry.weights:
        raise KitsplitModelError(f"model {entry.name!r} has no published weights ({entry.description})")
    return _file_location(entry, entry.weights, cache_dir or default_cache_dir())


def config_path(entry: ModelEntry, cache_dir: Path | None = None) -> Path | None:
    if not entry.config:
        return None
    return _file_location(entry, entry.config, cache_dir or default_cache_dir())


def verify(entry: ModelEntry, cache_dir: Path | None = None) -> list[Path]:
    """Check every file of the entry against its sha256. Raises KitsplitModelError."""
    if entry.kind == "oracle":
        return []
    paths = [weight_path(entry, cache_dir)]
    _check(paths[0], entry.sha256, f"weights of {entry.name}")
    cfg = config_path(entry, cache_dir)
    if cfg is not None:
        _check(cfg, entry.config.get("sha256"), f"config of {entry.name}")
        paths.append(cfg)
    return paths


def _download(url: str, dest: Path, expected: str | None, attempts: int = 2) -> None:
    """Download to dest. A sha256 mismatch is retried once (we have seen a transfer
    arrive with the right length but corrupted bytes); the pin stays the authority."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        actual = sha256_file(tmp) if expected is not None else None
        if actual == expected:
            tmp.replace(dest)
            return
        tmp.unlink()
        if attempt == attempts:
            raise KitsplitModelError(f"downloaded {url}: sha256 {actual} != expected {expected} ({attempts} attempts)")


def fetch(entry: ModelEntry, cache_dir: Path | None = None) -> list[Path]:
    """Download missing files (sha256-checked) into the cache, then verify all."""
    if entry.kind == "oracle":
        return []
    cache_dir = cache_dir or default_cache_dir()
    if not entry.weights:
        raise KitsplitModelError(f"model {entry.name!r} has no published weights ({entry.description})")
    for spec, expected in ((entry.weights, entry.sha256), (entry.config, (entry.config or {}).get("sha256"))):
        if not spec:
            continue
        dest = _file_location(entry, spec, cache_dir)
        if not dest.is_file():
            if "url" not in spec:
                raise KitsplitModelError(f"{dest} missing and no url to fetch it from")
            _download(spec["url"], dest, expected)
    return verify(entry, cache_dir)


def require_references(entry: ModelEntry) -> bool:
    return entry.kind == "oracle" and entry.arch == "ideal-ratio-mask"
