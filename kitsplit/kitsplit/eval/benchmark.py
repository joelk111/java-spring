"""Run N models on a Type 1 set, score them, write reports/<name>.md + .json.

Every model is scored at its native grouping (7 for kitsplit/B1, 4 for B2) AND at the
4-stem grouping, the only layout every model can be compared on.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from kitsplit._msst import PROJECT_ROOT
from kitsplit.errors import KitsplitInputError
from kitsplit.eval import metrics as M
from kitsplit.io import audio, stems as stems_io
from kitsplit.models import registry
from kitsplit.models.backend_torch import set_deterministic
from kitsplit.models.loader import load_backend
from kitsplit.separate import Settings, separate_array

SCHEMA = "kitsplit.benchmark.v1"
DISCLAIMER = (
    "Oracle rows (oracle-*) are given the answer: they test plumbing and say nothing about "
    "separation quality. Only numbers on the held-out REAL set (data/eval_real, plan §4.3) may "
    "be quoted as quality claims; any other dataset here is a development set."
)


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=PROJECT_ROOT, capture_output=True,
                               text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def manifest_hash(tracks: list[Path]) -> str:
    """sha256 over (relative path, sha256 of bytes) of every audio file in the set."""
    h = hashlib.sha256()
    root = tracks[0].parent
    for t in tracks:
        for f in sorted(t.glob("*.wav")):
            h.update(f"{f.relative_to(root).as_posix()}\0".encode())
            h.update(registry.sha256_file(f).encode())
    return h.hexdigest()


def _write_stems(dest: Path, stems7: np.ndarray, n: int, sr: int) -> None:
    for name, data in stems_io.group(stems7, n).items():
        audio.write(dest / f"{name}.wav", data, sr, "wav", "32f")


def run(data: Path, models: list[str], out: Path, metrics: list[str], preset: str = "quality",
        baseline_dirs: list[Path] | None = None, device: str = "auto", cache_dir: Path | None = None,
        limit: int | None = None) -> tuple[Path, Path]:
    tracks = [t for t in M.ref_tracks(Path(data)) if (t / "mixture.wav").is_file()]
    if not tracks:
        raise KitsplitInputError(f"{data}: benchmark needs Type 1 tracks with mixture.wav")
    total_tracks = len(tracks)
    if limit is not None:
        tracks = tracks[:limit]
    settings = Settings.from_preset(preset)
    out = Path(out)
    entries = [registry.resolve(m) for m in models]
    results: dict = {}
    for entry in entries:
        set_deterministic(0)
        native = entry.native_stems
        groupings = sorted({native, 4}, reverse=True)
        pred_root = out / "predictions" / entry.name
        rows = {n: [] for n in groupings}
        t0 = time.time()
        audio_sec = 0.0
        shared = None if registry.require_references(entry) else load_backend(entry, device=device, cache_dir=cache_dir)
        for t in tracks:
            x, info = audio.read(t / "mixture.wav")
            audio_sec += info.duration
            if shared is None:
                refs = np.stack([audio.read(t / f"{s}.wav")[0] for s in stems_io.STEMS7])
                be = load_backend(entry, device=device, references=refs, references_sr=info.sr)
            else:
                be = shared
            y = separate_array(x, info.sr, info.channels, be, settings)
            for n in groupings:
                d = pred_root / f"{n}stem" / t.name
                _write_stems(d, y, n, info.sr)
                rows[n].extend(M.evaluate_track(d, t, metrics, n))
        elapsed = time.time() - t0
        results[entry.name] = {
            "kind": entry.kind, "license": entry.license, "shippable": entry.shippable, "sha256": entry.sha256,
            "native_stems": native, "seconds": elapsed, "audio_seconds": audio_sec,
            "groupings": {str(n): {"rows": rows[n], "summary": M.summarize(rows[n], metrics)} for n in groupings},
        }
    for bdir in baseline_dirs or []:
        name = f"dir:{Path(bdir).name}"
        n = M.detect_grouping(next(p for p in Path(bdir).iterdir() if p.is_dir()))
        rows = []
        for t in tracks:
            rows.extend(M.evaluate_track(Path(bdir) / t.name, t, metrics, n))
        results[name] = {"kind": "precomputed", "native_stems": n,
                         "groupings": {str(n): {"rows": rows, "summary": M.summarize(rows, metrics)}}}

    report = {
        "schema": SCHEMA,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "dataset": {"path": str(data), "n_tracks": len(tracks), "n_tracks_available": total_tracks,
                    "limited": limit is not None, "manifest_sha256": manifest_hash(tracks)},
        "preset": {"name": preset, **settings.__dict__},
        "metrics": metrics,
        "models": results,
        "disclaimer": DISCLAIMER,
    }
    out.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark_{Path(data).name}"
    js = out / f"{stem}.json"
    md = out / f"{stem}.md"
    js.write_text(json.dumps(report, indent=2, sort_keys=True))
    md.write_text(to_markdown(report))
    return md, js


def _fmt(cell: dict | None) -> str:
    if not cell:
        return "n/a"
    return f"{cell['median']:.2f} ({cell['mean']:.2f})"


def to_markdown(report: dict) -> str:
    d = report["dataset"]
    lines = [
        f"# Benchmark: `{Path(d['path']).name}`",
        "",
        f"> {report['disclaimer']}",
        "",
        f"- created: {report['created']}",
        f"- git: `{report['git_sha']}`",
        f"- dataset: `{d['path']}` — {d['n_tracks']} tracks"
        + (f" (**limited** from {d['n_tracks_available']})" if d["limited"] else "")
        + f", manifest sha256 `{d['manifest_sha256']}`",
        f"- preset: `{report['preset']['name']}` (chunk {report['preset']['chunk_sec']} s, "
        f"overlap {report['preset']['overlap']}, shifts {report['preset']['shifts']})",
        "",
        "| model | kind | sha256 | license | shippable | CPU s per audio s |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in report["models"].items():
        rtf = f"{r['seconds'] / r['audio_seconds']:.2f}" if r.get("audio_seconds") else "n/a"
        sha = (r.get("sha256") or "—")[:12]
        lines.append(f"| {name} | {r['kind']} | `{sha}` | {r.get('license', '—')} | {r.get('shippable', '—')} | {rtf} |")
    groupings = sorted({g for r in report["models"].values() for g in r["groupings"]}, key=int, reverse=True)
    for g in groupings:
        names = [n for n, r in report["models"].items() if g in r["groupings"]]
        stems = list(stems_io.GROUPINGS[int(g)])
        for metric in report["metrics"]:
            lines += ["", f"## {g}-stem — {metric} — median (mean) over tracks", "",
                      "| stem | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
            for s in stems:
                cells = [_fmt(report["models"][n]["groupings"][g]["summary"].get(s, {}).get(metric)) for n in names]
                lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "n/a = reference stem silent in every track (not scored) or metric unavailable.", ""]
    return "\n".join(lines)
