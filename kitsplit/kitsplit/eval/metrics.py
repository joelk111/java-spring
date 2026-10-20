"""Metrics: MSST's implementations (comparable to MVSEP tables) + onset_snr.

onset_snr (kitsplit-specific): SDR computed only on the samples within +-ONSET_MS of
the reference onsets of that stem — "how intact are the transients". Onsets come from
the track's onsets.json (e.g. GMD MIDI) when present, else from detect_onsets() on the
reference stem (ASSUMPTION: the detector is unvalidated on dense real material).

Stems whose reference is silent (e.g. `residual` in StemGMD) are not scored (NaN):
SDR against silence is undefined.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from kitsplit._msst import ensure_msst_on_path
from kitsplit.errors import KitsplitInputError
from kitsplit.eval.onsets import detect_onsets
from kitsplit.io import audio, stems as stems_io

MSST_METRICS = ("sdr", "si_sdr", "l1_freq", "bleedless", "fullness", "log_wmse", "aura_stft", "aura_mrstft")
ALL_METRICS = MSST_METRICS + ("onset_snr",)
ONSET_MS = 5.0
SDR_CAP = 100.0  # pred == ref would give ~+130 dB from the eps; report it capped


def onset_snr(ref: np.ndarray, est: np.ndarray, sr: int, onsets: list[float], ms: float = ONSET_MS) -> float:
    if not onsets:
        return math.nan
    mask = np.zeros(ref.shape[-1], dtype=bool)
    half = int(round(ms * 1e-3 * sr))
    for t in onsets:
        c = int(round(t * sr))
        mask[max(0, c - half) : max(0, min(mask.size, c + half + 1))] = True
    r = ref[..., mask].astype(np.float64)
    e = est[..., mask].astype(np.float64)
    num = np.sum(r**2) + 1e-8
    den = np.sum((r - e) ** 2) + 1e-8
    return float(min(SDR_CAP, 10 * np.log10(num / den)))


def score(ref: np.ndarray, est: np.ndarray, mix: np.ndarray | None, sr: int, metrics: list[str],
          onsets: list[float] | None = None) -> dict[str, float]:
    unknown = set(metrics) - set(ALL_METRICS)
    if unknown:
        raise KitsplitInputError(f"unknown metrics {sorted(unknown)}; allowed: {', '.join(ALL_METRICS)}")
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[..., :n], est[..., :n]
    mix = mix[..., :n] if mix is not None else ref
    if not np.any(ref):
        return {m: math.nan for m in metrics}
    out: dict[str, float] = {}
    msst = [m for m in metrics if m in MSST_METRICS]
    if msst:
        ensure_msst_on_path()
        from utils.metrics import get_metrics  # type: ignore

        if "log_wmse" in msst and mix is ref:
            raise KitsplitInputError("log_wmse needs the mixture (mixture.wav in the reference track)")
        vals = get_metrics(msst, ref.astype(np.float32), est.astype(np.float32), mix.astype(np.float32))
        out.update({k: float(vals[k]) for k in msst})
        if "sdr" in out:
            out["sdr"] = min(SDR_CAP, out["sdr"])
    if "onset_snr" in metrics:
        out["onset_snr"] = onset_snr(ref, est, sr, onsets if onsets is not None else detect_onsets(ref, sr))
    return out


# ------------------------------------------------------------------ directories


def is_track(d: Path) -> bool:
    return (d / "kick.wav").is_file()


def ref_tracks(ref: Path) -> list[Path]:
    if is_track(ref):
        return [ref]
    tracks = sorted(d for d in ref.rglob("*") if d.is_dir() and is_track(d))
    if not tracks:
        raise KitsplitInputError(f"no Type 1 tracks under {ref}")
    return tracks


def detect_grouping(pred_dir: Path) -> int:
    names = {p.stem for p in pred_dir.iterdir() if p.suffix in (".wav", ".flac")} - {"mixture"}
    for n, g in stems_io.GROUPINGS.items():
        if names == set(g):
            return n
    raise KitsplitInputError(f"{pred_dir}: stem files {sorted(names)} match no 4/5/6/7-stem layout")


def _read_stem(d: Path, name: str) -> tuple[np.ndarray, int]:
    for ext in ("wav", "flac"):
        p = d / f"{name}.{ext}"
        if p.is_file():
            x, info = audio.read(p)
            return audio.restore_channels(x, info.channels), info.sr
    raise KitsplitInputError(f"missing {name}.wav in {d}")


def load_onsets(track: Path) -> dict[str, list[float]] | None:
    p = track / "onsets.json"
    if not p.is_file():
        return None
    return {k: [float(o[0]) for o in v] for k, v in json.loads(p.read_text())["onsets"].items()}


def evaluate_track(pred_dir: Path, ref_dir: Path, metrics: list[str], n: int | None = None) -> list[dict]:
    n = n or detect_grouping(pred_dir)
    refs7 = []
    sr = None
    for s in stems_io.STEMS7:
        x, sr = _read_stem(ref_dir, s)
        refs7.append(x)
    refs = stems_io.group(np.stack(refs7), n)
    mix = _read_stem(ref_dir, "mixture")[0] if (ref_dir / "mixture.wav").is_file() else None
    onsets7 = load_onsets(ref_dir)
    rows = []
    for name, members in stems_io.GROUPINGS[n].items():
        est, est_sr = _read_stem(pred_dir, name)
        if est_sr != sr:
            raise KitsplitInputError(f"{pred_dir / name}: sample rate {est_sr} != reference {sr}")
        onsets = sorted(t for m in members for t in onsets7.get(m, [])) if onsets7 is not None else None
        rows.append({"track": ref_dir.name, "stem": name, **score(refs[name], est, mix, sr, metrics, onsets)})
    return rows


def summarize(rows: list[dict], metrics: list[str]) -> dict:
    by_stem: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        for m in metrics:
            v = r.get(m, math.nan)
            if not math.isnan(v):
                by_stem.setdefault(r["stem"], {}).setdefault(m, []).append(v)
    return {s: {m: {"median": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v)} for m, v in ms.items()}
            for s, ms in by_stem.items()}


def evaluate_dir(pred: Path, ref: Path, metrics: list[str], n: int | None = None) -> dict:
    tracks = ref_tracks(ref)
    rows: list[dict] = []
    for t in tracks:
        pdir = pred if len(tracks) == 1 and not (pred / t.name).is_dir() else pred / t.name
        rows.extend(evaluate_track(pdir, t, metrics, n))
    return {"schema": "kitsplit.eval.v1", "metrics": metrics, "n_tracks": len(tracks),
            "rows": rows, "summary": summarize(rows, metrics)}


def summary_table(report: dict, per_track: bool = False) -> str:
    metrics = report["metrics"]
    lines = ["stem      " + "".join(f"{m:>22}" for m in metrics)]
    for stem, ms in report["summary"].items():
        cells = "".join(f"{ms[m]['median']:>10.2f} ({ms[m]['mean']:>7.2f})" if m in ms else f"{'n/a':>22}" for m in metrics)
        lines.append(f"{stem:10}{cells}")
    lines.append("cells: median (mean) over tracks")
    if per_track:
        for r in report["rows"]:
            lines.append(f"{r['track']} {r['stem']}: " + ", ".join(f"{m}={r.get(m, math.nan):.2f}" for m in metrics))
    return "\n".join(lines)
