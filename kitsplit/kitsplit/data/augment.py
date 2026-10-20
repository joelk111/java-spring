"""Drum-bus augmentation (plan §7.3): make clean summed stems look like a processed bus.

Every op takes explicit parameters and has identity parameters at which it returns its
input (within float rounding); see tests/unit/test_augment.py (U-AUG-1).

Keeping targets consistent — `target_mode` in the config:
  * "distribute" (default; usable with MSST Type 1 training, which re-sums stems):
    the processed stems still sum to the processed mixture.
      - linear ops (EQ tilt, stereo width) are applied to every stem;
      - gain-envelope ops (compressor incl. parallel blend, transient shaper, limiter)
        compute ONE gain envelope from the bus and apply it to every stem — exact,
        because g * sum(s) == sum(g * s);
      - non-linear ops (saturation) produce d = f(m) - m, which is split across stems
        with time-frequency power masks of the stems (masks sum to 1);
      - dither noise is assigned to `residual` (it is not drum sound).
  * "mixture_only": stems stay clean, only the mixture is processed. The sum is broken
    on purpose; it needs a custom training loop (Phase 2 A/B, plan §7.3 note).
ASSUMPTION: which mode trains better is unknown (the plan's least-sure point); it must
be A/B'd in Phase 2 and the result recorded in docs/ASSUMPTIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.signal import butter, fftconvolve, sosfilt

from kitsplit.dsp.stft import Stft, StftParams
from kitsplit.errors import KitsplitInputError
from kitsplit.io import audio
from kitsplit.io.stems import STEMS7

RESIDUAL = STEMS7.index("residual")
EPS = 1e-12
_BLOCK = 32  # samples per envelope step for the dynamics ops (~0.7 ms at 44.1 kHz)


def db2lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


# ============================================================ per-stem (pre-mix) ops


def gain(x: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
    return x * db2lin(gain_db)


def pan_narrow(x: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Scale the side signal: width 1 = unchanged, 0 = mono, >1 = wider."""
    if width == 1.0:
        return x
    mid = 0.5 * (x[0] + x[1])
    side = 0.5 * (x[0] - x[1]) * width
    return np.stack([mid + side, mid - side])


def highpass(x: np.ndarray, sr: int, cutoff_hz: float | None = None) -> np.ndarray:
    if cutoff_hz is None:  # a Butterworth filter has no continuous identity setting
        return x
    return sosfilt(butter(2, cutoff_hz, "highpass", fs=sr, output="sos"), x, axis=-1)


def lowpass(x: np.ndarray, sr: int, cutoff_hz: float | None = None) -> np.ndarray:
    if cutoff_hz is None or cutoff_hz >= 0.49 * sr:
        return x
    return sosfilt(butter(2, cutoff_hz, "lowpass", fs=sr, output="sos"), x, axis=-1)


def room(x: np.ndarray, sr: int, rng: np.random.Generator, wet_db: float = -np.inf, rt60: float = 0.3) -> np.ndarray:
    """Short synthetic room: exponentially decaying stereo noise IR at a low wet level."""
    n = max(1, int(rt60 * sr))
    t = np.arange(n) / sr
    ir = rng.standard_normal((2, n)) * np.exp(-6.91 * t / rt60)  # -60 dB at rt60
    ir /= np.sqrt(np.sum(ir**2, axis=1, keepdims=True)) + EPS
    if not np.isfinite(wet_db):
        return x
    wet = np.stack([fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])
    return x + db2lin(wet_db) * wet


# ============================================================ bus ops


def eq_tilt(x: np.ndarray, sr: int, db_per_oct: float = 0.0, pivot_hz: float = 1000.0) -> np.ndarray:
    """Tilt EQ as opposite low/high shelves around the pivot (RBJ cookbook, S = 0.5).
    ASSUMPTION: shelf gain = 2.5 * db_per_oct (approximates the slope over ~5 octaves)."""
    g = 2.5 * db_per_oct
    return _shelf(_shelf(x, sr, pivot_hz, -g, "low"), sr, pivot_hz, g, "high")


def _shelf(x: np.ndarray, sr: int, f0: float, gain_db: float, kind: str) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / 2 * np.sqrt((a + 1 / a) * (1 / 0.5 - 1) + 2)
    c, sq = np.cos(w0), 2 * np.sqrt(a) * alpha
    if kind == "low":
        b = [a * ((a + 1) - (a - 1) * c + sq), 2 * a * ((a - 1) - (a + 1) * c), a * ((a + 1) - (a - 1) * c - sq)]
        den = [(a + 1) + (a - 1) * c + sq, -2 * ((a - 1) + (a + 1) * c), (a + 1) + (a - 1) * c - sq]
    else:
        b = [a * ((a + 1) + (a - 1) * c + sq), -2 * a * ((a - 1) + (a + 1) * c), a * ((a + 1) + (a - 1) * c - sq)]
        den = [(a + 1) - (a - 1) * c + sq, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - sq]
    sos = np.array([[*(np.array(b) / den[0]), 1.0, den[1] / den[0], den[2] / den[0]]])
    return sosfilt(sos, x, axis=-1)


def stereo_width(x: np.ndarray, width: float = 1.0) -> np.ndarray:
    return pan_narrow(x, width)


def _block_env(x: np.ndarray) -> np.ndarray:
    """Channel-linked peak per block of _BLOCK samples -> [n_blocks]."""
    mono = np.max(np.abs(x), axis=0)
    n_blocks = -(-mono.size // _BLOCK)
    padded = np.zeros(n_blocks * _BLOCK)
    padded[: mono.size] = mono
    return padded.reshape(n_blocks, _BLOCK).max(axis=1)


def _ballistics(target: np.ndarray, sr: int, attack_ms: float, release_ms: float, falling_is_attack: bool,
                start: float) -> np.ndarray:
    """One-pole smoothing with separate attack/release time constants (per block)."""
    step = _BLOCK / sr
    a_att = np.exp(-step / max(attack_ms * 1e-3, step)) if attack_ms > 0 else 0.0
    a_rel = np.exp(-step / max(release_ms * 1e-3, step)) if release_ms > 0 else 0.0
    out = np.empty_like(target)
    state = start
    for i, t in enumerate(target):
        attacking = (t < state) if falling_is_attack else (t > state)
        coef = a_att if attacking else a_rel
        state = coef * state + (1 - coef) * t
        out[i] = state
    return out


def _expand(g_blocks: np.ndarray, n: int) -> np.ndarray:
    centers = (np.arange(g_blocks.size) + 0.5) * _BLOCK
    return np.interp(np.arange(n), centers, g_blocks)


def compressor_gain(x: np.ndarray, sr: int, threshold_db: float = 0.0, ratio: float = 1.0,
                    attack_ms: float = 10.0, release_ms: float = 100.0, makeup_db: float = 0.0) -> np.ndarray:
    """Feed-forward, linked, hard-knee compressor -> gain envelope [N] (1.0 = unchanged)."""
    env_db = 20 * np.log10(_block_env(x) + EPS)
    target = -np.maximum(env_db - threshold_db, 0.0) * (1.0 - 1.0 / ratio)
    gr = _ballistics(target, sr, attack_ms, release_ms, falling_is_attack=True, start=0.0)
    return _expand(10 ** ((gr + makeup_db) / 20), x.shape[1])


def parallel_gain(comp_gain: np.ndarray, mix: float = 0.0) -> np.ndarray:
    """(1 - mix) * dry + mix * compressed  ==  ((1 - mix) + mix * g) * dry."""
    return (1.0 - mix) + mix * comp_gain


def transient_gain(x: np.ndarray, sr: int, attack_db: float = 0.0, sustain_db: float = 0.0) -> np.ndarray:
    """Envelope-follower transient shaper: (fast - slow) envelope marks attacks."""
    env = 20 * np.log10(_block_env(x) + EPS)
    fast = _ballistics(env, sr, 1.0, 30.0, falling_is_attack=False, start=env[0])
    slow = _ballistics(env, sr, 30.0, 300.0, falling_is_attack=False, start=env[0])
    attack = np.clip((fast - slow) / 12.0, 0.0, 1.0)  # ASSUMPTION: a 12 dB rise = full attack
    g_db = attack * attack_db + (1.0 - attack) * sustain_db
    return _expand(10 ** (g_db / 20), x.shape[1])


def limiter_gain(x: np.ndarray, sr: int, ceiling_dbfs: float = 0.0, release_ms: float = 50.0) -> np.ndarray:
    """Brickwall: a smoothed gain with one block of look-ahead, then a per-sample
    min(g, ceiling / |x|) so the sample peak can never exceed the ceiling.
    The result is still a pure gain envelope."""
    ceiling = db2lin(ceiling_dbfs)
    peak = np.max(np.abs(x), axis=0)
    need = np.minimum(1.0, ceiling / np.maximum(peak, EPS))
    blocks = -(-need.size // _BLOCK)
    padded = np.ones(blocks * _BLOCK)
    padded[: need.size] = need
    need_db = 20 * np.log10(padded.reshape(blocks, _BLOCK).min(axis=1))
    need_db = np.minimum(need_db, np.concatenate([need_db[1:], [0.0]]))
    g = _expand(10 ** (_ballistics(need_db, sr, 0.0, release_ms, True, 0.0) / 20), x.shape[1])
    return np.minimum(g, need)


def saturate(x: np.ndarray, drive_db: float = 0.0, kind: str = "tape", wet: float = 0.0) -> np.ndarray:
    """tape = symmetric tanh, tube = biased tanh (adds even harmonics). wet 0 = bypass."""
    d = db2lin(drive_db)
    if kind == "tape":
        y = np.tanh(d * x) / np.tanh(d)
    elif kind == "tube":
        bias = 0.2
        y = (np.tanh(d * (x + bias)) - np.tanh(d * bias)) / (np.tanh(d * (1 + bias)) - np.tanh(d * bias))
    else:
        raise ValueError(f"unknown saturation kind {kind!r}")
    return x + wet * (y - x)


def dither(x: np.ndarray, rng: np.random.Generator, bits: int | None = None) -> np.ndarray:
    if bits is None:
        return x
    q = 2.0 ** -(bits - 1)
    tpdf = (rng.random(x.shape) - rng.random(x.shape)) * q
    return np.round((x + tpdf) / q) * q


# ============================================================ distributing to stems


def distribute(stems: np.ndarray, diff: np.ndarray, n_fft: int = 2048) -> np.ndarray:
    """Split diff [C, N] across stems [S, C, N] with the stems' power-ratio masks.
    TF bins where every stem is silent go to residual. Returns new stems whose sum is
    sum(stems) + diff; the iSTFT rounding leftover is added to residual."""
    st = Stft(StftParams(n_fft, n_fft // 4))
    n = diff.shape[-1]
    with torch.no_grad():
        p = st.forward(torch.from_numpy(stems)).abs() ** 2  # [S, C, F, T]
        total = p.sum(0, keepdim=True)
        masks = torch.where(total > 0, p / total.clamp_min(1e-300), torch.zeros_like(p))
        masks[RESIDUAL] += (total[0] == 0).to(p.dtype)
        d = st.forward(torch.from_numpy(diff))
        parts = st.inverse(masks * d.unsqueeze(0), n).numpy()
    out = stems + parts
    out[RESIDUAL] += (stems.sum(0) + diff) - out.sum(0)
    return out


def contaminate(stems: np.ndarray, contaminant: np.ndarray, level_db: float) -> np.ndarray:
    """Add non-drum material (bass / guitar bleed) to the residual target, at level_db
    relative to the drum mixture RMS. MUST come from licensed material."""
    drums = stems.sum(0)
    n = stems.shape[-1]
    c = contaminant[..., :n]
    if c.shape[-1] < n:
        c = np.pad(c, [(0, 0), (0, n - c.shape[-1])])
    scale = db2lin(level_db) * np.sqrt(np.mean(drums**2) + EPS) / np.sqrt(np.mean(c**2) + EPS)
    out = stems.copy()
    out[RESIDUAL] = out[RESIDUAL] + scale * c
    return out


# ============================================================ config-driven chain


@dataclass
class Example:
    stems: np.ndarray  # [7, C, N] float64
    mixture: np.ndarray  # [C, N] float64
    applied: list[dict]


def _draw(rng: np.random.Generator, spec: Any):
    """[lo, hi] (two numbers) = uniform range; any other list = discrete choice."""
    if isinstance(spec, list) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
        return float(rng.uniform(spec[0], spec[1]))
    if isinstance(spec, list):
        return spec[int(rng.integers(len(spec)))]
    return spec


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    mode = cfg.get("target_mode", "distribute")
    if mode not in ("distribute", "mixture_only"):
        raise KitsplitInputError(f"target_mode must be distribute|mixture_only, got {mode!r}")
    return cfg


BUS_ORDERABLE = ("compressor", "saturation", "transient_shaper", "eq_tilt", "stereo_width")
BUS_LAST = ("limiter", "dither")


def apply_chain(stems: np.ndarray, sr: int, cfg: dict, rng: np.random.Generator,
                contaminant: np.ndarray | None = None) -> Example:
    stems = np.array(stems, dtype=np.float64)
    applied: list[dict] = []

    def draw(section: dict, name: str) -> dict | None:
        spec = section.get(name)
        if not spec or rng.random() >= float(spec.get("p", 0.0)):
            return None
        params = {k: _draw(rng, v) for k, v in spec.items() if k != "p"}
        applied.append({"op": name, **params})
        return params

    # 1. per-stem pre-mix
    pre = cfg.get("per_stem") or {}
    for i in range(stems.shape[0]):
        s = stems[i]
        if (p := draw(pre, "gain")) is not None:
            s = gain(s, p["gain_db"])
        if (p := draw(pre, "pan_narrow")) is not None:
            s = pan_narrow(s, p["width"])
        if (p := draw(pre, "hpf")) is not None:
            s = highpass(s, sr, p["cutoff_hz"])
        if (p := draw(pre, "lpf")) is not None:
            s = lowpass(s, sr, p["cutoff_hz"])
        if (p := draw(pre, "room")) is not None:
            s = room(s, sr, rng, p["wet_db"], p["rt60"])
        stems[i] = s

    # 3. contamination happens "at the microphones", i.e. before the bus chain
    if contaminant is not None and (p := draw({"contamination": cfg.get("contamination") or {}}, "contamination")):
        stems = contaminate(stems, contaminant, p["level_db"])

    clean = stems.copy()
    mix = stems.sum(0)

    # 2. bus chain: random order "within realism"; limiter and dither always last
    bus = cfg.get("bus") or {}
    orderable = [k for k in BUS_ORDERABLE if k in bus]
    order = [orderable[i] for i in rng.permutation(len(orderable))] + [k for k in BUS_LAST if k in bus]
    for name in order:
        p = draw(bus, name)
        if p is None:
            continue
        if name in ("compressor", "transient_shaper", "limiter"):
            if name == "compressor":
                g = compressor_gain(mix, sr, p["threshold_db"], p["ratio"], p["attack_ms"], p["release_ms"],
                                    p.get("makeup_db", 0.0))
                g = parallel_gain(g, p.get("parallel_mix", 1.0))
            elif name == "transient_shaper":
                g = transient_gain(mix, sr, p["attack_db"], p["sustain_db"])
            else:
                g = limiter_gain(mix, sr, p["ceiling_dbfs"], p["release_ms"])
            stems, mix = stems * g, mix * g
        elif name == "eq_tilt":
            stems = np.stack([eq_tilt(s, sr, p["db_per_oct"]) for s in stems])
            mix = eq_tilt(mix, sr, p["db_per_oct"])
        elif name == "stereo_width":
            stems = np.stack([stereo_width(s, p["width"]) for s in stems])
            mix = stereo_width(mix, p["width"])
        elif name == "saturation":
            y = saturate(mix, p["drive_db"], p["kind"], p.get("wet", 1.0))
            stems, mix = distribute(stems, y - mix), y
        elif name == "dither":
            y = dither(mix, rng, int(p["bits"]))
            stems[RESIDUAL] += y - mix
            mix = y

    if cfg.get("target_mode", "distribute") == "mixture_only":
        return Example(clean, mix, applied)
    stems[RESIDUAL] += mix - stems.sum(0)  # float64 rounding leftover (~1e-16)
    return Example(stems, mix, applied)


def augment_dataset(inp: Path, out: Path, config: Path, seed: int = 0, copies: int = 1,
                    contaminants: list[np.ndarray] | None = None) -> int:
    """Type 1 -> Type 1: each track dir with the 7 stems gets `copies` augmented versions."""
    cfg = load_config(config)
    cont_cfg = cfg.get("contamination") or {}
    if contaminants is None and float(cont_cfg.get("p", 0)) > 0:
        if not cont_cfg.get("source_dir"):
            raise KitsplitInputError("contamination.p > 0 needs contamination.source_dir (licensed material only)")
        contaminants = [audio.read(p)[0].astype(np.float64) for p in sorted(Path(cont_cfg["source_dir"]).glob("*.wav"))]
    tracks = sorted(d for d in Path(inp).rglob("*") if d.is_dir() and (d / "kick.wav").is_file())
    if not tracks:
        raise KitsplitInputError(f"no Type 1 tracks (dirs with kick.wav) under {inp}")
    n = 0
    for ti, d in enumerate(tracks):
        loaded = [audio.read(d / f"{s}.wav") for s in STEMS7]
        sr = loaded[0][1].sr
        stems = np.stack([x for x, _ in loaded]).astype(np.float64)
        for k in range(copies):
            rng = np.random.default_rng([seed, ti, k])
            cont = contaminants[int(rng.integers(len(contaminants)))] if contaminants else None
            ex = apply_chain(stems, sr, cfg, rng, cont)
            dest = Path(out) / f"{d.relative_to(inp).as_posix().replace('/', '__')}__aug{k:02d}"
            for name, s in zip(STEMS7, ex.stems):
                audio.write(dest / f"{name}.wav", s.astype(np.float32), sr, "wav", "32f")
            audio.write(dest / "mixture.wav", ex.mixture.astype(np.float32), sr, "wav", "32f")
            (dest / "augment.yaml").write_text(yaml.safe_dump({"seed": [seed, ti, k], "ops": ex.applied}))
            n += 1
    return n
