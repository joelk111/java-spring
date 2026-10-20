"""One-shot x onset renderer.

Two uses:
- test fixtures: simple synthesised one-shots (sine-sweep kick, noise-burst snare,
  filtered-noise hats ...) on a fixed onset pattern;
- electronic-kit training data: user-supplied one-shot samples (velocity layers)
  placed on onsets (e.g. from Groove MIDI) — the IDM "drum machine" idea.

Every render returns the stems AND the mixture; the mixture is the float64 sum of the
stems, so sum(stems) == mixture holds exactly (bit-exact after optional quantisation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfilt

from kitsplit.io.stems import STEMS7

Onset = tuple[float, int]  # (time in seconds, MIDI velocity 1..127)


@dataclass
class Render:
    sr: int
    stems: np.ndarray  # [7, 2, N] float32, STEMS7 order
    mixture: np.ndarray  # [2, N] float32
    onsets: dict[str, list[Onset]] = field(default_factory=dict)


# ---------------------------------------------------------------- one-shots


def _env(sr: int, dur: float, decay: float) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.exp(-t / decay)


def _noise(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal(n)


def _filt(x: np.ndarray, sr: int, kind: str, freq) -> np.ndarray:
    sos = butter(4, freq, btype=kind, fs=sr, output="sos")
    return sosfilt(sos, x)


def kick(sr: int, rng: np.random.Generator) -> np.ndarray:
    dur = 0.45
    t = np.arange(int(dur * sr)) / sr
    freq = 45 + 105 * np.exp(-t / 0.04)  # sweep 150 -> 45 Hz
    phase = 2 * np.pi * np.cumsum(freq) / sr
    click = _filt(_noise(rng, len(t)), sr, "bandpass", (1500, 5000)) * _env(sr, dur, 0.003)
    return np.sin(phase) * _env(sr, dur, 0.12) + 0.2 * click


def snare(sr: int, rng: np.random.Generator) -> np.ndarray:
    dur = 0.3
    t = np.arange(int(dur * sr)) / sr
    body = np.sin(2 * np.pi * 185 * t) * _env(sr, dur, 0.04)
    rattle = _filt(_noise(rng, len(t)), sr, "bandpass", (1200, 7000)) * _env(sr, dur, 0.07)
    return 0.6 * body + 0.8 * rattle


def tom(sr: int, rng: np.random.Generator, f0: float) -> np.ndarray:
    dur = 0.6
    t = np.arange(int(dur * sr)) / sr
    freq = f0 * (1 + 0.3 * np.exp(-t / 0.05))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * _env(sr, dur, 0.18)


def hihat(sr: int, rng: np.random.Generator, open_: bool = False) -> np.ndarray:
    dur = 0.4 if open_ else 0.08
    x = _filt(_noise(rng, int(dur * sr)), sr, "highpass", min(7000, 0.4 * sr))
    return 0.5 * x * _env(sr, dur, 0.12 if open_ else 0.015)


def ride(sr: int, rng: np.random.Generator) -> np.ndarray:
    dur = 1.5
    t = np.arange(int(dur * sr)) / sr
    partials = [1.0, 1.47, 2.09, 2.56, 3.14]  # inharmonic "metal" ratios
    ping = sum(np.sin(2 * np.pi * 620 * p * t + i) / (i + 1) for i, p in enumerate(partials))
    wash = _filt(_noise(rng, len(t)), sr, "bandpass", (3000, 9000))
    return (0.3 * ping + 0.15 * wash) * _env(sr, dur, 0.5)


def crash(sr: int, rng: np.random.Generator) -> np.ndarray:
    dur = 2.0
    x = _filt(_noise(rng, int(dur * sr)), sr, "bandpass", (2500, min(12000, 0.45 * sr)))
    return 0.5 * x * _env(sr, dur, 0.6)


def residual_tone(sr: int, n: int) -> np.ndarray:
    """Low-level sustained bass note: stands in for non-drum bleed routed to residual."""
    t = np.arange(n) / sr
    return 0.05 * np.sin(2 * np.pi * 55 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))


# ---------------------------------------------------------------- pattern

# ASSUMPTION: stereo placement for the fixture kit (−1 = hard left).
PAN = {"kick": 0.0, "snare": -0.1, "toms": 0.3, "hh": -0.4, "ride": 0.45, "crash": -0.3, "residual": 0.0}


def fixed_pattern(duration: float, bpm: float = 120.0) -> dict[str, list[Onset]]:
    """Deterministic 4/4 groove used by the fixtures."""
    beat = 60.0 / bpm
    n_beats = int(duration / beat)
    p: dict[str, list[Onset]] = {s: [] for s in STEMS7 if s != "residual"}
    for b in range(n_beats):
        bar, pos = divmod(b, 4)
        t = b * beat
        last_bar = bar == (n_beats // 4) - 1
        if pos in (0, 2) and not (last_bar and pos == 2):
            p["kick"].append((t, 110))
        if pos in (1, 3) and not last_bar:
            p["snare"].append((t, 100))
        if bar % 2 == 0:
            p["hh"].extend([(t, 80), (t + beat / 2, 60)])
        else:
            p["ride"].append((t, 90))
        if pos == 0 and bar % 2 == 0:
            p["crash"].append((t, 100))
        if last_bar and pos >= 2:
            p["toms"].extend([(t, 100), (t + beat / 2, 90)])
    return {k: [o for o in v if o[0] < duration] for k, v in p.items()}


def _place(dst: np.ndarray, shot: np.ndarray, start: int, gain: float, pan: float) -> None:
    n = min(len(shot), dst.shape[1] - start)
    if n <= 0:
        return
    left = np.sqrt(0.5 * (1 - pan))
    right = np.sqrt(0.5 * (1 + pan))
    dst[0, start : start + n] += gain * left * shot[:n]
    dst[1, start : start + n] += gain * right * shot[:n]


def _quantize(x: np.ndarray, bits: int) -> np.ndarray:
    scale = float(2 ** (bits - 1))
    return np.round(x * scale) / scale


def render_fixture_kit(
    sr: int,
    duration: float,
    seed: int = 0,
    peak: float = 0.5,
    quantize_bits: int | None = 24,
    pattern: Mapping[str, Sequence[Onset]] | None = None,
) -> Render:
    """Synthesised kit on `fixed_pattern`. With quantize_bits, every stem lies on the
    PCM grid, so stems and mixture can be written as PCM without breaking the sum."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * sr))
    pattern = dict(pattern) if pattern is not None else fixed_pattern(duration)
    stems = np.zeros((len(STEMS7), 2, n), dtype=np.float64)
    tom_f = [210.0, 150.0, 100.0]
    for si, name in enumerate(STEMS7):
        if name == "residual":
            tone = residual_tone(sr, n)
            stems[si] = np.stack([tone, tone])
            continue
        for k, (t, vel) in enumerate(pattern.get(name, [])):
            if name == "kick":
                shot = kick(sr, rng)
            elif name == "snare":
                shot = snare(sr, rng)
            elif name == "toms":
                shot = tom(sr, rng, tom_f[k % 3])
            elif name == "hh":
                shot = hihat(sr, rng, open_=(k % 8 == 7))
            elif name == "ride":
                shot = ride(sr, rng)
            else:
                shot = crash(sr, rng)
            _place(stems[si], shot, int(round(t * sr)), vel / 127.0, PAN[name])
    mix = stems.sum(axis=0)
    gain = peak / max(np.max(np.abs(mix)), 1e-12)
    stems *= gain
    if quantize_bits:
        stems = _quantize(stems, quantize_bits)
    mixture = stems.sum(axis=0)
    return Render(sr, stems.astype(np.float32), mixture.astype(np.float32), {k: list(v) for k, v in pattern.items()})


def render_from_oneshots(
    oneshots: Mapping[str, Sequence[np.ndarray]],
    onsets: Mapping[str, Sequence[Onset]],
    sr: int,
    duration: float,
    pan: Mapping[str, float] | None = None,
) -> Render:
    """Electronic-kit renderer. oneshots[stem] = velocity layers (soft -> hard), each a
    mono array at `sr`. The layer is chosen from the velocity; gain = velocity / 127."""
    n = int(round(duration * sr))
    stems = np.zeros((len(STEMS7), 2, n), dtype=np.float64)
    pan = pan or {}
    for name, hits in onsets.items():
        if name not in STEMS7:
            raise ValueError(f"unknown stem {name!r}")
        layers = oneshots.get(name)
        if not layers:
            raise ValueError(f"no one-shots for stem {name!r}")
        si = STEMS7.index(name)
        for t, vel in hits:
            layer = layers[min(len(layers) - 1, int(vel / 128 * len(layers)))]
            _place(stems[si], np.asarray(layer, dtype=np.float64), int(round(t * sr)), vel / 127.0, pan.get(name, 0.0))
    mixture = stems.sum(axis=0)
    return Render(sr, stems.astype(np.float32), mixture.astype(np.float32), {k: list(v) for k, v in onsets.items()})


def write_render(render: Render, mixture_path: Path | None, stems_dir: Path, bit_depth: str = "24") -> None:
    from kitsplit.io import audio

    stems_dir.mkdir(parents=True, exist_ok=True)
    for name, s in zip(STEMS7, render.stems):
        audio.write(stems_dir / f"{name}.wav", s, render.sr, "wav", bit_depth)
    if mixture_path is not None:
        audio.write(mixture_path, render.mixture, render.sr, "wav", bit_depth)
