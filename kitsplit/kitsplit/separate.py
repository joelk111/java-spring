"""The pipeline: io -> resample/channel policy -> chunker -> backend -> resample back
-> consistency -> grouping -> io.

Deviation from the plan §6 diagram (deliberate): the consistency projection runs LAST,
after resampling back to the input sample rate and restoring the input channel count.
Projecting at the model rate would make the stems sum to the *resampled* input, not
to the file the user gave us.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from kitsplit._msst import PROJECT_ROOT
from kitsplit.dsp import chunker
from kitsplit.dsp.consistency import project
from kitsplit.errors import KitsplitInputError, KitsplitOutputExistsError
from kitsplit.io import audio, stems as stems_io

log = logging.getLogger("kitsplit")

PRESET_DIR = PROJECT_ROOT / "configs" / "inference"
# ASSUMPTION: below this peak the input is treated as silence (-120 dBFS).
SILENCE_PEAK = 1e-6


@dataclass
class Settings:
    chunk_sec: float = 10.0
    overlap: float = 0.75
    shifts: int = 4
    n_stems: int = 7
    fmt: str = "wav"
    bit_depth: str | None = None  # None = derived from the input
    consistency: bool = True
    overwrite: bool = False

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "Settings":
        path = PRESET_DIR / f"{preset}.yaml"
        if not path.is_file():
            allowed = "|".join(sorted(p.stem for p in PRESET_DIR.glob("*.yaml")))
            raise KitsplitInputError(f"unknown preset {preset!r}; allowed: {allowed}")
        values = yaml.safe_load(path.read_text())
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def validate(self) -> None:
        stems_io.check_n_stems(self.n_stems)
        if self.chunk_sec <= 0:
            raise KitsplitInputError("--chunk-sec must be > 0")
        if not 0 <= self.overlap < 1:
            raise KitsplitInputError("--overlap must be in [0, 1)")
        if self.shifts < 1:
            raise KitsplitInputError("--shifts must be >= 1")
        if self.bit_depth is not None:
            audio.validate_output_format(self.fmt, self.bit_depth)
        elif self.fmt not in ("wav", "flac"):
            raise KitsplitInputError(f"unknown format {self.fmt!r}; allowed: wav, flac")


@dataclass
class Result:
    input: str
    sr: int
    channels: int
    duration: float
    bit_depth: str
    outputs: dict[str, str] = field(default_factory=dict)
    sum_error: float = 0.0
    warnings: list[str] = field(default_factory=list)


def separate_array(x: np.ndarray, sr: int, channels: int, backend, settings: Settings) -> np.ndarray:
    """x: [2, N] float32 at `sr` (after the mono->stereo policy).
    Returns 7 stems [7, channels, N] at `sr` (consistency-projected if enabled)."""
    n = x.shape[1]
    msr = backend.sample_rate or audio.model_sr(sr)
    xm = audio.resample(x, sr, msr)
    if getattr(backend, "full_track", False):
        ym = backend.run_full(xm)
    else:
        chunk_len = backend.chunk_len or int(round(settings.chunk_sec * msr))
        ym = chunker.run_chunked(xm, backend.run, chunk_len, settings.overlap, settings.shifts)
    y = audio.resample(ym, msr, sr)
    if y.shape[-1] < n:
        y = np.concatenate([y, np.zeros((*y.shape[:-1], n - y.shape[-1]), np.float32)], axis=-1)
    y = y[..., :n]
    y = audio.restore_channels(y, channels)
    if settings.consistency:
        y = project(y, audio.restore_channels(x, channels))
    if not np.all(np.isfinite(y)):
        raise RuntimeError("model produced non-finite samples")
    return y


def output_dir_for(out: Path, inp: Path, multiple: bool) -> Path:
    return out / inp.stem if multiple else out


def separate_file(inp: Path, out_dir: Path, backend, settings: Settings) -> Result:
    x, info = audio.read(inp)
    bit_depth = settings.bit_depth or audio.default_bit_depth(info)
    audio.validate_output_format(settings.fmt, bit_depth)
    names = list(stems_io.GROUPINGS[settings.n_stems])
    targets = {name: out_dir / f"{name}.{settings.fmt}" for name in names}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing and not settings.overwrite:
        raise KitsplitOutputExistsError(f"output exists (use --overwrite): {', '.join(existing)}")

    result = Result(str(inp), info.sr, info.channels, info.duration, bit_depth)
    if float(np.max(np.abs(x))) < SILENCE_PEAK:
        msg = f"{inp}: input is silent (peak < {SILENCE_PEAK:g}); all stems will be silent"
        log.warning(msg)
        result.warnings.append(msg)

    y = separate_array(x, info.sr, info.channels, backend, settings)
    grouped = stems_io.group(y, settings.n_stems)
    for name, data in grouped.items():
        audio.write(targets[name], data, info.sr, settings.fmt, bit_depth)
        result.outputs[name] = str(targets[name])
    result.sum_error = stems_io.sum_error(grouped, audio.restore_channels(x, info.channels))
    return result
