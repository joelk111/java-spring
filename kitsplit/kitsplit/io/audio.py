"""Audio read/write, resampling and channel policy.

Internal representation everywhere: float32 numpy array shaped [channels, samples].
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from kitsplit.errors import KitsplitInputError

# ASSUMPTION: the model is trained at <= 48 kHz; anything above is resampled
# down for inference and the stems are resampled back (plan §6).
MAX_MODEL_SR = 48000

# soundfile subtype -> our bit-depth label
_SUBTYPE_TO_DEPTH = {
    "PCM_16": "16",
    "PCM_24": "24",
    "PCM_32": "32",
    "FLOAT": "32f",
    "DOUBLE": "32f",
}
_DEPTH_TO_SUBTYPE = {"16": "PCM_16", "24": "PCM_24", "32f": "FLOAT"}
_FORMATS = {"wav": "WAV", "flac": "FLAC"}


@dataclass(frozen=True)
class AudioInfo:
    sr: int
    channels: int  # channel count of the file on disk (before the mono->stereo policy)
    frames: int
    subtype: str
    format: str

    @property
    def duration(self) -> float:
        return self.frames / self.sr

    @property
    def bit_depth(self) -> str:
        return _SUBTYPE_TO_DEPTH.get(self.subtype, "unknown")


def probe(path: str | Path) -> AudioInfo:
    path = Path(path)
    if not path.is_file():
        raise KitsplitInputError(f"input file not found: {path}")
    try:
        info = sf.info(str(path))
    except Exception as exc:  # soundfile raises a variety of RuntimeError subclasses
        raise KitsplitInputError(f"not a readable audio file: {path} ({exc})") from exc
    if info.frames == 0:
        raise KitsplitInputError(f"audio file has 0 samples: {path}")
    if info.channels not in (1, 2):
        raise KitsplitInputError(
            f"{path}: {info.channels} channels; only mono or stereo drum buses are supported"
        )
    return AudioInfo(info.samplerate, info.channels, info.frames, info.subtype, info.format)


def read(path: str | Path) -> tuple[np.ndarray, AudioInfo]:
    """Read a file -> (float32 [2, N], info). Mono is duplicated to L/R; sr is kept."""
    info = probe(path)
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)  # [N, C]
    audio = np.ascontiguousarray(data.T)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return audio, info


def restore_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    """Undo the mono->stereo policy: [.., 2, N] -> [.., channels, N]."""
    if channels == 2:
        return audio
    if channels == 1:
        mean = audio.astype(np.float64).mean(axis=-2, keepdims=True)
        return mean.astype(np.float32)
    raise ValueError(f"unsupported channel count {channels}")


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling along the last axis (float64 internally, float32 out)."""
    if sr_in == sr_out:
        return audio
    ratio = Fraction(sr_out, sr_in)
    out = resample_poly(audio.astype(np.float64), ratio.numerator, ratio.denominator, axis=-1)
    return out.astype(np.float32)


def model_sr(sr: int) -> int:
    return min(sr, MAX_MODEL_SR)


def validate_output_format(fmt: str, bit_depth: str) -> None:
    if fmt not in _FORMATS:
        raise KitsplitInputError(f"unknown format {fmt!r}; allowed: {', '.join(_FORMATS)}")
    if bit_depth not in _DEPTH_TO_SUBTYPE:
        raise KitsplitInputError(
            f"unknown bit depth {bit_depth!r}; allowed: {', '.join(_DEPTH_TO_SUBTYPE)}"
        )
    if fmt == "flac" and bit_depth == "32f":
        raise KitsplitInputError("FLAC cannot store 32-bit float; use --format wav or --bit-depth 24")


def default_bit_depth(info: AudioInfo) -> str:
    """Output depth defaults to the input's. 16-bit and 32-bit int go to 24 / 32f.

    ASSUMPTION: 16-bit output cannot keep the stems' sum within 1e-6 of the input,
    so 16-bit inputs get 24-bit stems.
    """
    depth = info.bit_depth
    if depth == "32f" or depth == "32":
        return "32f"
    return "24"


def write(path: str | Path, audio: np.ndarray, sr: int, fmt: str = "wav", bit_depth: str = "24") -> None:
    validate_output_format(fmt, bit_depth)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"refusing to write non-finite samples to {path}")
    data = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).T)  # [N, C]
    with sf.SoundFile(str(path), "w", sr, data.shape[1], _DEPTH_TO_SUBTYPE[bit_depth], format=_FORMATS[fmt]) as f:
        _disable_peak_chunk(f)
        f.write(data)


# libsndfile writes a PEAK chunk with a wall-clock timestamp into float WAVs, which
# makes identical audio produce different bytes. Turn it off before writing.
_SFC_SET_ADD_PEAK_CHUNK = 0x1050


def _disable_peak_chunk(f: sf.SoundFile) -> None:
    from soundfile import _snd

    _snd.sf_command(f._file, _SFC_SET_ADD_PEAK_CHUNK, sf._ffi.NULL, 0)
