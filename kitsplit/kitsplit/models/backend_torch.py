"""Torch backends. All expose the same interface used by `separate`:

    sample_rate: int | None   model sample rate (None = any)
    chunk_len:  int | None    fixed chunk length in samples (None = from preset)
    run(chunks [K, 2, L], starts) -> [K, 7, 2, L]      (STEMS7 order)
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from kitsplit.dsp.consistency import masks_from_logits
from kitsplit.dsp.stft import Stft, StftParams
from kitsplit.errors import KitsplitDeviceError, KitsplitModelError
from kitsplit.io.stems import STEMS7


def resolve_device(device: str) -> str:
    """auto|cpu|cuda|dml -> torch device string. Raises KitsplitDeviceError."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise KitsplitDeviceError("--device cuda requested but no CUDA device is available")
        return "cuda"
    if device == "dml":
        raise KitsplitDeviceError("--device dml is only available with --backend onnx")
    raise KitsplitDeviceError(f"unknown device {device!r}; allowed: auto|cpu|cuda|dml")


def set_deterministic(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


# ----------------------------------------------------------- mask models


class TinyMaskNet(nn.Module):
    """Tiny random-init mask network. Used only to test export / backend equivalence.

    in : [B, 4, F, T]  (stereo complex spectrogram as 2ch x (re, im))
    out: [B, 7, 2, F, T] mask logits
    """

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, len(STEMS7) * 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        b, _, f, t = y.shape
        return y.reshape(b, len(STEMS7), 2, f, t)


ARCHS: dict[str, Callable[..., nn.Module]] = {"tiny-mask": TinyMaskNet}


def spec_to_input(spec: torch.Tensor) -> torch.Tensor:
    """[K, 2, F, T] complex -> [K, 4, F, T] real (L.re, L.im, R.re, R.im)."""
    k, c, f, t = spec.shape
    return torch.view_as_real(spec).permute(0, 1, 4, 2, 3).reshape(k, c * 2, f, t)


class MaskBackend:
    """Host STFT -> network (torch or onnx callable) -> softmax masks -> iSTFT."""

    def __init__(self, logits_fn: Callable[[np.ndarray], np.ndarray], stft_params: StftParams,
                 sample_rate: int | None, chunk_len: int | None = None):
        self.logits_fn = logits_fn
        self.stft = Stft(stft_params)
        self.sample_rate = sample_rate
        self.chunk_len = chunk_len

    def run(self, chunks: np.ndarray, starts: Sequence[int]) -> np.ndarray:
        length = chunks.shape[-1]
        with torch.no_grad():
            spec = self.stft.forward(torch.from_numpy(np.ascontiguousarray(chunks)))  # [K, 2, F, T]
            logits = torch.from_numpy(np.asarray(self.logits_fn(spec_to_input(spec).numpy())))
            masks = masks_from_logits(logits, stem_dim=1)  # [K, 7, 2, F, T], sums to 1
            est = self.stft.inverse(masks * spec.unsqueeze(1), length)
        return est.numpy().astype(np.float32)


def torch_logits_fn(model: nn.Module, device: str) -> Callable[[np.ndarray], np.ndarray]:
    model = model.to(device).eval()

    def fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(x).to(device)).cpu().numpy()

    return fn


# ----------------------------------------------------------- MSST / demucs


def _map_outputs(est: np.ndarray, names: Sequence[str], stem_map: dict[str, str]) -> np.ndarray:
    """est [K, S, C, L] with model stem names -> [K, 7, C, L]; unmapped slots stay 0."""
    out = np.zeros((est.shape[0], len(STEMS7), *est.shape[2:]), dtype=np.float32)
    for i, name in enumerate(names):
        if name not in stem_map:
            raise KitsplitModelError(f"model output {name!r} has no stem_map entry")
        out[:, STEMS7.index(stem_map[name])] += est[:, i]
    return out


class MsstBackend:
    """Waveform-in/waveform-out MSST model (e.g. MDX23C). STFT is inside the graph."""

    def __init__(self, model: nn.Module, names: Sequence[str], stem_map: dict[str, str],
                 sample_rate: int, chunk_len: int, device: str):
        self.model = model.to(device).eval()
        self.names = list(names)
        self.stem_map = stem_map
        self.sample_rate = sample_rate
        self.chunk_len = chunk_len
        self.device = device

    def run(self, chunks: np.ndarray, starts: Sequence[int]) -> np.ndarray:
        with torch.no_grad():
            est = self.model(torch.from_numpy(chunks).to(self.device)).cpu().numpy()
        if est.ndim == 3:  # single-target models return [K, C, L]
            est = est[:, None]
        return _map_outputs(est, self.names, self.stem_map)


class DemucsBackend:
    """inagoy/drumsep HDemucs via demucs.apply.apply_model on the full track
    (apply_model's own track-level normalisation and chunking, as the drumsep CLI does)."""

    full_track = True
    chunk_len = None

    def __init__(self, model, stem_map: dict[str, str], device: str, shifts: int = 1, overlap: float = 0.25):
        self.model = model.to(device).eval()
        self.stem_map = stem_map
        self.sample_rate = int(model.samplerate)
        self.device = device
        self.shifts = shifts
        self.overlap = overlap

    def run_full(self, x: np.ndarray) -> np.ndarray:
        import random

        from demucs.apply import apply_model

        random.seed(0)  # apply_model draws its shift offsets from `random`
        mix = torch.from_numpy(x)
        ref = mix.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            est = apply_model(self.model, ((mix - mean) / std)[None], shifts=self.shifts, split=True,
                              overlap=self.overlap, device=self.device, progress=False)[0]
        est = (est * std + mean).numpy()  # [S, C, N]
        return _map_outputs(est[None], self.model.sources, self.stem_map)[0]
