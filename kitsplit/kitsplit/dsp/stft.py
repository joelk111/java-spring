"""Host-side STFT/iSTFT.

The parameters are always loaded from the model config YAML (the same file the
training config is generated from), never duplicated as constants here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class StftParams:
    n_fft: int
    hop_length: int
    window: str = "hann"

    @classmethod
    def from_config(cls, config: str | Path | dict) -> "StftParams":
        """Read `audio.n_fft` / `audio.hop_length` / `audio.window` from a model config."""
        if not isinstance(config, dict):
            config = yaml.safe_load(Path(config).read_text())
        audio = config["audio"]
        return cls(int(audio["n_fft"]), int(audio["hop_length"]), str(audio.get("window", "hann")))

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1


class Stft:
    def __init__(self, params: StftParams):
        if params.window != "hann":
            raise ValueError(f"unsupported window {params.window!r}")
        self.params = params
        self._windows: dict[tuple[torch.dtype, torch.device], torch.Tensor] = {}

    def _window(self, ref: torch.Tensor) -> torch.Tensor:
        dtype = ref.real.dtype if ref.is_complex() else ref.dtype
        key = (dtype, ref.device)
        if key not in self._windows:
            self._windows[key] = torch.hann_window(self.params.n_fft, periodic=True, dtype=dtype, device=ref.device)
        return self._windows[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., N] real -> [..., F, T] complex."""
        lead = x.shape[:-1]
        spec = torch.stft(
            x.reshape(-1, x.shape[-1]),
            n_fft=self.params.n_fft,
            hop_length=self.params.hop_length,
            window=self._window(x),
            center=True,
            pad_mode="constant",
            return_complex=True,
        )
        return spec.reshape(*lead, *spec.shape[-2:])

    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """spec: [..., F, T] complex -> [..., length] real."""
        lead = spec.shape[:-2]
        x = torch.istft(
            spec.reshape(-1, *spec.shape[-2:]),
            n_fft=self.params.n_fft,
            hop_length=self.params.hop_length,
            window=self._window(spec),
            center=True,
            length=length,
        )
        return x.reshape(*lead, length)
