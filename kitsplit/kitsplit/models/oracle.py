"""Oracle "models" used ONLY to test plumbing.

They are given the answer (identity) or the true stems (ideal ratio mask). Any number
produced with them says nothing about how well a trained separator works.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from kitsplit.dsp.stft import Stft, StftParams
from kitsplit.io.stems import STEMS7

RESIDUAL = STEMS7.index("residual")


class IdentityOracle:
    """Everything goes to residual; named stems are silent."""

    name = "oracle-identity"
    sample_rate: int | None = None
    chunk_len: int | None = None
    needs_reference = False

    def run(self, chunks: np.ndarray, starts: Sequence[int]) -> np.ndarray:
        out = np.zeros((chunks.shape[0], len(STEMS7), *chunks.shape[1:]), dtype=np.float32)
        out[:, RESIDUAL] = chunks
        return out


class IdealRatioMaskOracle:
    """Applies the ideal ratio mask |S_i|^2 / sum_j |S_j|^2 (computed from the true
    stems) to the mixture STFT of each chunk."""

    name = "oracle-irm"
    chunk_len: int | None = None
    needs_reference = True
    # ASSUMPTION: oracle STFT size; only used by tests, not tied to any trained model.
    PARAMS = StftParams(n_fft=2048, hop_length=512)

    def __init__(self, references: np.ndarray, sample_rate: int):
        """references: [7, C, N] true stems at `sample_rate` (STEMS7 order)."""
        if references.shape[0] != len(STEMS7):
            raise ValueError("references must have 7 stems")
        self.references = references.astype(np.float32)
        self.sample_rate = sample_rate
        self.stft = Stft(self.PARAMS)

    def _ref_window(self, start: int, length: int) -> np.ndarray:
        s, c, n = self.references.shape
        out = np.zeros((s, c, length), dtype=np.float32)
        lo, hi = max(start, 0), min(start + length, n)
        if hi > lo:
            out[..., lo - start : hi - start] = self.references[..., lo:hi]
        return out

    def run(self, chunks: np.ndarray, starts: Sequence[int]) -> np.ndarray:
        length = chunks.shape[-1]
        refs = np.stack([self._ref_window(s, length) for s in starts])  # [K, 7, C, L]
        with torch.no_grad():
            mix_spec = self.stft.forward(torch.from_numpy(chunks).double())  # [K, C, F, T]
            ref_pow = self.stft.forward(torch.from_numpy(refs).double()).abs() ** 2  # [K, 7, C, F, T]
            mask = ref_pow / ref_pow.sum(dim=1, keepdim=True).clamp_min(1e-20)
            est = self.stft.inverse(mask * mix_spec.unsqueeze(1), length)
        return est.float().numpy()
