"""Mixture-consistency: make the 7 stems sum to the input.

Two parts (plan §6):
1. `masks_from_logits`: models output 7 mask logits per TF bin; softmax across stems
   makes the masks sum to 1, so the masked spectrograms sum to the mixture STFT.
2. `project`: the remaining leftover (iSTFT edge error, resampling, model output that
   is not mask-based) is added to `residual`.

`project` sets residual := residual + (mixture - sum(all 7)), which is algebraically
residual := mixture - sum(6 named). It is computed in the second form, in float64, so
the operation is idempotent bit-for-bit.
"""

from __future__ import annotations

import numpy as np
import torch

from kitsplit.io.stems import STEMS7

RESIDUAL_INDEX = STEMS7.index("residual")


def masks_from_logits(logits: torch.Tensor, stem_dim: int = 1) -> torch.Tensor:
    return torch.softmax(logits, dim=stem_dim)


def project(stems: np.ndarray, mixture: np.ndarray) -> np.ndarray:
    """stems: [7, C, N], mixture: [C, N] -> stems with residual replaced (float32)."""
    if stems.shape[0] != len(STEMS7) or stems.shape[1:] != mixture.shape:
        raise ValueError(f"shape mismatch: stems {stems.shape}, mixture {mixture.shape}")
    named = np.delete(stems, RESIDUAL_INDEX, axis=0).astype(np.float64)
    residual = mixture.astype(np.float64) - named.sum(axis=0)
    out = stems.astype(np.float32, copy=True)
    out[RESIDUAL_INDEX] = residual.astype(np.float32)
    return out
