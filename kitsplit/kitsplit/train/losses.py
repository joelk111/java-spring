"""Multi-resolution STFT + time-domain L1 loss with transient weighting (plan §7.2).

Frames within +-TRANSIENT_MS of a ground-truth onset get weight TRANSIENT_WEIGHT.
NOTE: not yet wired into MSST's training loop (see train/msst_adapter.py).
"""

from __future__ import annotations

from typing import Sequence

import torch

# ASSUMPTION: plan §7.2 values; to be ablated in Phase 3.
FFT_SIZES = (256, 1024, 4096)
TRANSIENT_MS = 10.0
TRANSIENT_WEIGHT = 3.0


def onset_weights(n: int, sr: int, onsets: Sequence[float], ms: float = TRANSIENT_MS,
                  weight: float = TRANSIENT_WEIGHT) -> torch.Tensor:
    """Per-sample weight [n]: `weight` within +-ms of any onset, else 1."""
    w = torch.ones(n)
    half = int(round(ms * 1e-3 * sr))
    for t in onsets:
        c = int(round(t * sr))
        w[max(0, c - half) : max(0, min(n, c + half + 1))] = weight
    return w


def _frame_weights(w: torch.Tensor, n_fft: int, hop: int, n_frames: int) -> torch.Tensor:
    """Max of the sample weights under each STFT frame (center=True framing)."""
    padded = torch.nn.functional.pad(w[None, None], (n_fft // 2, n_fft // 2), value=1.0)[0, 0]
    frames = padded.unfold(0, n_fft, hop)[:n_frames]
    return frames.max(dim=1).values


def mrstft_l1_loss(pred: torch.Tensor, target: torch.Tensor, sr: int,
                   weights: torch.Tensor | None = None, fft_sizes: Sequence[int] = FFT_SIZES) -> torch.Tensor:
    """pred/target: [..., N]. weights: [N] per-sample weights (onset_weights) or None.
    Returns a scalar: weighted time-domain L1 + mean over resolutions of weighted
    (spectral-convergence-free) log-magnitude + linear-magnitude L1."""
    n = pred.shape[-1]
    w = torch.ones(n, dtype=pred.dtype, device=pred.device) if weights is None else weights.to(pred)
    loss = ((pred - target).abs() * w).sum() / (w.sum() * pred[..., 0].numel())
    flat_p = pred.reshape(-1, n)
    flat_t = target.reshape(-1, n)
    for n_fft in fft_sizes:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, dtype=pred.dtype, device=pred.device)
        sp = torch.stft(flat_p, n_fft, hop, window=win, return_complex=True).abs()
        st = torch.stft(flat_t, n_fft, hop, window=win, return_complex=True).abs()
        fw = _frame_weights(w, n_fft, hop, sp.shape[-1])
        lin = ((sp - st).abs() * fw).sum() / (fw.sum() * sp.shape[0] * sp.shape[1])
        log = ((torch.log(sp + 1e-7) - torch.log(st + 1e-7)).abs() * fw).sum() / (fw.sum() * sp.shape[0] * sp.shape[1])
        loss = loss + (lin + log) / len(fft_sizes)
    return loss
