"""Deterministic overlap-add chunking with shift test-time augmentation.

Bit-exactness: weights and sums are accumulated in float64 and cast to float32
once at the end. For an identity model, sum(w_i * x) / sum(w_i) differs from x by
far less than half a float32 ulp, so the cast returns x exactly.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

# fn(chunks [K, C, L] float32, starts: absolute start sample of each chunk in the
# ORIGINAL signal (may be negative / run past the end because of padding))
#   -> [K, S, C, L] (S output stems)
ChunkFn = Callable[[np.ndarray, Sequence[int]], np.ndarray]


def hop_length(chunk_len: int, overlap: float) -> int:
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    return max(1, int(round(chunk_len * (1.0 - overlap))))


def chunk_starts(n: int, chunk_len: int, overlap: float) -> list[int]:
    hop = hop_length(chunk_len, overlap)
    starts = [0]
    while starts[-1] + chunk_len < n:
        starts.append(starts[-1] + hop)
    return starts


def split(x: np.ndarray, chunk_len: int, overlap: float) -> tuple[np.ndarray, list[int]]:
    """x: [C, N] -> (chunks [K, C, chunk_len], starts). The tail is zero-padded."""
    c, n = x.shape
    starts = chunk_starts(n, chunk_len, overlap)
    chunks = np.zeros((len(starts), c, chunk_len), dtype=np.float32)
    for k, s in enumerate(starts):
        seg = x[:, s : s + chunk_len]
        chunks[k, :, : seg.shape[1]] = seg
    return chunks, starts


def _window(chunk_len: int, first: bool, last: bool) -> np.ndarray:
    # Hann without the zero end-points, so every covered sample has weight > 0.
    w = np.hanning(chunk_len + 2)[1:-1].astype(np.float64)
    half = chunk_len // 2
    if first:
        w[:half] = 1.0  # nothing overlaps the start of the signal
    if last:
        w[half:] = 1.0  # nothing overlaps the end of the signal
    return w


def overlap_add(outputs: np.ndarray, starts: Sequence[int], n: int) -> np.ndarray:
    """outputs: [K, S, C, L] -> [S, C, n] (float32)."""
    k_total, s, c, chunk_len = outputs.shape
    acc = np.zeros((s, c, n), dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    for k, start in enumerate(starts):
        w = _window(chunk_len, k == 0, k == k_total - 1)
        end = min(start + chunk_len, n)
        m = end - start
        acc[..., start:end] += outputs[k, ..., :m].astype(np.float64) * w[:m]
        wsum[start:end] += w[:m]
    return (acc / wsum).astype(np.float32)


def shift_offsets(shifts: int, max_shift: int) -> list[int]:
    """Deterministic (not random) TTA offsets: 0, max/S, 2*max/S, ..."""
    if shifts < 1:
        raise ValueError("shifts must be >= 1")
    return [int(round(i * max_shift / shifts)) for i in range(shifts)]


def run_chunked(
    x: np.ndarray,
    fn: ChunkFn,
    chunk_len: int,
    overlap: float,
    shifts: int = 1,
    batch_size: int = 4,
) -> np.ndarray:
    """Apply fn chunk-wise to x [C, N] with overlap-add and shift-TTA -> [S, C, N]."""
    c, n = x.shape
    # ASSUMPTION: TTA shifts span at most one hop, which decorrelates chunk borders.
    offsets = shift_offsets(shifts, hop_length(chunk_len, overlap))
    total = None
    for off in offsets:
        padded = np.concatenate([np.zeros((c, off), dtype=np.float32), x], axis=1) if off else x
        chunks, starts = split(padded, chunk_len, overlap)
        abs_starts = [s - off for s in starts]
        parts = []
        for b in range(0, len(chunks), batch_size):
            out = np.asarray(fn(chunks[b : b + batch_size], abs_starts[b : b + batch_size]))
            if out.ndim != 4 or out.shape[0] != len(chunks[b : b + batch_size]) or out.shape[2:] != (c, chunk_len):
                raise ValueError(f"model returned shape {out.shape}, expected [K, S, {c}, {chunk_len}]")
            parts.append(out)
        y = overlap_add(np.concatenate(parts, axis=0), starts, padded.shape[1])[..., off : off + n]
        total = y.astype(np.float64) if total is None else total + y
    return (total / len(offsets)).astype(np.float32)
