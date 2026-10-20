"""Uncalibrated "drum-only-ness" heuristic for `kitsplit info` (warning only).

Median-filter harmonic/percussive split (Fitzgerald 2010) on the magnitude STFT:
score = percussive energy / total energy. Sustained tonal material (bass, pads,
vocals) lowers it. Never used to gate anything: it has not been calibrated on data.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import stft

# ASSUMPTION: analysis settings; ~40 ms window at 48 kHz, ~0.25 s / ~500 Hz filters.
N_FFT = 2048
HOP = 512


def drum_only_score(x: np.ndarray, sr: int, max_seconds: float = 30.0) -> float:
    mono = x.mean(axis=0)[: int(max_seconds * sr)].astype(np.float64)
    if not np.any(mono):
        return 0.0
    _, _, z = stft(mono, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
    mag = np.abs(z)
    frames = max(3, int(round(0.25 * sr / HOP)) | 1)
    bins = max(3, int(round(500 / (sr / N_FFT))) | 1)
    harm = median_filter(mag, size=(1, frames))
    perc = median_filter(mag, size=(bins, 1))
    p_mask = perc**2 / (perc**2 + harm**2 + 1e-20)
    return float(np.sum(p_mask * mag**2) / np.sum(mag**2))
