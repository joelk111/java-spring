"""Simple energy-rise onset detector.

Used (a) by onset_snr when a reference set ships no onset annotations, and (b) by the
synth tests. ASSUMPTION: thresholds below were chosen for isolated drum stems; they
are not validated on dense real recordings.
"""

from __future__ import annotations

import numpy as np

FRAME_SEC = 0.001  # 1 ms hop
WIN_SEC = 0.002  # 2 ms RMS window
LOOKBACK_SEC = 0.010
RISE_DB = 20.0
FLOOR_DB = -50.0  # relative to the stem's peak frame
REFRACTORY_SEC = 0.050


def detect_onsets(x: np.ndarray, sr: int) -> list[float]:
    """x: [C, N] or [N]. Returns onset times in seconds."""
    mono = np.abs(np.asarray(x, dtype=np.float64))
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    hop = max(1, int(round(FRAME_SEC * sr)))
    win = max(hop, int(round(WIN_SEC * sr)))
    n_frames = max(0, (len(mono) - win) // hop + 1)
    if n_frames == 0:
        return []
    sq = np.concatenate([[0.0], np.cumsum(mono**2)])
    starts = np.arange(n_frames) * hop
    rms = np.sqrt((sq[starts + win] - sq[starts]) / win)
    if rms.max() <= 0:
        return []
    db = 20 * np.log10(rms + 1e-12)
    floor = db.max() + FLOOR_DB
    look = max(1, int(round(LOOKBACK_SEC / FRAME_SEC)))
    refractory = int(round(REFRACTORY_SEC / FRAME_SEC))
    onsets: list[float] = []
    last = -refractory
    for t in range(n_frames):
        if db[t] < floor or t - last < refractory:
            continue
        prev = db[max(0, t - look) : t]
        base = prev.min() if prev.size else -240.0
        if db[t] - base >= RISE_DB:
            onsets.append(t * hop / sr)
            last = t
    return onsets
