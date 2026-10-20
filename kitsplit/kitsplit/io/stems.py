"""Stem naming, 7->6->5->4 grouping, and sum checks."""

from __future__ import annotations

import numpy as np

from kitsplit.errors import KitsplitInputError

STEMS7: tuple[str, ...] = ("kick", "snare", "toms", "hh", "ride", "crash", "residual")
NAMED_STEMS: tuple[str, ...] = STEMS7[:-1]
RESIDUAL = "residual"

# Each grouping maps output name -> members of STEMS7. Every grouping covers all 7
# stems exactly once, so the sum of a grouping is always the sum of the 7 stems.
GROUPINGS: dict[int, dict[str, tuple[str, ...]]] = {
    7: {s: (s,) for s in STEMS7},
    6: {
        "kick": ("kick",),
        "snare": ("snare",),
        "toms": ("toms",),
        "hh": ("hh",),
        "cymbals": ("ride", "crash"),
        "residual": ("residual",),
    },
    5: {
        "kick": ("kick",),
        "snare": ("snare",),
        "toms": ("toms",),
        "cymbals": ("hh", "ride", "crash"),
        "residual": ("residual",),
    },
    # ASSUMPTION: the 4-stem layout (kick, snare, toms, cymbals) has no residual
    # slot, so residual is folded into cymbals to keep the sum exact.
    4: {
        "kick": ("kick",),
        "snare": ("snare",),
        "toms": ("toms",),
        "cymbals": ("hh", "ride", "crash", "residual"),
    },
}


def check_n_stems(n: int) -> None:
    if n not in GROUPINGS:
        allowed = "|".join(str(k) for k in sorted(GROUPINGS))
        raise KitsplitInputError(f"--stems must be one of {allowed} (got {n})")


def group(stems7: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """stems7: [7, C, N] in STEMS7 order -> {name: [C, N]} for the n-stem layout."""
    check_n_stems(n)
    if stems7.shape[0] != len(STEMS7):
        raise ValueError(f"expected {len(STEMS7)} stems, got {stems7.shape[0]}")
    index = {name: i for i, name in enumerate(STEMS7)}
    out: dict[str, np.ndarray] = {}
    for name, members in GROUPINGS[n].items():
        acc = np.zeros(stems7.shape[1:], dtype=np.float64)
        for m in members:
            acc += stems7[index[m]]
        out[name] = acc.astype(np.float32)
    return out


def sum_error(stems: np.ndarray | dict[str, np.ndarray], mixture: np.ndarray) -> float:
    """max |sum(stems) - mixture|, summed in float64."""
    arrays = list(stems.values()) if isinstance(stems, dict) else list(stems)
    total = np.zeros(mixture.shape, dtype=np.float64)
    for a in arrays:
        total += a
    return float(np.max(np.abs(total - mixture))) if mixture.size else 0.0
