"""Real baseline weights (downloads ~600 MB). Run with KITSPLIT_NETWORK_TESTS=1.

These check that B1/B2 download with the pinned sha256, load, run, sum back to the
input, and that their stem mapping is not scrambled. On synthetic audio they say
nothing about separation quality."""

import numpy as np
import pytest

from kitsplit.io import audio
from kitsplit.io.stems import STEMS7
from kitsplit.models import registry
from kitsplit.models.loader import load_backend
from kitsplit.separate import Settings, separate_array

pytestmark = [pytest.mark.network, pytest.mark.plumbing]


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory):
    import os

    return tmp_path_factory.mktemp("cache") if not os.environ.get("KITSPLIT_TEST_CACHE") else os.environ["KITSPLIT_TEST_CACHE"]


@pytest.mark.parametrize("name,checked", [("B1", ["kick", "snare", "toms"]), ("B2", ["kick", "snare", "toms"])])
def test_baseline_fetch_run_and_mapping(fx, shared_cache, name, checked):
    from pathlib import Path

    entry = registry.resolve(name)
    assert entry.shippable is False
    registry.fetch(entry, Path(shared_cache))
    be = load_backend(entry, device="cpu", cache_dir=Path(shared_cache))
    x, info = audio.read(fx / "synth_kit_8s_48k.wav")
    y = separate_array(x, info.sr, 2, be, Settings(chunk_sec=6.0, overlap=0.5, shifts=1))
    assert np.max(np.abs(y.astype(np.float64).sum(0) - x)) < 1e-6
    refs = np.stack([audio.read(fx / "synth_kit_8s_48k" / f"{s}.wav")[0] for s in STEMS7]).astype(np.float64)
    for s in checked:
        i = STEMS7.index(s)
        corr = [np.sum(y[i] * r) / np.sqrt(np.sum(y[i] ** 2) * np.sum(r**2) + 1e-30) for r in refs]
        assert int(np.argmax(corr)) == i, (name, s, np.round(corr, 2))
