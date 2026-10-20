import numpy as np

from kitsplit.data import synth
from kitsplit.eval.onsets import detect_onsets
from kitsplit.io.stems import STEMS7


def test_u_synth_1_sum_exact_and_onsets():
    r = synth.render_fixture_kit(48000, 8.0, seed=0)
    assert np.array_equal(r.stems.astype(np.float64).sum(axis=0), r.mixture.astype(np.float64))
    # every rendered hit is in the pattern, and the kick stem shows exactly that many onsets
    pattern = synth.fixed_pattern(8.0)
    assert {k: len(v) for k, v in r.onsets.items()} == {k: len(v) for k, v in pattern.items()}
    kick = r.stems[STEMS7.index("kick")]
    detected = detect_onsets(kick, 48000)
    assert len(detected) == len(pattern["kick"])
    want = np.array([t for t, _ in pattern["kick"]])
    assert np.max(np.abs(np.array(detected) - want)) < 0.010


def test_render_from_oneshots_velocity_layers():
    soft, hard = np.full(100, 0.1), np.full(100, 1.0)
    r = synth.render_from_oneshots({"kick": [soft, hard]}, {"kick": [(0.0, 10), (0.5, 127)]}, 1000, 1.0)
    k = r.stems[STEMS7.index("kick")]
    assert np.isclose(k[0, 0], 0.1 * 10 / 127 * np.sqrt(0.5))
    assert np.isclose(k[0, 500], 1.0 * np.sqrt(0.5))
    assert np.array_equal(r.stems.astype(np.float64).sum(0), r.mixture.astype(np.float64))
