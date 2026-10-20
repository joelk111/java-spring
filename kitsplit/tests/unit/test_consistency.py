import numpy as np

from kitsplit.dsp.consistency import project


def _random(seed):
    rng = np.random.default_rng(seed)
    mix = rng.uniform(-1, 1, (2, 48000)).astype(np.float32)
    stems = (rng.standard_normal((7, 2, 48000)) * 0.3).astype(np.float32)
    return stems, mix


def test_u_cons_1_sum_matches_input():
    for seed in range(5):
        stems, mix = _random(seed)
        out = project(stems, mix)
        err = np.max(np.abs(out.astype(np.float64).sum(axis=0) - mix))
        assert err < 1e-6, err
        # float32 summation (what a naive user does) is also within budget
        assert np.max(np.abs(out.sum(axis=0) - mix)) < 1e-6
        np.testing.assert_array_equal(out[:6], stems[:6])  # named stems untouched


def test_u_cons_2_idempotent():
    stems, mix = _random(42)
    once = project(stems, mix)
    assert np.array_equal(project(once, mix), once)
