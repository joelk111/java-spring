import numpy as np
import pytest

from kitsplit.errors import KitsplitInputError
from kitsplit.io import stems


@pytest.mark.parametrize("n", [7, 6, 5, 4])
def test_u_stem_1_grouping(n):
    rng = np.random.default_rng(n)
    s7 = rng.standard_normal((7, 2, 1000)).astype(np.float32)
    g = stems.group(s7, n)
    expected_names = {7: list(stems.STEMS7),
                      6: ["kick", "snare", "toms", "hh", "cymbals", "residual"],
                      5: ["kick", "snare", "toms", "cymbals", "residual"],
                      4: ["kick", "snare", "toms", "cymbals"]}[n]
    assert list(g) == expected_names
    idx = {name: i for i, name in enumerate(stems.STEMS7)}
    for name, members in stems.GROUPINGS[n].items():
        want = s7[[idx[m] for m in members]].astype(np.float64).sum(axis=0)
        np.testing.assert_allclose(g[name], want, atol=1e-6)
    # every grouping covers each of the 7 stems exactly once
    members = sorted(m for ms in stems.GROUPINGS[n].values() for m in ms)
    assert members == sorted(stems.STEMS7)
    assert stems.sum_error(g, s7.astype(np.float64).sum(axis=0)) < 1e-5


def test_invalid_stem_count_lists_allowed():
    with pytest.raises(KitsplitInputError, match="4\\|5\\|6\\|7"):
        stems.check_n_stems(9)
