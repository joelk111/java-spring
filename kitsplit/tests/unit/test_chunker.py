import numpy as np
import pytest

from kitsplit.dsp import chunker
from kitsplit.models.oracle import IdentityOracle


def _identity_fn(chunks, starts):
    return chunks[:, None]


@pytest.mark.parametrize("overlap", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("n", [100, 1000, 1024, 4097, 10000])  # < 1 chunk, exact multiple, non-multiple
def test_u_chunk_1_identity_bit_exact(overlap, n):
    x = np.random.default_rng(n).standard_normal((2, n)).astype(np.float32)
    y = chunker.run_chunked(x, _identity_fn, chunk_len=1024, overlap=overlap, shifts=1)
    assert y.shape == (1, 2, n)
    assert np.array_equal(y[0], x)


@pytest.mark.parametrize("shifts", [1, 2, 4])
def test_u_chunk_2_shift_tta_identity(shifts):
    x = np.random.default_rng(shifts).standard_normal((2, 5000)).astype(np.float32)
    y = chunker.run_chunked(x, IdentityOracle().run, chunk_len=1024, overlap=0.75, shifts=shifts)
    assert np.array_equal(y[-1], x)  # residual == input
    assert not np.any(y[:-1])


def test_starts_passed_are_absolute():
    seen = []

    def fn(chunks, starts):
        seen.extend(starts)
        return chunks[:, None]

    chunker.run_chunked(np.zeros((2, 3000), np.float32), fn, 1000, 0.5, shifts=2)
    assert min(seen) < 0  # shifted pass starts before the signal
    assert 0 in seen
