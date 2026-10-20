import math

import numpy as np
import torch

from kitsplit.eval import metrics as M
from kitsplit.train.losses import mrstft_l1_loss, onset_weights

SR = 44100


def test_u_loss_1_zero_iff_equal_and_transient_weighting():
    g = torch.Generator().manual_seed(0)
    target = torch.randn(2, 2, SR, generator=g)
    assert float(mrstft_l1_loss(target, target, SR)) == 0.0
    pred = target + 0.01 * torch.randn(2, 2, SR, generator=g)
    assert float(mrstft_l1_loss(pred, target, SR)) > 0.0

    # an error far from any onset: weighting must not change the loss much...
    onsets = [0.25]
    w = onset_weights(SR, SR, onsets)
    far = target.clone()
    far[..., 30000:30100] += 0.5
    near = target.clone()
    near[..., 11025:11125] += 0.5  # 0.25 s = onset
    ratio_far = float(mrstft_l1_loss(far, target, SR, w)) / float(mrstft_l1_loss(far, target, SR))
    ratio_near = float(mrstft_l1_loss(near, target, SR, w)) / float(mrstft_l1_loss(near, target, SR))
    # ...while the same error at the onset is weighted up relative to the far one
    assert ratio_near > 1.5 * ratio_far
    assert torch.all(w[: int(0.2 * SR)] == 1.0) and w[int(0.25 * SR)] == 3.0


def test_u_met_1_pred_equals_ref():
    ref = np.random.default_rng(0).standard_normal((2, SR)).astype(np.float32)
    s = M.score(ref, ref.copy(), ref, SR, ["sdr", "bleedless", "fullness", "onset_snr"], onsets=[0.1, 0.5])
    assert s["sdr"] == M.SDR_CAP
    assert s["bleedless"] == 100.0 and s["fullness"] == 100.0
    assert s["onset_snr"] == M.SDR_CAP


def test_u_met_2_pred_zero():
    ref = np.random.default_rng(1).standard_normal((2, SR)).astype(np.float32)
    s = M.score(ref, np.zeros_like(ref), ref, SR, ["sdr", "onset_snr"], onsets=[0.1])
    assert s["sdr"] <= 0.0 and s["onset_snr"] <= 0.0


def test_silent_reference_not_scored():
    s = M.score(np.zeros((2, 100)), np.ones((2, 100)), None, SR, ["sdr"])
    assert math.isnan(s["sdr"])


def test_onset_snr_only_looks_at_transients():
    ref = np.random.default_rng(2).standard_normal((2, SR))
    est = ref.copy()
    est[:, 30000:] = 0  # damage only the region away from the onset
    assert M.onset_snr(ref, est, SR, [0.1]) == M.SDR_CAP
    assert M.score(ref, est, ref, SR, ["sdr"])["sdr"] < 10
