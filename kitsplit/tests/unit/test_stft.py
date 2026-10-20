from pathlib import Path

import numpy as np
import torch
import yaml

from kitsplit.dsp.stft import Stft, StftParams

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIGS = sorted((ROOT / "configs" / "model").glob("*.yaml"))


def test_u_stft_1_identity():
    params = StftParams(n_fft=2048, hop_length=512)
    stft = Stft(params)
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((2, 2, 44100)).astype(np.float32))
    y = stft.inverse(stft.forward(x), x.shape[-1])
    err_db = 10 * np.log10(float(((y - x) ** 2).sum() / (x**2).sum()))
    assert err_db < -100, err_db


def test_u_stft_2_params_come_from_model_config():
    assert MODEL_CONFIGS, "no model configs found"
    for path in MODEL_CONFIGS:
        raw = yaml.safe_load(path.read_text())["audio"]
        p = StftParams.from_config(path)
        assert (p.n_fft, p.hop_length) == (raw["n_fft"], raw["hop_length"]), path


def test_u_stft_2_no_duplicated_constants():
    """dsp/stft.py must not hard-code an n_fft / hop value."""
    src = (ROOT / "kitsplit" / "dsp" / "stft.py").read_text()
    for token in ("2048", "4096", "1024", "512", "8192"):
        assert token not in src
