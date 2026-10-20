"""Plumbing tests. They use oracle / random-weights models: they prove the pipeline
and metrics path, NOT separation quality. Never cite them as quality evidence."""

import numpy as np
import pytest
import soundfile as sf

from kitsplit.io import audio
from kitsplit.io.stems import STEMS7
from kitsplit.models import registry
from kitsplit.models.loader import load_backend
from kitsplit.models.oracle import IdealRatioMaskOracle, IdentityOracle
from kitsplit.separate import Settings, separate_array, separate_file

pytestmark = pytest.mark.plumbing


def _load_refs(d):
    return np.stack([audio.read(d / f"{s}.wav")[0] for s in STEMS7])


def _sdr(ref, est):
    return 10 * np.log10((np.sum(ref.astype(np.float64) ** 2) + 1e-8) / (np.sum((ref - est).astype(np.float64) ** 2) + 1e-8))


def _direct_irm(x, refs):
    """IRM on the whole signal in one STFT: no chunker, no TTA, no projection."""
    import torch

    stft = IdealRatioMaskOracle(refs, 48000).stft
    mix = stft.forward(torch.from_numpy(x).double())
    pw = stft.forward(torch.from_numpy(refs).double()).abs() ** 2
    return stft.inverse(pw / pw.sum(0, keepdim=True).clamp_min(1e-20) * mix, x.shape[1]).numpy()


def test_i_oracle_irm(fx):
    """Deviation from plan §9.3 (documented in docs/ASSUMPTIONS.md): the plan asks for
    per-stem SDR >= 15 dB. On this fixture even a direct, unchunked IRM only reaches
    5-32 dB (hats/crash/snare share broadband noise; the 55 Hz residual sits under the
    kick), so 15 dB tests the fixture, not the pipeline. The pipeline check is that
    chunking + TTA + projection reproduce the direct IRM."""
    x, info = audio.read(fx / "synth_kit_8s_48k.wav")
    refs = _load_refs(fx / "synth_kit_8s_48k")
    y = separate_array(x, info.sr, 2, IdealRatioMaskOracle(refs, info.sr), Settings(chunk_sec=2.0, overlap=0.5, shifts=2))
    direct = _direct_irm(x, refs)
    sdrs = {s: _sdr(refs[i], y[i]) for i, s in enumerate(STEMS7)}
    ref_sdrs = {s: _sdr(refs[i], direct[i]) for i, s in enumerate(STEMS7)}
    print("IRM oracle SDR via pipeline (plumbing only):", {k: round(v, 2) for k, v in sdrs.items()})
    print("IRM oracle SDR direct:", {k: round(v, 2) for k, v in ref_sdrs.items()})
    for s in STEMS7:
        assert abs(sdrs[s] - ref_sdrs[s]) < 0.1, (s, sdrs[s], ref_sdrs[s])
        assert sdrs[s] > 0.0, s
    assert np.max(np.abs(y.astype(np.float64).sum(0) - x)) < 1e-6


def test_i_oracle_irm_detects_misalignment(fx):
    """The check above must fail if the chunker hands the model wrong positions."""
    x, info = audio.read(fx / "synth_kit_8s_48k.wav")
    refs = _load_refs(fx / "synth_kit_8s_48k")
    oracle = IdealRatioMaskOracle(refs, info.sr)
    shifted = lambda chunks, starts: oracle.run(chunks, [s + 480 for s in starts])  # 10 ms off
    from kitsplit.dsp import chunker

    y = chunker.run_chunked(x, shifted, 96000, 0.5, 1)
    direct = _direct_irm(x, refs)
    worst = max(abs(_sdr(refs[i], y[i]) - _sdr(refs[i], direct[i])) for i in range(7))
    assert worst > 0.1


def test_i_oracle_id(fx):
    x, info = audio.read(fx / "synth_kit_8s_48k.wav")
    y = separate_array(x, info.sr, 2, IdentityOracle(), Settings(chunk_sec=1.0, overlap=0.75, shifts=4))
    assert np.array_equal(y[STEMS7.index("residual")], x)
    assert not np.any(y[:-1])


def test_i_det_run_twice_byte_identical(fx, tmp_path):
    be = load_backend(registry.resolve(str(fx / "models" / "tiny.yaml")), device="cpu")
    for depth in ("24", "32f"):
        outs = []
        for run in range(2):
            d = tmp_path / f"{depth}_{run}"
            separate_file(fx / "synth_kit_8s_48k.wav", d, be, Settings(chunk_sec=2.0, overlap=0.5, shifts=2, bit_depth=depth))
            outs.append({p.name: p.read_bytes() for p in sorted(d.iterdir())})
        assert outs[0] == outs[1]


@pytest.mark.parametrize("name", ["mono_44k.wav", "synth_kit_8s_48k.wav", "stereo_96k_24bit.wav"])
def test_i_sr_outputs_at_original_rate(fx, tmp_path, name):
    be = load_backend(registry.resolve(str(fx / "models" / "tiny.yaml")), device="cpu")
    r = separate_file(fx / name, tmp_path, be, Settings(chunk_sec=1.0, overlap=0.5, shifts=1))
    src = sf.info(str(fx / name))
    for p in r.outputs.values():
        out = sf.info(p)
        assert out.samplerate == src.samplerate
        assert out.channels == src.channels
        assert abs(out.frames - src.frames) <= 1
    assert r.sum_error < 1e-6


def test_i_backend_eq_torch_vs_onnx(fx, tmp_path):
    import shutil

    import torch
    import yaml

    from kitsplit.dsp.stft import StftParams
    from kitsplit.export.onnx_export import export
    from kitsplit.models.backend_torch import TinyMaskNet

    models = tmp_path / "models"
    shutil.copytree(fx / "models", models)
    cfg = yaml.safe_load((models / "tiny_mask_config.yaml").read_text())
    net = TinyMaskNet(**cfg["model"])
    net.load_state_dict(torch.load(models / "tiny_mask.pt", weights_only=True))
    export(net, StftParams.from_config(cfg), models / "tiny_mask.onnx")
    entry = registry.resolve(str(models / "tiny.yaml"))
    x, info = audio.read(fx / "synth_kit_8s_48k.wav")
    s = Settings(chunk_sec=2.0, overlap=0.5, shifts=1)
    y_t = separate_array(x, info.sr, 2, load_backend(entry, "cpu", "torch"), s)
    y_o = separate_array(x, info.sr, 2, load_backend(entry, "cpu", "onnx"), s)
    diff_rms = np.sqrt(np.mean((y_t.astype(np.float64) - y_o) ** 2))
    assert 20 * np.log10(diff_rms + 1e-30) < -60
    assert np.any(y_t[:-1])  # random-weights model actually routes energy to named stems
