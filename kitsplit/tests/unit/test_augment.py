from pathlib import Path

import numpy as np
import pytest

from kitsplit.data import augment as A
from kitsplit.data import synth
from kitsplit.io.stems import STEMS7

SR = 44100
CFG = Path(__file__).resolve().parents[2] / "configs" / "augment" / "drumbus_v1.yaml"


@pytest.fixture(scope="module")
def kit():
    return synth.render_fixture_kit(SR, 3.0, seed=5, quantize_bits=None).stems.astype(np.float64)


def _db(err, ref):
    return 10 * np.log10((np.sum(err**2) + 1e-300) / np.sum(ref**2))


def test_u_aug_1_identity_params_bypass(kit):
    x = kit.sum(0)
    rng = np.random.default_rng(0)
    one = np.ones(x.shape[1])
    cases = {
        "gain": A.gain(x, 0.0),
        "pan_narrow": A.pan_narrow(x, 1.0),
        "hpf": A.highpass(x, SR, None),
        "lpf": A.lowpass(x, SR, None),
        "room": A.room(x, SR, rng, -np.inf),
        "eq_tilt": A.eq_tilt(x, SR, 0.0),
        "stereo_width": A.stereo_width(x, 1.0),
        "compressor": x * A.compressor_gain(x, SR, threshold_db=0.0, ratio=1.0),
        "parallel": x * A.parallel_gain(A.compressor_gain(x, SR, -20, 4), mix=0.0),
        "transient": x * A.transient_gain(x, SR, 0.0, 0.0),
        "limiter": x * A.limiter_gain(x, SR, ceiling_dbfs=6.0),  # ceiling above the peak
        "saturation": A.saturate(x, 12.0, "tape", wet=0.0),
        "dither": A.dither(x, rng, None),
    }
    for name, y in cases.items():
        assert y.shape == x.shape, name
        assert np.array_equal(y, x) or _db(y - x, x) < -120, (name, _db(y - x, x))
    assert np.array_equal(A.compressor_gain(x, SR, 0.0, 1.0), one)


def _full_cfg():
    cfg = A.load_config(CFG)
    for section in ("per_stem", "bus"):
        for op in cfg[section].values():
            op["p"] = 1.0  # every op on
    return cfg


def test_u_aug_2_seeded_random_params(kit):
    cfg = _full_cfg()
    outs = [A.apply_chain(kit, SR, cfg, np.random.default_rng(123)) for _ in range(2)]
    assert np.array_equal(outs[0].stems, outs[1].stems) and np.array_equal(outs[0].mixture, outs[1].mixture)
    assert outs[0].applied == outs[1].applied
    other = A.apply_chain(kit, SR, cfg, np.random.default_rng(124))
    assert not np.array_equal(other.mixture, outs[0].mixture)
    for seed in range(10):
        ex = A.apply_chain(kit, SR, cfg, np.random.default_rng(seed))
        assert ex.stems.shape == kit.shape and ex.mixture.shape == kit.shape[1:]
        assert np.all(np.isfinite(ex.stems)) and np.all(np.isfinite(ex.mixture))
        ceiling = next(o for o in ex.applied if o["op"] == "limiter")["ceiling_dbfs"]
        # TPDF dither spans +-1 LSB and rounding adds up to 0.5 LSB
        dither_lsb = 1.5 * 2.0 ** -(int(next(o for o in ex.applied if o["op"] == "dither")["bits"]) - 1)
        assert np.max(np.abs(ex.mixture)) <= min(1.0, A.db2lin(ceiling) + dither_lsb)
        # distribute mode: processed stems still sum to the processed mixture
        assert np.max(np.abs(ex.stems.sum(0) - ex.mixture)) < 1e-12


def test_bus_chain_actually_changes_the_mix(kit):
    ex = A.apply_chain(kit, SR, _full_cfg(), np.random.default_rng(7))
    assert _db(ex.mixture - kit.sum(0), kit.sum(0)) > -20


def test_mixture_only_mode_keeps_clean_stems(kit):
    cfg = _full_cfg()
    cfg["target_mode"] = "mixture_only"
    cfg["per_stem"] = {}
    ex = A.apply_chain(kit, SR, cfg, np.random.default_rng(3))
    np.testing.assert_array_equal(ex.stems, kit)
    assert not np.allclose(ex.mixture, kit.sum(0))


def test_u_aug_3_contamination_goes_to_residual(kit):
    t = np.arange(kit.shape[-1]) / SR
    bass = np.stack([np.sin(2 * np.pi * 82.4 * t)] * 2)  # licensed-by-construction test tone
    cfg = {"contamination": {"p": 1.0, "level_db": -20.0}}
    ex = A.apply_chain(kit, SR, cfg, np.random.default_rng(0), contaminant=bass)
    np.testing.assert_array_equal(ex.stems[:6], kit[:6])  # named stems untouched
    added = ex.stems[STEMS7.index("residual")] - kit[STEMS7.index("residual")]
    corr = np.sum(added * bass) / np.sqrt(np.sum(added**2) * np.sum(bass**2))
    assert corr > 0.999
    # with gain-envelope bus ops the contaminant stays (scaled) in residual only
    cfg["bus"] = {"compressor": {"p": 1.0, "threshold_db": -25, "ratio": 4, "attack_ms": 5, "release_ms": 100},
                  "limiter": {"p": 1.0, "ceiling_dbfs": -1.0, "release_ms": 50}}
    ex2 = A.apply_chain(kit, SR, cfg, np.random.default_rng(0), contaminant=bass)
    for i in range(6):
        c = np.sum(ex2.stems[i] * bass) / np.sqrt(np.sum(ex2.stems[i] ** 2) * np.sum(bass**2) + 1e-30)
        k = np.sum(kit[i] * bass) / np.sqrt(np.sum(kit[i] ** 2) * np.sum(bass**2) + 1e-30)
        assert abs(c) <= abs(k) + 0.02, (STEMS7[i], c, k)


def test_distribute_keeps_sum(kit):
    diff = np.random.default_rng(1).standard_normal(kit.shape[1:]) * 0.01
    out = A.distribute(kit, diff)
    assert np.max(np.abs(out.sum(0) - (kit.sum(0) + diff))) < 1e-12


def test_augment_dataset_cli(fx, tmp_path):
    from typer.testing import CliRunner

    from kitsplit.cli import app

    r = CliRunner().invoke(app, ["augment", "--in", str(fx / "type1_mini"), "--out", str(tmp_path / "aug"),
                                 "--config", str(CFG), "--seed", "1", "--copies", "2"])
    assert r.exit_code == 0, r.output
    dirs = sorted(p for p in (tmp_path / "aug").iterdir())
    assert len(dirs) == 4 and all((d / "mixture.wav").is_file() and (d / "augment.yaml").is_file() for d in dirs)
