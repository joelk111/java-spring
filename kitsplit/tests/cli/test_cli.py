"""CLI tests C-1..C-18, C-21, C-22 (plan §9.4). Oracle/random models = plumbing only."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
from typer.testing import CliRunner

from kitsplit.cli import app
from kitsplit.io import audio

pytestmark = pytest.mark.plumbing
runner = CliRunner()
SEP_ID = ["--model", "oracle-identity", "--preset", "fast"]


def run(*args):
    return runner.invoke(app, [str(a) for a in args])


def test_c1_help():
    r = run("--help")
    assert r.exit_code == 0
    for cmd in ("separate", "info", "models", "dataset", "augment", "train", "eval", "benchmark", "export-onnx", "doctor"):
        assert cmd in r.output


def test_c1_subprocess_smoke():
    r = subprocess.run([sys.executable, "-m", "kitsplit.cli", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "separate" in r.stdout


def test_c2_identity_seven_files(fx, tmp_path):
    r = run("separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, *SEP_ID)
    assert r.exit_code == 0, r.output
    assert sorted(p.stem for p in tmp_path.iterdir()) == sorted(["kick", "snare", "toms", "hh", "ride", "crash", "residual"])
    res, _ = sf.read(tmp_path / "residual.wav", dtype="int32")
    inp, _ = sf.read(fx / "synth_kit_8s_48k.wav", dtype="int32")
    assert np.array_equal(res, inp)


def test_c3_four_stems(fx, tmp_path):
    r = run("separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, "--stems", 4, *SEP_ID)
    assert r.exit_code == 0, r.output
    assert sorted(p.stem for p in tmp_path.iterdir()) == ["cymbals", "kick", "snare", "toms"]


def test_c4_bad_stem_count(fx, tmp_path):
    r = run("separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, "--stems", 9, *SEP_ID)
    assert r.exit_code == 2
    assert "4|5|6|7" in r.output


def test_c5_missing_file(tmp_path):
    assert run("separate", tmp_path / "missing.wav", "--out", tmp_path, *SEP_ID).exit_code == 2


def test_c6_not_audio(fx, tmp_path):
    assert run("separate", fx / "not_audio.txt", "--out", tmp_path, *SEP_ID).exit_code == 2


def test_c7_c8_overwrite(fx, tmp_path):
    args = ("separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, *SEP_ID)
    assert run(*args).exit_code == 0
    assert run(*args).exit_code == 5
    assert run(*args, "--overwrite").exit_code == 0


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a CPU-only box")
def test_c9_cuda_on_cpu_box(fx, tmp_path):
    tiny = fx / "models" / "tiny.yaml"
    r = run("--device", "cuda", "separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, "--model", tiny)
    assert r.exit_code == 4


def test_c10_unknown_model(fx, tmp_path):
    r = run("separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, "--model", "nonexistent")
    assert r.exit_code == 3


def test_c10b_default_model_not_trained_yet(fx, tmp_path):
    r = run("separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path)
    assert r.exit_code == 3
    assert "no published weights" in r.output


def test_c11_tampered(fx, tmp_path):
    r = run("separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, "--model", fx / "models" / "tampered.yaml")
    assert r.exit_code == 3
    assert "sha256 mismatch" in r.output


def test_c12_json(fx, tmp_path):
    r = run("--json", "separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, *SEP_ID)
    assert r.exit_code == 0
    d = json.loads(r.stdout)
    assert d["schema"] == "kitsplit.separate.v1"
    res = d["results"][0]
    assert res["sr"] == 48000 and abs(res["duration"] - 8.0) < 1e-9
    assert len(res["outputs"]) == 7 and all(p.endswith(".wav") for p in res["outputs"].values())


def test_c12b_json_error_schema(tmp_path):
    r = run("--json", "separate", tmp_path / "nope.wav", "--out", tmp_path, *SEP_ID)
    assert r.exit_code == 2
    d = json.loads(r.stdout)
    assert d["schema"] == "kitsplit.error.v1" and d["exit_code"] == 2


def test_c13_one_subdir_per_input(fx, tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    shutil.copy(fx / "mono_44k.wav", a)
    shutil.copy(fx / "stereo_48k_float.wav", b)
    out = tmp_path / "out"
    r = run("separate", a, b, "--out", out, *SEP_ID)
    assert r.exit_code == 0, r.output
    assert sorted(p.name for p in out.iterdir()) == ["a", "b"]
    assert len(list((out / "a").iterdir())) == 7


def test_c14_keeps_96k_24bit(fx, tmp_path):
    r = run("separate", fx / "stereo_96k_24bit.wav", "--out", tmp_path, *SEP_ID)
    assert r.exit_code == 0
    info = sf.info(str(tmp_path / "kick.wav"))
    assert info.samplerate == 96000 and info.subtype == "PCM_24"


def test_c15_flac_32f(fx, tmp_path):
    r = run("separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, "--format", "flac", "--bit-depth", "32f", *SEP_ID)
    assert r.exit_code == 2
    assert "FLAC cannot store 32-bit float" in r.output


def test_c16_silence(fx, tmp_path, caplog):
    r = run("separate", fx / "silence_2s.wav", "--out", tmp_path, *SEP_ID)
    assert r.exit_code == 0
    assert "silent" in r.output or "silent" in caplog.text
    for p in tmp_path.iterdir():
        x, _ = sf.read(p)
        assert np.all(np.isfinite(x)) and not np.any(x)


def test_c17_info(fx):
    r = run("info", fx / "synth_kit_8s_48k.wav")
    assert r.exit_code == 0 and "sr: 48000" in r.output and "channels: 2" in r.output
    d = json.loads(run("--json", "info", fx / "synth_kit_8s_48k.wav").stdout)
    assert d["schema"] == "kitsplit.info.v1" and d["sr"] == 48000 and abs(d["duration"] - 8) < 1e-9


def test_c18_models_list():
    r = run("models", "list")
    assert r.exit_code == 0
    assert "oracle-identity" in r.output and "LICENSE" in r.output


def test_c21_doctor():
    r = run("--json", "doctor")
    d = json.loads(r.stdout)
    assert r.exit_code == 0, [c for c in d["checks"] if c["status"] == "FAIL"]
    assert any(c["name"].startswith("submodule") for c in d["checks"])


def test_c22_export_onnx_then_separate(fx, tmp_path):
    models = tmp_path / "models"
    shutil.copytree(fx / "models", models)
    r = run("export-onnx", "--ckpt", models / "tiny_mask.pt", "--config", models / "tiny_mask_config.yaml",
            "--out", models / "tiny_mask.onnx")
    assert r.exit_code == 0, r.output
    o, t = tmp_path / "onnx", tmp_path / "torch"
    for backend, out in (("onnx", o), ("torch", t)):
        r = run("--device", "cpu", "separate", fx / "synth_kit_8s_48k.wav", "--out", out, "--model", models / "tiny.yaml",
                "--backend", backend, "--preset", "fast", "--bit-depth", "32f")
        assert r.exit_code == 0, r.output
    for p in t.iterdir():
        a, _ = sf.read(p)
        b, _ = sf.read(o / p.name)
        assert 20 * np.log10(np.sqrt(np.mean((a - b) ** 2)) + 1e-30) < -60


def test_dc_offset_and_clipped_inputs_keep_sum(fx, tmp_path):
    for name in ("dc_offset.wav", "clipped.wav"):
        r = run("--json", "--device", "cpu", "separate", fx / name, "--out", tmp_path / name,
                "--model", fx / "models" / "tiny.yaml", "--preset", "fast")
        assert r.exit_code == 0, r.output
        assert json.loads(r.stdout)["results"][0]["sum_error"] < 1e-6


def test_c19_eval_after_c2(fx, tmp_path):
    assert run("separate", fx / "synth_kit_8s_48k.wav", "--out", tmp_path, *SEP_ID).exit_code == 0
    r = run("--json", "eval", "--pred", tmp_path, "--ref", fx / "synth_kit_8s_48k", "--out", tmp_path / "e.json")
    assert r.exit_code == 0, r.output
    d = json.loads(r.stdout)
    assert {row["stem"] for row in d["rows"]} == {"kick", "snare", "toms", "hh", "ride", "crash", "residual"}


def test_c20_benchmark_oracles(fx, tmp_path):
    r = run("benchmark", "--data", fx / "type1_mini", "--models", "oracle-identity,oracle-irm", "--out", tmp_path,
            "--preset", "fast", "--metrics", "sdr")
    assert r.exit_code == 0, r.output
    md, js = tmp_path / "benchmark_type1_mini.md", tmp_path / "benchmark_type1_mini.json"
    assert md.is_file() and js.is_file()
    d = json.loads(js.read_text())
    irm = d["models"]["oracle-irm"]["groupings"]["7"]["summary"]
    ident = d["models"]["oracle-identity"]["groupings"]["7"]["summary"]
    for stem in ("kick", "snare", "toms", "hh", "ride", "crash", "residual"):
        assert irm[stem]["sdr"]["median"] > ident[stem]["sdr"]["median"], stem
    assert "plumbing" in md.read_text()
