"""I-MSST-CFG, I-EVAL-LAYOUT (plumbing)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from kitsplit._msst import ensure_msst_on_path
from kitsplit.io.stems import STEMS7
from kitsplit.train import msst_adapter

pytestmark = pytest.mark.plumbing
ROOT = Path(__file__).resolve().parents[2]
MODEL_CFG = ROOT / "configs" / "model" / "mdx23c_7stem.yaml"
AUG_CFG = ROOT / "configs" / "augment" / "drumbus_v1.yaml"


def test_i_msst_cfg_loads_in_msst_and_runs(tmp_path, fx):
    msst_yaml = msst_adapter.write_msst_config(MODEL_CFG, AUG_CFG, tmp_path / "msst.yaml")
    ensure_msst_on_path()
    from utils.settings import get_model_from_config

    model, cfg = get_model_from_config("mdx23c", str(msst_yaml))
    assert list(cfg.training.instruments) == list(STEMS7)
    with torch.no_grad():
        y = model(torch.zeros(1, 2, cfg.audio.chunk_size))
    assert tuple(y.shape) == (1, 7, 2, cfg.audio.chunk_size)

    # MSST's own Type 1 dataset loader accepts our fixture layout and yields 7-stem batches
    from utils.dataset import MSSDataset

    cfg.training.batch_size = 2
    cfg.training.num_steps = 2
    ds = MSSDataset(cfg, [str(fx / "type1_mini")], metadata_path=str(tmp_path / "meta.pkl"), dataset_type=1,
                    verbose=False)
    stems, mix = ds[0]
    assert tuple(stems.shape) == (7, 2, cfg.audio.chunk_size) and tuple(mix.shape) == (2, cfg.audio.chunk_size)


def test_train_dry_run_writes_command(tmp_path, fx):
    cmd = msst_adapter.train(MODEL_CFG, AUG_CFG, [fx / "type1_mini"], [fx / "type1_mini"], tmp_path, dry_run=True)
    assert cmd[1].endswith("train.py") and "--dataset_type" in cmd
    assert (tmp_path / "msst_config.yaml").is_file() and (tmp_path / "command.txt").is_file()
    two = msst_adapter.train_command(MODEL_CFG, tmp_path / "msst_config.yaml", [fx], [fx], tmp_path, gpus="0,1")
    assert two[1].endswith("train_ddp.py")


def test_instruments_in_model_config_rejected(tmp_path):
    from kitsplit.errors import KitsplitInputError

    cfg = yaml.safe_load(MODEL_CFG.read_text())
    cfg["training"]["instruments"] = ["kick"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(KitsplitInputError):
        msst_adapter.build_msst_config(bad)


def test_i_eval_layout(tmp_path, fx):
    from typer.testing import CliRunner

    from kitsplit.cli import app

    r = CliRunner().invoke(app, ["benchmark", "--data", str(fx / "type1_mini"), "--models", "oracle-identity",
                                 "--out", str(tmp_path), "--preset", "fast", "--metrics", "sdr"])
    assert r.exit_code == 0, r.output
    pred = tmp_path / "predictions" / "oracle-identity" / "7stem"
    r = CliRunner().invoke(app, ["eval", "--pred", str(pred), "--ref", str(fx / "type1_mini"),
                                 "--metrics", "sdr,l1_freq,bleedless,fullness,log_wmse,onset_snr"])
    assert r.exit_code == 0, r.output
    report = json.loads((pred / "eval.json").read_text())
    assert len(report["rows"]) == 2 * 7  # one row per track per stem
    assert {(row["track"], row["stem"]) for row in report["rows"]} == {(t, s) for t in ("track_a", "track_b") for s in STEMS7}
