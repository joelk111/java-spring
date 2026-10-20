"""configs/model/*.yaml (+ configs/augment/*.yaml) -> MSST YAML -> MSST train.py.

What the adapter does NOT do (stated, not hidden):
  * kitsplit's drum-bus augmentation is not an MSST in-loop augmentation. MSST's
    Type 1 loader re-sums the stems, so the bus chain runs OFFLINE (`kitsplit augment`
    in "distribute" mode, whose stems still sum to the processed mix) and the cache is
    passed as an extra --data path. MSST's own augmentations are disabled.
  * kitsplit's transient-weighted loss (train/losses.py) is not wired into MSST's
    training loop yet: MSST's --loss choices are fixed. The adapter uses MSST's
    `masked_loss` + `multistft_loss` + `l1_loss` (closest available).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from kitsplit._msst import msst_root
from kitsplit.errors import KitsplitError, KitsplitInputError
from kitsplit.io.stems import STEMS7

KITSPLIT_ONLY_AUDIO_KEYS = ("window",)
# ASSUMPTION: loss mix; MSST's default multistft coef (0.001) is kept.
MSST_LOSSES = ["masked_loss", "multistft_loss", "l1_loss"]
MSST_METRICS = ["sdr", "bleedless", "fullness", "l1_freq"]


def build_msst_config(model_config: Path, augment_config: Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(model_config).read_text())
    for key in ("arch", "audio", "model", "training"):
        if key not in cfg:
            raise KitsplitInputError(f"{model_config}: missing section {key!r}")
    if "instruments" in cfg["training"]:
        raise KitsplitInputError(f"{model_config}: do not list instruments; they come from STEMS7")
    audio = {k: v for k, v in cfg["audio"].items() if k not in KITSPLIT_ONLY_AUDIO_KEYS}
    training = dict(cfg["training"], instruments=list(STEMS7))
    out = {
        "audio": audio,
        "model": cfg["model"],
        "training": training,
        "augmentations": {"enable": False},
        "inference": cfg.get("inference", {"batch_size": 1, "num_overlap": 4}),
    }
    if augment_config is not None:
        # provenance only: the offline cache must have been built with this file
        out["kitsplit_augment"] = yaml.safe_load(Path(augment_config).read_text())
    return out


def write_msst_config(model_config: Path, augment_config: Path | None, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(build_msst_config(model_config, augment_config), sort_keys=False))
    return dest


def train_command(model_config: Path, msst_yaml: Path, data: list[Path], valid: list[Path], results: Path,
                  gpus: str = "", resume: Path | None = None, extra: list[str] | None = None) -> list[str]:
    arch = yaml.safe_load(Path(model_config).read_text())["arch"]
    ids = [g for g in gpus.split(",") if g.strip()] if gpus else []
    script = "train_ddp.py" if len(ids) >= 2 else "train.py"
    cmd = [sys.executable, str(msst_root() / script), "--model_type", arch, "--config_path", str(msst_yaml),
           "--results_path", str(results), "--data_path", *map(str, data), "--valid_path", *map(str, valid),
           "--dataset_type", "1", "--seed", "0", "--loss", *MSST_LOSSES, "--metrics", *MSST_METRICS,
           "--metric_for_scheduler", "sdr"]
    if ids:
        cmd += ["--device_ids", *ids]
    if resume is not None:
        cmd += ["--start_check_point", str(resume)]
    return cmd + list(extra or [])


def train(model_config: Path, augment_config: Path | None, data: list[Path], valid: list[Path], results: Path,
          gpus: str = "", resume: Path | None = None, dry_run: bool = False, extra: list[str] | None = None) -> list[str]:
    # MSST runs with cwd = its own root, so every path must be absolute
    results = Path(results).resolve()
    data = [Path(d).resolve() for d in data]
    valid = [Path(v).resolve() for v in valid]
    resume = Path(resume).resolve() if resume is not None else None
    msst_yaml = write_msst_config(model_config, augment_config, results / "msst_config.yaml")
    cmd = train_command(model_config, msst_yaml, data, valid, results, gpus, resume, extra)
    (results / "command.txt").write_text(" ".join(cmd) + "\n")
    if not dry_run:
        log_path = results / "train.log"
        with open(log_path, "w") as log:
            code = subprocess.run(cmd, cwd=msst_root(), stdout=log, stderr=subprocess.STDOUT).returncode
        if code != 0:
            tail = "".join(log_path.read_text().splitlines(keepends=True)[-5:])
            raise KitsplitError(f"MSST training exited with {code}; see {log_path}\n{tail}")
    return cmd
