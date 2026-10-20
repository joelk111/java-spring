"""Registry entry -> runnable backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from kitsplit.errors import KitsplitInputError, KitsplitModelError
from kitsplit.models import registry
from kitsplit.models.oracle import IdealRatioMaskOracle, IdentityOracle


def _load_state(path: Path) -> dict:
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise KitsplitModelError(f"cannot load weights {path}: {exc}") from exc
    for key in ("state_dict", "model_state_dict", "state"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    return state


def load_backend(
    entry: registry.ModelEntry,
    device: str = "cpu",
    backend: str = "torch",
    references: np.ndarray | None = None,
    references_sr: int | None = None,
    cache_dir: Path | None = None,
):
    if backend not in ("torch", "onnx"):
        raise KitsplitInputError(f"--backend must be torch|onnx (got {backend!r})")

    if entry.kind == "oracle":
        if entry.arch == "identity":
            return IdentityOracle()
        if references is None or references_sr is None:
            raise KitsplitInputError(f"{entry.name} needs the true stems; it only runs inside eval/benchmark")
        return IdealRatioMaskOracle(references, references_sr)

    if backend == "onnx" and entry.kind != "kitsplit":
        raise KitsplitInputError(f"--backend onnx is not available for {entry.name} (kind {entry.kind})")

    from kitsplit.models.backend_torch import resolve_device

    torch_device = resolve_device(device) if backend == "torch" else "cpu"
    paths = registry.verify(entry, cache_dir)

    if entry.kind == "kitsplit":
        from kitsplit.dsp.stft import StftParams
        from kitsplit.models.backend_torch import ARCHS, MaskBackend, torch_logits_fn

        cfg_file = registry.config_path(entry, cache_dir)
        if cfg_file is None:
            raise KitsplitModelError(f"{entry.name}: kitsplit models need a config")
        cfg = yaml.safe_load(cfg_file.read_text())
        params = StftParams.from_config(cfg)
        sr = int(cfg["audio"]["sample_rate"])
        if backend == "onnx":
            from kitsplit.models.backend_onnx import onnx_logits_fn

            onnx_spec = entry.raw.get("onnx")
            if not onnx_spec:
                raise KitsplitModelError(f"{entry.name} has no ONNX export (`onnx:` entry)")
            onnx_file = Path(onnx_spec["path"])
            if not onnx_file.is_absolute() and entry.base_dir is not None:
                onnx_file = entry.base_dir / onnx_file
            if not onnx_file.is_file():
                raise KitsplitModelError(f"ONNX file missing: {onnx_file}")
            if onnx_spec.get("sha256") and registry.sha256_file(onnx_file) != onnx_spec["sha256"]:
                raise KitsplitModelError(f"ONNX file sha256 mismatch: {onnx_file}")
            return MaskBackend(onnx_logits_fn(onnx_file, device), params, sr)
        if entry.arch not in ARCHS:
            raise KitsplitModelError(f"unknown kitsplit arch {entry.arch!r}")
        model = ARCHS[entry.arch](**cfg.get("model", {}))
        model.load_state_dict(_load_state(paths[0]))
        return MaskBackend(torch_logits_fn(model, torch_device), params, sr)

    if entry.kind == "msst":
        from kitsplit._msst import ensure_msst_on_path
        from kitsplit.models.backend_torch import MsstBackend

        ensure_msst_on_path()
        from utils.settings import get_model_from_config  # type: ignore

        model, cfg = get_model_from_config(entry.arch, str(paths[1]))
        model.load_state_dict(_load_state(paths[0]))
        names = list(cfg.training.instruments)
        # kitsplit's own MSST-trained models output STEMS7 names: identity map by default
        stem_map = entry.stem_map or {n: n for n in names}
        return MsstBackend(model, names, stem_map, int(cfg.audio.get("sample_rate", entry.sample_rate)),
                           int(cfg.audio.chunk_size), torch_device)

    if entry.kind == "demucs":
        from kitsplit.models.backend_torch import DemucsBackend

        try:
            from demucs.states import load_model
        except ImportError as exc:
            raise KitsplitModelError("B2 needs the `demucs` package (pip install demucs)") from exc
        # A demucs package is a pickle (weights_only=False runs code from it). This only
        # happens after registry.verify() matched its pinned sha256.
        package = torch.load(str(paths[0]), map_location="cpu", weights_only=False)
        return DemucsBackend(load_model(package), entry.stem_map or {}, torch_device)

    raise KitsplitModelError(f"unsupported model kind {entry.kind!r}")
