"""ONNX Runtime backend. The graph contains only the network: the host computes the
STFT (same params as training) and feeds [B, 4, F, T] (2ch x re/im), see MaskBackend."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from kitsplit.errors import KitsplitDeviceError

_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
}


def providers_for(device: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == "auto":
        for d in ("cuda", "dml"):
            if _PROVIDERS[d] in available:
                return [_PROVIDERS[d], "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if device not in _PROVIDERS:
        raise KitsplitDeviceError(f"unknown device {device!r}; allowed: auto|cpu|cuda|dml")
    if _PROVIDERS[device] not in available:
        raise KitsplitDeviceError(f"--device {device}: onnxruntime provider {_PROVIDERS[device]} not available ({available})")
    return [_PROVIDERS[device]]


def onnx_logits_fn(path: str | Path, device: str) -> Callable[[np.ndarray], np.ndarray]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # deterministic reductions
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers_for(device))
    name = sess.get_inputs()[0].name

    def fn(x: np.ndarray) -> np.ndarray:
        return sess.run(None, {name: x.astype(np.float32)})[0]

    return fn
