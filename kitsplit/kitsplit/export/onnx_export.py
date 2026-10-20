"""Export the mask network (no STFT in the graph) to ONNX, opset 17."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from kitsplit.dsp.stft import StftParams
from kitsplit.errors import KitsplitError

OPSET = 17


def export(model: nn.Module, stft: StftParams, out: str | Path, fp16: bool = False) -> Path:
    if fp16:
        # Not implemented yet (Phase 4). Refuse instead of silently exporting fp32.
        raise KitsplitError("--fp16 export is not implemented yet")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.zeros(1, 4, stft.n_bins, 16)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["spec"],
        output_names=["mask_logits"],
        dynamic_axes={"spec": {0: "batch", 3: "frames"}, "mask_logits": {0: "batch", 4: "frames"}},
        opset_version=OPSET,
        dynamo=False,
    )
    return out
