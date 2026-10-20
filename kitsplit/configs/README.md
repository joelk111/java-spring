# configs

- `model/` — kitsplit model configs (MSST format minus `instruments`, plus `arch`).
  Only `mdx23c_7stem.yaml` exists so far; `scnet_7stem.yaml` and
  `melband_roformer_7stem.yaml` (plan §5) are not written yet.
- `augment/drumbus_v1.yaml` — offline drum-bus augmentation (`kitsplit augment`).
- `inference/{quality,fast}.yaml` — chunk / overlap / shift presets.

## GPU memory (plan A4)

The reference box is a 48 GB GPU; `batch_size: 12` is copied from the B1 config and
was **not measured here** (no GPU in the development container). With less VRAM,
lower `training.batch_size` and raise `training.gradient_accumulation_steps` so that
their product stays the same, e.g. 24 GB → `batch_size: 6`,
`gradient_accumulation_steps: 2`. Disable `use_amp` only if you see NaNs.
