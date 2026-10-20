# Architecture (inference path)

```
file ─▶ io.audio.read ─▶ channel / sr policy ─▶ dsp.chunker.run_chunked ─▶ backend.run
        (float32 [2,N],   (mono → dup L/R;        (Hann w/o zero ends,       (batch of chunks →
         sr kept)          > 48 kHz → resample)    float64 overlap-add,       7 stems)
                                                   deterministic shift-TTA)
                    ─▶ resample back to input sr ─▶ restore channels ─▶ dsp.consistency.project
                    ─▶ io.stems.group(n) ─▶ io.audio.write (input sr, input depth / --bit-depth)
```

## Key decisions

- **Mixture consistency, two layers.** Mask models output 7 logits per TF bin;
  softmax across stems makes the masks sum to 1 (`MaskBackend`). Then
  `consistency.project` puts whatever is left (iSTFT edge error, resampling, models
  that are not mask-based such as MDX23C/HDemucs baselines) into `residual`:
  `residual := input − Σ(6 named)`, computed in float64 (bit-for-bit idempotent).
  Tested: `max|Σ stems − input| < 1e-6` (U-CONS-1, Q-4 on real files).
- **Projection is the last step**, after resampling back and restoring mono (see
  ASSUMPTIONS deviation 4).
- **STFT lives in the host** for kitsplit mask models; the ONNX graph takes
  `[B, 2ch×2(re,im), F, T]` (opset 17). STFT parameters are read from the model
  config YAML (U-STFT-2). MSST models used as baselines (MDX23C) keep their own STFT
  in the graph and run only on the torch backend.
- **Backends** share `run(chunks, starts) -> [K, 7, 2, L]`; `starts` are absolute
  sample positions so reference-based oracles can line up with the chunk.
  Torch vs ONNX on the same random-weights model: diff RMS < −60 dB (I-BACKEND-EQ).
- **Deterministic.** Fixed seeds, `torch.use_deterministic_algorithms(True)`,
  deterministic TTA offsets, single-threaded ONNX Runtime reductions, and no
  timestamped PEAK chunk in float WAVs → `kitsplit separate` twice gives
  byte-identical files (I-DET).
- **Registry.** `kitsplit/models/registry.yaml` (+ `KITSPLIT_REGISTRY`, or a one-entry
  YAML via `--model PATH`). Every entry has `sha256` and `license`; weights are
  downloaded to `~/.cache/kitsplit` and re-verified before every load. The demucs
  (B2) package is a pickle and is only unpickled after its sha256 matched.

## Training path

`configs/model/*.yaml` (kitsplit format, no `instruments`) → `train/msst_adapter.py`
→ MSST YAML (instruments = STEMS7, MSST augmentations off) → pinned MSST
`train.py` / `train_ddp.py`. Drum-bus augmentation runs offline (`kitsplit augment`,
"distribute" mode keeps Σ stems = processed mix, because MSST re-sums stems).
