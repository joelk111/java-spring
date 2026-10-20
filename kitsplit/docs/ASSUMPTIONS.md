# Assumptions, deviations and open questions

Rule (plan §0.4): every guessed constant carries `# ASSUMPTION:` in code and a line
here. `tests/unit/test_docs.py` fails if a file with an `ASSUMPTION` marker is not
listed below.

## Guessed constants (code → assumption → how to resolve)

| Where | Assumption | Resolve by |
|---|---|---|
| `kitsplit/io/audio.py` (MAX_MODEL_SR) | Models run at ≤ 48 kHz; higher rates are resampled for inference and back. | Check whether 96 kHz-trained models are worth it (Phase 3+). |
| `kitsplit/io/audio.py` (default_bit_depth) | 16-bit inputs get 24-bit stems: 16-bit output cannot keep Σ stems within 1e-6 of the input. | — (follows from quantisation maths: 7 × 2⁻¹⁶ ≫ 1e-6). |
| `kitsplit/io/stems.py` | 4-stem layout has no residual slot, so residual is folded into `cymbals` to keep the sum exact. | Product decision: confirm with users / UI. |
| `kitsplit/dsp/chunker.py` | TTA shift offsets are deterministic and span ≤ one hop. | Ablate shift range on eval_real (Phase 3). |
| `kitsplit/separate.py` (SILENCE_PEAK) | Peak < 1e-6 (−120 dBFS) = silent input → warning. | — (only affects a log message). |
| `kitsplit/models/oracle.py` | IRM oracle STFT 2048/512. | — (tests only). |
| `kitsplit/data/synth.py` | Fixture kit pan positions. | — (fixtures only). |
| `kitsplit/data/synth_dataset.py` | Electronic-kit renders use one random fixed pan per stem. | Compare with real electronic multitracks once acquired. |
| `kitsplit/data/augment.py` (module) | Which `target_mode` trains better ("distribute" vs "mixture_only") is **unknown** — the plan's least-sure design point. | Phase 2 A/B on eval_real; record the result here. |
| `kitsplit/data/augment.py` (eq_tilt) | Tilt EQ = opposite shelves with gain 2.5 × dB/oct. | Measure the realised slope; adjust if needed. |
| `kitsplit/data/augment.py` (transient_gain) | A 12 dB fast-vs-slow envelope rise = full attack. | Listen / compare with a reference transient designer. |
| `configs/augment/drumbus_v1.yaml` | Every probability and range. | Fit to statistics of real processed drum buses (Phase 2). |
| `kitsplit/train/losses.py` | MR-STFT sizes 256/1024/4096, ±10 ms transient window, weight 3. | Ablation in Phase 3 (plan §7.2). |
| `kitsplit/train/msst_adapter.py` | Loss mix `masked_loss + multistft_loss + l1_loss` with MSST default coefficients. | Phase 1 runs on a GPU box. |
| `configs/model/mdx23c_7stem.yaml` | Batch 12 fits a 48 GB GPU (copied from the B1 config, not measured). | First Phase 1 run; see `configs/README.md`. |
| `configs/inference/{quality,fast}.yaml` | Chunk/overlap/shift values from plan §5, untuned. | Tune on eval_real once a trained model exists. |
| `kitsplit/eval/onsets.py` | Energy-rise detector thresholds (20 dB rise, −50 dB floor, 50 ms refractory). Validated only on StemGMD kick/snare stems (see below). | Validate on real multitracks with annotated onsets. |
| `kitsplit/eval/metrics.py` | onset_snr falls back to detected onsets when a set has no onset annotations. | Prefer annotated onsets (GMD MIDI) wherever available. |
| `kitsplit/eval/drumness.py` | HPSS settings and the 0.5 warning threshold (`kitsplit/cli.py`) for "drum-only-ness" are uncalibrated. | Calibrate on drum-only vs full-mix files; it only warns. |
| `kitsplit/cli.py` | Same 0.5 drum-only threshold as above. | Same as above. |

## Deviations from the plan (deliberate)

1. **Fixtures are generated, not committed** (plan §9.1). The listed set is ~35 MB of
   WAV (> 5 MB budget). `scripts/make_fixtures.py` regenerates them; the committed
   `tests/fixtures/MANIFEST.sha256` pins every byte, so non-determinism fails
   `tests/unit/test_fixtures.py` and `kitsplit doctor`.
2. **`stereo_48k_float.flac` → `stereo_48k_float.wav` + `stereo_48k_24bit.flac`.** FLAC
   cannot store float (the plan's own C-15 says so).
3. **I-ORACLE-IRM threshold.** The plan asks for per-stem SDR ≥ 15 dB. On the fixture
   kit even a direct, unchunked IRM reaches only 5.4–32 dB (residual 5.4, hh 11.7,
   snare 13.5, crash 13.8): the synthetic hats/crash/snare share broadband noise and the
   55 Hz residual sits under the kick. Making the synth easier would make the test
   always pass without testing anything. The test now asserts the pipeline reproduces
   the direct IRM within 0.1 dB per stem (plus a check that a 10 ms misalignment is
   detected).
4. **Consistency projection runs last**, after resampling back to the input rate and
   restoring mono — otherwise stems would sum to the resampled input, not the file.
5. **StemGMD mapping** (plan §4.1): StemGMD has no pedal hi-hat class (pedal is folded
   into closed hi-hat by the GMD reduction). `hh = closed + open`. Consequence:
   **`residual` is always silent in StemGMD data** — its only training signal is the
   contamination augmentation (needs licensed material, currently disabled) or real
   multitracks.
6. **MSST via the pinned submodule, not `pip install msst`.** msst 0.1.0 pins
   `torch<2.12`. `msst check` (I-MSST-CFG) is replaced by loading the emitted YAML with
   MSST's `get_model_from_config` and loading `type1_mini` with MSST's `MSSDataset`.
7. **Exit code 6** added for license-gate halts (`fetch_stemgmd.py`); the plan's list
   had no code for it.
8. **`kitsplit train` cannot use kitsplit's bus augmentation in-loop** (MSST re-sums
   stems): it runs offline via `kitsplit augment`. The transient-weighted loss in
   `train/losses.py` is implemented and tested but **not wired into MSST training**.
9. **`export-onnx --fp16` is not implemented** (refuses with an error instead of
   silently exporting fp32).

## Verified facts that used to be assumptions

- StemGMD license: `cc-by-4.0` on both Zenodo records (API, 2026-10-19).
- GMD license: CC BY 4.0 (`groove/LICENSE` inside the MIDI zip); zip sha256 matches the
  digest published on the Magenta page.
- GMD MIDI → StemGMD class table: all 22 rows match the "Paper Mapping" column at
  magenta.tensorflow.org/datasets/groove.
- GMD MIDI is time-aligned with StemGMD audio: on the 34-track eval subset, snare MIDI
  onsets vs detected audio onsets: median −1.2 ms, p5..p95 −2.5..+3.7 ms (kick median
  −0.9 ms; kick p5 −21 ms is most likely the detector matching a neighbouring hit).
- libsndfile float↔24-bit round trip is exact; quantisation error ≤ 2⁻²³.
- libsndfile writes a time-stamped PEAK chunk into float WAVs → disabled, otherwise
  identical audio gives different bytes (found by the fixture-determinism check).

## Open questions (not resolved here)

- Weights licenses of B1 (aufr33/jarredou) and B2 (inagoy) — both eval-only.
- B4 (MVSEP DrumSep MelBand RoFormer): weights' availability not checked; not used.
- Whether B1 was trained on StemGMD (would make StemGMD numbers for B1 optimistic).
