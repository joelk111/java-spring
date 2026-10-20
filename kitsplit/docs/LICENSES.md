# Licenses of third-party code, weights and data

Rule (plan §0.5): anything without a stated license is **eval-only** — never
shipped, never fine-tuned from, never redistributed.

All entries below were checked by reading the files at the pinned commit
(or the Zenodo API record for datasets) on 2026-10-19. "SPDX" is what the
LICENSE file actually says; it is not inferred from a README badge.

## Code (git submodules under `third_party/`)

| Submodule | Pinned commit | LICENSE file | SPDX | Notes |
|---|---|---|---|---|
| ZFTurbo/Music-Source-Separation-Training | `6fdad13686bb7e84d029ec66f9dbe76e65c93ebf` | `LICENSE` | `MIT` | `pyproject.toml` also declares `license = "MIT"`. |
| inagoy/drumsep | `c1cea3f47bacd410412c7f563109f0a227b0e784` | `LICENSE` | `MIT` | Covers the repo code. The **weights are not in the repo** (downloaded from Google Drive id `1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC` by `drumsepInstall`) and no file states that the MIT license covers them → weights treated as **eval-only** until the author confirms. |
| openmirlab/mdxnet-infer | `cb4779eedeca4f55ba498df8e234e2b5f4b90aaf` | `LICENSE` | `MIT` | Code only. Its `checkpoints.toml` states for the DrumSep checkpoint: *"Unknown; treat as non-commercial-safe only pending author confirmation"* → B1 weights are **eval-only**. |
| bernardo-torres/inverse-drum-machine | `456656868538205ef756912c7cf5b0fd936de8af` | `LICENSE` | `Apache-2.0` | Repo contains `pretrained/idm-44-train-kits`; the README makes no separate statement about the weights. Assumption: covered by the repo's Apache-2.0. Re-check before shipping anything derived from it. |
| polimi-ispl/larsnet | `17d631fd18e77ee2f1d23ee7b3fc0fb46ae629e2` | **NONE FOUND** | **NONE FOUND** | No LICENSE file in the repo → code is **eval-only / reference only**. README states LarsNet checkpoints are **CC BY-NC 4.0** (non-commercial). |
| betweentwomidnights/stems.cpp | `a9d2b6c06d8e7ad9121c0696ae5795415febadac` | `LICENSE` | `MIT` | Not used in v1. Several GGUF models listed in its README are marked "no upstream license" there. `ggml/` and `vendor/` carry their own licenses (not reviewed — not used). |

## Model weights

| Registry name | Source | License | `shippable` |
|---|---|---|---|
| `b1-drumsep-mdx23c-6stem` | openmirlab/mdxnet-infer release `weights-drumsep-v1` (mirror of aufr33/jarredou) | Unknown (none published) | `false` — eval-only |
| `b2-drumsep-hdemucs-4stem` | inagoy/drumsep Google Drive file | Not stated for weights | `false` — eval-only |
| `kitsplit-v1` | (not trained yet) | — | — |

## Datasets

| Dataset | Source | License | Use |
|---|---|---|---|
| StemGMD | Zenodo 7860223 (part 1) + 7882857 (part 2) | `cc-by-4.0` (Zenodo API `metadata.license.id`, both records) | Training allowed incl. commercial, **attribution required** (cite Mezza et al. 2024, Pattern Recognition Letters 183). |
| ENST-Drums | — | research-use (plan A3) — not obtained | eval-only |
| Groove MIDI Dataset v1.0.0 (MIDI only: onsets + train/validation/test folds) | `storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip` (sha256 `651cbc52…dcc1e`) | `CC-BY-4.0` (quoted from `groove/LICENSE` in the zip) | Fold assignment + onset annotations; attribution required. |
