"""kitsplit command line. Thin: all logic lives in modules.

Exit codes: 0 ok, 2 bad input/args, 3 model missing/corrupt, 4 device unavailable,
5 output exists without --overwrite, 6 license gate, 10 internal error.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional

import typer

from kitsplit import __version__
from kitsplit.errors import KitsplitError, KitsplitInputError

SEPARATE_SCHEMA = "kitsplit.separate.v1"
INFO_SCHEMA = "kitsplit.info.v1"
ERROR_SCHEMA = "kitsplit.error.v1"

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Stereo drum bus -> per-drum stems.")
models_app = typer.Typer(no_args_is_help=True, help="Model registry: list / fetch / verify.")
dataset_app = typer.Typer(no_args_is_help=True, help="Dataset builders.")
app.add_typer(models_app, name="models")
app.add_typer(dataset_app, name="dataset")


class State:
    json: bool = False
    device: str = "auto"
    cache_dir: Optional[Path] = None


state = State()
log = logging.getLogger("kitsplit")


def _emit(obj: dict) -> None:
    typer.echo(json.dumps(obj, indent=2, sort_keys=True))


def handle_errors(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KitsplitError as exc:
            code = exc.exit_code
            msg = str(exc)
        except typer.Exit:
            raise
        except Exception as exc:  # noqa: BLE001 - map anything unexpected to exit 10
            log.exception("internal error")
            code, msg = 10, f"internal error: {exc}"
        if state.json:
            _emit({"schema": ERROR_SCHEMA, "error": msg, "exit_code": code})
        typer.echo(f"error: {msg}", err=True)
        raise typer.Exit(code)

    return wrapper


@app.callback()
def main(
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON on stdout."),
    log_level: str = typer.Option("WARNING", "--log-level"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda|dml"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Model cache (default ~/.cache/kitsplit)."),
    version: bool = typer.Option(False, "--version", is_eager=True, help="Print version and exit."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit(0)
    logging.basicConfig(level=log_level.upper(), stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    state.json, state.device, state.cache_dir = json_out, device, cache_dir


# ------------------------------------------------------------------ separate


@app.command()
@handle_errors
def separate(
    inputs: List[Path] = typer.Argument(..., help="Drum-only input file(s)."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    model: str = typer.Option("kitsplit-v1", "--model", help="Registry name or path to a model YAML."),
    backend: str = typer.Option("torch", "--backend", help="torch|onnx"),
    stems: int = typer.Option(7, "--stems", help="4|5|6|7"),
    preset: str = typer.Option("quality", "--preset", help="quality|fast"),
    chunk_sec: Optional[float] = typer.Option(None, "--chunk-sec"),
    overlap: Optional[float] = typer.Option(None, "--overlap"),
    shifts: Optional[int] = typer.Option(None, "--shifts"),
    fmt: str = typer.Option("wav", "--format", help="wav|flac"),
    bit_depth: Optional[str] = typer.Option(None, "--bit-depth", help="24|32f (default: from input)"),
    no_consistency: bool = typer.Option(False, "--no-consistency", help="DEBUG: skip the sum projection."),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Separate drum bus file(s) into stems."""
    from kitsplit.models import registry
    from kitsplit.models.backend_torch import set_deterministic
    from kitsplit.models.loader import load_backend
    from kitsplit.separate import Settings, output_dir_for, separate_file

    settings = Settings.from_preset(
        preset, chunk_sec=chunk_sec, overlap=overlap, shifts=shifts, n_stems=stems, fmt=fmt,
        bit_depth=bit_depth, consistency=not no_consistency, overwrite=overwrite,
    )
    settings.validate()
    for p in inputs:  # fail fast on bad inputs before loading a model
        if not p.is_file():
            raise KitsplitInputError(f"input file not found: {p}")
    set_deterministic(0)
    entry = registry.resolve(model)
    be = load_backend(entry, device=state.device, backend=backend, cache_dir=state.cache_dir)
    results = []
    for p in inputs:
        r = separate_file(p, output_dir_for(out, p, len(inputs) > 1), be, settings)
        results.append(r.__dict__)
        if not state.json:
            typer.echo(f"{p} -> {len(r.outputs)} stems in {Path(next(iter(r.outputs.values()))).parent}"
                       f" (max |sum - input| = {r.sum_error:.2e})")
    if state.json:
        _emit({"schema": SEPARATE_SCHEMA, "model": entry.name, "results": results})


# ------------------------------------------------------------------ info


@app.command()
@handle_errors
def info(inp: Path = typer.Argument(...)) -> None:
    """Print sample rate, channels, duration, peak and a drum-only-ness estimate."""
    from kitsplit.eval.drumness import drum_only_score
    from kitsplit.io import audio

    x, meta = audio.read(inp)
    peak = float(abs(x).max())
    score = drum_only_score(x, meta.sr)
    warn = score < 0.5  # ASSUMPTION: uncalibrated threshold, warning only
    d = {
        "schema": INFO_SCHEMA, "path": str(inp), "sr": meta.sr, "channels": meta.channels,
        "duration": meta.duration, "frames": meta.frames, "bit_depth": meta.bit_depth,
        "peak": peak, "drum_only_score": score, "drum_only_warning": warn,
    }
    if state.json:
        _emit(d)
    else:
        for k in ("sr", "channels", "duration", "frames", "bit_depth", "peak", "drum_only_score"):
            typer.echo(f"{k}: {d[k]}")
    if warn:
        log.warning("input may not be drum-only (uncalibrated heuristic score %.2f)", score)


# ------------------------------------------------------------------ models


@models_app.command("list")
@handle_errors
def models_list() -> None:
    from kitsplit.models import registry

    rows = [{"name": e.name, "kind": e.kind, "arch": e.arch, "license": e.license,
             "shippable": e.shippable, "description": e.description} for e in registry.load_registry()]
    if state.json:
        _emit({"schema": "kitsplit.models.v1", "models": rows})
        return
    typer.echo(f"{'NAME':28} {'KIND':9} {'SHIPPABLE':9} LICENSE")
    for r in rows:
        typer.echo(f"{r['name']:28} {r['kind']:9} {str(r['shippable']):9} {r['license']}")


@models_app.command("fetch")
@handle_errors
def models_fetch(name: str = typer.Argument(...)) -> None:
    from kitsplit.models import registry

    entry = registry.resolve(name)
    typer.echo(f"license: {entry.license} (shippable: {entry.shippable})", err=True)
    for p in registry.fetch(entry, state.cache_dir):
        typer.echo(f"ok {p}")


@models_app.command("verify")
@handle_errors
def models_verify(name: str = typer.Argument(...)) -> None:
    from kitsplit.models import registry

    entry = registry.resolve(name)
    for p in registry.verify(entry, state.cache_dir):
        typer.echo(f"sha256 ok {p}")


# ------------------------------------------------------------------ dataset / augment / train


@dataset_app.command("build-stemgmd")
@handle_errors
def dataset_build_stemgmd(
    src: Path = typer.Option(..., "--src", help="Unzipped StemGMD root (contains audio/)."),
    out: Path = typer.Option(..., "--out"),
    map_file: Optional[Path] = typer.Option(None, "--map", help="stem_map.yaml"),
    gmd: Optional[Path] = typer.Option(None, "--gmd", help="groove-v1.0.0-midionly.zip: folds + onsets.json"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only the first N clips (smoke runs)."),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Remap StemGMD (9 classes) to MSST Type 1 with the 7 kitsplit stems + mixture.wav."""
    from kitsplit.data import stemgmd

    n = stemgmd.build(src, out, map_file, gmd=gmd, limit=limit, overwrite=overwrite)
    typer.echo(f"wrote {n} tracks to {out}")


@dataset_app.command("synth")
@handle_errors
def dataset_synth(
    onsets: Path = typer.Option(..., "--onsets", help="Directory of GMD MIDI files."),
    oneshots: Path = typer.Option(..., "--oneshots", help="Directory: <stem>/<layer>.wav"),
    out: Path = typer.Option(..., "--out"),
    seed: int = typer.Option(0, "--seed"),
    sr: int = typer.Option(44100, "--sr"),
) -> None:
    """Electronic-kit data: one-shots x GMD onsets -> MSST Type 1."""
    from kitsplit.data import synth_dataset

    n = synth_dataset.build(onsets, oneshots, out, seed=seed, sr=sr)
    typer.echo(f"wrote {n} tracks to {out}")


@app.command()
@handle_errors
def augment(
    inp: Path = typer.Option(..., "--in", help="MSST Type 1 directory."),
    out: Path = typer.Option(..., "--out"),
    config: Path = typer.Option(..., "--config", help="configs/augment/*.yaml"),
    seed: int = typer.Option(0, "--seed"),
    copies: int = typer.Option(1, "--copies"),
) -> None:
    """Offline drum-bus augmentation (Type 1 -> Type 1 cache)."""
    from kitsplit.data import augment as aug

    n = aug.augment_dataset(inp, out, config, seed=seed, copies=copies)
    typer.echo(f"wrote {n} augmented tracks to {out}")


@app.command()
@handle_errors
def train(
    model_config: Path = typer.Option(..., "--model-config"),
    augment_config: Optional[Path] = typer.Option(None, "--augment-config"),
    data: List[Path] = typer.Option(..., "--data"),
    valid: List[Path] = typer.Option(..., "--valid"),
    results: Path = typer.Option(..., "--results"),
    gpus: str = typer.Option("", "--gpus", help="e.g. 0,1 (>=2 uses train_ddp.py)"),
    resume: Optional[Path] = typer.Option(None, "--resume"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write the MSST YAML and print the command only."),
) -> None:
    """Write the MSST YAML, then call MSST train.py / train_ddp.py."""
    from kitsplit.train import msst_adapter

    cmd = msst_adapter.train(model_config, augment_config, data, valid, results, gpus, resume, dry_run)
    typer.echo(" ".join(cmd))


# ------------------------------------------------------------------ eval / benchmark


@app.command("eval")
@handle_errors
def eval_cmd(
    pred: Path = typer.Option(..., "--pred", help="Directory of predicted stems (one subdir per track, or flat)."),
    ref: Path = typer.Option(..., "--ref", help="Type 1 reference (dir of tracks, or a single track dir)."),
    metrics: str = typer.Option("sdr,bleedless,fullness,onset_snr", "--metrics"),
    per_track: bool = typer.Option(False, "--per-track"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write JSON here (default <pred>/eval.json)."),
) -> None:
    """Score one output directory against references."""
    from kitsplit.eval import metrics as m

    report = m.evaluate_dir(pred, ref, metrics.split(","))
    dest = out or pred / "eval.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True))
    if state.json:
        _emit(report)
    else:
        typer.echo(m.summary_table(report, per_track=per_track))
        typer.echo(f"wrote {dest}")


@app.command()
@handle_errors
def benchmark(
    data: Path = typer.Option(..., "--data", help="Type 1 set with mixture.wav per track."),
    models: str = typer.Option(..., "--models", help="Comma-separated registry names."),
    out: Path = typer.Option(..., "--out"),
    metrics: str = typer.Option("sdr,bleedless,fullness,onset_snr", "--metrics"),
    preset: str = typer.Option("quality", "--preset"),
    include_baseline_dirs: List[Path] = typer.Option([], "--include-baseline-dirs"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only the first N tracks (state it in reports!)."),
) -> None:
    """Run N models on a Type 1 set, score them, write .md and .json."""
    from kitsplit.eval import benchmark as b

    md, js = b.run(data, models.split(","), out, metrics.split(","), preset, include_baseline_dirs,
                   device=state.device, cache_dir=state.cache_dir, limit=limit)
    typer.echo(f"wrote {md} and {js}")


# ------------------------------------------------------------------ export / doctor


@app.command("export-onnx")
@handle_errors
def export_onnx(
    ckpt: Path = typer.Option(..., "--ckpt"),
    config: Path = typer.Option(..., "--config", help="kitsplit model config (audio + model sections)."),
    out: Path = typer.Option(..., "--out"),
    fp16: bool = typer.Option(False, "--fp16"),
) -> None:
    """Export a kitsplit mask model to ONNX (STFT stays in the host)."""
    import torch
    import yaml

    from kitsplit.dsp.stft import StftParams
    from kitsplit.export.onnx_export import export
    from kitsplit.models.backend_torch import ARCHS

    cfg = yaml.safe_load(config.read_text())
    arch = cfg.get("arch", "tiny-mask")
    if arch not in ARCHS:
        raise KitsplitInputError(f"export-onnx supports archs {list(ARCHS)}; got {arch!r}")
    model = ARCHS[arch](**cfg.get("model", {}))
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu", weights_only=True))
    typer.echo(f"wrote {export(model, StftParams.from_config(cfg), out, fp16)}")


@app.command()
@handle_errors
def doctor() -> None:
    """Check deps, devices, cache, submodule pins, fixture hashes."""
    from kitsplit import doctor as d

    report = d.run(state.cache_dir)
    if state.json:
        _emit(report)
    else:
        for check in report["checks"]:
            typer.echo(f"[{check['status']:4}] {check['name']}: {check['detail']}")
    if not report["ok"]:
        raise typer.Exit(10)


if __name__ == "__main__":
    app()
