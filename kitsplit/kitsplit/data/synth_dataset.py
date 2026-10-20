"""Electronic-kit training data: user one-shots x Groove MIDI onsets -> MSST Type 1.

    oneshots/<stem>/<layer>.wav   one or more velocity layers per stem, sorted by name
                                  (soft -> hard); only rights-cleared samples!
    onsets/**/*.mid               GMD MIDI files (CC BY 4.0)

Each MIDI file becomes one track dir with the 7 stems, mixture.wav and onsets.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kitsplit.data import synth
from kitsplit.data.stemgmd import load_map
from kitsplit.errors import KitsplitInputError
from kitsplit.io import audio
from kitsplit.io.stems import STEMS7

TAIL_SEC = 2.0  # room for the last one-shot to ring out


def load_oneshots(root: Path, sr: int) -> dict[str, list[np.ndarray]]:
    shots: dict[str, list[np.ndarray]] = {}
    for stem in STEMS7:
        files = sorted((root / stem).glob("*.wav"))
        layers = []
        for f in files:
            x, info = audio.read(f)
            layers.append(audio.resample(x, info.sr, sr).mean(axis=0).astype(np.float64))
        if layers:
            shots[stem] = layers
    if not shots:
        raise KitsplitInputError(f"no one-shots found under {root}/<stem>/*.wav")
    return shots


def midi_onsets(path: Path, midi_to_stem: dict[int, str]) -> dict[str, list[tuple[float, int]]]:
    import mido

    out: dict[str, list[tuple[float, int]]] = {}
    t = 0.0
    for msg in mido.MidiFile(str(path)):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0 and msg.note in midi_to_stem:
            out.setdefault(midi_to_stem[msg.note], []).append((t, msg.velocity))
    return out


def build(onsets_dir: Path, oneshots_dir: Path, out: Path, seed: int = 0, sr: int = 44100) -> int:
    m = load_map()
    class_to_stem = {c: s for s, cs in m["map"].items() for c in cs}
    midi_to_stem = {int(k): class_to_stem[v] for k, v in m["midi_to_class"].items()}
    shots = load_oneshots(Path(oneshots_dir), sr)
    rng = np.random.default_rng(seed)
    pan = {s: float(rng.uniform(-0.5, 0.5)) for s in STEMS7}  # ASSUMPTION: random fixed pan per stem
    midis = sorted(Path(onsets_dir).rglob("*.mid"))
    if not midis:
        raise KitsplitInputError(f"no .mid files under {onsets_dir}")
    n = 0
    for mid in midis:
        on = {s: v for s, v in midi_onsets(mid, midi_to_stem).items() if s in shots}
        if not on:
            continue
        dur = max(t for v in on.values() for t, _ in v) + TAIL_SEC
        r = synth.render_from_oneshots(shots, on, sr, dur, pan)
        dest = Path(out) / mid.relative_to(onsets_dir).with_suffix("").as_posix().replace("/", "__")
        synth.write_render(r, dest / "mixture.wav", dest, "32f")
        (dest / "onsets.json").write_text(json.dumps({"source": f"MIDI {mid.name}", "onsets": r.onsets}))
        n += 1
    return n
