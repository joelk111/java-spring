"""StemGMD -> MSST Type 1 with the 7 kitsplit stems + mixture.wav.

Source layout (verified on the Zenodo archive listing):
    <src>/audio/<drummer>/<session>/<kit>/<class>/<clip>.wav
Output layout:
    <out>/<split>/<drummer>__<session>__<kit>__<clip>/{kick,...,residual,mixture}.wav
    (+ onsets.json when GMD MIDI is supplied)

`split` is the Groove MIDI fold (train / validation / test) looked up in GMD's
info.csv; without it every track goes to `<out>/all/`. StemGMD mixtures are the plain
sum of the stems, so mixture.wav is written as that sum.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from kitsplit.errors import KitsplitInputError, KitsplitOutputExistsError
from kitsplit.io import audio
from kitsplit.io.stems import STEMS7

DEFAULT_MAP = Path(__file__).with_name("stem_map.yaml")


def load_map(path: Path | None = None) -> dict:
    m = yaml.safe_load((path or DEFAULT_MAP).read_text())
    classes = list(m["stemgmd_classes"])
    members = [c for stem in STEMS7 for c in m["map"].get(stem, [])]
    if sorted(members) != sorted(classes) or len(members) != len(set(members)):
        raise KitsplitInputError(f"stem map must cover every StemGMD class exactly once: {members} vs {classes}")
    if set(m["map"]) != set(STEMS7):
        raise KitsplitInputError(f"stem map keys must be exactly {STEMS7}")
    return m


def scan(src: Path, classes: list[str]) -> dict[tuple[str, str, str, str], dict[str, Path]]:
    """-> {(drummer, session, kit, clip): {class: path}}"""
    root = src / "audio" if (src / "audio").is_dir() else src
    tracks: dict[tuple[str, str, str, str], dict[str, Path]] = defaultdict(dict)
    for wav in sorted(root.glob("*/*/*/*/*.wav")):
        drummer, session, kit, cls = wav.parts[-5:-1]
        if cls not in classes:
            raise KitsplitInputError(f"unexpected StemGMD class folder {cls!r} in {wav}")
        tracks[(drummer, session, kit, wav.stem)][cls] = wav
    if not tracks:
        raise KitsplitInputError(f"no StemGMD clips found under {root}")
    return dict(tracks)


class Gmd:
    """Groove MIDI metadata (folds) and onsets, read from groove-v1.0.0-midionly.zip
    or its extracted `groove/` directory."""

    def __init__(self, path: Path):
        self.path = path
        self._zip = zipfile.ZipFile(path) if path.suffix == ".zip" else None
        rows = csv.DictReader(io.StringIO(self._read("info.csv").decode()))
        self.rows = {r["midi_filename"][: -len(".mid")]: r for r in rows}

    def _read(self, rel: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(f"groove/{rel}")
        return (self.path / rel).read_bytes()

    def key(self, drummer: str, session: str, clip: str) -> str:
        return f"{drummer}/{session}/{clip}"

    def split(self, drummer: str, session: str, clip: str) -> str:
        row = self.rows.get(self.key(drummer, session, clip))
        if row is None:
            raise KitsplitInputError(f"clip {drummer}/{session}/{clip} not in GMD info.csv")
        return row["split"]

    def onsets(self, drummer: str, session: str, clip: str, midi_to_class: dict[int, str]) -> dict[str, list[list[float]]]:
        """-> {StemGMD class: [[time_s, velocity], ...]} from the clip's MIDI."""
        import mido

        midi = mido.MidiFile(file=io.BytesIO(self._read(self.key(drummer, session, clip) + ".mid")))
        out: dict[str, list[list[float]]] = defaultdict(list)
        t = 0.0
        for msg in midi:  # iterating a MidiFile yields messages with .time in seconds (tempo-aware)
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                cls = midi_to_class.get(msg.note)
                if cls is None:
                    raise KitsplitInputError(f"MIDI pitch {msg.note} has no StemGMD class mapping")
                out[cls].append([round(t, 6), msg.velocity])
        return dict(out)


def _write_exact(path: Path, x: np.ndarray, sr: int) -> None:
    """Smallest lossless format: 16-bit when every sample is on the 16-bit grid (StemGMD
    sources and sums of them), else 24-bit, else float when the sum clips."""
    peak = float(np.max(np.abs(x), initial=0.0))
    if peak >= 1.0:
        depth = "32f"
    elif np.array_equal(np.round(x * 32768.0), x * 32768.0):
        depth = "16"
    else:
        depth = "24"
    audio.write(path, x, sr, "wav", depth)


def build(src: Path, out: Path, map_path: Path | None = None, gmd: Path | None = None,
          limit: int | None = None, overwrite: bool = False) -> int:
    m = load_map(map_path)
    classes = list(m["stemgmd_classes"])
    midi_map = {int(k): v for k, v in m["midi_to_class"].items()}
    gmd_meta = Gmd(gmd) if gmd else None
    tracks = scan(Path(src), classes)
    n = 0
    for (drummer, session, kit, clip), files in sorted(tracks.items()):
        if limit is not None and n >= limit:
            break
        missing = sorted(set(classes) - set(files))
        if missing:
            raise KitsplitInputError(f"{drummer}/{session}/{kit}/{clip}: missing classes {missing}")
        split = gmd_meta.split(drummer, session, clip) if gmd_meta else "all"
        dest = Path(out) / split / f"{drummer}__{session}__{kit}__{clip}"
        if (dest / "mixture.wav").exists() and not overwrite:
            raise KitsplitOutputExistsError(f"{dest} exists (use --overwrite)")
        data = {}
        sr = None
        for cls, p in files.items():
            x, info = sf.read(str(p), dtype="float64", always_2d=True)
            if sr is not None and info != sr:
                raise KitsplitInputError(f"{p}: sample rate {info} != {sr}")
            sr = info
            data[cls] = x.T
        length = {v.shape[1] for v in data.values()}
        if len(length) != 1:
            raise KitsplitInputError(f"{drummer}/{session}/{kit}/{clip}: class files differ in length {length}")
        shape = next(iter(data.values())).shape
        mixture = np.zeros(shape)
        dest.mkdir(parents=True, exist_ok=True)
        for stem in STEMS7:
            acc = np.zeros(shape)
            for cls in m["map"][stem]:
                acc += data[cls]
            mixture += acc
            _write_exact(dest / f"{stem}.wav", acc, sr)
        _write_exact(dest / "mixture.wav", mixture, sr)
        if gmd_meta:
            per_class = gmd_meta.onsets(drummer, session, clip, midi_map)
            per_stem = {stem: sorted(o for cls in m["map"][stem] for o in per_class.get(cls, [])) for stem in STEMS7}
            (dest / "onsets.json").write_text(json.dumps({"source": "GMD MIDI", "onsets": per_stem}))
        n += 1
    return n
