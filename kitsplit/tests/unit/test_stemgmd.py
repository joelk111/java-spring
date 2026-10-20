import io
import json
import zipfile

import mido
import numpy as np
import pytest
import soundfile as sf

from kitsplit.data import stemgmd
from kitsplit.errors import KitsplitInputError
from kitsplit.io.stems import STEMS7

import fetch_stemgmd

CLASSES = ["kick", "snare", "hi_tom", "mid_tom", "low_tom", "hihat_closed", "hihat_open", "crash_left", "ride"]


def test_u_stem_2_map_covers_all_classes_once():
    m = stemgmd.load_map()
    assert sorted(m["stemgmd_classes"]) == sorted(CLASSES)
    members = [c for s in STEMS7 for c in m["map"][s]]
    assert sorted(members) == sorted(CLASSES) and len(members) == len(set(members))
    assert m["map"]["residual"] == []  # StemGMD has no non-drum class
    assert set(m["midi_to_class"].values()) == set(CLASSES)
    assert m["midi_to_class"][44] == "hihat_closed"  # pedal hi-hat folds into closed


def test_bad_map_rejected(tmp_path):
    bad = tmp_path / "m.yaml"
    bad.write_text("stemgmd_classes: [kick, snare]\nmap: {kick: [kick], snare: [kick], toms: [], hh: [], ride: [], crash: [], residual: []}\n")
    with pytest.raises(KitsplitInputError):
        stemgmd.load_map(bad)


def _fake_stemgmd(root, clips=("1_funk-groove1_138_beat_4-4",), kits=("bluebird", "heavy")):
    rng = np.random.default_rng(0)
    for kit in kits:
        for cls in CLASSES:
            d = root / "audio" / "drummer1" / "eval_session" / kit / cls
            d.mkdir(parents=True)
            for clip in clips:
                x = np.round(rng.uniform(-0.3, 0.3, (4410, 2)) * 32768) / 32768
                sf.write(str(d / f"{clip}.wav"), x, 44100, subtype="PCM_16")


def _fake_gmd(path):
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))  # 120 bpm
    for note, dt in ((36, 0), (44, 480), (38, 480)):  # kick @0 s, pedal hh @0.5 s, snare @1.0 s
        tr.append(mido.Message("note_on", note=note, velocity=100, time=dt))
        tr.append(mido.Message("note_off", note=note, velocity=0, time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    info = ("drummer,session,id,style,bpm,beat_type,time_signature,midi_filename,audio_filename,duration,split\n"
            "drummer1,drummer1/eval_session,drummer1/eval_session/1,funk/groove1,138,beat,4-4,"
            "drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid,x.wav,1.0,test\n")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("groove/info.csv", info)
        z.writestr("groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid", buf.getvalue())


def test_build_remaps_sums_and_folds(tmp_path):
    _fake_stemgmd(tmp_path / "src")
    _fake_gmd(tmp_path / "gmd.zip")
    n = stemgmd.build(tmp_path / "src", tmp_path / "out", gmd=tmp_path / "gmd.zip")
    assert n == 2
    track = tmp_path / "out" / "test" / "drummer1__eval_session__heavy__1_funk-groove1_138_beat_4-4"
    src = tmp_path / "src" / "audio" / "drummer1" / "eval_session" / "heavy"
    read = lambda p: sf.read(str(p), dtype="float64")[0]
    toms = read(src / "hi_tom" / "1_funk-groove1_138_beat_4-4.wav") + read(src / "mid_tom" / "1_funk-groove1_138_beat_4-4.wav") \
        + read(src / "low_tom" / "1_funk-groove1_138_beat_4-4.wav")
    np.testing.assert_array_equal(read(track / "toms.wav"), toms)
    stems = sum(read(track / f"{s}.wav") for s in STEMS7)
    np.testing.assert_array_equal(stems, read(track / "mixture.wav"))
    assert not read(track / "residual.wav").any()
    on = json.loads((track / "onsets.json").read_text())["onsets"]
    assert on["kick"] == [[0.0, 100]] and on["hh"] == [[0.5, 100]] and on["snare"] == [[1.0, 100]]


def test_build_refuses_overwrite(tmp_path):
    from kitsplit.errors import KitsplitOutputExistsError

    _fake_stemgmd(tmp_path / "src", kits=("bluebird",))
    stemgmd.build(tmp_path / "src", tmp_path / "out")
    with pytest.raises(KitsplitOutputExistsError):
        stemgmd.build(tmp_path / "src", tmp_path / "out")


def test_missing_class_is_an_error(tmp_path):
    _fake_stemgmd(tmp_path / "src", kits=("bluebird",))
    next((tmp_path / "src").rglob("ride/*.wav")).unlink()
    with pytest.raises(KitsplitInputError, match="missing classes"):
        stemgmd.build(tmp_path / "src", tmp_path / "out")


@pytest.mark.parametrize("lic,allow_nc,code", [("cc-by-4.0", False, 0), ("cc-by-nc-4.0", False, 6),
                                               ("cc-by-nc-4.0", True, 0), (None, False, 6)])
def test_license_gate(lic, allow_nc, code, capsys):
    rec = {"metadata": {"license": {"id": lic} if lic else None}, "files": []}
    argv = ["--license-only"] + (["--allow-nc"] if allow_nc else [])
    assert fetch_stemgmd.main(argv, fetch_record=lambda _id: rec) == code
    if code == 6:
        assert "HALT" in capsys.readouterr().err


def test_dataset_synth_cli(tmp_path):
    from typer.testing import CliRunner

    from kitsplit.cli import app

    for stem, freq in (("kick", 60.0), ("snare", 200.0)):
        d = tmp_path / "shots" / stem
        d.mkdir(parents=True)
        for layer, amp in (("a_soft", 0.2), ("b_hard", 0.8)):
            t = np.arange(4410) / 44100
            sf.write(str(d / f"{layer}.wav"), amp * np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.05), 44100)
    gmd = tmp_path / "gmd.zip"
    _fake_gmd(gmd)
    (tmp_path / "midi").mkdir()
    (tmp_path / "midi" / "a.mid").write_bytes(zipfile.ZipFile(gmd).read(
        "groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid"))
    r = CliRunner().invoke(app, ["dataset", "synth", "--onsets", str(tmp_path / "midi"), "--oneshots",
                                 str(tmp_path / "shots"), "--out", str(tmp_path / "out")])
    assert r.exit_code == 0, r.output
    track = tmp_path / "out" / "a"
    stems = sum(sf.read(str(track / f"{s}.wav"), dtype="float64")[0] for s in STEMS7)
    np.testing.assert_array_equal(stems, sf.read(str(track / "mixture.wav"), dtype="float64")[0])
    on = json.loads((track / "onsets.json").read_text())["onsets"]
    assert [o[0] for o in on["kick"]] == [0.0] and [o[0] for o in on["snare"]] == [1.0]
    assert "hh" not in on  # no hh one-shots supplied -> pedal-hh onset skipped
