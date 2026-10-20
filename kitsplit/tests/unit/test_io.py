import numpy as np
import pytest
import soundfile as sf

from kitsplit.errors import KitsplitInputError
from kitsplit.io import audio

READABLE = ["synth_kit_8s_48k.wav", "mono_44k.wav", "stereo_96k_24bit.wav", "stereo_48k_float.wav",
            "stereo_48k_24bit.flac", "silence_2s.wav", "dc_offset.wav", "clipped.wav"]


@pytest.mark.parametrize("name", READABLE)
def test_u_io_1_read_every_fixture(fx, name):
    x, info = audio.read(fx / name)
    assert x.dtype == np.float32
    assert x.ndim == 2 and x.shape[0] == 2 and x.shape[1] == info.frames
    assert info.sr == sf.info(str(fx / name)).samplerate


def test_u_io_1_mono_is_duplicated(fx):
    x, info = audio.read(fx / "mono_44k.wav")
    assert info.channels == 1
    np.testing.assert_array_equal(x[0], x[1])


@pytest.mark.parametrize("fmt,depth,tol", [("wav", "24", 2.0**-23), ("flac", "24", 2.0**-23), ("wav", "32f", 0.0)])
def test_u_io_2_roundtrip(tmp_path, fmt, depth, tol):
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.99, 0.99, size=(2, 4800)).astype(np.float32)
    p = tmp_path / f"x.{fmt}"
    audio.write(p, x, 48000, fmt, depth)
    y, info = audio.read(p)
    assert info.sr == 48000
    assert float(np.max(np.abs(y - x))) <= tol


@pytest.mark.parametrize("name", ["empty.wav", "not_audio.txt", "does_not_exist.wav"])
def test_u_io_3_bad_inputs(fx, name):
    with pytest.raises(KitsplitInputError):
        audio.read(fx / name)


def test_flac_cannot_store_float():
    with pytest.raises(KitsplitInputError, match="FLAC"):
        audio.validate_output_format("flac", "32f")


def test_float_wav_is_byte_deterministic(tmp_path):
    x = np.random.default_rng(1).standard_normal((2, 1000)).astype(np.float32) * 0.1
    audio.write(tmp_path / "a.wav", x, 48000, "wav", "32f")
    import time; time.sleep(1.1)  # PEAK chunk timestamps have 1 s resolution
    audio.write(tmp_path / "b.wav", x, 48000, "wav", "32f")
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


def test_resample_roundtrip_length():
    x = np.random.default_rng(2).standard_normal((2, 96000)).astype(np.float32)
    y = audio.resample(audio.resample(x, 96000, 48000), 48000, 96000)
    assert y.shape == x.shape
