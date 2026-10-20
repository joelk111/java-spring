"""Quality gates Q-1..Q-4 (plan §9.5). Need --real-data (data/eval_real) and a trained
kitsplit-v1; never run in default CI."""

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.quality
GATED = ("kick", "snare", "toms")


@pytest.fixture(scope="module")
def bench(request, tmp_path_factory):
    from kitsplit.eval import benchmark

    data = Path(request.config.getoption("--real-data"))
    out = tmp_path_factory.mktemp("quality")
    _, js = benchmark.run(data, ["kitsplit-v1", "b1-drumsep-mdx23c-6stem"], out,
                          ["sdr", "bleedless", "fullness", "onset_snr"], "quality")
    return json.loads(js.read_text())


def _median(bench, model, stem, metric):
    return bench["models"][model]["groupings"]["7"]["summary"][stem][metric]["median"]


def test_q1_beats_b1_on_kick_snare_toms(bench):
    for s in GATED:
        for m in ("sdr", "bleedless"):
            assert _median(bench, "kitsplit-v1", s, m) >= _median(bench, "b1-drumsep-mdx23c-6stem", s, m), (s, m)


def test_q2_cymbals_reported_only(bench):
    for s in ("hh", "ride", "crash"):
        print(s, _median(bench, "kitsplit-v1", s, "sdr"))


def test_q3_no_regression_vs_last_release(bench):
    last = Path(__file__).resolve().parents[2] / "reports" / "last_release.json"
    if not last.is_file():
        pytest.skip("no tagged release report yet")
    prev = json.loads(last.read_text())
    for s in ("kick", "snare", "toms", "hh", "ride", "crash"):
        drop = _median(prev, "kitsplit-v1", s, "sdr") - _median(bench, "kitsplit-v1", s, "sdr")
        assert drop <= 0.3, (s, drop)


def test_q4_consistency_on_real_files(request, tmp_path):
    from kitsplit.io import audio
    from kitsplit.models import registry
    from kitsplit.models.loader import load_backend
    from kitsplit.separate import Settings, separate_array

    data = Path(request.config.getoption("--real-data"))
    be = load_backend(registry.resolve("kitsplit-v1"))
    for mix in sorted(data.rglob("mixture.wav")):
        x, info = audio.read(mix)
        y = separate_array(x, info.sr, info.channels, be, Settings.from_preset("fast"))
        assert np.max(np.abs(y.astype(np.float64).sum(0) - audio.restore_channels(x, info.channels))) < 1e-6
