import re

import pytest

from kitsplit.errors import KitsplitModelError
from kitsplit.models import registry


def test_u_reg_1_every_entry_has_sha_and_license():
    entries = registry.load_registry()
    assert entries
    for e in entries:
        assert "sha256" in e.raw and "license" in e.raw and e.license
        if e.kind != "oracle" and e.weights:
            assert re.fullmatch(r"[0-9a-f]{64}", e.sha256 or ""), e.name
        if any(w in e.license.lower() for w in ("unknown", "not stated", "none", "tbd", "n/a", "test")):
            assert e.shippable is False, e.name


def test_baselines_are_eval_only():
    for name in ("B1", "B2"):
        assert registry.resolve(name).shippable is False


def test_unknown_model():
    with pytest.raises(KitsplitModelError):
        registry.resolve("nonexistent")


def test_tampered_fixture_fails_verify(fx):
    with pytest.raises(KitsplitModelError, match="sha256 mismatch"):
        registry.verify(registry.resolve(str(fx / "models" / "tampered.yaml")))


def test_untrained_kitsplit_v1_is_reported_missing():
    with pytest.raises(KitsplitModelError, match="no published weights"):
        registry.verify(registry.resolve("kitsplit-v1"))
