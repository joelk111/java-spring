import hashlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "scripts"))


def pytest_addoption(parser):
    parser.addoption("--real-data", action="store", default=None,
                     help="Path to a real Type 1 eval set (enables `quality` tests).")


def pytest_collection_modifyitems(config, items):
    real = config.getoption("--real-data")
    net = os.environ.get("KITSPLIT_NETWORK_TESTS") == "1"
    for item in items:
        if "quality" in item.keywords and not real:
            item.add_marker(pytest.mark.skip(reason="quality tests need --real-data"))
        if "network" in item.keywords and not net:
            item.add_marker(pytest.mark.skip(reason="network tests need KITSPLIT_NETWORK_TESTS=1"))


def _fixtures_ok() -> bool:
    manifest = FIXTURES / "MANIFEST.sha256"
    for line in manifest.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        p = FIXTURES / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            return False
    return True


@pytest.fixture(scope="session", autouse=True)
def generated_fixtures():
    """Regenerate the (deterministic, git-ignored) fixtures when any is missing/stale."""
    if not _fixtures_ok():
        import make_fixtures

        make_fixtures.generate()
    return FIXTURES


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("KITSPLIT_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def fx():
    return FIXTURES
