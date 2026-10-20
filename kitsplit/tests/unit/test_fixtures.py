"""Fixtures are regenerated, not committed: their bytes must match MANIFEST.sha256."""

import hashlib

from make_fixtures import manifest_lines, generate


def test_fixture_bytes_match_manifest(fx, tmp_path):
    files = generate(tmp_path)
    got = manifest_lines(files, tmp_path)
    want = (fx / "MANIFEST.sha256").read_text().splitlines()
    assert got == want
