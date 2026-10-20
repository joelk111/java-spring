import re
from pathlib import Path

import update_readme_claims as claims

ROOT = Path(__file__).resolve().parents[2]


def test_every_assumption_marker_is_documented():
    doc = (ROOT / "docs" / "ASSUMPTIONS.md").read_text()
    files = set()
    for base in ("kitsplit", "configs", "scripts"):
        for p in (ROOT / base).rglob("*"):
            if p.suffix in (".py", ".yaml") and "ASSUMPTION" in p.read_text():
                files.add(p.relative_to(ROOT).as_posix())
    missing = [f for f in sorted(files) if f not in doc and not any(
        g in doc and re.fullmatch(g.replace("{quality,fast}", "(quality|fast)"), f) for g in re.findall(r"`([^`]+)`", doc))]
    assert not missing, f"ASSUMPTION markers not listed in docs/ASSUMPTIONS.md: {missing}"


def test_readme_claims_block_is_generated():
    text = (ROOT / "README.md").read_text()
    assert claims.render(text, claims.claims_block()) == text


def test_no_claims_without_eval_real(tmp_path):
    assert "No quality claims" in claims.claims_block(tmp_path / "missing.json")
