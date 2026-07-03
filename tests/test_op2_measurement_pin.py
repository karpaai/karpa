"""RALPH_EXPECTED_MEASUREMENT pins op2's expected container_measurement to the
published image, decoupling it from the validator box's live (possibly dev-branch)
tree so a proof-source change on the box can't silently break op2 for all miners.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ralph_bootstrap  # noqa: F401,E402
from validator.validator import _expected_container_measurement  # noqa: E402


def test_pin_hit_ignores_tree(monkeypatch):
    # a valid 64-hex pin is returned verbatim, without touching the live tree
    monkeypatch.setenv("RALPH_EXPECTED_MEASUREMENT", "b" * 64)
    assert _expected_container_measurement(Path("/does/not/matter")) == "b" * 64


def test_pin_lowercased_and_validated(monkeypatch):
    monkeypatch.setenv("RALPH_EXPECTED_MEASUREMENT", "A" * 64)
    assert _expected_container_measurement(Path("/x")) == "a" * 64


def test_invalid_pin_falls_back_to_live_compute(monkeypatch):
    # a malformed pin is ignored -> compute from the live tree (legacy behaviour).
    # validator.py is not in list_proof_sources, so these edits are measurement-neutral.
    monkeypatch.setenv("RALPH_EXPECTED_MEASUREMENT", "tooshort")
    repo = Path(__file__).resolve().parent.parent
    v = _expected_container_measurement(repo)
    assert len(v) == 64 and all(c in "0123456789abcdef" for c in v) and v != "tooshort"
