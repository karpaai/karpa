"""Pre-crown re-derivation gate (proof-of-EXECUTION).

op_rederive_trajectory re-runs the first steps of a challenger's declared training on
the validator's CANONICAL data and requires the loss trajectory to reproduce. It is:
  * OFF by default; shadow logs the verdict but never blocks; enforce withholds the crown
    on a genuine trajectory mismatch (RALPH_REDERIVE, _rederive_mode).
  * run ONLY for a bundle that would take the crown (router gate) -> cost O(king-changes),
    not O(submissions).
  * fail-OPEN on every ambiguity (disabled / no canonical data / timeout / too-few points)
    so a validator glitch never wrongly withholds a crown; a failed verdict only PREVENTS a
    promotion, it never dethrones the sitting king.
  * env-tunable tolerances (_rederive_tols) so the honest cross-GPU spread is calibrated in
    shadow without a code change.
Also guards the seed-clobber fix: re-derivation must NOT pass --seed (train.py --seed
clobbers both init_seed AND data_seed -> false-reject of any honest init_seed!=data_seed run).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401,E402
import validator.router as router  # noqa: E402
from validator.validator import (  # noqa: E402
    _apply_rederive_mode,
    _rederive_mode,
    _rederive_tols,
    op_rederive_trajectory,
)


# ---------------------------------------------------------------- mode parsing
@pytest.mark.parametrize("val,expect", [
    (None, "off"), ("", "off"), ("0", "off"), ("off", "off"),
    ("1", "enforce"), ("on", "enforce"), ("true", "enforce"),
    ("enforce", "enforce"), ("ENFORCE", "enforce"),
    ("shadow", "shadow"), ("Shadow", "shadow"),
])
def test_rederive_mode(monkeypatch, val, expect):
    if val is None:
        monkeypatch.delenv("RALPH_REDERIVE", raising=False)
    else:
        monkeypatch.setenv("RALPH_REDERIVE", val)
    assert _rederive_mode() == expect


# ------------------------------------------------------------ tolerance knobs
def test_tols_default(monkeypatch):
    for k in ("STEP0", "ABS", "REL"):
        monkeypatch.delenv(f"RALPH_REDERIVE_{k}_TOL", raising=False)
    assert _rederive_tols() == (0.10, 0.40, 0.10)


def test_tols_env_override(monkeypatch):
    monkeypatch.setenv("RALPH_REDERIVE_STEP0_TOL", "0.30")
    monkeypatch.setenv("RALPH_REDERIVE_ABS_TOL", "0.60")
    monkeypatch.setenv("RALPH_REDERIVE_REL_TOL", "0.15")
    assert _rederive_tols() == (0.30, 0.60, 0.15)


def test_tols_malformed_falls_back(monkeypatch):
    monkeypatch.setenv("RALPH_REDERIVE_STEP0_TOL", "not-a-float")
    assert _rederive_tols()[0] == 0.10


# --------------------------------------------------------------- shadow wrap
def test_shadow_never_gates_on_fail():
    ok, detail = _apply_rederive_mode("shadow", False, "mismatch at step 0")
    assert ok is True and "would_gate=FAIL" in detail  # logged, but never blocks


def test_shadow_reports_pass():
    ok, detail = _apply_rederive_mode("shadow", True, "reproduced 3/3")
    assert ok is True and "would_gate=PASS" in detail


def test_enforce_passes_verdict_through():
    assert _apply_rederive_mode("enforce", False, "x") == (False, "x")
    assert _apply_rederive_mode("enforce", True, "y") == (True, "y")


# ------------------------------------------------------------- fail-open skips
def test_disabled_is_fail_open(monkeypatch, tmp_path):
    monkeypatch.delenv("RALPH_REDERIVE", raising=False)
    ok, detail = op_rederive_trajectory(tmp_path, tmp_path)
    assert ok is True and "disabled" in detail


def test_no_canonical_data_is_fail_open(monkeypatch, tmp_path):
    # enabled, but the validator has not materialized the canonical shards/manifest ->
    # skip (never a false-reject), the load-bearing pre-P2 behaviour.
    monkeypatch.setenv("RALPH_REDERIVE", "1")
    import ralph_bootstrap as rb
    monkeypatch.setattr(rb, "RECIPE_DIR", tmp_path / "no_recipe_here")
    ok, detail = op_rederive_trajectory(tmp_path, tmp_path)
    assert ok is True and "not materialized" in detail


# ------------------------------------------------- seed-clobber fix (source guard)
def test_rederive_does_not_pass_seed():
    """train.py --seed clobbers both init_seed and data_seed; re-derivation must rely on
    the full cfg via --config instead, or any honest init_seed!=data_seed run false-rejects."""
    import inspect
    src = inspect.getsource(op_rederive_trajectory)
    assert '"--seed"' not in src and "'--seed'" not in src


# ============================================================ router crown gate
def _fake_judge(*, val_bpb, benchmark_accuracy, tier="verified"):
    return types.SimpleNamespace(
        rejected=None, miner_hotkey="5F_challenger", bundle_hash="bh_challenger",
        handshake_nonce="nonce",
        hidden_eval=types.SimpleNamespace(val_bpb=val_bpb, benchmark_accuracy=benchmark_accuracy),
        calibration={"matmul_ms": 10.0}, training_summary={"wall_clock_s": 1.0},
        operations={"op2_attestation": {"tier": tier}}, to_dict=lambda: {},
    )


@pytest.fixture
def patched(monkeypatch):
    holder: dict = {"rederive": (True, "disabled")}
    calls: list = []

    def set_judge(**kw):
        holder["result"] = _fake_judge(**kw)

    def set_rederive(ok, detail="v"):
        holder["rederive"] = (ok, detail)

    def _spy(ralph_root, proof_dir):
        calls.append(proof_dir)
        return holder["rederive"]

    monkeypatch.setattr(router, "judge_submission", lambda *a, **k: holder["result"])
    monkeypatch.setattr(router, "op_rederive_trajectory", _spy)
    return types.SimpleNamespace(set_judge=set_judge, set_rederive=set_rederive, calls=calls)


def test_would_beat_king_and_rederive_pass_crowns(tmp_path, patched):
    patched.set_judge(val_bpb=2.0, benchmark_accuracy=0.4)          # genesis
    patched.set_rederive(True)
    out = router.process_submission(tmp_path, tmp_path / "p_a")
    assert out["status"] == "accepted"
    assert router._load_king(tmp_path)["val_bpb"] == 2.0
    assert len(patched.calls) == 1                                   # ran for the crown


def test_would_beat_king_but_rederive_fail_withholds_crown(tmp_path, patched):
    # genesis crowns cleanly (rederive passes), sets the bar at 2.0
    patched.set_judge(val_bpb=2.0, benchmark_accuracy=0.4)
    patched.set_rederive(True)
    router.process_submission(tmp_path, tmp_path / "p_a")

    # a decisively-better challenger, but re-derivation FAILS -> crown withheld, king holds
    patched.set_judge(val_bpb=1.0, benchmark_accuracy=0.4)
    patched.set_rederive(False, "re-derivation mismatch at step 0")
    out = router.process_submission(tmp_path, tmp_path / "p_b")

    assert out["status"] == "rederive_withheld"
    assert out["event"]["accepted_as_king"] is False
    assert out["event"]["rederive_blocked"] is True
    assert router._load_king(tmp_path)["val_bpb"] == 2.0            # sitting king NOT dethroned


def test_below_threshold_challenger_skips_rederive(tmp_path, patched):
    # genesis crown @ 1.0
    patched.set_judge(val_bpb=1.0, benchmark_accuracy=0.4)
    patched.set_rederive(True)
    router.process_submission(tmp_path, tmp_path / "p_a")
    patched.calls.clear()

    # a worse challenger (2.0 > 1.0 val_bpb, does not beat the bar) -> rederive NOT invoked
    patched.set_judge(val_bpb=2.0, benchmark_accuracy=0.4)
    out = router.process_submission(tmp_path, tmp_path / "p_b")
    assert out["status"] == "below_threshold"
    assert patched.calls == []                                      # cost gate: O(king-changes)


# ============================================ PRODUCTION crown gate (service.py)
def _svc_result(val_bpb=1.0):
    return types.SimpleNamespace(
        rejected=None,
        hidden_eval=types.SimpleNamespace(
            val_bpb=val_bpb, benchmark_accuracy=0.5, val_seq_len=1024,
            sealed_stream_manifest_hash="h", tail_val_bpb=None,
        ),
        calibration={"matmul_ms": 10.0}, training_summary={"wall_clock_s": 1.0},
        operations={"op2_attestation": {"tier": "verified"}},
        miner_hotkey="5F", miner_github="gh", pr_url=None, bundle_hash="bh",
    )


class _Chain:
    def __init__(self, king=None):
        self._king = king

    def get_king(self):
        return self._king

    def get_high_water_mark(self):
        return None


@pytest.fixture
def svc(monkeypatch):
    import validator.service as S
    calls: list = []
    seen: dict = {}

    monkeypatch.setattr(S, "_verify_pr_if_required", lambda *a, **k: (True, "ok"))

    def _classify(*, decisively, **k):
        seen["decisively"] = decisively
        return ("king_change" if decisively else "plain_failure",
                1.0 if decisively else 0.0)

    monkeypatch.setattr(S, "_classify_outcome", _classify)

    def set_judge(val_bpb):
        monkeypatch.setattr(S, "judge_submission", lambda *a, **k: _svc_result(val_bpb))

    def set_rederive(ok, detail="v"):
        def _spy(ralph_root, bundle_dir):
            calls.append(bundle_dir)
            return (ok, detail)
        monkeypatch.setattr(S, "op_rederive_trajectory", _spy)

    return types.SimpleNamespace(
        S=S, calls=calls, seen=seen, set_judge=set_judge, set_rederive=set_rederive,
    )


def test_service_rederive_pass_allows_crown(tmp_path, svc):
    svc.set_judge(1.0)
    svc.set_rederive(True)
    out = svc.S.score_and_decide(_Chain(king=None), tmp_path, 0.02)  # genesis
    assert out["status"] == "accepted" and out["accepted"] is True
    assert out["rederive_withheld"] is False
    assert svc.seen["decisively"] is True and len(svc.calls) == 1


def test_service_rederive_fail_withholds_crown(tmp_path, svc):
    svc.set_judge(1.0)
    svc.set_rederive(False, "re-derivation mismatch at step 0")
    out = svc.S.score_and_decide(_Chain(king=None), tmp_path, 0.02)
    assert out["status"] == "rederive_withheld"
    assert out["rederive_withheld"] is True and out["accepted"] is False
    assert svc.seen["decisively"] is False           # gate flipped it before classify


def test_service_non_decisive_skips_rederive(tmp_path, svc):
    # sitting king @ 1.0; a worse challenger @ 2.0 does not beat it -> rederive NOT called
    king = types.SimpleNamespace(val_bpb=1.0, benchmark_accuracy=0.5)
    svc.set_judge(2.0)
    svc.set_rederive(True)
    out = svc.S.score_and_decide(_Chain(king=king), tmp_path, 0.02)
    assert out["accepted"] is False and out["rederive_withheld"] is False
    assert svc.calls == []                           # cost gate: O(king-changes)
