"""check_training_log_integrity — reject a fabricated training_log used to disguise a
forged-down wall_clock_s, without false-positiving on a legit low-jitter run.

Root incident: taohunter v0.3.3 reported wall_clock_s low enough to hit exactly 50%
MFU (under the 70% ceiling) and shipped a perfectly-linear log (step-0 elapsed 0.0)
that agreed with it, keeping normalized-H100h under the cap.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ralph_bootstrap  # noqa: F401,E402
from validator.integrity import check_training_log_integrity  # noqa: E402

TOK = 50 * 5243  # tokens per logged 50-step interval


def _row(step, elapsed, tokens_seen, tps):
    return {"step": step, "loss": 3.0, "tokens_seen": tokens_seen,
            "tokens_per_sec": tps, "elapsed_s": elapsed}


def _real(n=120, step=50, dt=0.808, step0=6.5, jitter=0.02, seed=1):
    rng = random.Random(seed)
    rows, e = [], 0.0
    for i in range(n):
        e = step0 if i == 0 else e + step * dt * (1 + rng.uniform(-jitter, jitter))
        ts = (i + 1) * TOK
        rows.append(_row(i * step, e, ts, ts / e))  # tps == tokens/elapsed identity
    return rows


def test_real_jittered_log_passes():
    rows = _real()
    ok, reason, _ = check_training_log_integrity(rows, {"wall_clock_s": rows[-1]["elapsed_s"]})
    assert ok, reason


def test_step0_zero_rejected():
    # taohunter-style: perfectly linear, step-0 elapsed 0.0
    rows = [_row(i * 50, (i * 50) * 0.808, (i + 1) * TOK, 324500.0) for i in range(120)]
    ok, reason, _ = check_training_log_integrity(rows, {"wall_clock_s": rows[-1]["elapsed_s"]})
    assert not ok and "step-0" in reason


def test_tps_identity_evasion_rejected():
    # sloppy fabrication: elapsed and tps produced inconsistently -> the MAJORITY of
    # rows break the tokens_seen/elapsed_s identity (a uniform offset instead reads as
    # a non-canonical tps definition and is warn-only, tested separately).
    rows = _real()
    for i, r in enumerate(rows):
        if i % 3 != 0 and r["elapsed_s"] > 0:  # ~2/3 of rows inconsistent
            r["tokens_per_sec"] = r["tokens_seen"] / r["elapsed_s"] * 1.5
    ok, reason, _ = check_training_log_integrity(rows, {"wall_clock_s": rows[-1]["elapsed_s"]})
    assert not ok and "tokens_per_sec" in reason


def test_wall_mismatch_rejected():
    rows = _real()
    ok, reason, _ = check_training_log_integrity(rows, {"wall_clock_s": rows[-1]["elapsed_s"] * 0.5})
    assert not ok and "wall_clock_s" in reason


def test_perfectly_linear_but_valid_step0_and_identity_warns_not_rejects():
    # FP-safety: a perfectly-linear log with a REAL step-0 + valid tps identity is NOT
    # rejected (a legit stable torch.compile run can look like this) — only warned/audited.
    rows = []
    for i in range(120):
        e = 6.5 + (i * 50) * 0.808  # linear after a real step-0
        ts = (i + 1) * TOK
        rows.append(_row(i * 50, e, ts, ts / e))
    ok, reason, warns = check_training_log_integrity(rows, {"wall_clock_s": rows[-1]["elapsed_s"]})
    assert ok, reason
    assert any("perfectly linear" in w for w in warns)  # flagged for audit


def test_too_few_rows_deferred():
    ok, _, _ = check_training_log_integrity([{"step": 0, "elapsed_s": 5.0}], {})
    assert ok  # <2 rows -> deferred (op3 handles empty)


def test_noncanonical_tps_semantics_warns_not_rejects():
    # a recipe that logs INSTANTANEOUS tps (never == cumulative) must not be rejected
    rows = _real()
    for r in rows:
        r["tokens_per_sec"] = 999999.0  # matches nothing
    ok, reason, warns = check_training_log_integrity(rows, {"wall_clock_s": rows[-1]["elapsed_s"]})
    assert ok, reason  # identity skipped (no row matches) rather than rejected
    assert any("non-canonical tps" in w for w in warns)
