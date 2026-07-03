"""PR #91 gap-closing: startup benchmark self-test hard-fail + idle-epoch audit gate.

Both behaviors were previously inline in service.main() (untestable / warn-only).
They are now extracted into _benchmark_selftest and _run_idle_audit and covered here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import ralph_bootstrap  # noqa: F401,E402
from validator import service as S  # noqa: E402


def _forgeable(n=300):
    # target systematically small, distractors uniform-large -> a blind score=-id
    # sweep beats chance (the exact deployed break).
    rng = np.random.default_rng(0)
    return [
        {
            "context_ids": [1, 2, 3],
            "target_id": int(rng.integers(0, 5000)),
            "distractors": [int(rng.integers(20000, 50000)) for _ in range(4)],
        }
        for _ in range(n)
    ]


def _write_bench(root: Path, examples):
    d = root / "eval" / "private"
    d.mkdir(parents=True, exist_ok=True)
    (d / "active_benchmark.json").write_text(json.dumps(examples))


# --- _benchmark_selftest: warns by default, REFUSES when crown enabled + forgeable ---

def test_selftest_refuses_when_crown_enabled_and_forgeable(tmp_path, monkeypatch):
    _write_bench(tmp_path, _forgeable())
    monkeypatch.setenv("RALPH_BENCHMARK_CROWN", "1")
    with pytest.raises(SystemExit):
        S._benchmark_selftest(tmp_path)


def test_selftest_forgeable_but_crown_off_only_warns(tmp_path, monkeypatch):
    _write_bench(tmp_path, _forgeable())
    monkeypatch.delenv("RALPH_BENCHMARK_CROWN", raising=False)
    S._benchmark_selftest(tmp_path)  # neutralized default -> warn, no raise


def test_selftest_missing_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("RALPH_BENCHMARK_CROWN", "1")
    S._benchmark_selftest(tmp_path)  # no active_benchmark.json -> return, no raise


# --- _run_idle_audit: default-on gate, RALPH_AUDIT_LOOP_OFF=1 disables, errors swallowed ---

def test_idle_audit_gate_calls_then_off(tmp_path, monkeypatch):
    calls = []
    import validator.audit_scheduler as sched
    monkeypatch.setattr(
        sched, "run_pending_audits",
        lambda *a, **k: (calls.append(1), {"processed": 0})[1],
    )
    monkeypatch.delenv("RALPH_AUDIT_LOOP_OFF", raising=False)
    S._run_idle_audit(object(), tmp_path, 0.013)
    assert len(calls) == 1  # gate default-on -> called
    monkeypatch.setenv("RALPH_AUDIT_LOOP_OFF", "1")
    S._run_idle_audit(object(), tmp_path, 0.013)
    assert len(calls) == 1  # gate off -> NOT called again


def test_idle_audit_swallows_errors(tmp_path, monkeypatch):
    import validator.audit_scheduler as sched

    def _boom(*a, **k):
        raise RuntimeError("audit blew up")

    monkeypatch.setattr(sched, "run_pending_audits", _boom)
    monkeypatch.delenv("RALPH_AUDIT_LOOP_OFF", raising=False)
    assert S._run_idle_audit(object(), tmp_path, 0.013) is None  # never raises
