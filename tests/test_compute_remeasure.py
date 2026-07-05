"""On-GPU throughput re-measurement gate (the airtight compute-forgery gate).

op_compute_remeasure re-runs a would-be king's exact config on the validator's GPU and
rejects a declared throughput that can't be reproduced. The GPU probe itself needs
hardware (validated in shadow on the box), but the mode-parsing + the pure verdict
(GPU-scaled margin comparison) + fail-open behaviour are unit-tested here against the
real 2026-07 numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401,E402
from validator.validator import (  # noqa: E402
    _remeasure_mode,
    _remeasure_verdict,
    op_compute_remeasure,
)

H100 = "NVIDIA H100 80GB HBM3"
H200 = "NVIDIA H200"


# ---------------------------------------------------------------- mode parsing
@pytest.mark.parametrize("val,expect", [
    (None, "off"), ("", "off"), ("0", "off"), ("off", "off"),
    ("1", "enforce"), ("on", "enforce"), ("enforce", "enforce"), ("ENFORCE", "enforce"),
    ("shadow", "shadow"), ("Shadow", "shadow"),
])
def test_remeasure_mode(monkeypatch, val, expect):
    if val is None:
        monkeypatch.delenv("RALPH_COMPUTE_REMEASURE", raising=False)
    else:
        monkeypatch.setenv("RALPH_COMPUTE_REMEASURE", val)
    assert _remeasure_mode() == expect


# --------------------------------------------------------------- verdict logic
def test_rejects_kaizen_forgery():
    # Kaizen b1930c27: declared 356k on H100; validator re-measures the same uncompiled
    # Muon config at ~86k. Same GPU -> scale 1.0. ratio 4.1x >> 1.6 margin -> REJECT.
    ok, detail = _remeasure_verdict(356_000, 86_000, H100, H100, margin=1.6)
    assert not ok and "ratio 4.1x" in detail


def test_passes_danielortega_with_hardware_and_torchver_variance():
    # danielortega: declared 239k on H200; validator (H100) re-measures his COMPILED recipe
    # at ~172k. peak scale H200/H100 == 1.0; margin 1.6 -> ceiling 275k. 239k < 275k -> PASS
    # (absorbs the ~15% torch-version gap + H200 memory-bandwidth edge).
    ok, _ = _remeasure_verdict(239_000, 172_000, H200, H100, margin=1.6)
    assert ok


def test_gpu_scaling_up_for_slower_validator():
    # Miner on H200 (fast), validator on A100 (312 TF). A validator that re-measures 100k
    # on its A100 scales UP by 989/312 = 3.17 -> ceiling 100k*3.17*1.6 = 507k. A declared
    # 300k on H200 is achievable -> PASS (don't false-reject a genuinely-faster GPU).
    ok, detail = _remeasure_verdict(300_000, 100_000, H200, "NVIDIA A100", margin=1.6)
    assert ok and "scale 3.17" in detail


def test_rejects_gross_forgery_even_with_scaling():
    ok, _ = _remeasure_verdict(800_000, 100_000, H100, H100, margin=1.6)  # 8x
    assert not ok


def test_fail_open_when_no_measurement():
    ok, detail = _remeasure_verdict(300_000, 0, H100, H100, margin=1.6)
    assert ok and "inconclusive" in detail


def test_margin_boundary():
    # exactly at the margin passes; just over rejects (same GPU, scale 1.0)
    assert _remeasure_verdict(160_000, 100_000, H100, H100, margin=1.6)[0]
    assert not _remeasure_verdict(160_001, 100_000, H100, H100, margin=1.6)[0]


# ------------------------------------------------------------- op fail-open/off
def test_op_off_is_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("RALPH_COMPUTE_REMEASURE", raising=False)
    ok, detail = op_compute_remeasure(tmp_path, tmp_path)
    assert ok and "disabled" in detail


def test_op_skips_without_final_state(monkeypatch, tmp_path):
    # enabled, but no final_state -> fail-open skip (never a false-reject on a glitch)
    monkeypatch.setenv("RALPH_COMPUTE_REMEASURE", "1")
    ok, detail = op_compute_remeasure(tmp_path, tmp_path)
    assert ok and "no final_state" in detail
