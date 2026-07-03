"""Tests for validator.scoring.score_bundle.

Covers:
  - NaN/Inf safety (deep_review_2026-05-31 high #3)
  - decisively_beats_king OR-of-AND condition
  - v1.2 single-tier semantics (no α discount)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from validator.scoring import ALPHA_VERIFIED, score_bundle

_BASE = dict(
    benchmark_accuracy=0.5,
    king_val_bpb=1.5,
    king_benchmark=0.45,
    noise_floor_margin=0.013,
    matmul_ms=5.0,
    wall_clock_s=60.0,
)


def test_decisive_on_bpb_gain():
    """val_bpb 0.05 better than king (>> noise floor) → decisive."""
    sr = score_bundle(val_bpb=1.45, **_BASE)
    assert sr.decisively_beats_king is True
    assert sr.quality_gain > _BASE["noise_floor_margin"]


def test_not_decisive_inside_noise_band():
    """val_bpb 0.005 better AND bench 0.005 better — both inside noise band
    → not decisive. The OR-of-AND condition requires at least one axis to
    exceed the noise floor."""
    sr = score_bundle(
        val_bpb=1.495,  # 0.005 better — inside noise band
        benchmark_accuracy=0.455,  # 0.005 better — inside noise band
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False


def test_not_decisive_when_bench_regresses_and_quality_below_dominant(monkeypatch):
    """bpb wins past noise but BELOW DOMINANT_QUALITY_MULTIPLIER, and bench
    regresses past -noise — not decisive. The quality_gain (0.02 ≈ 1.5x
    noise) doesn't clear the dominant-quality branch (3x noise = 0.039)
    and the paired-axes branch fails because bench drop is too large.
    (Legacy benchmark-in-crown behavior — now behind RALPH_BENCHMARK_CROWN.)"""
    monkeypatch.setenv("RALPH_BENCHMARK_CROWN", "1")
    sr = score_bundle(
        val_bpb=1.48,                # 0.02 better — above noise, below dominant
        benchmark_accuracy=0.40,     # 0.05 worse — outside -noise_floor
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False


def test_v0_10_branch_a_no_longer_crowns_when_bench_regresses(monkeypatch):
    """v0.10 Goodhart closure: under the GUARDED dominant-quality clause, a
    big quality_gain (~6x noise) does NOT crown if benchmark regressed past
    the noise floor. This is the seed-search backdoor the v0.10 fix closes
    — see DOMINANT_QUALITY_MULTIPLIER comment in validator/scoring.py.

    Prior to v0.10 this case returned True via the unguarded Branch A; under
    the v0.10 guard the case is plain_failure because the benchmark drop
    indicates the win may be spurious. Branch B and Branch C also fail
    (Branch B needs benchmark-no-regress; Branch C needs benchmark > noise).
    (Legacy benchmark-in-crown behavior — now behind RALPH_BENCHMARK_CROWN.)
    """
    monkeypatch.setenv("RALPH_BENCHMARK_CROWN", "1")
    sr = score_bundle(
        val_bpb=1.42,                # 0.08 better — well above 3x noise
        benchmark_accuracy=0.40,     # 0.05 worse — outside -noise_floor
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False


def test_v0_10_branch_a_still_crowns_when_bench_within_noise():
    """The guard does NOT block legitimate wins where benchmark moves are
    inside the noise band. A dominant-quality win paired with a 0.005
    benchmark drop (well inside ±0.013) still crowns via Branch A."""
    sr = score_bundle(
        val_bpb=1.42,                # 0.08 better
        benchmark_accuracy=0.445,    # 0.005 worse — INSIDE -noise_floor
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is True


def test_dominant_quality_threshold_just_above_3x_noise():
    """Just above 3x noise_floor with benchmark-no-regress: Branch A fires.
    Picks a delta safely past floating-point noise on the threshold itself,
    and keeps benchmark inside the noise band so the v0.10 guard does not
    interfere."""
    sr = score_bundle(
        val_bpb=1.460,               # gain = 0.040 > 3 * 0.013 = 0.039
        benchmark_accuracy=0.443,    # 0.007 worse — INSIDE -noise_floor (0.013)
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is True


def test_dominant_quality_threshold_does_not_fire_just_below(monkeypatch):
    """Boundary the other side: 2.9x noise (below 3x) doesn't trigger the
    dominant-quality branch. If bench also regresses past -noise the
    submission falls back to non-decisive."""
    monkeypatch.setenv("RALPH_BENCHMARK_CROWN", "1")
    sr = score_bundle(
        val_bpb=1.4623,              # gain = 0.0377 = 2.9 * 0.013
        benchmark_accuracy=0.00,     # bench tanked
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False


def test_decisive_via_benchmark_axis(monkeypatch):
    """Benchmark wins, val_bpb roughly tied — decisive via the benchmark arm of
    the OR-of-AND condition. (Legacy behavior — now behind RALPH_BENCHMARK_CROWN.)"""
    monkeypatch.setenv("RALPH_BENCHMARK_CROWN", "1")
    sr = score_bundle(
        val_bpb=1.50,  # tied
        benchmark_accuracy=0.50,  # 0.05 better — outside noise floor
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is True


def test_benchmark_neutralized_by_default(monkeypatch):
    """Default (no RALPH_BENCHMARK_CROWN): the forgeable in-container benchmark can
    neither crown a challenger alone (Branch C gone) nor block a genuine val_bpb win."""
    monkeypatch.delenv("RALPH_BENCHMARK_CROWN", raising=False)
    # bench-only "win", val_bpb tied -> NO crown.
    assert score_bundle(
        val_bpb=1.50, benchmark_accuracy=0.50, king_val_bpb=1.5, king_benchmark=0.45,
        noise_floor_margin=0.013, matmul_ms=5.0, wall_clock_s=60.0,
    ).decisively_beats_king is False
    # val_bpb win with benchmark tanked -> STILL crowns (benchmark ignored).
    assert score_bundle(
        val_bpb=1.42, benchmark_accuracy=0.0, king_val_bpb=1.5, king_benchmark=0.45,
        noise_floor_margin=0.013, matmul_ms=5.0, wall_clock_s=60.0,
    ).decisively_beats_king is True


def test_nan_val_bpb_not_decisive():
    """NaN val_bpb must NOT crown a king — non-finite metrics rejected."""
    sr = score_bundle(val_bpb=float("nan"), **_BASE)
    assert sr.decisively_beats_king is False
    assert sr.quality_gain == 0.0


def test_inf_val_bpb_not_decisive():
    sr = score_bundle(val_bpb=float("inf"), **_BASE)
    assert sr.decisively_beats_king is False
    assert sr.quality_gain == 0.0


def test_nan_benchmark_not_decisive():
    sr = score_bundle(
        val_bpb=1.45,
        benchmark_accuracy=float("nan"),
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False


def test_first_submission_no_king():
    """king_val_bpb=None → no decisive comparison possible; caller treats
    is_first separately."""
    sr = score_bundle(
        val_bpb=1.45,
        benchmark_accuracy=0.5,
        king_val_bpb=None,
        king_benchmark=None,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False
    assert sr.quality_gain == 0.0


def test_v12_single_tier_no_alpha_discount():
    """ScoreReport.tier always returns 'verified' (v1.2) regardless of
    requested tier; cost_effective == cost (no α factor)."""
    sr = score_bundle(val_bpb=1.45, tier="verified", **_BASE)
    assert sr.tier == "verified"
    # cost_effective must equal cost (no α division)
    assert math.isclose(sr.compute_cost_effective, sr.compute_cost)


def test_alpha_verified_kept_for_back_compat():
    """ALPHA_VERIFIED remains importable for any external caller but
    represents the no-discount default."""
    assert ALPHA_VERIFIED == 1.0
