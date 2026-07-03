"""
Scoring — the `score = quality + benchmark + stability - cost - complexity -
regression` formula from whitepaper §5.5.

v1.2 §5.4: single attested-execution tier. The two-tier credibility factor
(α = 1.0 verified / α = 0.5 unverified) is RETIRED. Every miner runs the
official Ralph container inside CC; anything without a valid attestation
chain is rejected at op2. No discount factor on the cost denominator.

For Phase 0 we keep it minimal and tunable: only quality and cost terms are
implemented; stability, complexity, regression are zero placeholders with
hooks so we can wire them in once we have multi-seed runs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# Kept for back-compat with any external caller that still imports it; v1.2
# code never multiplies by it (single attested-execution tier).
ALPHA_VERIFIED = 1.0

# A quality_gain >= DOMINANT_QUALITY_MULTIPLIER * noise_floor_margin lets a
# challenger crown via the dominant-quality branch (Branch A). The multiplier
# is intentionally generous so this branch fires only on unambiguous wins.
#
# v0.10 CHANGE: Branch A now ALSO requires benchmark_gain >= -noise_floor_margin.
# The unguarded prior form was a Goodhart vector — a miner could seed-search 200
# random init seeds, pick the best val_bpb (expected ≈ √(2 ln N) · σ ≈ 0.042 bpb,
# i.e. exactly at the 3x noise-floor threshold) and crown regardless of what
# benchmark did. The matching condition with the other two clauses removes that
# backdoor. See docs/king_criterion_review/00_RECOMMENDATION.md §6 (v0.10) and
# the published verdict at docs/direction_reframe/00_VERDICT.md §6.
DOMINANT_QUALITY_MULTIPLIER = 3.0

# Calibrated H100 matmul_ms reference for cost normalization. Measured on an
# NVIDIA H100 PCIe (float32) running the run_calibration workload (~0.51 ms). The
# previous 5.0 was a placeholder that inflated normalized H100-hours ~14x (it made
# a real ~7 wall-clock-hour run look like 102 "H100-hours"). Override per host
# with RALPH_H100_MATMUL_MS_REF (e.g. a different reference GPU).
DEFAULT_H100_MATMUL_MS_REF = 0.51


def _h100_matmul_ms_ref() -> float:
    try:
        v = float(os.environ.get("RALPH_H100_MATMUL_MS_REF", ""))
    except (TypeError, ValueError):
        return DEFAULT_H100_MATMUL_MS_REF
    return v if v > 0 else DEFAULT_H100_MATMUL_MS_REF


# bpb-points charged per normalized H100-hour in the net-score crown gate.
# Exchange rate: 1/weight = H100-hours of compute that one bpb-point of quality
# improvement "pays for". At 0.002 a typical 0.05-bpb incremental win stays
# net-positive up to ~25 H100h and is rejected beyond that — efficiency pressure
# without freezing incremental competition. Tune UP for more pressure.
DEFAULT_COMPUTE_COST_WEIGHT = 0.002


def _compute_cost_weight() -> float:
    """Weight on normalized H100-hours — the cost term in `score`, and thus in the
    net-score crown gate. Tunable via RALPH_COMPUTE_COST_WEIGHT (default
    DEFAULT_COMPUTE_COST_WEIGHT). Higher = more efficiency pressure (a wasteful run
    needs proportionally larger quality/benchmark gains to stay net-positive)."""
    try:
        v = float(os.environ.get("RALPH_COMPUTE_COST_WEIGHT", ""))
    except (TypeError, ValueError):
        return DEFAULT_COMPUTE_COST_WEIGHT
    return v if v >= 0 else DEFAULT_COMPUTE_COST_WEIGHT


def _compute_crown_gate_enabled() -> bool:
    """Net-score crown gate. On by default now the cost reference is calibrated;
    RALPH_COMPUTE_CROWN_GATE in {0,false,no,off} reverts to quality/benchmark-only
    crowning (escape hatch if the cost model ever misbehaves on mainnet)."""
    return os.environ.get("RALPH_COMPUTE_CROWN_GATE", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def get_king_rule() -> str:
    """Return the currently-active king-selection rule.

    Reads the RALPH_KING_RULE environment variable; defaults to "legacy" which
    is the v0.10-guarded version of the original score_bundle gate. The
    forthcoming "cross_scale_v1" rule (B3 of the Cross-Scale Downstream Pareto
    build) will be selected via RALPH_KING_RULE=cross_scale_v1 once it ships;
    until then this function only returns "legacy" and a request for any other
    value falls back to "legacy" with a one-line warning to stderr.
    """
    val = os.environ.get("RALPH_KING_RULE", "legacy").strip()
    if val in ("legacy", "cross_scale_v1"):
        return val
    import sys as _sys
    print(
        f"[scoring] WARNING: RALPH_KING_RULE={val!r} not recognised; "
        "falling back to 'legacy'. Valid values: legacy, cross_scale_v1.",
        file=_sys.stderr,
    )
    return "legacy"


@dataclass
class ScoreReport:
    val_bpb: float
    benchmark_accuracy: float
    quality_gain: float
    benchmark_gain: float
    compute_cost: float
    compute_cost_effective: float
    tier: str
    score: float
    king_val_bpb: float | None
    king_benchmark: float | None
    decisively_beats_king: bool


def _hours_to_normalized_h100(
    matmul_ms: float,
    wall_clock_s: float,
    h100_matmul_ms_ref: float | None = None,
) -> float:
    """Translate wall-clock into normalized H100-hours via the calibration
    benchmark's matmul timing. The miner's machine took (matmul_ms / h100_ref)
    times as long as an H100 would have, so each wall-clock hour counts as
    (h100_ref / matmul_ms) normalized H100-hours.

    Reference: h100_matmul_ms_ref is the matmul_ms on a reference H100 for the
    calibration workload (DEFAULT_H100_MATMUL_MS_REF, calibrated on H100 PCIe),
    overridable via RALPH_H100_MATMUL_MS_REF.
    """
    if h100_matmul_ms_ref is None:
        h100_matmul_ms_ref = _h100_matmul_ms_ref()
    if matmul_ms <= 0:
        return wall_clock_s / 3600.0
    speed_factor = h100_matmul_ms_ref / matmul_ms
    return (wall_clock_s / 3600.0) * speed_factor


def score_bundle(
    val_bpb: float,
    benchmark_accuracy: float,
    king_val_bpb: float | None,
    king_benchmark: float | None,
    noise_floor_margin: float,
    matmul_ms: float,
    wall_clock_s: float,
    tier: str = "verified",
    bpb_weight: float = 1.0,
    benchmark_weight: float = 1.0,
    cost_weight: float | None = None,
) -> ScoreReport:
    """
    Quality gain on val_bpb is computed as (king - challenger) since lower
    val_bpb is better. Benchmark gain is (challenger - king) since higher
    accuracy is better.

    v1.2 §5.4: single attested-execution tier. No α discount on the cost
    denominator — verification is binary, enforced at op2.

    NaN/Inf-safe (deep_review_2026-05-31 high #3): if val_bpb or
    benchmark_accuracy is non-finite the ScoreReport is returned with
    decisively_beats_king=False and quality_gain/benchmark_gain zero so the
    caller can reject without crowning a null king.
    """
    # The in-container benchmark_accuracy is blind-forgeable until the host-reduced
    # benchmark is enforced (a score=-token_id sweep beats chance on a non-content-
    # whitened active_benchmark.json). So benchmark_gain is NEUTRALIZED in the crown
    # by default: it can neither crown a challenger (Branch C) nor mask a quality
    # regression (the benchmark-no-regress guards go vacuous). val_bpb quality_gain
    # alone decides. Re-enable once the benchmark is host-reduced: RALPH_BENCHMARK_CROWN=1.
    _bench_crown = os.environ.get("RALPH_BENCHMARK_CROWN") == "1"
    finite_metrics = (
        isinstance(val_bpb, (int, float)) and math.isfinite(val_bpb)
        and isinstance(benchmark_accuracy, (int, float)) and math.isfinite(benchmark_accuracy)
    )

    if not finite_metrics or king_val_bpb is None:
        quality_gain = 0.0
        benchmark_gain = 0.0
        decisively = False
    else:
        quality_gain = (king_val_bpb - val_bpb)
        benchmark_gain = (benchmark_accuracy - (king_benchmark or 0.0))
        # Three branches, all v0.10-guarded so that no clause crowns a
        # challenger that regressed past the noise floor on the OTHER axis:
        #   Branch A: dominant quality (≥3× noise) AND benchmark-no-regress
        #   Branch B: quality > noise              AND benchmark-no-regress
        #   Branch C: benchmark > noise            AND quality-no-regress  (bench-crown only)
        # The Branch A guard (the AND ... clause) is the v0.10 Goodhart fix.
        _bg = benchmark_gain if _bench_crown else 0.0
        decisively = (
            (quality_gain >= DOMINANT_QUALITY_MULTIPLIER * noise_floor_margin
             and _bg >= -noise_floor_margin)
            or
            (quality_gain > noise_floor_margin and _bg >= -noise_floor_margin)
            or
            (_bench_crown and benchmark_gain > noise_floor_margin and quality_gain >= -noise_floor_margin)
        )

    if cost_weight is None:
        cost_weight = _compute_cost_weight()
    cost_h100h = _hours_to_normalized_h100(matmul_ms, wall_clock_s)
    # v1.2: no α factor.
    cost_effective = cost_h100h

    score = (
        bpb_weight * quality_gain
        + (benchmark_weight * benchmark_gain if _bench_crown else 0.0)
        - cost_weight * cost_effective
    )

    # Compute-aware crown gate (calibrated net-score): a challenger that beats the
    # king on quality/benchmark must ALSO be net-positive once compute is charged
    # — quality_gain + benchmark_gain must outweigh cost_weight * normalized
    # H100-hours (i.e. score > 0). Rewards efficiency, rejects runaway compute,
    # with NO hard hour cap. `decisively` is only ever True against a real
    # incumbent (genesis crowns via is_first in the caller), so the first king is
    # never blocked. Escape hatch: RALPH_COMPUTE_CROWN_GATE=0.
    if decisively and _compute_crown_gate_enabled() and score <= 0:
        decisively = False

    return ScoreReport(
        val_bpb=val_bpb,
        benchmark_accuracy=benchmark_accuracy,
        quality_gain=quality_gain,
        benchmark_gain=benchmark_gain,
        compute_cost=cost_h100h,
        compute_cost_effective=cost_effective,
        tier="verified",  # v1.2 single tier
        score=score,
        king_val_bpb=king_val_bpb,
        king_benchmark=king_benchmark,
        decisively_beats_king=decisively,
    )
