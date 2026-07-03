"""
Submission router + score aggregation + merge.

For Phase 0 we model the full lifecycle on a local "chain" (JSON-file ledger):
  - Miner submits via miner.submit; the proof bundle lands in runs/.
  - The router picks up new submissions, runs validators against each, and
    aggregates results.
  - If a submission decisively beats the current king (whitepaper §5.7), the
    router writes a merge event to the chain and updates `chain/king.json`.

Phase 0.5+ replaces this with Bittensor weight-setting + automated PR-merge.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator.scoring import score_bundle
from validator.validator import judge_submission, op_rederive_trajectory


def _chain_dir(ralph_root: Path) -> Path:
    d = ralph_root / "chain"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_king(ralph_root: Path) -> dict | None:
    path = _chain_dir(ralph_root) / "king.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_king(ralph_root: Path, king: dict) -> None:
    (_chain_dir(ralph_root) / "king.json").write_text(json.dumps(king, indent=2, sort_keys=True))


def _load_high_water(ralph_root: Path) -> dict | None:
    """The last-crowned king's quality bar, persisted separately from king.json so
    it survives a cleared/missing throne. None only at genuine genesis."""
    path = _chain_dir(ralph_root) / "high_water.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d if (isinstance(d, dict) and "val_bpb" in d) else None


def _write_high_water(ralph_root: Path, val_bpb: float, benchmark_accuracy: float) -> None:
    (_chain_dir(ralph_root) / "high_water.json").write_text(
        json.dumps(
            {"val_bpb": float(val_bpb), "benchmark_accuracy": float(benchmark_accuracy)},
            indent=2,
            sort_keys=True,
        )
    )


def _append_event(ralph_root: Path, event: dict) -> None:
    path = _chain_dir(ralph_root) / "events.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def process_submission(
    ralph_root: Path,
    proof_dir: Path,
    noise_floor_margin: float = 0.02,
) -> dict:
    """
    Full Phase-0 lifecycle for one submission:
      1. judge_submission (4 ops + hidden eval)
      2. compute score vs king
      3. emit merge or reject event
    """
    proof_dir = Path(proof_dir)
    result = judge_submission(ralph_root, proof_dir)

    if result.rejected is not None:
        event = {
            "type": "submission_rejected",
            "timestamp": time.time(),
            "miner_hotkey": result.miner_hotkey,
            "bundle_hash": result.bundle_hash,
            "reason": result.rejected.reason,
            "detail": result.rejected.detail,
            "operations": result.operations,
        }
        _append_event(ralph_root, event)
        return {"status": "rejected", "result": result.to_dict(), "event": event}

    king = _load_king(ralph_root)
    # Bar to beat: the live king if present, else the last-crowned king's
    # persisted high-water mark. This closes the gain-0 free-crown gap — a
    # challenger arriving while the throne is transiently empty must still
    # decisively beat the prior bar instead of auto-crowning.
    hwm = _load_high_water(ralph_root)
    bar = king if king else hwm
    bar_val_bpb = bar["val_bpb"] if bar else None
    bar_benchmark = bar["benchmark_accuracy"] if bar else None

    tier = result.operations.get("op2_attestation", {}).get("tier", "unverified")
    score = score_bundle(
        val_bpb=result.hidden_eval.val_bpb,
        benchmark_accuracy=result.hidden_eval.benchmark_accuracy,
        king_val_bpb=bar_val_bpb,
        king_benchmark=bar_benchmark,
        noise_floor_margin=noise_floor_margin,
        matmul_ms=result.calibration["matmul_ms"],
        wall_clock_s=result.training_summary["wall_clock_s"],
        tier=tier,
    )

    # is_first is true ONLY at genuine genesis (no king has ever been crowned);
    # a transiently-empty throne with a high-water mark must beat the bar.
    is_first = king is None and hwm is None
    accepted = score.decisively_beats_king if bar else False

    # Pre-crown proof-of-EXECUTION: re-run the first steps on CANONICAL data and require
    # the declared loss trajectory to reproduce. Run ONLY for a bundle that would take the
    # crown, so cost is O(king-changes) not O(submissions). off/shadow never block a crown
    # (shadow logs the verdict for calibration); enforce withholds the crown on a genuine
    # trajectory mismatch. Every ambiguity (disabled, no canonical data, timeout, too-few
    # points) stays fail-OPEN, so a validator glitch never wrongly withholds a crown, and a
    # failed re-derivation only PREVENTS a promotion — it never dethrones the sitting king.
    rederive_blocked = False
    if accepted or is_first:
        ok_rd, detail_rd = op_rederive_trajectory(ralph_root, proof_dir)
        result.operations["op_rederive"] = {"ok": ok_rd, "detail": detail_rd}
        if not ok_rd:
            rederive_blocked = True
            accepted = False
            is_first = False

    event = {
        "type": "submission_scored",
        "timestamp": time.time(),
        "miner_hotkey": result.miner_hotkey,
        "bundle_hash": result.bundle_hash,
        "handshake_nonce": result.handshake_nonce,
        "val_bpb": result.hidden_eval.val_bpb,
        "benchmark_accuracy": result.hidden_eval.benchmark_accuracy,
        "quality_gain": score.quality_gain,
        "benchmark_gain": score.benchmark_gain,
        "compute_cost_h100h": score.compute_cost,
        "score": score.score,
        "decisively_beats_king": score.decisively_beats_king,
        "accepted_as_king": accepted or is_first,
        "rederive_blocked": rederive_blocked,
        "operations": result.operations,
    }
    _append_event(ralph_root, event)

    if accepted or is_first:
        new_king = {
            "miner_hotkey": result.miner_hotkey,
            "bundle_hash": result.bundle_hash,
            "val_bpb": result.hidden_eval.val_bpb,
            "benchmark_accuracy": result.hidden_eval.benchmark_accuracy,
            "compute_cost_h100h": score.compute_cost,
            "crowned_at": time.time(),
            "proof_dir": str(proof_dir),
        }
        if king:
            new_king["previous_king"] = king
        _write_king(ralph_root, new_king)
        _write_high_water(
            ralph_root, new_king["val_bpb"], new_king["benchmark_accuracy"]
        )
        _append_event(ralph_root, {
            "type": "king_changed" if king else "initial_king",
            "timestamp": time.time(),
            "new_king": new_king,
        })

    if accepted or is_first:
        status = "accepted"
    elif rederive_blocked:
        status = "rederive_withheld"
    else:
        status = "below_threshold"
    return {
        "status": status,
        "result": result.to_dict(),
        "score": asdict(score),
        "event": event,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ralph-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--proof-dir", type=Path, required=True)
    p.add_argument("--noise-floor", type=float, default=0.02)
    args = p.parse_args()

    out = process_submission(args.ralph_root, args.proof_dir, args.noise_floor)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
