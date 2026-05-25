"""
Stage 5 audit — probabilistic full re-run.

The validator re-executes the miner's declared patch with the declared seed
and data order, producing an independent checkpoint. Because GPU training is
not bit-reproducible, the audit checks score-and-trajectory equivalence:

  1. The re-run's hidden-eval val_bpb must lie within the noise-floor margin
     of the miner's reported score.
  2. The re-run's loss-trajectory over the first 5-10% of steps must track
     the miner's training log within a tolerance band.

Failure on either test is treated as fraud (whitepaper §5.7).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import run_hidden_eval
from model import KarpathianBase, KarpathianConfig
from proof.runner import run_proof_test


@dataclass
class AuditResult:
    passed: bool
    miner_val_bpb: float
    audit_val_bpb: float
    bpb_diff: float
    bpb_within_margin: bool
    trajectory_match: bool
    trajectory_max_deviation: float
    trajectory_steps_checked: int
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _load_training_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_trajectory(
    miner_log: list[dict],
    audit_log: list[dict],
    fraction: float = 0.1,
    tolerance: float = 0.5,
) -> tuple[bool, float, int]:
    """
    Compare the loss-trajectory of the first `fraction` of steps. Returns
    (match, max_deviation, n_steps_checked).

    Tolerance is the maximum allowed absolute difference in loss at any step.
    On CPU this will be ~0 (bit-identical); on GPU we see divergence from
    non-deterministic kernels. A tolerance of 0.5 nats is generous enough to
    accommodate GPU variance while catching fundamentally different runs.
    """
    n = max(1, int(len(miner_log) * fraction))
    n = min(n, len(miner_log), len(audit_log))
    max_dev = 0.0
    for i in range(n):
        diff = abs(miner_log[i]["loss"] - audit_log[i]["loss"])
        max_dev = max(max_dev, diff)
    return max_dev <= tolerance, max_dev, n


def run_audit(
    karpathian_root: Path,
    submission_dir: Path,
    miner_proof_dir: Path,
    audit_out_dir: Path,
    noise_floor_margin: float = 0.05,
    trajectory_fraction: float = 0.1,
    trajectory_tolerance: float = 0.5,
) -> AuditResult:
    """
    Full Stage 5 audit. Re-runs the proof test from scratch, compares with
    the miner's submitted results.
    """
    miner_proof = Path(miner_proof_dir)
    miner_manifest = json.loads((miner_proof / "bundle_manifest.json").read_text())
    miner_state = json.loads((miner_proof / "training" / "final_state.json").read_text())
    miner_log = _load_training_log(miner_proof / "training" / "training_log.jsonl")

    # Re-run the proof test with the same submission.
    audit_bundle = run_proof_test(
        karpathian_root=karpathian_root,
        submission_dir=submission_dir,
        out_dir=audit_out_dir,
    )

    # Hidden-eval on the audit checkpoint.
    ckpt = torch.load(audit_bundle.checkpoint_path, weights_only=False, map_location="cpu")
    saved = ckpt["config"]
    cfg = KarpathianConfig(
        vocab_size=saved["vocab_size"], dim=saved["dim"],
        n_layers=saved["n_layers"], n_heads=saved["n_heads"],
        head_dim=saved["head_dim"], ffn_mult=saved["ffn_mult"],
        max_seq_len=saved["max_seq_len"],
    )
    model = KarpathianBase(cfg)
    model.load_state_dict(ckpt["model"])
    if torch.cuda.is_available():
        model = model.cuda()
    eval_result = run_hidden_eval(
        model, karpathian_root / "eval" / "private",
        seq_len=cfg.max_seq_len // 2,
    )

    # Hidden-eval on the miner's checkpoint for comparison.
    miner_ckpt = torch.load(
        miner_proof / "training" / "checkpoint.pt",
        weights_only=False, map_location="cpu",
    )
    miner_model = KarpathianBase(cfg)
    miner_model.load_state_dict(miner_ckpt["model"])
    if torch.cuda.is_available():
        miner_model = miner_model.cuda()
    miner_eval = run_hidden_eval(
        miner_model, karpathian_root / "eval" / "private",
        seq_len=cfg.max_seq_len // 2,
    )

    miner_bpb = miner_eval.val_bpb
    audit_bpb = eval_result.val_bpb
    bpb_diff = abs(miner_bpb - audit_bpb)
    bpb_ok = bpb_diff <= noise_floor_margin

    # Trajectory check.
    audit_log = _load_training_log(audit_bundle.training_log_path)
    traj_ok, max_dev, n_steps = check_trajectory(
        miner_log, audit_log,
        fraction=trajectory_fraction,
        tolerance=trajectory_tolerance,
    )

    passed = bpb_ok and traj_ok
    detail_parts = []
    if not bpb_ok:
        detail_parts.append(
            f"val_bpb mismatch: miner={miner_bpb:.4f} audit={audit_bpb:.4f} "
            f"diff={bpb_diff:.4f} > margin={noise_floor_margin:.4f}"
        )
    if not traj_ok:
        detail_parts.append(
            f"trajectory divergence: max_dev={max_dev:.4f} > tolerance={trajectory_tolerance:.4f} "
            f"over first {n_steps} steps"
        )
    if passed:
        detail_parts.append(
            f"audit passed: bpb_diff={bpb_diff:.4f} traj_max_dev={max_dev:.4f}"
        )

    return AuditResult(
        passed=passed,
        miner_val_bpb=miner_bpb,
        audit_val_bpb=audit_bpb,
        bpb_diff=bpb_diff,
        bpb_within_margin=bpb_ok,
        trajectory_match=traj_ok,
        trajectory_max_deviation=max_dev,
        trajectory_steps_checked=n_steps,
        detail="; ".join(detail_parts),
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--karpathian-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--submission-dir", type=Path, required=True)
    p.add_argument("--miner-proof-dir", type=Path, required=True)
    p.add_argument("--audit-out-dir", type=Path, required=True)
    p.add_argument("--noise-floor-margin", type=float, default=0.05)
    args = p.parse_args()

    result = run_audit(
        karpathian_root=args.karpathian_root,
        submission_dir=args.submission_dir,
        miner_proof_dir=args.miner_proof_dir,
        audit_out_dir=args.audit_out_dir,
        noise_floor_margin=args.noise_floor_margin,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
