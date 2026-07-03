"""
Benchmark mix scoring — placeholder for Phase 0.

The whitepaper specifies ~1500 examples drawn from a held-out benchmark mix
(MMLU/HellaSwag/ARC subsets + code/math). For Phase 0 we ship a tiny
placeholder set so the validator scoring pipeline has something to call into;
real benchmarks land in Phase 0.5 once we plug in lm-evaluation-harness or
build our own held-out mix.

The placeholder computes a stable, model-quality-correlated score on a small
synthetic completion task: given a short context, the model should rank the
correct next-token continuation above k random distractors. This isn't real
benchmark accuracy — it's a stand-in that varies with model quality so the
end-to-end pipeline has a non-trivial signal to score against.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_benchmark_score(
    model: torch.nn.Module,
    examples: list[dict],
    device: torch.device | None = None,
) -> dict:
    """
    Each example is {context_ids: list[int], target_id: int, distractors: list[int]}.
    Score = fraction where target_id has highest log-prob under model among
    (target_id + distractors).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ex in examples:
            ctx = torch.tensor([ex["context_ids"]], dtype=torch.long, device=device)
            logits, _ = model(ctx)
            last_logits = logits[0, -1]
            candidates = [ex["target_id"]] + list(ex["distractors"])
            scores = last_logits[candidates]
            best = int(scores.argmax().item())
            if best == 0:  # index 0 == target
                correct += 1
            total += 1
    accuracy = correct / max(total, 1)
    return {"benchmark_accuracy": accuracy, "n_examples": total, "n_correct": correct}


def make_placeholder_examples(
    n: int = 50, seed: int = 7777, vocab_size: int = 50257, n_candidates: int = 5
) -> list[dict]:
    """Generate stable placeholder examples for the Phase 0 hidden-eval set.

    CONTENT-WHITENED: the target and the distractors are drawn from ONE
    exchangeable pool (a single `rng.choice` without replacement) and the target
    slot is chosen uniformly. So the candidate token-ids share the same marginal
    and NO monotone function of the token-id (e.g. "pick the smallest id") can
    distinguish the answer — a host-reduced benchmark on a non-whitened file is
    still forgeable from the id distribution (the deployed file must be
    regenerated with this generator).
    """
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n):
        ctx_len = int(rng.integers(8, 24))
        context_ids = rng.integers(0, vocab_size, size=ctx_len).tolist()
        cands = rng.choice(vocab_size, size=n_candidates, replace=False)
        target_slot = int(rng.integers(0, n_candidates))
        target_id = int(cands[target_slot])
        distractors = [int(c) for i, c in enumerate(cands) if i != target_slot]
        examples.append({
            "context_ids": context_ids,
            "target_id": target_id,
            "distractors": distractors,
        })
    return examples


def benchmark_blind_forgeable(examples: list[dict], sigma: float = 3.0) -> tuple[bool, str]:
    """Detect a non-content-whitened (blind-forgeable) benchmark file.

    compute_benchmark_score ranks [target_id] + distractors by the model's logits.
    If the correct target is distinguishable from the distractors by a monotone
    function of the token-id alone (e.g. targets are systematically small while
    distractors are uniform), a model that never learned anything wins by scoring
    `-token_id` — the exact 0.785-vs-0.2 break found on the deployed file. A
    content-whitened file (make_placeholder_examples: target + distractors from one
    exchangeable rng.choice pool, uniform target slot) makes every blind monotone
    score chance-level.

    Returns (forgeable, reason). forgeable=True -> the file must be regenerated
    with make_placeholder_examples before it can gate the crown.
    """
    import numpy as np

    if not examples:
        return False, "empty benchmark (nothing to forge)"
    n = len(examples)
    k = 1 + len(examples[0].get("distractors", []))
    if k <= 1:
        return False, "single-candidate benchmark"
    chance = 1.0 / k
    sd = (chance * (1.0 - chance) / n) ** 0.5
    bar = chance + sigma * sd
    worst = 0.0
    worst_label = ""
    for sign, label in ((-1.0, "-id"), (1.0, "+id")):
        correct = 0
        for e in examples:
            cands = [e["target_id"]] + list(e["distractors"])
            if int(np.argmax([sign * float(t) for t in cands])) == 0:
                correct += 1
        acc = correct / n
        if acc > worst:
            worst, worst_label = acc, label
    if worst > bar:
        return True, (
            f"blind-forgeable benchmark: score={worst_label} sweep scores {worst:.3f} "
            f"> chance {chance:.3f} + {sigma:.0f}sigma ({bar:.3f}); regenerate "
            f"active_benchmark.json with the content-whitened make_placeholder_examples"
        )
    return False, f"benchmark not blind-forgeable (worst blind ±id = {worst:.3f} <= {bar:.3f})"
