"""The blind-forgeability detector must flag an id-biased benchmark (the deployed
0.785 break) and pass the content-whitened generator."""
from __future__ import annotations

import numpy as np

from eval.benchmark import benchmark_blind_forgeable, make_placeholder_examples


def test_whitened_generator_not_forgeable():
    ex = make_placeholder_examples(n=300, seed=1, n_candidates=5)
    forgeable, reason = benchmark_blind_forgeable(ex)
    assert not forgeable, reason


def test_id_biased_benchmark_is_flagged():
    # Targets systematically small, distractors uniform-large -> blind score=-id wins
    # (the exact deployed break: target mean ~4550 vs distractor mean ~24917).
    rng = np.random.default_rng(0)
    ex = [
        {
            "context_ids": [1, 2, 3],
            "target_id": int(rng.integers(0, 5000)),
            "distractors": [int(rng.integers(20000, 50000)) for _ in range(4)],
        }
        for _ in range(300)
    ]
    forgeable, reason = benchmark_blind_forgeable(ex)
    assert forgeable and "blind-forgeable" in reason


def test_empty_and_single_candidate_are_safe():
    assert not benchmark_blind_forgeable([])[0]
    assert not benchmark_blind_forgeable([{"target_id": 1, "distractors": []}])[0]
