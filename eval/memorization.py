"""op4b — held-out MEMORIZATION gate (difference-in-differences).

Catches the training-time held-out-memorization fraud class (2026-07-06 fraud king
2c3d59e3: a canonical checkpoint that passed every gate but had TRAINED ON the
leaked eval shard -> artificially low val_bpb, true generalization worse than GPT-2).
Compute/arch/attestation gates are blind to this: the fraud is in the loss surface,
not the code path, so HOSB (eval-time answer-blanking) does not touch it either.

The signal (the exact control that unmasked the fraud king): a memorizer scores the
PINNED shard P (what op4 crowns on) anomalously LOW relative to FRESH disjoint text F,
compared to how a CLEAN reference model R (public GPT-2, never trained on P) ranks the
same two. Difference-in-differences cancels "P is just intrinsically easier/harder text":

    DD = (meanNLL_M(P) - meanNLL_M(F)) - (meanNLL_R(P) - meanNLL_R(F))

Honest model: DD >= 0 (it finds P about as hard as, or harder than, F -- same as R).
Memorizer:    DD << 0 (it finds P much easier than F, against R's verdict).
  Fraud king measured: DD = (3.42-3.81) - (3.61-3.38) = -0.62.
  #1593 (honest):      DD = (3.44-2.94) - (3.61-3.38) = +0.27.

Plus a TAIL statistic for CONCENTRATED (near-verbatim) recall that a diffuse memorizer
can hide from the mean: fraction of P-windows the model predicts near-perfectly.

This module keeps `memorization_verdict` pure (numpy only, no torch) so it is unit-
tested against the real measured numbers; the GPU harness is lazily imported.
"""
from __future__ import annotations

import numpy as np

# Defaults; overridable via RALPH_MEMGATE_TAU / _TAU_TAIL / _TAIL_FLOOR env in the caller.
DEFAULT_TAU = 0.15        # DD reject threshold (nats). Fraud king -0.62; honest >= +0.2. Wide margin.
DEFAULT_TAU_TAIL = 0.05   # max fraction of near-verbatim windows before reject.
DEFAULT_TAIL_FLOOR = 0.5  # a window mean-NLL below this (nats) is near-verbatim recall.


def window_means(flat_nlls: np.ndarray, seq_len: int) -> np.ndarray:
    """Reshape a flat per-token NLL array (len == n_windows*seq_len, as produced by
    eval.val_bpb.per_position_nlls) into per-window mean NLL."""
    flat = np.asarray(flat_nlls, dtype=np.float64)
    n = (len(flat) // seq_len) * seq_len
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return flat[:n].reshape(-1, seq_len).mean(axis=1)


def memorization_verdict(
    m_on_p: np.ndarray,
    m_on_f: np.ndarray,
    r_on_p: np.ndarray,
    r_on_f: np.ndarray,
    *,
    tau: float = DEFAULT_TAU,
    tau_tail: float = DEFAULT_TAU_TAIL,
    tail_floor: float = DEFAULT_TAIL_FLOOR,
) -> dict:
    """Pure diff-in-diff + tail verdict on per-window mean NLLs.

    Args are per-window mean-NLL arrays: submission model M and clean reference R,
    each on the pinned shard P and the fresh disjoint control F.
    Returns a dict: ok + every stat (so a CPU auditor can reproduce the decision).
    """
    for name, a in (("m_on_p", m_on_p), ("m_on_f", m_on_f), ("r_on_p", r_on_p), ("r_on_f", r_on_f)):
        if len(a) == 0:
            return {"ok": True, "skipped": True, "detail": f"memgate skipped: empty {name}"}
    m_p, m_f = float(np.mean(m_on_p)), float(np.mean(m_on_f))
    r_p, r_f = float(np.mean(r_on_p)), float(np.mean(r_on_f))
    m_adv = m_p - m_f          # M's pinned-vs-fresh gap (negative => M finds P easier)
    r_adv = r_p - r_f          # clean reference's intrinsic pinned-vs-fresh gap
    dd = m_adv - r_adv
    tail_frac = float(np.mean(np.asarray(m_on_p, dtype=np.float64) < tail_floor))
    reject_dd = dd < -tau
    reject_tail = tail_frac > tau_tail
    ok = not (reject_dd or reject_tail)
    reasons = []
    if reject_dd:
        reasons.append(
            f"diff-in-diff {dd:.3f} < -{tau}: scores the pinned eval shard {-dd:.3f} nats "
            f"easier than a clean reference would rank it vs fresh text — consistent with "
            f"having TRAINED ON the held-out (memorization)"
        )
    if reject_tail:
        reasons.append(
            f"tail_frac {tail_frac:.3f} > {tau_tail}: near-verbatim recall on "
            f"{tail_frac*100:.1f}% of pinned windows (mean-NLL < {tail_floor})"
        )
    detail = (
        f"memgate ok: dd={dd:.3f} (M {m_adv:+.3f} vs ref {r_adv:+.3f}), tail={tail_frac:.3f}"
        if ok else "MEMORIZATION REJECT: " + "; ".join(reasons)
    )
    return {
        "ok": ok, "dd": round(dd, 4), "m_adv": round(m_adv, 4), "r_adv": round(r_adv, 4),
        "tail_frac": round(tail_frac, 4), "m_p": round(m_p, 4), "m_f": round(m_f, 4),
        "r_p": round(r_p, 4), "r_f": round(r_f, 4),
        "tau": tau, "tau_tail": tau_tail, "n_windows_p": int(len(m_on_p)), "detail": detail,
    }


# ---------------------------------------------------------------------------
# GPU harness (lazily imports torch/transformers so the verdict above stays pure).
# ---------------------------------------------------------------------------

class _GPT2Ref:
    """Adapter so eval.val_bpb.per_position_nlls (which calls `logits, _ = model(x)`)
    can score public GPT-2 (same gpt2 BPE / 50257 vocab as the held-out)."""

    def __init__(self, device):
        import torch
        from transformers import GPT2LMHeadModel
        self.m = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval().to(torch.bfloat16)
        self.device = device

    def eval(self):
        return self

    def parameters(self):
        return self.m.parameters()

    def __call__(self, inp):
        return self.m(inp).logits, None


def reference_window_means(p_tokens, f_tokens, seq_len, device, batch_size=16):
    """Per-window mean NLLs of the clean GPT-2 reference on P and F. Deterministic in
    (P, F, seq_len) — the caller caches these keyed by the shard fingerprints so GPT-2
    runs once per shard-pair, not once per submission."""
    from eval.val_bpb import per_position_nlls
    ref = _GPT2Ref(device)
    r_p = window_means(per_position_nlls(ref, p_tokens, seq_len, batch_size, device), seq_len)
    r_f = window_means(per_position_nlls(ref, f_tokens, seq_len, batch_size, device), seq_len)
    return r_p, r_f


def model_window_means(model, tokens, seq_len, device, batch_size=16):
    """Per-window mean NLLs of the submission model on a token stream."""
    from eval.val_bpb import per_position_nlls
    return window_means(per_position_nlls(model, tokens, seq_len, batch_size, device), seq_len)


def run_memorization_gate(
    model, p_tokens, f_tokens, seq_len, device, *,
    ref_window_means=None, batch_size=16,
    tau=DEFAULT_TAU, tau_tail=DEFAULT_TAU_TAIL, tail_floor=DEFAULT_TAIL_FLOOR,
):
    """End-to-end op4b: score M on the pinned shard P and the fresh control F, obtain the
    clean GPT-2 reference on the same (computed here, or passed pre-cached as
    (r_on_p, r_on_f)), and return the memorization_verdict dict.
    ref_window_means lets the caller cache GPT-2 across submissions (it never changes for
    a fixed shard-pair)."""
    m_on_p = model_window_means(model, p_tokens, seq_len, device, batch_size)
    m_on_f = model_window_means(model, f_tokens, seq_len, device, batch_size)
    if ref_window_means is None:
        r_on_p, r_on_f = reference_window_means(p_tokens, f_tokens, seq_len, device, batch_size)
    else:
        r_on_p, r_on_f = ref_window_means
    return memorization_verdict(
        m_on_p, m_on_f, r_on_p, r_on_f, tau=tau, tau_tail=tau_tail, tail_floor=tail_floor)
