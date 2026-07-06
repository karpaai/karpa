"""op4b memorization-gate verdict tests, calibrated on the REAL measured numbers from
the 2026-07-06 investigation (per-window mean NLLs, nats):
  fraud king 2c3d59e3: P=3.42 F=3.81 ; GPT-2 ref: P=3.61 F=3.38  -> DD=-0.62 (memorized)
  #1593 (honest):      P=3.44 F=2.94 ; same ref                   -> DD=+0.27 (honest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pure verdict tests need only numpy; eval.memorization lazily imports torch/transformers
from eval.memorization import memorization_verdict, window_means  # noqa: E402

REF_P, REF_F = 3.61, 3.38  # public GPT-2 reference (never trained on the shard)


def _const(mean, n=200):
    return np.full(n, mean, dtype=np.float64)


def test_fraud_king_rejected():
    v = memorization_verdict(_const(3.42), _const(3.81), _const(REF_P), _const(REF_F))
    assert not v["ok"], v
    assert v["dd"] < -0.15
    assert "MEMORIZATION" in v["detail"]


def test_1593_honest_passes():
    v = memorization_verdict(_const(3.44), _const(2.94), _const(REF_P), _const(REF_F))
    assert v["ok"], v
    assert v["dd"] > 0  # finds pinned HARDER than fresh -> not memorized


def test_symmetric_honest_passes():
    # a model that ranks P vs F exactly like the reference -> DD == 0 -> pass
    v = memorization_verdict(_const(4.0), _const(3.77), _const(REF_P), _const(REF_F))
    assert v["ok"] and abs(v["dd"]) < 1e-9


def test_borderline_below_tau_passes():
    # small negative DD within noise must NOT reject at tau=0.15:
    # dd = (3.60-3.45) - (3.61-3.38) = 0.15 - 0.23 = -0.08  (|dd| < tau)
    v = memorization_verdict(_const(3.60), _const(3.45), _const(REF_P), _const(REF_F))
    assert v["ok"], v
    assert -0.15 < v["dd"] < 0


def test_concentrated_memorizer_caught_by_tail():
    # diffuse mean looks fine (DD ~ 0) but 20% of pinned windows are near-verbatim (~0 nats)
    m_p = np.concatenate([_const(0.02, 40), _const(4.5, 160)])  # mean ~3.6, matches ref-ish
    v = memorization_verdict(m_p, _const(3.4), _const(REF_P), _const(REF_F))
    assert not v["ok"], v
    assert v["tail_frac"] >= 0.2


def test_empty_skips():
    v = memorization_verdict(np.array([]), _const(3.0), _const(REF_P), _const(REF_F))
    assert v["ok"] and v.get("skipped")


def test_window_means_reshape():
    flat = np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float32)  # seq_len=2 -> windows [1,3],[2,4]
    wm = window_means(flat, 2)
    assert np.allclose(wm, [2.0, 3.0])


def test_stats_present_for_auditor():
    v = memorization_verdict(_const(3.42), _const(3.81), _const(REF_P), _const(REF_F))
    for k in ("dd", "m_adv", "r_adv", "tail_frac", "m_p", "m_f", "r_p", "r_f", "n_windows_p"):
        assert k in v
