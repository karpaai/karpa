"""op4 must FAIL CLOSED on an arch-divergent checkpoint (one that can't load into the
validator's canonical model) rather than falling back to the miner's own eval code.

The exploit (Kaizen0304, 2026-07-05): a patch adds value_embeddings to model/_v4skip.py,
so the checkpoint carries params the canonical RalphBase lacks -> load_state_dict fails ->
op4 fell back to _patched_hidden_eval, which runs the MINER's model code to compute the
crown metric. With HOSB off that eval is not answer-blanked, so the crown was never
independently verified. Fail-closed refuses to crown a model the validator can't build.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401,E402
import validator.validator as V  # noqa: E402


def _force_arch_mismatch(tmp_path, monkeypatch):
    (tmp_path / "training").mkdir(parents=True, exist_ok=True)
    (tmp_path / "training" / "checkpoint.pt").write_bytes(b"x")  # must exist
    monkeypatch.setenv("RALPH_SANDBOX", "0")
    saved = {"vocab_size": 50257, "dim": 1024, "n_layers": 16, "n_heads": 16,
             "head_dim": 64, "ffn_mult": 2.6875, "max_seq_len": 1024}
    monkeypatch.setattr(V, "_safe_load_checkpoint_config", lambda p: saved)
    monkeypatch.setattr(V, "_safe_load_checkpoint_weights", lambda p: {"value_embed.weight": 1})

    class _FakeModel:
        def __init__(self, cfg):
            pass

        def load_state_dict(self, sd):
            raise RuntimeError("Unexpected key(s) in state_dict: value_embed.weight; size mismatch")

    monkeypatch.setattr(V, "RalphBase", _FakeModel)
    monkeypatch.setattr(V, "_is_state_dict_shape_mismatch", lambda e: True)
    calls = []
    monkeypatch.setattr(V, "_patched_hidden_eval",
                        lambda *a, **k: (calls.append(1) or (True, "ran MINER code", object())))
    return calls


def test_arch_divergent_fails_closed(tmp_path, monkeypatch):
    calls = _force_arch_mismatch(tmp_path, monkeypatch)
    monkeypatch.delenv("RALPH_ALLOW_PATCHED_EVAL", raising=False)
    ok, detail, res = V._legacy_hidden_eval(tmp_path, tmp_path)
    assert not ok and res is None
    assert "arch-divergent" in detail and "not independently verifiable" in detail
    assert calls == []  # the miner's eval code was NEVER run


def test_override_restores_patched_eval(tmp_path, monkeypatch):
    # RALPH_ALLOW_PATCHED_EVAL=1 (testnet/debug or once HOSB is enforced) restores fallback.
    calls = _force_arch_mismatch(tmp_path, monkeypatch)
    monkeypatch.setenv("RALPH_ALLOW_PATCHED_EVAL", "1")
    ok, detail, res = V._legacy_hidden_eval(tmp_path, tmp_path)
    assert ok and calls == [1]  # fallback ran
