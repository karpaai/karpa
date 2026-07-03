"""The trainedness guard must reject the random-init / log-mismatched checkpoints
that slipped through (e.g. the uid155 fraud) while never rejecting a real model."""
from __future__ import annotations

import math

import pytest

from validator.integrity import (
    check_canonical_data_source,
    check_checkpoint_not_blocklisted,
    check_checkpoint_trained,
    check_compute_budget,
    check_compute_plausibility,
    check_model_size,
    check_recipe_config_matches_proof,
    check_training_timing,
    compare_loss_trajectory,
    nats_per_token_from_bpb,
)


# --- compute-budget cap (fair 1x H100-class contest) -------------------------
def test_budget_rejects_over_cap():
    # martyniukr (H200): wall 19321s -> ~7.96 norm-H100h at the fixed 0.344 ref.
    ok, reason = check_compute_budget({"wall_clock_s": 19321.0}, {"matmul_ms": 0.344}, budget=5.0, mm_ref_hopper=0.344)
    assert not ok and "over compute budget" in reason


def test_budget_accepts_under_cap():
    # revert king c48ad59d: wall 7117s -> ~2.93 norm-H100h.
    assert check_compute_budget({"wall_clock_s": 7117.0}, {"matmul_ms": 0.344}, budget=5.0, mm_ref_hopper=0.344)[0]


def test_budget_spoofproof_matmul_ms():
    # A fabricated large matmul_ms must NOT shrink the cost under the cap (fixed ref).
    over = {"wall_clock_s": 19321.0}
    assert not check_compute_budget(over, {"matmul_ms": 5.0}, budget=5.0, mm_ref_hopper=0.344)[0]


def test_budget_skips_without_wall():
    assert check_compute_budget({"wall_clock_s": 0}, {}, budget=5.0, mm_ref_hopper=0.344)[0]


# --- model-size cap (fair fixed-arch contest) --------------------------------
def test_model_size_rejects_big_undertrained():
    # taohunter251 / gradient-artist: 1.2B model that fits under the compute budget.
    ok, reason = check_model_size({"n_params": 1_216_087_552}, max_n_params=400_000_000)
    assert not ok and "model too large" in reason


def test_model_size_accepts_canonical_254m():
    assert check_model_size({"n_params": 253_874_184}, max_n_params=400_000_000)[0]


def test_model_size_skips_without_n_params():
    assert check_model_size({}, max_n_params=400_000_000)[0]
    assert check_model_size({"n_params": 0}, max_n_params=400_000_000)[0]


# --- Hopper-only GPU arch bind (proof.gpu_arch) ------------------------------
def _fake_jwt(payload: dict) -> str:
    import base64
    import json as _j
    seg = base64.urlsafe_b64encode(_j.dumps(payload).encode()).rstrip(b"=").decode()
    return "aGRy." + seg + ".c2ln"  # header.<payload>.sig (sig unchecked here; op2 verifies the real sig)


def _gpu_token(hwmodel: str, *, break_digest: bool = False) -> str:
    import hashlib
    import json as _j
    filler = "x" * 80  # keep each JWT > 80 chars so _all_jwts picks it up
    detached = _fake_jwt({"hwmodel": hwmodel, "eat_nonce": "n", "pad": filler})
    digest = "deadbeef" * 8 if break_digest else hashlib.sha256(detached.encode()).hexdigest()
    wrapper = _fake_jwt({"submods": {"GPU-0": ["DIGEST", ["SHA-256", digest]]}, "eat_nonce": "n", "pad": filler})
    return _j.dumps([wrapper, detached])


def test_arch_allows_gh100():
    from proof.gpu_arch import verify_gpu_arch_allowed
    ok, _r, die = verify_gpu_arch_allowed({"epochs": [{"gpu_token": _gpu_token("GH100")}]}, allow={"GH100"})
    assert ok and die == "GH100"


def test_arch_denies_blackwell():
    from proof.gpu_arch import verify_gpu_arch_allowed
    ok, _r, die = verify_gpu_arch_allowed({"epochs": [{"gpu_token": _gpu_token("GB20X")}]}, allow={"GH100"})
    assert not ok and die == "GB20X"


def test_arch_denies_digest_mismatch():
    # A swapped-in hwmodel EAT whose sha256 != the signed wrapper commitment.
    from proof.gpu_arch import verify_gpu_arch_allowed
    assert not verify_gpu_arch_allowed({"epochs": [{"gpu_token": _gpu_token("GH100", break_digest=True)}]})[0]


def test_arch_denies_missing_token():
    from proof.gpu_arch import verify_gpu_arch_allowed
    assert not verify_gpu_arch_allowed({"epochs": [{"gpu_token": None}]}, allow={"GH100"})[0]
    assert not verify_gpu_arch_allowed({"epochs": []}, allow={"GH100"})[0]


# --- training-timing gate (anti off-protocol) --------------------------------
def test_timing_rejects_run_longer_than_canonical_code_age():
    # aa427cd1/6a25bdc8: ~6.95h wall_clock but the canonical code was ~2h old.
    fs = {"wall_clock_s": 25011.0}
    ok, reason = check_training_timing(fs, canonical_code_epoch=1_000_000.0, now_epoch=1_000_000.0 + 7380, slack_s=7200)
    assert not ok and "off-protocol" in reason


def test_timing_accepts_run_within_code_age():
    fs = {"wall_clock_s": 21600.0}  # 6h, code existed 8h
    assert check_training_timing(fs, canonical_code_epoch=1e6, now_epoch=1e6 + 8 * 3600, slack_s=7200)[0]


def test_timing_skips_without_wall_clock_or_epoch():
    assert check_training_timing({"wall_clock_s": 0}, canonical_code_epoch=1e6, now_epoch=1e6 + 10, slack_s=7200)[0]
    assert check_training_timing({"wall_clock_s": 9e9}, canonical_code_epoch=None, now_epoch=1e6, slack_s=7200)[0]


# --- fraud-checkpoint blocklist ----------------------------------------------
def test_blocklist_rejects_known_fraud_checkpoint():
    fraud = "f06a9090548978a1987c2b3ada48746348d8134e68f598e0be982e4d5f26f7ab"
    ok, reason = check_checkpoint_not_blocklisted(fraud, {fraud})
    assert not ok and "blocklisted" in reason


def test_blocklist_passes_unknown_and_none():
    assert check_checkpoint_not_blocklisted("d" * 64, {"a" * 64})[0]
    assert check_checkpoint_not_blocklisted(None, {"a" * 64})[0]
    assert check_checkpoint_not_blocklisted("a" * 64, set())[0]


# --- pre-crown re-derivation trajectory compare ------------------------------
_HONEST = [(0, 11.02), (50, 6.77), (100, 5.78), (150, 5.20)]


def test_rederive_accepts_matching_trajectory():
    assert compare_loss_trajectory(_HONEST, [(0, 11.03), (50, 6.80), (100, 5.75), (150, 5.24)])[0]


def test_rederive_rejects_fabricated_trajectory():
    ok, reason = compare_loss_trajectory(_HONEST, [(0, 11.0), (50, 8.9), (100, 8.1), (150, 7.6)])
    assert not ok and "trajectory mismatch" in reason


def test_rederive_rejects_offprotocol_faster_drop():
    # A different (e.g. bigger) off-protocol model / different data drops differently.
    ok, reason = compare_loss_trajectory(_HONEST, [(0, 11.01), (50, 5.1), (100, 4.0), (150, 3.4)])
    assert not ok


def test_rederive_rejects_step0_fingerprint_mismatch():
    ok, reason = compare_loss_trajectory(_HONEST, [(0, 9.8), (50, 6.8), (100, 5.8), (150, 5.2)])
    assert not ok and "step 0" in reason


def test_rederive_tolerates_one_noisy_point():
    assert compare_loss_trajectory(_HONEST, [(0, 11.02), (50, 6.77), (100, 6.30), (150, 5.20)])[0]


def test_rederive_needs_min_points():
    assert not compare_loss_trajectory(_HONEST, [(0, 11.0)])[0]

VOCAB = 50257
RANDOM_NATS = math.log(VOCAB)  # ~10.82


# --- compute-plausibility: anti compute-gaming -------------------------------
H100 = {"gpu_name": "NVIDIA H100 80GB HBM3"}


def test_rejects_fabricated_compute_the_5ctaoqf1_king():
    # 5.557B tokens in 6788s on ONE H100 => 818k tok/s => ~126% MFU = impossible.
    fs = {"tokens_seen": 5_557_452_800, "wall_clock_s": 6787.88, "n_params": 253_874_184}
    ok, reason = check_compute_plausibility(fs, H100)
    assert not ok and "fabricated compute" in reason and "MFU" in reason


def test_accepts_a_real_30h_run():
    fs = {"tokens_seen": 5_557_452_800, "wall_clock_s": 109_000, "n_params": 253_874_184}  # ~51k tok/s
    assert check_compute_plausibility(fs, H100)[0]


def test_accepts_an_optimized_run_under_the_ceiling():
    fs = {"tokens_seen": 5_557_452_800, "wall_clock_s": 22_000, "n_params": 253_874_184}  # ~250k tok/s, ~39% MFU
    assert check_compute_plausibility(fs, H100)[0]


def test_incomplete_training_summary_is_skipped_not_rejected():
    assert check_compute_plausibility({"tokens_seen": 0, "wall_clock_s": 0}, {})[0]
    assert check_compute_plausibility({}, None)[0]


def test_unknown_gpu_uses_fastest_peak_to_avoid_false_reject():
    fs = {"tokens_seen": 5_557_452_800, "wall_clock_s": 22_000, "n_params": 253_874_184}
    assert check_compute_plausibility(fs, {"gpu_name": "Some Future GPU"})[0]


# --- declared-recipe-matches-proof -------------------------------------------
def test_rejects_config_step_mismatch_the_5ctaoqf1_king():
    patch = '+++ b/configs/muon_wsd_qknorm_b20593.json\n+{\n+  "total_steps": 40000,\n+  "qk_norm": true\n+}\n'
    ok, reason = check_recipe_config_matches_proof(patch, {"steps": 10600})
    assert not ok and "mismatch" in reason


def test_accepts_matching_config_steps():
    patch = '+++ b/configs/run.json\n+{\n+  "total_steps": 10600\n+}\n'
    assert check_recipe_config_matches_proof(patch, {"steps": 10600})[0]


def test_config_match_skips_when_no_config_or_no_steps():
    assert check_recipe_config_matches_proof("+++ b/model/x.py\n+x = 1\n", {"steps": 10600})[0]
    assert check_recipe_config_matches_proof('+++ b/configs/c.json\n+{"total_steps": 5}\n', {})[0]


# --- canonical data source (anti data-lock-bypass) ---------------------------
def test_accepts_container_mount_absolute_data_path():
    # The canonical runner pins --manifest/--data-base-dir to the RESOLVED-ABSOLUTE
    # container path (proof/runner.py); train.py records it verbatim. Different CC
    # containers mount the recipe at different roots (/workspace, /dstack, /home/...),
    # all ending in the canonical .../data tree — that's the honest bundle (the
    # runner's own pin), not a swap. They fold to the relative tail and are accepted;
    # rejecting them was the miner-reported op1 breakage.
    for mp in (
        "/workspace/recipe/data/data_manifest.json",
        "/dstack/persistent/canon/recipe/data/data_manifest.json",
        "/home/root/diony/recipe/data/data_manifest.json",
    ):
        assert check_canonical_data_source({"config": {"manifest_path": mp}})[0], mp
    assert check_canonical_data_source({"config": {"data_base_dir": "/workspace/recipe/data"}})[0]


def test_rejects_mnt_data_base_dir():
    # No canonical "data" segment -> stays absolute -> rejected.
    assert not check_canonical_data_source({"config": {"data_base_dir": "/mnt/scratch/SN40/data_50b"}})[0]


def test_rejects_absolute_path_outside_data_tree():
    for mp in ("/home/attacker/evil.bin", "/mnt/scratch/other.json"):
        assert not check_canonical_data_source({"config": {"manifest_path": mp}})[0], mp


def test_rejects_any_absolute_or_escaping_path():
    # Allowlist: nothing absolute/escaping passes, no matter the prefix.
    for p in ("/anything/at/all/manifest.json", "~/data/manifest.json", "../../escape/manifest.json"):
        assert not check_canonical_data_source({"config": {"manifest_path": p}})[0], p


def test_accepts_canonical_relative_data_path_7fd43cef():
    fs = {"config": {"manifest_path": "data/data_manifest.json", "data_base_dir": "data"}}
    assert check_canonical_data_source(fs)[0]


def test_data_source_skips_when_no_config():
    assert check_canonical_data_source({})[0]
    assert check_canonical_data_source({"config": {}})[0]


def test_rejects_the_uid155_random_king():
    # Measured in the incident: ~11.0 nats/token, log claimed final_loss 3.05.
    ok, reason = check_checkpoint_trained(11.0, VOCAB, claimed_final_loss=3.0496)
    assert not ok
    assert "untrained" in reason


def test_rejects_random_even_without_a_claimed_loss():
    ok, reason = check_checkpoint_trained(RANDOM_NATS, VOCAB)
    assert not ok and "untrained" in reason


def test_accepts_a_real_trained_model():
    # A legit king sits at val_bpb ~1.3-1.6 -> ~3.6-4.4 nats/token.
    for val_bpb in (1.306, 1.336, 1.581):
        nats = nats_per_token_from_bpb(val_bpb, bytes_per_token=4.0)
        ok, reason = check_checkpoint_trained(nats, VOCAB, claimed_final_loss=3.0496)
        assert ok, f"false-rejected a real model (val_bpb={val_bpb}, nats={nats:.2f}): {reason}"


def test_catches_subtle_log_mismatch_below_random():
    # Not fully random (7 nats), but the log claims a much better 2.0 -> the
    # scored checkpoint clearly isn't from the declared run.
    ok, reason = check_checkpoint_trained(7.0, VOCAB, claimed_final_loss=2.0)
    assert not ok and "mismatch" in reason


def test_generous_to_normal_train_test_gap():
    # Held-out modestly worse than training must NOT trip the mismatch check.
    ok, _ = check_checkpoint_trained(4.4, VOCAB, claimed_final_loss=3.05)
    assert ok


def test_bpb_inversion_roundtrips():
    nats = nats_per_token_from_bpb(1.5, 4.0)
    assert nats == pytest.approx(1.5 * math.log(2) * 4.0)


def test_rejects_non_finite_and_bad_vocab():
    assert not check_checkpoint_trained(float("nan"), VOCAB)[0]
    assert not check_checkpoint_trained(3.5, 1)[0]


# --- patch scan: manifest regeneration (mechanism-based, path-agnostic) -------
def test_patch_scan_flags_build_manifest_regen():
    from proof.runner import scan_diff_for_exploit_patterns
    patch = (
        "+++ b/recipe/train.py\n"
        "+        from data.manifest import build_manifest\n"
        "+        build_manifest('x', 'gpt2', 50257, 'uint16', shards, base).write(_mpath)\n"
    )
    assert any("data manifest at runtime" in r for r, _ in scan_diff_for_exploit_patterns(patch))


def test_patch_scan_clean_model_change_passes():
    from proof.runner import scan_diff_for_exploit_patterns
    patch = "+++ b/model/gpt.py\n+    self.norm = RMSNorm(dim)\n+    x = self.norm(x)\n"
    assert scan_diff_for_exploit_patterns(patch) == []
