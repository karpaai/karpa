"""
Validator client. Four cheap operations per submission (§5.4), plus the
audit decision.

  Operation 1: diff scan + bundle integrity
  Operation 2: attestation chain verification
  Operation 3: training-log plausibility
  Operation 4: hidden-eval inference

Phase 0 attestation is mock (HMAC-signed JSON). Phase 0.5+ swaps in real
TDX + nvtrust quote verification — only proof.mock_attest.verify_mock_attestation
changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import RalphBase, RalphConfig

from eval import HiddenEvalResult
from miner.submit import lookup_handshake, verify_signature
from proof.mock_attest import (
    MockAttestation,
    verify_mock_attestation,
)
from proof.real_attest import (
    RealAttestation,
)
from proof.real_attest import (
    verify_attestation as verify_real_attestation,
)
from proof.runner import _load_restricted_paths, scan_diff_for_exploit_patterns, scan_diff_for_restricted
from proof.sources import compute_container_measurement
from validator.integrity import (
    check_canonical_data_source,
    check_checkpoint_not_blocklisted,
    check_compute_budget,
    check_compute_plausibility,
    check_model_size,
    check_recipe_config_matches_proof,
    check_training_timing,
)

# Hard-coded sanity bounds for the miner-submitted model config. The validator
# loads checkpoint['config'] from an attacker-controlled file; without bounds
# a malicious config could OOM the host or allocate gigabytes of memory before
# the actual RCE prevention (weights_only=True) gets a chance. These bounds
# are 10x the largest legitimate config we ship today (h100_scale.json).
MAX_VOCAB_SIZE = 200_000
MAX_DIM = 8192
MAX_N_LAYERS = 64
MAX_N_HEADS = 64
MAX_HEAD_DIM = 256
MAX_MAX_SEQ_LEN = 8192
MAX_CHECKPOINT_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB


@dataclass
class ValidatorReject:
    reason: str
    detail: str = ""


@dataclass
class ValidatorResult:
    miner_hotkey: str
    bundle_hash: str
    handshake_nonce: str
    miner_github: str = ""  # self-declared attribution; informational only
    pr_url: str = ""  # PR against RalphLabsAI/recipe (verified later in service.py)
    hidden_eval: HiddenEvalResult | None = None
    training_summary: dict | None = None
    calibration: dict | None = None
    operations: dict = field(default_factory=dict)
    rejected: ValidatorReject | None = None

    def to_dict(self) -> dict:
        return {
            "miner_hotkey": self.miner_hotkey,
            "miner_github": self.miner_github,
            "pr_url": self.pr_url,
            "bundle_hash": self.bundle_hash,
            "handshake_nonce": self.handshake_nonce,
            # Use to_legacy_dict() so the chain payload stays byte-equivalent
            # to pre-v0.11 form when downstream is None (B1-D12 contract).
            "hidden_eval": self.hidden_eval.to_legacy_dict() if self.hidden_eval else None,
            "training_summary": self.training_summary,
            "calibration": self.calibration,
            "operations": self.operations,
            "rejected": asdict(self.rejected) if self.rejected else None,
        }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_load_checkpoint_config(ckpt_path: Path) -> dict:
    """Load checkpoint['config'] safely.

    We need the config (vocab_size, dim, n_layers, ...) to instantiate the
    model BEFORE loading state_dict. The legacy on-disk format stores config
    inside the checkpoint pickle, which means loading the config exposes us
    to the same pickle-RCE we're trying to avoid.

    Strategy:
      1. Prefer a sibling `checkpoint_config.json` (new format, JSON-only).
      2. Fall back to weights_only=True + extra unpickler patches.
      3. Reject if both fail.

    Bounds-check every field before returning — the values are then passed to
    RalphConfig which allocates tensors proportional to them.
    """
    sidecar = ckpt_path.parent / "checkpoint_config.json"
    if sidecar.exists():
        cfg = json.loads(sidecar.read_text())
    else:
        # Legacy path: extract config from the pickle. We use weights_only=True
        # which since PyTorch 2.4 raises on any arbitrary class reducer, but
        # still permits primitives. The "config" key is plain dict-of-prims so
        # this works for honest miners; malicious pickles raise.
        try:
            ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"checkpoint config could not be loaded safely: {e}. "
                f"Miner must ship checkpoint_config.json alongside checkpoint.pt."
            )
        cfg = ckpt.get("config", {})
        if not isinstance(cfg, dict):
            raise RuntimeError("checkpoint['config'] is not a dict")
    bounds = [
        ("vocab_size", MAX_VOCAB_SIZE),
        ("dim", MAX_DIM),
        ("n_layers", MAX_N_LAYERS),
        ("n_heads", MAX_N_HEADS),
        ("head_dim", MAX_HEAD_DIM),
        ("max_seq_len", MAX_MAX_SEQ_LEN),
    ]
    for key, ceiling in bounds:
        v = cfg.get(key)
        if not isinstance(v, int) or v <= 0 or v > ceiling:
            raise RuntimeError(f"checkpoint config {key}={v!r} out of bounds (max {ceiling})")
    ffn_mult = cfg.get("ffn_mult", 8 / 3)
    if not isinstance(ffn_mult, (int, float)) or not (0.1 <= float(ffn_mult) <= 32.0):
        raise RuntimeError(f"checkpoint config ffn_mult={ffn_mult!r} out of bounds")
    return cfg


def _safe_load_checkpoint_weights(ckpt_path: Path, expected_keys: set[str] | None = None) -> dict:
    """Load checkpoint['model'] state_dict via weights_only=True.

    The size cap is checked BEFORE torch.load to bound memory and prevent
    20GB checkpoint DoS. weights_only=True (PyTorch 2.4+) rejects any
    non-tensor / non-primitive payload so a malicious pickle can't fire a
    reducer.
    """
    size = ckpt_path.stat().st_size
    if size > MAX_CHECKPOINT_BYTES:
        raise RuntimeError(f"checkpoint too large: {size} bytes (max {MAX_CHECKPOINT_BYTES})")
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    state_dict = ckpt.get("model", ckpt if isinstance(ckpt, dict) and "model" not in ckpt else None)
    if state_dict is None:
        raise RuntimeError("checkpoint has no 'model' state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"checkpoint['model'] is not a dict: {type(state_dict).__name__}")
    return state_dict


def _git_commit_epoch(repo_dir: str) -> float | None:
    """Unix commit time (UTC seconds) of HEAD in repo_dir, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "show", "-s", "--format=%ct", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def _fraud_checkpoints(ralph_root: Path) -> set:
    """Load the fraud/off-protocol checkpoint blocklist (chain/fraud_checkpoints.json).
    Entries may be bare SHA-256 strings or {"checkpoint_sha256": ...} dicts. A missing
    or malformed file yields an empty set (fail-open)."""
    p = ralph_root / "chain" / "fraud_checkpoints.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    out: set = set()
    for e in data if isinstance(data, list) else []:
        if isinstance(e, str):
            out.add(e)
        elif isinstance(e, dict) and isinstance(e.get("checkpoint_sha256"), str):
            out.add(e["checkpoint_sha256"])
    return out


def _canonical_code_epoch() -> float | None:
    """Wall-clock epoch the attested canonical code was 'born' — the commit time of
    the PINNED canonical recipe (RECIPE_DIR HEAD), or RALPH_CANONICAL_CODE_EPOCH if
    set. Anchored to the pinned recipe (not ralph HEAD) so validator-only deploys
    don't reset the clock and retro-invalidate honest in-flight runs; only a
    measurement cutover re-pins RECIPE_DIR. Used by the training-timing gate."""
    env = os.environ.get("RALPH_CANONICAL_CODE_EPOCH", "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:
        from ralph_bootstrap import RECIPE_DIR

        return _git_commit_epoch(str(RECIPE_DIR))
    except Exception:  # noqa: BLE001 — bootstrap/git failure -> gate skips (fail-open)
        return None


def op1_diff_and_integrity(
    ralph_root: Path,
    submission_payload: dict,
    proof_dir: Path,
    chain=None,
) -> tuple[bool, str]:
    """Verify miner signature + bundle integrity + restricted-file scan.

    Whitepaper §5.4 Operation 1. Hardened per deep_review_2026-05-31:
      #8  patch.diff is now integrity-checked vs manifest['patch_sha256'],
          and bundle_hash is recomputed from disk and required to match
          submission_payload['bundle_hash'] + manifest['bundle_hash'].
      #9  if on-chain handshake is enforced, patch_hash from the chain
          entry must equal manifest['patch_sha256'].
      #12 the restricted-file scanner is now invoked on the validator
          side (was previously only miner-side).
    """
    # Verify miner signature — hypothesis is part of the signed payload
    # post-fix so miners can't replace a "we're investigating X" rationale
    # with "we proved X" after the fact.
    submission_hypothesis = submission_payload.get("hypothesis", "")
    ok = verify_signature(
        miner_hotkey=submission_payload["miner_hotkey"],
        bundle_hash=submission_payload["bundle_hash"],
        handshake_nonce=submission_payload["handshake_nonce"],
        signature_hex=submission_payload["signature_hex"],
        public_key_hex=submission_payload["public_key_hex"],
        hypothesis=submission_hypothesis,
    )
    if not ok:
        return False, "submission signature invalid"

    # Verify bundle manifest hashes match what's on disk.
    manifest_path = proof_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        return False, "missing bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    pairs = [
        ("checkpoint", proof_dir / "training" / "checkpoint.pt", manifest.get("checkpoint_sha256")),
        ("training_log", proof_dir / "training" / "training_log.jsonl", manifest.get("training_log_sha256")),
        ("calibration", proof_dir / "calibration.json", manifest.get("calibration_sha256")),
        # final_state is REQUIRED. The anti-gaming gates (compute/timing/data/config)
        # all read final_state.config, and the runner folds final_state into bundle_hash
        # UNCONDITIONALLY (proof.runner). If a bundle could omit it, the validator would
        # (a) skip every gate (the `if fs_path.exists()` block below) and (b) recompute a
        # 4-component hash that still self-consistently matches a tampered submission —
        # a one-file bypass of the entire anti-gaming surface. Requiring it here closes that.
        ("final_state", proof_dir / "training" / "final_state.json", manifest.get("final_state_sha256")),
    ]
    # Attestation is now required (single attested-execution tier).
    if manifest.get("attestation_sha256"):
        pairs.append(("attestation", proof_dir / "attestation.json", manifest["attestation_sha256"]))
    # patch.diff: integrity-check whenever manifest declares a hash for it.
    patch_path = proof_dir / "patch.diff"
    patch_sha = manifest.get("patch_sha256")
    if patch_sha:
        pairs.append(("patch", patch_path, patch_sha))
    for name, path, expected in pairs:
        if expected is None:
            return False, f"manifest missing {name} hash"
        if not path.exists():
            return False, f"missing artifact {name} at {path}"
        actual = _file_sha256(path)
        if actual != expected:
            return False, f"{name} hash mismatch (expected {expected[:8]}, got {actual[:8]})"

    # Fraud-checkpoint blocklist: reject re-submission of a checkpoint previously
    # dethroned as off-protocol/fabricated (identical model, new bundle hash +
    # adjusted metadata). manifest['checkpoint_sha256'] is authenticated by the loop above.
    ok_bl, detail_bl = check_checkpoint_not_blocklisted(
        manifest.get("checkpoint_sha256"), _fraud_checkpoints(ralph_root)
    )
    if not ok_bl:
        return False, detail_bl

    # Recompute bundle_hash from disk and require it match BOTH the
    # submission's signed-over hash AND the manifest's declared bundle hash.
    # The recipe for bundle_hash is the same as in proof.runner.run_proof_test.
    bundle_components: list[bytes] = []
    if patch_path.exists():
        bundle_components.append(_file_sha256(patch_path).encode())
    else:
        # Baseline bundles can ship without patch.diff (empty patch).
        bundle_components.append(b"")
    bundle_components.append(_file_sha256(proof_dir / "training" / "checkpoint.pt").encode())
    bundle_components.append(_file_sha256(proof_dir / "training" / "training_log.jsonl").encode())
    bundle_components.append(_file_sha256(proof_dir / "calibration.json").encode())
    # final_state is guaranteed present by the required-artifact loop above, so append
    # it unconditionally — matching proof.runner's unconditional 5-component hash.
    bundle_components.append(_file_sha256(proof_dir / "training" / "final_state.json").encode())
    recomputed = hashlib.sha256(b"".join(bundle_components)).hexdigest()
    if submission_payload.get("bundle_hash") != recomputed:
        return False, (
            f"bundle_hash mismatch: signed={submission_payload.get('bundle_hash','?')[:12]}, "
            f"recomputed={recomputed[:12]}"
        )
    if manifest.get("bundle_hash") != recomputed:
        return False, (
            f"manifest bundle_hash mismatch: manifest={manifest.get('bundle_hash','?')[:12]}, "
            f"recomputed={recomputed[:12]}"
        )

    # Verify handshake nonce was committed on-chain. Until on-chain commits
    # work reliably for the test, the lookup may fail for legitimate cross-host
    # miners — set RALPH_SKIP_HANDSHAKE=1 to skip in that mode.
    import os as _os
    if not _os.environ.get("RALPH_SKIP_HANDSHAKE"):
        # Verify the handshake binds (hotkey, patch_hash, nonce). With a chain
        # handle this queries the miner's LIVE on-chain commitment (works for
        # any external miner — #9 patch cross-check is folded into the hash).
        # Without one, fall back to the local handshakes.jsonl record.
        if chain is not None and hasattr(chain, "verify_handshake_onchain"):
            ok_hs, detail_hs = chain.verify_handshake_onchain(
                submission_payload["miner_hotkey"],
                patch_sha or "",
                submission_payload["handshake_nonce"],
            )
            if not ok_hs:
                return False, detail_hs
        else:
            chain_entry = lookup_handshake(ralph_root, submission_payload["handshake_nonce"])
            if chain_entry is None:
                return False, "handshake nonce not found on chain"
            if chain_entry.get("miner_hotkey") != submission_payload["miner_hotkey"]:
                return False, "handshake nonce was committed by a different miner"
            chain_patch_hash = chain_entry.get("patch_hash")
            if chain_patch_hash and patch_sha and chain_patch_hash != patch_sha:
                return False, (
                    f"on-chain patch_hash mismatch: chain={chain_patch_hash[:12]}, "
                    f"bundle={patch_sha[:12]}"
                )

    # #12: restricted-file scanner now runs on the validator side. The
    # miner-side scan in proof.runner was bypassable by a miner who skipped
    # invoking the runner.
    if patch_path.exists():
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        restricted_yaml = ralph_root / "restricted_files.yaml"
        if restricted_yaml.exists():
            patterns = _load_restricted_paths(restricted_yaml)
            violations = scan_diff_for_restricted(patch_text, patterns)
            if violations:
                return False, f"patch touches restricted paths: {violations}"
        exploit_hits = scan_diff_for_exploit_patterns(patch_text)
        if exploit_hits:
            return False, f"patch injects off-protocol inputs: {exploit_hits[0][0]} :: {exploit_hits[0][1]}"

    # Compute-plausibility + declared-recipe-matches-proof (anti compute-gaming):
    # wall_clock_s is miner-declared (not in bundle_hash), so it can be under-claimed
    # to win the compute-weighted crown. Reject physically-impossible training
    # throughput + a submitted config whose step count the proof never ran.
    fs_path = proof_dir / "training" / "final_state.json"
    if fs_path.exists():
        try:
            final_state = json.loads(fs_path.read_text(encoding="utf-8", errors="replace"))
        except (ValueError, OSError):
            final_state = {}
        calibration: dict = {}
        cal_path = proof_dir / "calibration.json"
        if cal_path.exists():
            try:
                calibration = json.loads(cal_path.read_text(encoding="utf-8", errors="replace"))
            except (ValueError, OSError):
                calibration = {}
        ok_c, detail_c = check_compute_plausibility(final_state, calibration)
        if not ok_c:
            return False, detail_c
        # Compute-budget cap (fair 1x H100-class contest). HARD reject over
        # RALPH_H100H_BUDGET (default 5.0) normalized H100-hours; spoof-proof via a
        # fixed Hopper reference. Disable: RALPH_H100H_GATE_OFF=1.
        if os.environ.get("RALPH_H100H_GATE_OFF") != "1":
            def _envf(name, default):
                try:
                    return float(os.environ.get(name, "") or default)
                except (TypeError, ValueError):
                    return default
            ok_b, detail_b = check_compute_budget(
                final_state,
                calibration,
                budget=_envf("RALPH_H100H_BUDGET", 5.0),
                mm_ref_hopper=_envf("RALPH_H100H_MM_REF_HOPPER", 0.344),
            )
            if not ok_b:
                return False, detail_b
        # Model-size cap (fair fixed-arch contest; blocks the big-under-trained
        # memorizer that fits under the compute budget). Disable: RALPH_MODEL_SIZE_GATE_OFF=1.
        if os.environ.get("RALPH_MODEL_SIZE_GATE_OFF") != "1":
            try:
                _maxn = int(float(os.environ.get("RALPH_MAX_N_PARAMS", "") or 400_000_000))
            except (TypeError, ValueError):
                _maxn = 400_000_000
            ok_sz, detail_sz = check_model_size(final_state, max_n_params=_maxn)
            if not ok_sz:
                return False, detail_sz
        # Off-protocol training: a checkpoint attesting to the canonical code cannot
        # have trained longer than that code has existed. Pairs with the MFU gate
        # above (too-long wall_clock here, too-short there). Disable: RALPH_TIMING_GATE_OFF=1.
        if os.environ.get("RALPH_TIMING_GATE_OFF") != "1":
            try:
                _slack = float(os.environ.get("RALPH_TIMING_SLACK_S") or 7200.0)
            except ValueError:
                _slack = 7200.0
            ok_t, detail_t = check_training_timing(
                final_state,
                canonical_code_epoch=_canonical_code_epoch(),
                now_epoch=time.time(),
                slack_s=_slack,
            )
            if not ok_t:
                return False, detail_t
        ok_d, detail_d = check_canonical_data_source(final_state)
        if not ok_d:
            return False, detail_d
        if patch_path.exists():
            ok_m, detail_m = check_recipe_config_matches_proof(
                patch_path.read_text(encoding="utf-8", errors="replace"), final_state
            )
            if not ok_m:
                return False, detail_m

    return True, "ok"


def _check_canonical_source_version(proof_dir: Path) -> tuple[bool, str]:
    """Actionable version-drift check for op2 — OFF unless
    RALPH_CANONICAL_SOURCE_COMMITS is set (format 'ralph=<sha>,recipe=<sha>').

    When set, the bundle's declared source commits (bundle_manifest.json:
    ralph_source_commit / recipe_source_commit) must match the canonical pair,
    else REJECT with a clear message. Best-effort: a missing manifest/field is
    deferred to the measurement-hash check — this only turns a *known* version
    drift into an actionable error; it is NOT the security gate (a miner who
    spoofs the declared commit still fails the measurement check below).
    """
    import os
    spec = os.environ.get("RALPH_CANONICAL_SOURCE_COMMITS", "").strip()
    if not spec:
        return True, ""
    canon: dict[str, str] = {}
    for kv in spec.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            canon[k.strip()] = v.strip()
    try:
        m = json.loads((proof_dir / "bundle_manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return True, ""

    def _drift(field: str, want: str) -> str | None:
        got = (m.get(field) or "").strip()
        if not want or not got:
            return None
        # tolerate short vs full sha (either is a prefix of the other)
        if want.startswith(got) or got.startswith(want):
            return None
        return f"{field.split('_')[0]}={got} (canonical {want})"

    bad = [d for d in (
        _drift("ralph_source_commit", canon.get("ralph", "")),
        _drift("recipe_source_commit", canon.get("recipe", "")),
    ) if d]
    if bad:
        return False, (
            "built against non-canonical sources: " + "; ".join(bad)
            + " — rebuild on the canonical ralph-prooftest image"
        )
    return True, ""


def op2_attestation_verify(
    ralph_root: Path,
    submission_payload: dict,
    proof_dir: Path,
) -> tuple[bool, str, str]:
    """Verify attestation. Returns (ok, detail, tier).

    Whitepaper v1.2 §5.4 — single attested-execution tier. There is no
    "unverified" path; a submission without a valid attestation chain is
    REJECTED outright.

    On mainnet (RALPH_ALLOW_MOCK_ATTESTATION unset or != "1") only real_*
    attestation types are accepted. Mock attestations exist in the repo as
    open-source code so anyone can forge them — they are explicitly rejected
    on mainnet. Testnet operators can set RALPH_ALLOW_MOCK_ATTESTATION=1 to
    accept mocks (with a loud warning).

    A real attestation must CARRY the TEE/CC hardware evidence its level demands
    (verify_attestation no longer accepts empty quotes): RALPH_REQUIRE_ATTEST_LEVEL
    defaults to "tdx_nvcc" (Intel TDX + NVIDIA CC, both required); set it to
    "nvcc_only" to relax to CC-GPU-without-TDX on testnet.

    See deep_review_2026-05-31 critical #3/#4/#5.
    """
    import os as _os

    att_path = proof_dir / "attestation.json"
    if not att_path.exists():
        # Whitepaper v1.2: no attestation = rejection. The legacy
        # "unverified tier α=0.5" path is retired.
        return False, "missing attestation.json — single attested-execution tier required (v1.2 §5.4)", "rejected"

    att_text = att_path.read_text()
    att_data = json.loads(att_text)
    from ralph_bootstrap import RECIPE_DIR
    expected_measurement = compute_container_measurement(ralph_root, recipe_dir=RECIPE_DIR)

    # B-pin: actionable version-drift error (off unless RALPH_CANONICAL_SOURCE_COMMITS
    # is set). Turns the opaque "container measurement mismatch" into a clear
    # "built against ralph=X, canonical is ralph=Y — rebuild on the canonical
    # image" for the common honest-miner-on-the-wrong-version case. The
    # measurement-hash check below remains the security gate.
    _vok, _vdetail = _check_canonical_source_version(proof_dir)
    if not _vok:
        return False, _vdetail, "rejected"

    allow_mock = _os.environ.get("RALPH_ALLOW_MOCK_ATTESTATION") == "1"

    # Auto-detect attestation format: real (has attestation_type field) vs legacy mock
    if "attestation_type" in att_data:
        att = RealAttestation.from_json(att_text)
        att_type_label = att.attestation_type
        is_real = att_type_label.startswith("real_")
        if not is_real and not allow_mock:
            return False, (
                f"attestation_type={att_type_label!r} is not real_*; mock "
                "attestations rejected on mainnet (set RALPH_ALLOW_MOCK_ATTESTATION=1 "
                "for testnet)"
            ), "rejected"
        if not is_real and allow_mock:
            import sys as _sys
            print(
                "[attest] WARNING: RALPH_ALLOW_MOCK_ATTESTATION=1 — accepting "
                f"mock attestation_type={att_type_label!r}. MUST NOT BE SET ON MAINNET.",
                file=_sys.stderr,
            )
        ok, errors = verify_real_attestation(
            att,
            expected_container_measurement=expected_measurement,
            expected_handshake_nonce=submission_payload["handshake_nonce"],
            expected_bundle_hash=submission_payload["bundle_hash"],
        )
    else:
        # Legacy mock format (no attestation_type field).
        if not allow_mock:
            return False, (
                "legacy mock attestation rejected on mainnet "
                "(set RALPH_ALLOW_MOCK_ATTESTATION=1 for testnet)"
            ), "rejected"
        import sys as _sys
        print(
            "[attest] WARNING: RALPH_ALLOW_MOCK_ATTESTATION=1 — accepting "
            "legacy mock attestation. MUST NOT BE SET ON MAINNET.",
            file=_sys.stderr,
        )
        att = MockAttestation.from_json(att_text)
        ok, errors = verify_mock_attestation(
            att,
            expected_container_measurement=expected_measurement,
            expected_handshake_nonce=submission_payload["handshake_nonce"],
            expected_bundle_hash=submission_payload["bundle_hash"],
        )
        att_type_label = "mock"
        is_real = False

    if not ok:
        return False, f"attestation verification failed ({att_type_label}): " + "; ".join(errors), "rejected"

    # Hopper-only hardware bind (fair 1x H100-class contest). Runs AFTER the NRAS
    # ES384 + eat_nonce==handshake_nonce + TDX checks above, and only for real_*
    # attestations (mocks are already rejected on mainnet). The die rides the
    # signed, nonce-bound NRAS hwmodel (proof.gpu_arch); GH100 (H100+H200) allowed,
    # Blackwell/unknown denied. Disable: RALPH_ARCH_GATE_OFF=1.
    if is_real and os.environ.get("RALPH_ARCH_GATE_OFF") != "1":
        from proof.gpu_arch import verify_gpu_arch_allowed
        allow = {c.strip().upper() for c in os.environ.get("RALPH_ALLOWED_GPU_DIES", "GH100").split(",") if c.strip()}
        arch_ok, arch_reason, _die = verify_gpu_arch_allowed(att_data, allow=allow or {"GH100"})
        if not arch_ok:
            return False, f"GPU arch bind failed: {arch_reason}", "rejected"

    # v1.2: single attested-execution tier. No α discount.
    detail = f"attestation verified ({att_type_label})"
    return True, detail, "verified"


def op3_log_plausibility(proof_dir: Path) -> tuple[bool, str]:
    """Cheap sanity checks on the training log."""
    log_path = proof_dir / "training" / "training_log.jsonl"
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    if not lines:
        return False, "empty training log"
    losses = [e["loss"] for e in lines]
    # No NaN / Inf
    for v in losses:
        if v != v or math.isinf(v):
            return False, "NaN/Inf in training loss"
    # Loss should not have suddenly exploded.
    if losses[-1] > 50.0:
        return False, f"final loss suspiciously high ({losses[-1]:.2f})"
    # No suspicious resume (training time should monotonically increase).
    elapsed = [e["elapsed_s"] for e in lines]
    for prev, cur in zip(elapsed, elapsed[1:]):
        if cur < prev - 0.5:  # tolerate small clock jitter
            return False, f"non-monotonic elapsed_s at step (prev={prev:.2f}, cur={cur:.2f})"
    return True, f"loss[0]={losses[0]:.3f} -> loss[-1]={losses[-1]:.3f}"


def _is_state_dict_shape_mismatch(err: Exception) -> bool:
    """Detect when a load_state_dict failure is due to architecture divergence
    between the validator's canonical RalphBase and the miner's trained model.

    These are the recoverable cases — the miner's patch added/removed/renamed
    parameters relative to canonical. We can retry under a patched-workdir
    subprocess. Other RuntimeErrors (corrupt tensors, etc.) re-raise.
    """
    msg = str(err)
    return (
        "Unexpected key" in msg
        or "Missing key" in msg
        or "size mismatch" in msg
    )


def _run_eval_subprocess(
    workdir: Path,
    ckpt_path: Path,
    ralph_root: Path,
    label: str,
) -> tuple[bool, str, HiddenEvalResult | None]:
    """Run eval_in_workdir.py in a child process and parse its result.

    Isolates the model build + GPU forward in a subprocess so a fatal CUDA fault
    (illegal memory access / device-side assert) or a hang kills ONLY the child —
    the validator rejects the bundle instead of aborting (a C++ CUDA abort can't
    be caught in-process). `workdir` supplies the `model/` package the child
    imports (canonical RECIPE_DIR or a patched copy); `label` prefixes messages.

    SECURITY: the child runs miner-controlled model code, so it gets an
    allowlist-only env (never the seal privkey / wallet / tokens) and its stderr
    is redacted — same discipline as the op4 env-sanitize stopgap (PR#70).
    """
    import os
    import subprocess

    from proof.runner import _redacted, _sanitized_env

    helper = Path(__file__).resolve().parent / "eval_in_workdir.py"
    if not helper.exists():
        return False, f"{label}: helper script missing at {helper}", None
    # Allowlist-only env (no seal key / wallet / tokens). One exception: forward
    # the validator's OWN RALPH_ALLOW_SYNTHETIC_EVAL — run_hidden_eval (canonical
    # eval harness, runs IN the child) reads it to allow the testnet/CI synthetic
    # fallback when no held-out shard is deployed. It is never set on mainnet
    # (=> fail-closed there) and is not miner-settable, so forwarding it past the
    # secret blocklist matches the in-process op4 semantics without weakening
    # mainnet. The actual enforcement toggles (SKIP_HANDSHAKE, ALLOW_MOCK_
    # ATTESTATION, TEST_MODE) stay blocked.
    child_env = _sanitized_env(extra={"PYTHONPATH": str(workdir)})
    if os.environ.get("RALPH_ALLOW_SYNTHETIC_EVAL"):
        child_env["RALPH_ALLOW_SYNTHETIC_EVAL"] = os.environ["RALPH_ALLOW_SYNTHETIC_EVAL"]
    try:
        res = subprocess.run(
            [sys.executable, str(helper), str(workdir), str(ckpt_path), str(ralph_root)],
            capture_output=True,
            text=True,
            timeout=240,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return False, f"{label} subprocess timed out (>240s)", None
    if res.returncode != 0:
        tail = _redacted(res.stderr or "")[-300:]
        return False, f"{label} subprocess exit={res.returncode}: {tail}", None

    marker = "RALPH_EVAL_RESULT "
    line = next((ln for ln in (res.stdout or "").splitlines() if ln.startswith(marker)), None)
    if line is None:
        return False, f"{label}: no RALPH_EVAL_RESULT line in stdout", None
    fields: dict[str, str] = {}
    for tok in line[len(marker):].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            fields[k] = v
    required = ("val_bpb", "benchmark_acc", "tokens_evaluated", "benchmark_examples", "eval_set_hash")
    if not all(k in fields for k in required):
        return False, f"{label}: malformed result line: {line!r}", None

    def _opt_float(key: str) -> float | None:
        v = fields.get(key)
        if v is None or v == "none":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _opt_int(key: str) -> int | None:
        v = fields.get(key)
        if v is None or v == "none":
            return None
        try:
            return int(v)
        except ValueError:
            return None

    def _opt_str(key: str) -> str | None:
        v = fields.get(key)
        return None if (v is None or v == "none") else v

    try:
        result = HiddenEvalResult(
            val_bpb=float(fields["val_bpb"]),
            benchmark_accuracy=float(fields["benchmark_acc"]),
            tokens_evaluated=int(fields["tokens_evaluated"]),
            benchmark_examples=int(fields["benchmark_examples"]),
            eval_set_hash=fields["eval_set_hash"],
            val_seq_len=_opt_int("val_seq_len"),
            sealed_stream_manifest_hash=_opt_str("sealed_stream_manifest_hash"),
            tail_val_bpb=_opt_float("tail_val_bpb"),
        )
    except ValueError as e:
        return False, f"{label}: result line parse error: {e}", None
    return True, f"val_bpb={result.val_bpb:.4f} bench={result.benchmark_accuracy:.3f} ({label})", result


def _patched_hidden_eval(
    ralph_root: Path,
    proof_dir: Path,
    ckpt_path: Path,
) -> tuple[bool, str, HiddenEvalResult | None]:
    """Fallback path when canonical RalphBase can't load the miner's checkpoint.

    Creates a temp workdir, applies the miner's patch.diff against a copy of
    the canonical recipe, and runs eval_in_workdir.py as a subprocess so the
    patched model code is loaded fresh (no module-reload hazards in our own
    interpreter). Returns the same triple shape as op4_hidden_eval so callers
    don't need to branch on the path.
    """
    import shutil
    import tempfile

    from proof.runner import apply_patch

    patch_path = proof_dir / "patch.diff"
    if not patch_path.exists():
        return False, "state_dict mismatch and no patch.diff to retry with", None

    try:
        from ralph_bootstrap import RECIPE_DIR
    except Exception as e:
        return False, f"patched-eval setup: bootstrap import failed: {e}", None

    with tempfile.TemporaryDirectory(prefix="ralph_patched_eval_") as tmp:
        workdir = Path(tmp) / "workdir"
        workdir.mkdir(parents=True)
        # Mirror the proof.runner layout: recipe sources from RECIPE_DIR, eval
        # / calibration from the ralph protocol root.
        for sub in ("model", "recipe", "data", "configs"):
            src = RECIPE_DIR / sub
            if src.exists():
                shutil.copytree(src, workdir / sub, dirs_exist_ok=True)
        for sub in ("eval", "calibration"):
            src = ralph_root / sub
            if src.exists():
                shutil.copytree(src, workdir / sub, dirs_exist_ok=True)

        try:
            apply_patch(workdir, patch_path)
        except Exception as e:
            return False, f"patched-eval: patch apply failed: {str(e)[:200]}", None

        return _run_eval_subprocess(workdir, ckpt_path, ralph_root, "patched-eval")


def _sandboxed_hidden_eval(
    ralph_root: Path,
    proof_dir: Path,
) -> tuple[bool, str, HiddenEvalResult | None]:
    """op4 hidden-eval run inside the hardened container (RALPH_SANDBOX=1).

    The miner's (possibly patched) model executes contained — no network, no
    secrets, non-root. The container emits per-position NLLs + benchmark
    accuracy; the HOST reduces the crown-critical val_bpb from the NLLs and
    computes the eval-set hash itself. FAIL-CLOSED: if the sandbox runtime can't
    be verified, the submission is rejected — never a bare-exec fallback.
    """
    import os
    import shutil
    import tempfile

    import numpy as np

    from eval.host_reduce import (
        expected_token_count,
        hash_target_stream,
        reduce_token_nlls,
    )
    from eval.val_bpb import DEFAULT_BYTES_PER_TOKEN, load_eval_tokens, pinned_eval_seq_len
    from ralph_bootstrap import RECIPE_DIR
    from validator.sandbox import Mount, SandboxConfig, SandboxUnavailable, is_pinned_image, run_in_sandbox

    ckpt_path = proof_dir / "training" / "checkpoint.pt"
    eval_dir = ralph_root / "eval" / "private"
    if not ckpt_path.exists():
        return False, f"missing checkpoint at {ckpt_path}", None
    if not (eval_dir / "active_tokens.bin").exists():
        return False, f"missing held-out shard at {eval_dir}", None

    image = os.environ.get("RALPH_SANDBOX_IMAGE", "")
    if not is_pinned_image(image):
        return False, "RALPH_SANDBOX=1 but RALPH_SANDBOX_IMAGE is not pinned (need name@sha256:… or sha256:…)", None
    gpu = int(os.environ.get("RALPH_SANDBOX_GPU", "0")) if torch.cuda.is_available() else None
    cfg = SandboxConfig(image=image, gpu_device=gpu)

    # Per-submission host scratch for the container's /out. The container itself
    # is ephemeral (docker --rm); this dir holds nlls.npy + manifest.json only
    # long enough to host-reduce, then is removed on EVERY exit path (finally)
    # so /tmp doesn't accumulate ~12 MB per submission.
    out_dir = Path(tempfile.mkdtemp(prefix="ralph_sbx_out_"))
    try:
        mounts = [
            Mount(Path(RECIPE_DIR), "/canon", ro=True),
            Mount(proof_dir, "/in", ro=True),
            Mount(eval_dir, "/eval-private", ro=True),
        ]
        # ENTRYPOINT already runs `python -m validator.sandbox_eval`; pass args only.
        container_argv = [
            "/canon", "/in/patch.diff", "/in/training/checkpoint.pt", "/eval-private", "/out",
        ]
        try:
            res = run_in_sandbox(
                cfg,
                container_argv=container_argv,
                mounts=mounts,
                out_dir=out_dir,
                timeout_s=int(os.environ.get("RALPH_SANDBOX_TIMEOUT_S", "900")),
            )
        except SandboxUnavailable as e:
            return False, f"op4 sandbox unavailable (FAIL-CLOSED, not falling back): {e}", None
        if res.returncode != 0:
            return False, f"op4 sandbox eval failed (rc={res.returncode}): {res.stderr[-300:]}", None

        nll_path = out_dir / "nlls.npy"
        man_path = out_dir / "manifest.json"
        if not (nll_path.exists() and man_path.exists()):
            return False, "op4 sandbox produced no nlls/manifest output", None

        manifest = json.loads(man_path.read_text())
        # Host-PIN the eval window: re-derive seq_len from the checkpoint config
        # ourselves and REJECT if the container echoed anything else. The manifest
        # value is verified, never trusted — a miner cannot widen (or otherwise
        # pick) the eval window from inside the container.
        try:
            expected_seq_len = pinned_eval_seq_len(
                _safe_load_checkpoint_config(ckpt_path)["max_seq_len"]
            )
        except (KeyError, ValueError, RuntimeError, OSError) as e:
            return False, f"op4 sandbox could not derive host seq_len: {e}", None
        seq_len = int(manifest["seq_len"])
        if seq_len != expected_seq_len:
            return False, (
                f"op4 sandbox seq_len mismatch: container echoed {seq_len}, host "
                f"pins {expected_seq_len} (miner cannot choose the eval window)"
            ), None
        tokens = np.asarray(load_eval_tokens(eval_dir / "active_tokens.bin"))
        eval_set_hash = hash_target_stream(tokens)  # HOST-computed, not miner-supplied
        try:
            reduced = reduce_token_nlls(
                np.load(nll_path),
                seq_len=seq_len,
                bytes_per_token=DEFAULT_BYTES_PER_TOKEN,
                expected_tokens=expected_token_count(len(tokens), seq_len),
                eval_set_hash=eval_set_hash,
            )
        except ValueError as e:
            return False, f"op4 host-reduction rejected the emitted nlls: {e}", None

        result = HiddenEvalResult(
            val_bpb=reduced.val_bpb,
            benchmark_accuracy=round(float(manifest.get("benchmark_accuracy", 0.0)), 3),
            tokens_evaluated=reduced.tokens_evaluated,
            benchmark_examples=int(manifest.get("benchmark_examples", 0)),
            eval_set_hash=eval_set_hash,
            val_seq_len=seq_len,
            tail_val_bpb=reduced.tail_val_bpb,
        )
        return True, f"val_bpb={result.val_bpb:.4f} bench={result.benchmark_accuracy:.3f} (sandboxed)", result
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# --- Hidden-eval result cache -------------------------------------------------
# A deferred challenger (king min-tenure guard) is re-scored EVERY epoch while it
# waits out the incumbent's tenure (~300 blocks). The bundle and the held-out
# eval shard are both immutable across those epochs, so the op4 GPU eval (~90 s)
# returns an identical result each time — pure waste that also blocks the GPU
# from processing new submissions. Cache it, keyed on a fingerprint of the eval
# shard so the cache auto-invalidates the moment the shard is rotated. Stored as
# a dotfile inside the bundle dir; op1 integrity is manifest-based (verifies only
# the declared files), so the extra file is ignored.
_EVAL_CACHE_FIELDS = (
    "val_bpb", "benchmark_accuracy", "tokens_evaluated", "benchmark_examples",
    "eval_set_hash", "val_seq_len", "sealed_stream_manifest_hash", "tail_val_bpb",
)


def _eval_shard_fingerprint(eval_dir: Path) -> str:
    """sha256 over the held-out eval shard (tokens + benchmark). Changes iff the
    eval set is rotated — exactly when a cached score MUST be discarded."""
    h = hashlib.sha256()
    for name in ("active_tokens.bin", "active_benchmark.json"):
        p = eval_dir / name
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def _eval_cache_path(proof_dir: Path) -> Path:
    # A dotfile INSIDE the bundle dir. op1 integrity is manifest-based (it verifies
    # only the declared files — checkpoint/training_log/calibration/attestation/
    # patch — and recomputes bundle_hash from those four), so this extra file is
    # ignored. It is archived with the bundle (harmless) and absent on a fresh
    # re-download -> correct re-eval. Per-bundle, so no cross-bundle collision.
    return proof_dir / ".hidden_eval_cache.json"


def _load_cached_hidden_eval(proof_dir: Path, shard_fp: str) -> HiddenEvalResult | None:
    try:
        d = json.loads(_eval_cache_path(proof_dir).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None
    if d.get("eval_shard_fingerprint") != shard_fp:
        return None  # eval shard rotated since this was cached
    r = d.get("result")
    if not isinstance(r, dict):
        return None
    try:
        return HiddenEvalResult(**{k: r[k] for k in _EVAL_CACHE_FIELDS if k in r})
    except (TypeError, KeyError):
        return None


def _save_cached_hidden_eval(proof_dir: Path, shard_fp: str, result: HiddenEvalResult) -> None:
    # A downstream (CSDP) report is a nested object we don't round-trip here —
    # skip the cache rather than drop it; the next epoch re-evals.
    if getattr(result, "downstream", None) is not None:
        return
    payload = {
        "eval_shard_fingerprint": shard_fp,
        "result": {k: getattr(result, k) for k in _EVAL_CACHE_FIELDS},
    }
    try:
        p = _eval_cache_path(proof_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))
    except OSError:
        pass  # cache is a pure optimization — never fail scoring on a write error


# --- HOSB (Host-Owned Suffix-Blanked) op4 scoring -------------------------------
# Look-ahead is useless BY CONSTRUCTION: the host builds the eval grid with the
# answer (input[t+1]) blanked to filler and holds the real target out-of-band; the
# (sandboxed) model emits per-position NLLs; the HOST reduces val_bpb + runs the
# witnesses. Gated by RALPH_HOSB: "off" (default, legacy) | "shadow" (legacy
# crowns, HOSB logged to hosb_shadow.jsonl for tolerance calibration) | "enforce"
# (HOSB is the crown; fail-closed on a witness rejection, missing chain entropy,
# or no pinned sandbox image). Tolerances are env-tunable for on-box calibration.


def _hosb_mode() -> str:
    import os
    m = os.environ.get("RALPH_HOSB", "off").strip().lower()
    return m if m in ("off", "shadow", "enforce") else "off"


def _hosb_tolerances() -> dict:
    import os

    def _f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    return {
        "tol_witness": _f("RALPH_HOSB_TOL_WITNESS", 0.05),
        "wrong_target_floor": _f("RALPH_HOSB_WRONG_FLOOR", 1.0),
        "wrong_target_max_low_frac": _f("RALPH_HOSB_WRONG_MAX_LOW", 0.1),
        "n_scored_per_window": int(_f("RALPH_HOSB_N_SCORED", 16)),
    }


def derive_grid_seed(
    block_hash_hex: str, eval_shard_fp: str, bundle_fp: str, secret: str = ""
) -> bytes:
    """32-byte HOSB grid seed = blake2b(block_hash || shard_fp || bundle_fp || secret).

    Security note (do NOT overclaim): the grid seed selects WHICH positions are
    scored / which are witness cells. The PRIMARY look-ahead defense does not
    depend on this being secret — the answer is physically absent from every
    scored row (filler), so a look-ahead forward gains nothing regardless. The
    seed's job is only to keep the witness-cell LAYOUT unpredictable. Its secrecy
    rests on the held-out `eval_shard_fp` (the shard is secret) and, when set, the
    validator-only `secret` (RALPH_HOSB_SECRET, never serialized) — NOT on the
    block hash, which is public and (for a same-epoch score) may predate the
    submission. `block_hash` + `bundle_fp` make the seed unique per (epoch, bundle)
    and reproducible for audit; they are not the secret.
    """
    h = hashlib.blake2b(digest_size=32)
    for part in (block_hash_hex, eval_shard_fp, bundle_fp, secret):
        h.update(b"\x00")
        h.update(str(part).encode("utf-8"))
    return h.digest()


def _bundle_fp(proof_dir: Path) -> str:
    try:
        return hashlib.sha256((proof_dir / "submission.json").read_bytes()).hexdigest()
    except OSError:
        return proof_dir.name


def _hosb_epoch_seed(chain, eval_shard_fp: str, bundle_fp: str):
    """(seed, epoch_tag) anchored to the current weight-epoch window so the grid is
    STABLE within an epoch (deferred-challenger cache hits) and ROTATES across
    epochs. (None, None) without chain entropy — the caller MUST fail closed in
    enforce mode rather than use a fixed seed. Mixes a validator-only secret
    (RALPH_HOSB_SECRET) so the witness layout doesn't collapse onto shard_fp."""
    import os
    if chain is None:
        return None, None
    try:
        e = max(1, int(os.environ.get("RALPH_HOSB_EPOCH_BLOCKS", "360")))
        anchor = (int(chain.get_current_block()) // e) * e
        bh = str(chain.get_block_hash(anchor))
    except Exception:
        return None, None
    secret = os.environ.get("RALPH_HOSB_SECRET", "") or os.environ.get("RALPH_VALIDATOR_HOTKEY", "")
    seed = derive_grid_seed(bh, eval_shard_fp, bundle_fp, secret)
    return seed, f"{anchor}:{bh[:16]}"


def _hosb_sandbox_nlls(ralph_root: Path, proof_dir: Path, idx_grid, tgt_grid, seed: bytes = b""):
    """Run the (untrusted) model over the HOST grid in the hardened container and
    return the per-cell NLLs + host-reduced benchmark the host computes ITSELF.

    LEAK-FREE: the container is given ONLY idx_grid (blanked rows) + scored_idx
    (the scored COLUMN per row — not the answer) + a SHUFFLED benchmark candidate
    grid (no correct-index marker, no raw answer file). It NEVER receives the
    targets or the benchmark answer key. It emits the top-K logits at each scored
    cell + a score per shuffled benchmark candidate; the HOST computes
    cross-entropy (`ce_from_topk_logits`) and benchmark accuracy
    (`reduce_benchmark_scores`) against the private answers it kept. Returns
    (nlls_2d, manifest) with a HOST benchmark_accuracy. Raises on failure.
    """
    import os
    import shutil
    import tempfile

    import numpy as np

    from eval.host_reduce import ce_from_topk_logits
    from ralph_bootstrap import RECIPE_DIR
    from validator.sandbox import Mount, SandboxConfig, SandboxUnavailable, is_pinned_image, run_in_sandbox

    image = os.environ.get("RALPH_SANDBOX_IMAGE", "")
    if not is_pinned_image(image):
        raise SandboxUnavailable("HOSB requires a pinned RALPH_SANDBOX_IMAGE (name@sha256:… or sha256:…)")
    gpu = int(os.environ.get("RALPH_SANDBOX_GPU", "0")) if torch.cuda.is_available() else None
    cfg = SandboxConfig(image=image, gpu_device=gpu)
    # Large K so the residual lower-bound-Z deflation (-log(1-tail_K)) stays under
    # the crown margin; calibrate per shard on-box (heavy-tail/low-bpt → larger K).
    top_k = int(os.environ.get("RALPH_HOSB_TOPK", "4096"))

    idx_grid = np.asarray(idx_grid)
    tgt_grid = np.asarray(tgt_grid)
    # The scored column per row + the private target there (host-side only).
    scored_idx = (tgt_grid != -100).argmax(axis=1)
    rows = np.arange(idx_grid.shape[0])
    targets = tgt_grid[rows, scored_idx]

    grid_dir = Path(tempfile.mkdtemp(prefix="ralph_hosb_grid_"))
    out_dir = Path(tempfile.mkdtemp(prefix="ralph_hosb_out_"))
    # userns-remap maps the container --user to an unprivileged host uid that
    # cannot touch root-owned 0700 mkdtemp dirs; open the dir bits so the (ro)
    # grid is readable and the (rw) out is writable from the remapped namespace.
    os.chmod(grid_dir, 0o755)
    os.chmod(out_dir, 0o777)
    try:
        np.save(grid_dir / "idx_grid.npy", idx_grid)
        np.save(grid_dir / "scored_idx.npy", scored_idx)  # positions, NOT the answer
        (grid_dir / "job.json").write_text(json.dumps({"top_k": top_k}))
        # Host-reduced benchmark: build the SHUFFLED candidate grid host-side and
        # mount it WITHOUT the correct-index marker / answer key; keep correct_pos
        # private. The raw active_benchmark.json is never mounted.
        bench_correct_pos = None
        bpath = ralph_root / "eval" / "private" / "active_benchmark.json"
        if bpath.exists():
            from eval.val_bpb import build_benchmark_grid
            examples = json.loads(bpath.read_text())
            ctx_flat, ctx_off, cands_shuf, bench_correct_pos = build_benchmark_grid(examples, seed)
            np.save(grid_dir / "bench_context.npy", ctx_flat)
            np.save(grid_dir / "bench_ctx_offsets.npy", ctx_off)
            np.save(grid_dir / "bench_cands.npy", cands_shuf)  # shuffled — no correct marker
        mounts = [
            Mount(Path(RECIPE_DIR), "/canon", ro=True),
            Mount(proof_dir, "/in", ro=True),
            Mount(grid_dir, "/grid", ro=True),  # idx_grid + scored_idx — NO targets, NO raw shard
        ]
        # ENTRYPOINT already runs `python -m validator.sandbox_eval`; pass args only.
        container_argv = [
            "--grid",
            "/canon", "/in/patch.diff", "/in/training/checkpoint.pt", "/grid", "/out",
        ]
        res = run_in_sandbox(
            cfg, container_argv=container_argv, mounts=mounts, out_dir=out_dir,
            timeout_s=int(os.environ.get("RALPH_SANDBOX_TIMEOUT_S", "2400")),
        )
        if res.returncode != 0:
            raise RuntimeError(f"HOSB grid container failed (rc={res.returncode}): {res.stderr[-300:]}")
        mpath = out_dir / "manifest.json"
        needed = [out_dir / f for f in ("topk_logits.npy", "topk_indices.npy")]
        if not (mpath.exists() and all(p.exists() for p in needed)):
            raise RuntimeError("HOSB grid container produced no topk_logits/topk_indices/manifest")
        manifest = json.loads(mpath.read_text())
        # V from the HOST checkpoint config (bounded to MAX_VOCAB_SIZE; the
        # RALPH_VOCAB_SIZE==50257 lock is enforced at op1/ladder, not here), used
        # ONLY to bound the emitted indices — never the container manifest. CE no
        # longer depends on V: Z_hat is the HOST's own logsumexp over the emitted
        # top-K (a lower bound on the true Z), so no container value can deflate it.
        host_vocab = int(_safe_load_checkpoint_config(proof_dir / "training" / "checkpoint.pt")["vocab_size"])
        k_expected = min(top_k, host_vocab)
        topk_logits, topk_idx = (np.load(p) for p in needed)
        m = idx_grid.shape[0]
        if topk_logits.shape != (m, k_expected) or topk_idx.shape != (m, k_expected):
            raise RuntimeError(
                f"HOSB top-K shape mismatch: got {topk_logits.shape}/{topk_idx.shape}, expected ({m},{k_expected})"
            )
        ce = ce_from_topk_logits(topk_logits, topk_idx, targets, host_vocab)
        # Scatter the host-computed CE back to (M, L) at each row's scored column.
        nlls_2d = np.zeros(idx_grid.shape, dtype=np.float64)
        nlls_2d[rows, scored_idx] = ce

        # HOST-reduce the benchmark (HRB): argmax the container's per-candidate
        # scores against the PRIVATE correct slot — the container never saw it.
        if bench_correct_pos is not None and (out_dir / "bench_scores.npy").exists():
            from eval.host_reduce import reduce_benchmark_scores
            acc, _stderr = reduce_benchmark_scores(np.load(out_dir / "bench_scores.npy"), bench_correct_pos)
            manifest["benchmark_accuracy"] = round(float(acc), 3)
            manifest["benchmark_examples"] = int(bench_correct_pos.shape[0])
        else:
            manifest["benchmark_accuracy"] = 0.0
            manifest["benchmark_examples"] = 0
        return nlls_2d, manifest
    finally:
        shutil.rmtree(grid_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def _hosb_eval(ralph_root: Path, proof_dir: Path, chain, nll_provider=None):
    """HOSB op4: host builds the blanked grid (block-hash entropy), the (sandboxed)
    model emits per-position NLLs, the HOST reduces val_bpb + runs the witnesses.
    `nll_provider(idx_grid, tgt_grid) -> (nlls, manifest)` defaults to the sandbox;
    tests inject an in-process provider. Returns (ok, detail, HiddenEvalResult|None);
    a witness rejection / missing entropy / grid failure is FAIL-CLOSED."""
    import numpy as np

    from eval.host_reduce import NonCausalModelError, hash_target_stream, reduce_blanked_nlls
    from eval.val_bpb import DEFAULT_BYTES_PER_TOKEN, build_blanked_grid, load_eval_tokens, pinned_eval_seq_len

    eval_dir = ralph_root / "eval" / "private"
    ckpt_path = proof_dir / "training" / "checkpoint.pt"
    tok_path = eval_dir / "active_tokens.bin"
    if not ckpt_path.exists():
        return False, f"HOSB: missing checkpoint at {ckpt_path}", None
    if not tok_path.exists():
        return False, f"HOSB: missing held-out shard at {eval_dir}", None

    shard_fp = _eval_shard_fingerprint(eval_dir)
    seed, _tag = _hosb_epoch_seed(chain, shard_fp, _bundle_fp(proof_dir))
    if seed is None:
        return False, "HOSB: no chain entropy for the grid seed (fail-closed)", None
    try:
        seq_len = pinned_eval_seq_len(_safe_load_checkpoint_config(ckpt_path)["max_seq_len"])
    except (KeyError, ValueError, RuntimeError, OSError) as e:
        return False, f"HOSB: could not derive seq_len: {e}", None

    tokens = np.asarray(load_eval_tokens(tok_path))
    tol = _hosb_tolerances()
    provider = nll_provider or (lambda ig, tg: _hosb_sandbox_nlls(ralph_root, proof_dir, ig, tg, seed))
    try:
        # Filler = a DISJOINT pseudo-random corpus (seeded for audit), NOT the
        # secret held-out shard. A causal model ignores input[>e] entirely, so the
        # filler value never affects a scored cell; drawing it from the shard would
        # mount verbatim secret slices (a second answer-recovery path). Build is
        # inside the try so a malformed/short shard fails CLOSED, not crash.
        filler_seed = hashlib.blake2b(seed + b"hosb-filler", digest_size=8).digest()
        filler_rng = np.random.default_rng(int.from_bytes(filler_seed, "little"))
        filler = filler_rng.integers(0, int(tokens.max()) + 1, size=max(len(tokens), 2 * seq_len + 2))
        idx_grid, tgt_grid, layout = build_blanked_grid(tokens, filler, seq_len, seed)
        # Wrong-target witness must not be silently empty (it gates the look-ahead
        # backstop); require a real quota on a non-trivial grid.
        n_real = sum(1 for c in layout if c.target_kind == "real" and c.filler_set == "A")
        n_wrong = sum(1 for c in layout if c.target_kind == "wrong")
        if n_real >= 100 and n_wrong < max(8, int(0.03 * n_real)):
            return False, f"HOSB: too few wrong-target witness cells ({n_wrong}) — fail-closed", None
        nlls, manifest = provider(idx_grid, tgt_grid)
    except (ValueError, AssertionError) as e:  # grid build / fraction / leak-invariant
        return False, f"HOSB: grid build failed (fail-closed): {e}", None
    except Exception as e:  # noqa: BLE001 — SandboxUnavailable/RuntimeError/etc → fail closed
        return False, f"HOSB: grid eval failed (fail-closed): {e}", None

    try:
        reduced = reduce_blanked_nlls(
            np.asarray(nlls), layout, seq_len=seq_len, bytes_per_token=DEFAULT_BYTES_PER_TOKEN,
            eval_set_hash=hash_target_stream(tokens),
            tol_witness=tol["tol_witness"], wrong_target_floor=tol["wrong_target_floor"],
            wrong_target_max_low_frac=tol["wrong_target_max_low_frac"],
        )
    except NonCausalModelError as e:
        return False, f"HOSB REJECTED (non-causal/look-ahead): {e}", None
    except ValueError as e:
        return False, f"HOSB: malformed grid output (fail-closed): {e}", None

    result = HiddenEvalResult(
        val_bpb=reduced.val_bpb,
        benchmark_accuracy=round(float(manifest.get("benchmark_accuracy", 0.0)), 3),
        tokens_evaluated=reduced.tokens_evaluated,
        benchmark_examples=int(manifest.get("benchmark_examples", 0)),
        eval_set_hash=reduced.eval_set_hash,
        val_seq_len=seq_len,
        tail_val_bpb=reduced.tail_val_bpb,
    )
    return True, f"val_bpb={result.val_bpb:.4f} bench={result.benchmark_accuracy:.3f} (HOSB)", result


def _hosb_shadow_log(ralph_root: Path, proof_dir: Path, chain, legacy) -> None:
    """Run HOSB alongside the crowned legacy score and append the comparison to
    hosb_shadow.jsonl — the owner's calibration signal (legacy vs HOSB val_bpb,
    witness pass/reject). Never affects the crown; never raises."""
    rec: dict = {"bundle": proof_dir.name, "legacy_val_bpb": round(float(legacy.val_bpb), 6)}
    try:
        ok, detail, hosb = _hosb_eval(ralph_root, proof_dir, chain)
        rec["hosb_ok"] = bool(ok)
        rec["hosb_detail"] = detail
        if ok and hosb is not None:
            rec["hosb_val_bpb"] = round(float(hosb.val_bpb), 6)
            rec["delta_vs_legacy"] = round(float(hosb.val_bpb - legacy.val_bpb), 6)
            rec["hosb_tail_val_bpb"] = hosb.tail_val_bpb
            rec["hosb_benchmark"] = hosb.benchmark_accuracy
    except Exception as e:  # noqa: BLE001 — shadow logging must never break scoring
        rec["hosb_ok"] = False
        rec["hosb_detail"] = f"shadow error: {e}"
    try:
        with (ralph_root / "hosb_shadow.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _legacy_hidden_eval(
    ralph_root: Path,
    proof_dir: Path,
) -> tuple[bool, str, HiddenEvalResult | None]:
    """Pre-HOSB op4 scoring: sandbox window-NLLs (RALPH_SANDBOX=1) OR the
    canonical/patched subprocess. No caching — the op4 dispatcher owns the cache."""
    import os
    if os.environ.get("RALPH_SANDBOX", "0") == "1":
        return _sandboxed_hidden_eval(ralph_root, proof_dir)

    ckpt_path = proof_dir / "training" / "checkpoint.pt"
    if not ckpt_path.exists():
        return False, f"missing checkpoint at {ckpt_path}", None
    # Load config + weights using the SAFE path — no pickle reducers, bounds-checked.
    saved = _safe_load_checkpoint_config(ckpt_path)
    state_dict = _safe_load_checkpoint_weights(ckpt_path)
    cfg = RalphConfig(
        vocab_size=saved["vocab_size"],
        dim=saved["dim"],
        n_layers=saved["n_layers"],
        n_heads=saved["n_heads"],
        head_dim=saved["head_dim"],
        ffn_mult=saved.get("ffn_mult", 8 / 3),
        max_seq_len=saved["max_seq_len"],
    )
    try:
        model = RalphBase(cfg)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Architecture divergence (a structural patch that adds parameters):
        # retry under the patched workdir so the actually-trained model scores.
        if _is_state_dict_shape_mismatch(e):
            return _patched_hidden_eval(ralph_root, proof_dir, ckpt_path)
        raise
    # The CPU load above only ROUTED canonical-vs-patched; free it and run the GPU
    # forward in a SUBPROCESS so a fatal CUDA fault kills only the child.
    del model, state_dict
    from ralph_bootstrap import RECIPE_DIR
    return _run_eval_subprocess(RECIPE_DIR, ckpt_path, ralph_root, "canonical-eval")


def _read_training_log_points(log_path: Path, max_step: int | None = None) -> list:
    """Parse (step, loss) points from a training_log.jsonl, up to max_step (inclusive)."""
    out: list = []
    try:
        for ln in log_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            e = json.loads(ln)
            s = int(e["step"])
            if max_step is not None and s > max_step:
                break
            out.append((s, float(e["loss"])))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return out


def op_rederive_trajectory(ralph_root: Path, proof_dir: Path) -> tuple[bool, str]:
    """Pre-crown proof-of-EXECUTION: re-run the FIRST few steps of the declared
    training on CANONICAL data and require the loss trajectory to reproduce.

    op2 attests code PRESENCE, not EXECUTION; the timing gate + fraud blocklist only
    raise the cost of off-protocol training. This is the check that actually proves the
    declared run happened: apply the miner's patch to the canonical recipe, run the real
    train.py with the miner's exact config+seed but pinned to CANONICAL data, collect the
    first few logged points, and compare the trajectory (validator.integrity.compare_loss_trajectory).

    OFF by default (RALPH_REDERIVE=1 to enable). Fail-OPEN (skip) when disabled or when the
    canonical training data is not materialized on the validator — enable only AFTER
    materializing data/data_manifest.json + shards and calibrating the tolerance band on a
    known-honest bundle. Expensive (GPU-minutes); the caller should run it only on a bundle
    that would actually beat the king. NOTE: written but UNVALIDATED until canonical data
    exists on a validator to run it against.
    """
    import os
    import shutil
    import subprocess
    import tempfile
    import time

    if os.environ.get("RALPH_REDERIVE") != "1":
        return True, "re-derivation disabled (RALPH_REDERIVE!=1)"
    from ralph_bootstrap import RECIPE_DIR

    canon_manifest = RECIPE_DIR / "data" / "data_manifest.json"
    if not canon_manifest.exists():
        return True, "re-derivation skipped: canonical data_manifest.json not materialized on validator"

    fs_path = proof_dir / "training" / "final_state.json"
    log_path = proof_dir / "training" / "training_log.jsonl"
    if not (fs_path.exists() and log_path.exists()):
        return True, "re-derivation skipped: missing final_state/training_log"
    try:
        cfg = (json.loads(fs_path.read_text()).get("config")) or {}
    except (OSError, ValueError):
        return True, "re-derivation skipped: unreadable final_state"

    n_points = int(os.environ.get("RALPH_REDERIVE_POINTS", "3"))
    log_every = int(cfg.get("log_every", 50) or 50)
    max_step = log_every * (n_points - 1)
    declared = _read_training_log_points(log_path, max_step=max_step)
    if len(declared) < 2:
        return True, "re-derivation skipped: declared log too short to compare"

    from proof.runner import _sanitized_env, apply_patch

    patch_path = proof_dir / "patch.diff"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        workdir = tmp_p / "workdir"
        for sub in ("model", "recipe", "data", "configs"):
            src = RECIPE_DIR / sub
            if src.exists():
                shutil.copytree(src, workdir / sub, dirs_exist_ok=True)
        try:
            if patch_path.exists():
                apply_patch(workdir, patch_path)
        except Exception as e:  # noqa: BLE001
            return False, f"re-derivation: patch apply failed: {str(e)[:150]}"

        # Reproduce the miner's exact config (hence LR schedule) but pin data to canonical.
        cfg_file = tmp_p / "rederive_config.json"
        cfg_file.write_text(json.dumps(cfg))
        out_dir = tmp_p / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        rlog = out_dir / "training_log.jsonl"
        train_py = workdir / "recipe" / "train.py"
        if not train_py.exists():
            return True, "re-derivation skipped: recipe/train.py not found in workdir"
        cmd = [
            sys.executable, str(train_py),
            "--config", str(cfg_file),
            "--manifest", str(canon_manifest.resolve()),
            "--data-base-dir", str((RECIPE_DIR / "data").resolve()),
            "--out-dir", str(out_dir),
        ]
        seed = cfg.get("init_seed", cfg.get("data_seed"))
        if seed is not None:
            cmd += ["--seed", str(int(seed))]

        env = _sanitized_env(extra={"PYTHONPATH": str(workdir)})
        timeout_s = int(os.environ.get("RALPH_REDERIVE_TIMEOUT_S", "1200"))
        proc = subprocess.Popen(cmd, cwd=str(workdir), env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Watch the log until the target step is written (schedule preserved, we just
        # stop early), then terminate — a faithful partial re-run, not a full retrain.
        start = time.time()
        rederived: list = []
        try:
            while time.time() - start < timeout_s:
                if rlog.exists():
                    rederived = _read_training_log_points(rlog, max_step=max_step)
                    if rederived and max(s for s, _ in rederived) >= max_step:
                        break
                if proc.poll() is not None:  # train.py finished (ran fewer than max_step)
                    rederived = _read_training_log_points(rlog, max_step=max_step)
                    break
                time.sleep(2)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if len(rederived) < 2:
            return True, f"re-derivation inconclusive: only {len(rederived)} point(s) produced in {timeout_s}s (skip)"

    from validator.integrity import compare_loss_trajectory
    return compare_loss_trajectory(declared, rederived)


def op4_hidden_eval(
    ralph_root: Path,
    proof_dir: Path,
    chain=None,
) -> tuple[bool, str, HiddenEvalResult | None]:
    """Compute the crown-deciding val_bpb. RALPH_HOSB selects the path:
    off (default) = legacy; shadow = legacy crowns + HOSB logged for calibration;
    enforce = HOSB is the crown (fail-closed on witness reject / no entropy)."""
    eval_dir = ralph_root / "eval" / "private"
    shard_fp = _eval_shard_fingerprint(eval_dir)
    mode = _hosb_mode()

    if mode == "enforce":
        # GUARD: enforce is NOT yet cheat-proof — two verified container-side
        # forgeries remain open (the top-K logsumexp/partition function and the
        # container-reported benchmark_accuracy; see the HOSB enforce-gating
        # checklist). Refuse to crown on it without an explicit ack so it can't be
        # flipped on mainnet by accident. Fail-CLOSED (reject), never crown a cheat.
        import os
        if os.environ.get("RALPH_HOSB_ENFORCE_ACK", "0") != "1":
            return False, (
                "HOSB enforce: val_bpb (lower-bound-Z) and benchmark (host-reduced) are now both "
                "host-owned, but the king is not yet HOSB-re-scored and on-box tolerance/K "
                "calibration has not run; flip only after that GO/NO-GO. Set "
                "RALPH_HOSB_ENFORCE_ACK=1 to override for testing, or use RALPH_HOSB=shadow"
            ), None
        _seed, tag = _hosb_epoch_seed(chain, shard_fp, _bundle_fp(proof_dir))
        cache_fp = f"{shard_fp}|hosb-enforce|{tag or 'noentropy'}"
        cached = _load_cached_hidden_eval(proof_dir, cache_fp)
        if cached is not None:
            return True, f"val_bpb={cached.val_bpb:.4f} bench={cached.benchmark_accuracy:.3f} (cached HOSB)", cached
        ok, detail, result = _hosb_eval(ralph_root, proof_dir, chain)
        if ok and result is not None:
            _save_cached_hidden_eval(proof_dir, cache_fp, result)
        return ok, detail, result

    # off / shadow: the LEGACY score is crowned (shadow additionally logs HOSB).
    cached = _load_cached_hidden_eval(proof_dir, shard_fp)
    if cached is not None:
        return True, f"val_bpb={cached.val_bpb:.4f} bench={cached.benchmark_accuracy:.3f} (cached)", cached
    ok, detail, result = _legacy_hidden_eval(ralph_root, proof_dir)
    if ok and result is not None:
        _save_cached_hidden_eval(proof_dir, shard_fp, result)
        if mode == "shadow":
            _hosb_shadow_log(ralph_root, proof_dir, chain, result)
    return ok, detail, result


def judge_submission(
    ralph_root: Path,
    proof_dir: Path,
    chain=None,
) -> ValidatorResult:
    """Run the four ops in order. Any failure shorts out and returns a rejection.

    `chain`, when provided, lets op1 verify the handshake against the live
    on-chain commitment instead of the local handshakes.jsonl record.
    """
    sub_path = proof_dir / "submission.json"
    if not sub_path.exists():
        return ValidatorResult(
            miner_hotkey="?",
            bundle_hash="?",
            handshake_nonce="?",
            rejected=ValidatorReject("missing_submission_json", str(sub_path)),
        )
    submission = json.loads(sub_path.read_text())
    result = ValidatorResult(
        miner_hotkey=submission["miner_hotkey"],
        miner_github=submission.get("miner_github", ""),
        pr_url=submission.get("pr_url", ""),
        bundle_hash=submission["bundle_hash"],
        handshake_nonce=submission["handshake_nonce"],
    )

    ok, detail = op1_diff_and_integrity(ralph_root, submission, proof_dir, chain=chain)
    result.operations["op1_diff_integrity"] = {"ok": ok, "detail": detail}
    if not ok:
        result.rejected = ValidatorReject("op1_diff_integrity", detail)
        return result

    ok, detail, tier = op2_attestation_verify(ralph_root, submission, proof_dir)
    result.operations["op2_attestation"] = {"ok": ok, "detail": detail, "tier": tier}
    if not ok:
        result.rejected = ValidatorReject("op2_attestation", detail)
        return result

    ok, detail = op3_log_plausibility(proof_dir)
    result.operations["op3_log_plausibility"] = {"ok": ok, "detail": detail}
    if not ok:
        result.rejected = ValidatorReject("op3_log_plausibility", detail)
        return result

    # Pre-crown proof-of-EXECUTION (OFF by default; RALPH_REDERIVE=1). Re-runs the
    # first training steps on CANONICAL data and requires the loss trajectory to
    # reproduce — the only check that off-protocol training can't satisfy without
    # actually running the canonical recipe. Skips cleanly when disabled or when the
    # canonical data isn't materialized. Expensive: for efficiency the crown path
    # should ideally invoke this only for a bundle that would beat the king (it still
    # short-circuits behind op1-op3 here).
    ok, detail = op_rederive_trajectory(ralph_root, proof_dir)
    result.operations["op_rederive"] = {"ok": ok, "detail": detail}
    if not ok:
        result.rejected = ValidatorReject("op_rederive_trajectory", detail)
        return result

    ok, detail, hidden_eval = op4_hidden_eval(ralph_root, proof_dir, chain=chain)
    result.operations["op4_hidden_eval"] = {"ok": ok, "detail": detail}
    if not ok:
        # op4 failed — e.g. the checkpoint won't load into the validator's
        # RalphBase (load_state_dict shape mismatch) AND the patched-workdir
        # re-eval subprocess also failed, so op4 returns (False, detail, None).
        # Reject cleanly (mirrors op1-op3) instead of returning a "passing"
        # result with hidden_eval=None, which crashes scoring on
        # NoneType.val_bpb and takes down the whole epoch loop — a DoS surface
        # for any unloadable checkpoint.
        result.rejected = ValidatorReject("op4_hidden_eval", detail)
        return result
    result.hidden_eval = hidden_eval

    # Attach training + calibration summaries for downstream scoring.
    final_state_path = proof_dir / "training" / "final_state.json"
    if final_state_path.exists():
        result.training_summary = json.loads(final_state_path.read_text())
    cal_path = proof_dir / "calibration.json"
    if cal_path.exists():
        result.calibration = json.loads(cal_path.read_text())

    return result


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ralph-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--proof-dir", type=Path, required=True)
    args = p.parse_args()

    res = judge_submission(args.ralph_root, args.proof_dir)
    print(json.dumps(res.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
