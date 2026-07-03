"""Checkpoint-trainedness / log-consistency guard.

Motivation (real incident): a submission whose checkpoint was random-INITIALISED
(every weight at init std) shipped a `training_log.jsonl` claiming a full run and
got crowned — because op4 was scoring against random tokens at the time and the
re-train audit that would have caught the log/checkpoint mismatch never ran. The
checkpoint measured ~ln(vocab) nats/token (uniform output) yet the log claimed a
final loss of ~3 nats.

This is a CHEAP guard (no extra GPU work — it consumes the val_bpb op4 already
computes plus the miner's own declared `final_loss`):

  (a) UNTRAINED: the held-out loss sits within `random_fraction` of the random
      baseline ln(vocab_size) -> the checkpoint carries ~no learned signal.
  (b) LOG/CHECKPOINT MISMATCH: a declared training `final_loss` exists but the
      held-out loss is implausibly worse than it -> the scored checkpoint did
      not come from the declared training run.

Thresholds are deliberately generous so an honest model (held-out a bit worse
than training, never near random) is NEVER rejected; only garbage / fabricated
checkpoints trip it. Returns (ok, reason); ok=False means reject as fraud/broken.
"""
from __future__ import annotations

import math
import re

# Reject if held-out loss >= this fraction of the random baseline ln(vocab).
# A real ~254M model sits at ~3-4.5 nats/token; random is ~10.8 for vocab 50257.
# 0.80 -> reject above ~8.6 nats, leaving a wide safety margin under legit models.
DEFAULT_RANDOM_FRACTION = 0.80

# Reject if held-out loss > claimed_final_loss * FACTOR + MARGIN. Generous: a
# normal train->held-out gap is well under 1.5x; this only fires on gross
# mismatch (e.g. claimed 3.0, measured 9.0).
DEFAULT_MISMATCH_FACTOR = 2.5
DEFAULT_MISMATCH_MARGIN = 1.0


def nats_per_token_from_bpb(val_bpb: float, bytes_per_token: float) -> float:
    """Invert val_bpb = nats / (ln2 * bytes_per_token)."""
    return float(val_bpb) * math.log(2) * float(bytes_per_token)


def check_checkpoint_trained(
    measured_nats_per_token: float,
    vocab_size: int,
    *,
    claimed_final_loss: float | None = None,
    random_fraction: float = DEFAULT_RANDOM_FRACTION,
    mismatch_factor: float = DEFAULT_MISMATCH_FACTOR,
    mismatch_margin: float = DEFAULT_MISMATCH_MARGIN,
) -> tuple[bool, str]:
    """Cheap guard against untrained / log-mismatched checkpoints.

    Args:
      measured_nats_per_token: held-out cross-entropy (nats/token) the validator
        actually measured for this checkpoint (e.g. from op4's val_bpb via
        `nats_per_token_from_bpb`).
      vocab_size: the checkpoint's vocab — sets the random baseline ln(vocab).
      claimed_final_loss: the miner's declared training `final_loss` (nats/token)
        from final_state.json, if present. Enables the log-mismatch check.

    Returns (ok, reason). ok=False -> reject.
    """
    if not (isinstance(measured_nats_per_token, (int, float)) and math.isfinite(measured_nats_per_token)):
        return False, f"non-finite measured loss: {measured_nats_per_token!r}"
    if not (isinstance(vocab_size, int) and vocab_size > 1):
        return False, f"invalid vocab_size: {vocab_size!r}"

    random_baseline = math.log(vocab_size)  # nats/token of a uniform predictor
    if measured_nats_per_token >= random_fraction * random_baseline:
        return False, (
            f"untrained checkpoint: held-out {measured_nats_per_token:.2f} nats/token "
            f">= {random_fraction:.0%} of random baseline {random_baseline:.2f} "
            f"(vocab {vocab_size}) — weights appear at initialization"
        )

    if claimed_final_loss is not None and isinstance(claimed_final_loss, (int, float)) and claimed_final_loss > 0:
        bound = claimed_final_loss * mismatch_factor + mismatch_margin
        if measured_nats_per_token > bound:
            return False, (
                f"log/checkpoint mismatch: held-out {measured_nats_per_token:.2f} nats/token "
                f">> declared training final_loss {claimed_final_loss:.2f} "
                f"(plausible bound {bound:.2f}) — scored checkpoint not from the declared run"
            )

    return True, "ok"


# --- Compute-plausibility (anti compute-gaming) -------------------------------
#
# `wall_clock_s` is MINER-DECLARED (and not in bundle_hash), so a miner can
# under-claim it to look efficient and win the compute-weighted crown — train a
# real model over ~30 H100h but report ~2h. The give-away is physics: the implied
# model-FLOP rate (~6*N*tok/s) cannot exceed the GPU's bf16 matmul peak, and real
# sustained TRAINING MFU is ~30-55%. An implied MFU above the ceiling means the
# wall_clock_s (hence the compute cost) is fabricated.
MAX_PLAUSIBLE_MFU = 0.7
# bf16 dense matmul peak (TFLOP/s) per GPU family — the hard physical ceiling.
_GPU_BF16_PEAK_TFLOPS = {
    "a100": 312.0, "a800": 312.0, "l4": 121.0, "l40": 362.0, "4090": 165.0,
    "h100": 989.0, "h200": 989.0, "h800": 989.0,
    "b100": 1800.0, "b200": 2250.0, "gb200": 2500.0,
}
# Unknown GPU -> assume the fastest known part, so we NEVER false-reject; the gate
# only fires when even the fastest plausible GPU cannot explain the throughput.
_DEFAULT_PEAK_TFLOPS = 2500.0


def _gpu_bf16_peak_flops(gpu_name: str | None) -> float:
    g = (gpu_name or "").lower()
    for key, tflops in _GPU_BF16_PEAK_TFLOPS.items():
        if key in g:
            return tflops * 1e12
    return _DEFAULT_PEAK_TFLOPS * 1e12


def check_compute_plausibility(
    final_state: dict,
    calibration: dict | None = None,
    *,
    max_mfu: float = MAX_PLAUSIBLE_MFU,
) -> tuple[bool, str]:
    """Reject a bundle whose declared training throughput is physically impossible.

    tokens_seen / wall_clock_s implies ~6*N FLOPs/token; over the declared GPU's
    bf16 peak that is the achieved MFU. An implied MFU > `max_mfu` means the
    wall_clock_s (and the efficiency-gate compute cost it drives) is fabricated.
    Best-effort: a missing/incomplete training_summary is skipped (deferred to the
    other gates), not rejected. Returns (ok, reason); ok=False -> reject.
    """
    fs = final_state or {}
    try:
        tokens = float(fs.get("tokens_seen", 0) or 0)
        wall = float(fs.get("wall_clock_s", 0) or 0)
        n = float(fs.get("n_params", 0) or 0)
    except (TypeError, ValueError):
        return True, "compute-plausibility: non-numeric training_summary (skipped)"
    if tokens <= 0 or wall <= 0 or n <= 0:
        return True, "compute-plausibility: incomplete training_summary (skipped)"
    gpu = (calibration or {}).get("gpu_name") or fs.get("gpu_name") or fs.get("device") or ""
    flops_per_s = 6.0 * n * tokens / wall  # 6N FLOPs/token (fwd+bwd)
    mfu = flops_per_s / _gpu_bf16_peak_flops(gpu)
    if mfu > max_mfu:
        return False, (
            f"fabricated compute: {tokens / wall:,.0f} tok/s for a {n / 1e6:.0f}M model on "
            f"'{gpu or 'unknown'}' => {mfu * 100:.0f}% MFU (> {max_mfu * 100:.0f}% physical max); "
            f"wall_clock_s={wall:.0f}s for {tokens:,.0f} tokens is not achievable"
        )
    return True, f"compute plausible: {tokens / wall:,.0f} tok/s, {mfu * 100:.0f}% MFU"


# --- Compute-budget cap (fair "1x H100-class" contest) ------------------------
#
# Cap total normalized H100-hours so the crown is a FIXED compute-budget contest
# (best recipe wins) rather than "whoever buys the most tokens wins". SPOOF-PROOF:
# the denominator is a FIXED validator-side Hopper reference (mm_ref_hopper), NOT
# the miner-reported calibration matmul_ms — a fabricated large matmul_ms cannot
# shrink the cost under the cap. wall_clock_s is the only miner input, and it is
# independently lower-bounded by check_compute_plausibility's MFU gate (too-short
# wall => impossible throughput => rejected there). Pairs with the Hopper arch
# bind (op2): the die is attested GH100, so one fixed Hopper ref is the right
# normalizer. Tunable via RALPH_H100H_BUDGET / RALPH_H100H_MM_REF_HOPPER.
DEFAULT_H100H_BUDGET = 5.0
DEFAULT_MM_REF_HOPPER = 0.344  # fastest attested Hopper (H200) matmul_ms; conservative (never under-charges)


def check_compute_budget(
    final_state: dict,
    calibration: dict | None = None,
    *,
    budget: float = DEFAULT_H100H_BUDGET,
    h100_matmul_ms_ref: float | None = None,
    mm_ref_hopper: float = DEFAULT_MM_REF_HOPPER,
) -> tuple[bool, str]:
    """Reject a bundle whose normalized H100-hours exceed `budget`.
    norm_h100h = (wall_clock_s/3600) * (h100_ref / mm_ref_hopper), using FIXED
    references so a spoofed calibration matmul_ms cannot duck the cap. Best-effort:
    incomplete/non-finite wall_clock_s is skipped. Returns (ok, reason)."""
    fs = final_state or {}
    try:
        wall = float(fs.get("wall_clock_s", 0) or 0)
    except (TypeError, ValueError):
        return True, "compute-budget: non-numeric wall_clock_s (skipped)"
    if wall <= 0 or not math.isfinite(wall):
        return True, "compute-budget: incomplete/non-finite wall_clock_s (skipped)"
    if h100_matmul_ms_ref is None:
        try:
            from validator.scoring import _h100_matmul_ms_ref
            h100_matmul_ms_ref = _h100_matmul_ms_ref()
        except Exception:  # noqa: BLE001 — fall back to the calibrated H100 ref
            h100_matmul_ms_ref = 0.51
    if mm_ref_hopper <= 0:
        mm_ref_hopper = DEFAULT_MM_REF_HOPPER
    norm_h100h = (wall / 3600.0) * (h100_matmul_ms_ref / mm_ref_hopper)
    if norm_h100h > budget:
        return False, (
            f"over compute budget: normalized_H100_hours={norm_h100h:.2f} > cap {budget:.1f} "
            f"(wall {wall / 3600:.2f}h at fixed Hopper ref {mm_ref_hopper:.3f}ms; miner matmul_ms "
            f"ignored to prevent spoofing) — exceeds the fair 1x H100-class budget"
        )
    return True, f"compute budget ok: {norm_h100h:.2f} H100h <= cap {budget:.1f}"


# --- Model-size cap (fair fixed-arch contest) ---------------------------------
#
# The compute-budget cap limits FLOPs/wall, not CAPACITY. A big under-trained model
# (e.g. 1.2B at <1 token/param) fits under the H100-hour budget yet wins on raw
# capacity / held-out MEMORIZATION rather than recipe quality — the recurring
# "1.2B, 0.6 tok/param, val_bpb 1.33" fraud class. n_params is un-forgeable: op4
# builds the model from the declared config and load_state_dict fails if the
# checkpoint's real shape differs, so a big model cannot masquerade as small.
# Pins the contest to the canonical ~254M arch class (tunable RALPH_MAX_N_PARAMS).
DEFAULT_MAX_N_PARAMS = 400_000_000


def check_model_size(final_state: dict, *, max_n_params: int = DEFAULT_MAX_N_PARAMS) -> tuple[bool, str]:
    """Reject a model larger than max_n_params. Best-effort: skipped if n_params
    is absent/non-numeric. Returns (ok, reason); ok=False -> reject."""
    fs = final_state or {}
    try:
        n = float(fs.get("n_params", 0) or 0)
    except (TypeError, ValueError):
        return True, "model-size: non-numeric n_params (skipped)"
    if n <= 0:
        return True, "model-size: no n_params (skipped)"
    if n > max_n_params:
        return False, (
            f"model too large: n_params={n / 1e6:.0f}M > cap {max_n_params / 1e6:.0f}M — the 1x-H100 "
            f"contest is a fixed ~254M-class recipe competition; a bigger under-trained model wins on "
            f"capacity/held-out-memorization, not recipe quality"
        )
    return True, f"model size ok: {n / 1e6:.0f}M <= {max_n_params / 1e6:.0f}M"


# --- Training-timing plausibility (anti off-protocol-training) -----------------
#
# op2 attestation proves the canonical recipe/runner CODE was present in the
# enclave (container_measurement) and that the bundle is BOUND to it (report_data),
# but NOT that the code EXECUTED to produce the checkpoint. A miner with real CC
# hardware can train a model OFF-PROTOCOL (own box, any data/compute, no data-lock,
# no step/compute gate), then spin up the canonical container and mint an
# attestation over the pre-trained checkpoint + a fabricated final_state.
#
# The physical tell: a checkpoint that attests to the canonical code cannot have
# been trained for longer than that code has EXISTED. If the declared wall_clock_s
# exceeds the wall-clock time elapsed since the canonical code was committed, the
# run necessarily started before this code existed -> it was produced off-protocol
# and the enclave only attested a pre-trained model. Pairs with
# check_compute_plausibility: too-LONG wall_clock trips THIS gate; too-SHORT trips
# the MFU gate. Together they box in the off-protocol class (a real model needs
# real FLOPs => a minimum wall_clock the window cannot contain).
def check_training_timing(
    final_state: dict,
    *,
    canonical_code_epoch: float | None,
    now_epoch: float,
    slack_s: float = 7200.0,
) -> tuple[bool, str]:
    """Reject a checkpoint whose declared training duration exceeds the lifetime of
    the canonical code it attests to. Best-effort: skipped (ok) when the canonical
    code epoch is unknown or no wall_clock_s is declared. Returns (ok, reason);
    ok=False -> reject as off-protocol."""
    fs = final_state or {}
    if canonical_code_epoch is None:
        return True, "timing: unknown canonical code epoch (skipped)"
    try:
        wall = float(fs.get("wall_clock_s", 0) or 0)
    except (TypeError, ValueError):
        return True, "timing: non-numeric wall_clock_s (skipped)"
    if wall <= 0:
        return True, "timing: no declared wall_clock_s (skipped)"
    code_age = float(now_epoch) - float(canonical_code_epoch)
    if wall > code_age + slack_s:
        return False, (
            f"off-protocol training: declared wall_clock_s={wall:,.0f}s ({wall / 3600:.1f}h) "
            f"exceeds canonical code age {code_age:,.0f}s ({code_age / 3600:.1f}h, "
            f"+{slack_s / 3600:.1f}h slack) — the attested canonical recipe is younger than "
            f"the claimed run, so the checkpoint was trained before this code existed"
        )
    return True, f"timing plausible: wall {wall / 3600:.1f}h <= code age {code_age / 3600:.1f}h"


def check_checkpoint_not_blocklisted(checkpoint_sha256: str | None, blocked: set) -> tuple[bool, str]:
    """Reject a checkpoint whose SHA-256 was previously dethroned as fraud/off-protocol.

    Stopgap against re-submitting the IDENTICAL off-protocol model under a fresh
    bundle hash + adjusted final_state metadata. The timing gate weakens as the
    canonical code ages (a 7h claim becomes "possible" 7h after the cutover), so an
    unchanged fraud checkpoint can otherwise be re-crowned by simply waiting. The
    caller passes manifest['checkpoint_sha256'], which is authenticated against the
    on-disk checkpoint by the artifact-integrity loop. Returns (ok, reason)."""
    if isinstance(checkpoint_sha256, str) and checkpoint_sha256 in blocked:
        return False, (
            f"blocklisted checkpoint {checkpoint_sha256[:16]}… — this exact model was "
            f"previously dethroned as off-protocol/fabricated; re-derive on the canonical "
            f"code to resubmit"
        )
    return True, "checkpoint not blocklisted"


# --- Pre-crown re-derivation (proof of EXECUTION, not just presence) -----------
#
# op2 attests that the canonical code was PRESENT; nothing proves it EXECUTED to
# produce the checkpoint. The timing gate + fraud blocklist raise the cost but a
# patient attacker retrains a fresh off-protocol checkpoint and waits out the
# timing window. The only check that proves the declared run actually happened is
# to RE-RUN a slice of it: apply the miner's patch to the canonical recipe, run the
# real train.py for the first N steps on CANONICAL data with the miner's config +
# seed, and compare the re-derived per-step loss trajectory against the declared
# training_log.jsonl.
#
# Why it works: the step-0 loss (init-seed weights forward on the first canonical
# batch) is a near-deterministic FINGERPRINT of (arch, seed, data) — an off-protocol
# run on different data/arch, or a fabricated log, misses it. The next few logged
# points can't be reproduced without actually running the canonical optimizer on
# canonical data, so faking them == honestly training (the attacker gains nothing).
# Coverage limit: partial re-derivation proves the run STARTED honestly; a
# "run N canonical steps then swap the final checkpoint" attack needs the attacker
# to actually run N canonical steps AND the swapped checkpoint still faces op4 — it
# raises cost sharply but only full re-derivation closes it completely.
def compare_loss_trajectory(
    declared,
    rederived,
    *,
    step0_tol: float = 0.10,
    abs_tol: float = 0.40,
    rel_tol: float = 0.10,
    min_points: int = 2,
) -> tuple[bool, str]:
    """Compare a declared vs a re-derived training-loss trajectory.

    Args:
      declared/rederived: iterables of (step, loss), matched by step number.
      step0_tol: tight band for the step-0 fingerprint (init forward, deterministic).
      abs_tol/rel_tol: looser band for later steps (benign GPU/compile nondeterminism);
        a step passes if |declared - rederived| <= max(abs_tol, rel_tol*|declared|).
      min_points: minimum matched steps required to render a verdict.

    Returns (ok, reason). ok=False => the declared run was not reproduced on canonical
    data (off-protocol / fabricated log)."""
    dd = {int(s): float(v) for s, v in declared if v == v}  # drop NaN
    rr = {int(s): float(v) for s, v in rederived if v == v}
    common = sorted(set(dd) & set(rr))
    if len(common) < min_points:
        return False, (
            f"re-derivation produced too few comparable points ({len(common)} < {min_points}) "
            f"— cannot confirm the declared training ran on canonical code/data"
        )
    # Step-0 fingerprint: init-seed weights forwarded on the first canonical batch.
    # A miss here means different arch/seed/data than the canonical recipe.
    if 0 in common and abs(dd[0] - rr[0]) > step0_tol:
        return False, (
            f"re-derivation mismatch at step 0: declared loss {dd[0]:.3f} vs re-derived "
            f"{rr[0]:.3f} (tol {step0_tol:.2f}) — different init/data/arch than the canonical "
            f"recipe: the checkpoint was trained off-protocol or the log is fabricated"
        )
    fails = []
    for s in common:
        tol = step0_tol if s == 0 else max(abs_tol, rel_tol * abs(dd[s]))
        if abs(dd[s] - rr[s]) > tol:
            fails.append((s, dd[s], rr[s], tol))
    # Tolerate a single noisy point; a majority outside band = systematic divergence.
    if len(fails) > max(0, (len(common) - 1) // 2):
        s, d, r, t = fails[0]
        return False, (
            f"re-derivation trajectory mismatch: {len(fails)}/{len(common)} steps outside band "
            f"(e.g. step {s}: declared {d:.3f} vs re-derived {r:.3f}, tol {t:.2f}) — the declared "
            f"training was not reproduced on canonical data (off-protocol)"
        )
    return True, f"re-derivation reproduced {len(common) - len(fails)}/{len(common)} trajectory points"


def _added_config_jsons(patch_text: str) -> list[dict]:
    """Parse every NEW/whole configs/*.json the patch adds (best-effort)."""
    import json

    out: list[dict] = []
    path: str | None = None
    buf: list[str] = []

    def _flush() -> None:
        if path and path.endswith(".json") and "config" in path and buf:
            try:
                out.append(json.loads("\n".join(buf)))
            except Exception:  # noqa: BLE001 — partial/edited config, skip
                pass

    for ln in (patch_text or "").splitlines():
        if ln.startswith("+++ b/"):
            _flush()
            path, buf = ln[6:], []
        elif ln.startswith("+") and not ln.startswith("+++"):
            buf.append(ln[1:])
    _flush()
    return out


def check_recipe_config_matches_proof(patch_text: str, final_state: dict) -> tuple[bool, str]:
    """A submitted training config (configs/*.json) must match what the proof ran.

    If the patch declares `total_steps` that differs from the steps the proof
    recorded, the crowned checkpoint was NOT produced by the declared recipe (the
    submitted config is a decoy). Best-effort: skipped when no config is added or no
    proof step count exists. Returns (ok, reason); ok=False -> reject.
    """
    fs = final_state or {}
    proof_steps = fs.get("steps")
    if proof_steps is None:
        proof_steps = (fs.get("config") or {}).get("total_steps")
    if proof_steps is None:
        return True, "config-match: no proof step count (skipped)"
    for cfg in _added_config_jsons(patch_text):
        declared = cfg.get("total_steps")
        if declared is None:
            continue
        try:
            if int(declared) != int(proof_steps):
                return False, (
                    f"declared recipe mismatch: submitted config total_steps={declared} but the "
                    f"proof ran {proof_steps} steps — crowned checkpoint not from the submitted recipe"
                )
        except (TypeError, ValueError):
            continue
    return True, "config matches proof"


# The canonical data_manifest is the container's relative data/ tree. A config
# pointing manifest_path/data_base_dir at a home ("~") or parent-escape ("..")
# path is outside that tree — reject.
#
# BUT the canonical proof runner (proof/runner.py) intentionally pins
# --manifest/--data-base-dir to the RESOLVED-ABSOLUTE container path
# ((RECIPE_DIR/"data"...).resolve()), and recipe train.py records that verbatim
# into final_state.config. So the honest, canonical output IS an absolute path
# like "/workspace/recipe/data/data_manifest.json" or "/dstack/.../recipe/data/…".
# Rejecting every absolute path therefore false-rejects every honest bundle built
# by the canonical harness (the miner-reported op1 breakage). We fold an absolute
# path that passes through the canonical "data" tree back to its relative tail
# ("…/data/data_manifest.json" -> "data/data_manifest.json") before the check, so
# the runner's own resolved paths pass; ~ home and .. escapes stay rejected.
# NOTE: this path check is NOT the data-lock security boundary — the runner PINS
# the load dir (its CLI arg overrides any config the miner sets), and the
# container filesystem is miner-owned regardless of the string, so a determined
# swap is only deterred by re-derivation, not by this regex. Its job is only to
# not false-reject the honest canonical output while flagging obviously-off paths.
_NONCANONICAL_PATH_RE = re.compile(r"^\s*(?:~|\.\.|/)")
_CANONICAL_DATA_TAIL_RE = re.compile(r"(?:^|/)(data(?:/.*)?)$")


def _canonical_relative_tail(v: str) -> str:
    """Fold an absolute path that passes through the canonical 'data' tree back to
    its container-relative tail (…/data or …/data/… -> 'data[/…]'). ~ home and ..
    escapes are left untouched (they stay rejected by the allowlist)."""
    s = v.strip()
    if s.startswith("/"):
        m = _CANONICAL_DATA_TAIL_RE.search(s)
        if m:
            return m.group(1)
    return s


def check_canonical_data_source(final_state: dict) -> tuple[bool, str]:
    """Reject a bundle whose training config points the data manifest/dir outside
    the canonical data tree. The canonical runner's RESOLVED-ABSOLUTE paths are
    folded to their relative tail first (so they pass); only ~/.. escapes and
    absolute paths with no canonical 'data' segment are rejected. Lives in
    `final_state.config` (not the patch), so op1 must check it here. Best-effort:
    skipped when there is no config. Returns (ok, reason)."""
    cfg = (final_state or {}).get("config") or {}
    for key in ("manifest_path", "data_base_dir", "data_dir", "data_path"):
        v = cfg.get(key)
        if isinstance(v, str) and v.strip() and _NONCANONICAL_PATH_RE.match(_canonical_relative_tail(v)):
            return False, (
                f"non-canonical data source: config.{key}={v!r} does not resolve to the "
                f"canonical container-relative data/ tree (~/.. escape or an absolute path "
                f"outside data/) — points at a miner-controlled location"
            )
    return True, "canonical data source"
