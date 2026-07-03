#!/usr/bin/env python3
"""
Continuous validator service — the 24/7 loop that makes Ralph a real subnet.

Each epoch (~tempo blocks):
  1. Sync metagraph
  2. Poll submission queue for new proof bundles
  3. Score each new submission (4 ops + hidden eval)
  4. Compare to current king
  5. Set weights on-chain
  6. Sleep until next epoch

Submissions arrive via the queue directory:
  queue/pending/<bundle_id>/     — new submissions waiting to be scored
  queue/scored/<bundle_id>/      — scored submissions (archived)
  queue/rejected/<bundle_id>/    — rejected submissions

A miner submits by placing their proof bundle in queue/pending/.
In production this is replaced by HuggingFace Hub polling.

Usage:
    python -m validator.service

    # Or with explicit config:
    python -m validator.service --queue-dir /path/to/queue --epoch-blocks 100
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401  — injects RALPH_RECIPE_DIR onto sys.path
from chain_layer.config import get_chain
from validator.hf_poller import DEFAULT_REPO as DEFAULT_HF_REPO
from validator.hf_poller import poll_hub
from validator.scoring import score_bundle
from validator.validator import judge_submission


# Bittensor's bt.logging hijacks Python's logging module and raises the root
# level to WARNING, silencing our INFO messages. Use direct prints with
# timestamps so our output always appears regardless of what bittensor does.
def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log_info(msg: str) -> None:
    print(f"{_ts()} [INFO] {msg}", flush=True)

def log_warn(msg: str) -> None:
    print(f"{_ts()} [WARN] {msg}", flush=True)

def log_err(msg: str) -> None:
    print(f"{_ts()} [ERROR] {msg}", flush=True)

def log_debug(msg: str) -> None:
    if os.environ.get("RALPH_DEBUG"):
        print(f"{_ts()} [DEBUG] {msg}", flush=True)


# Three-class submission classification — see RUN_PLAN_meaningful_failure.md.
# A submission that passes all four cheap ops is then sorted into one of:
#   - king_change       weight 1.0   (decisively beats king past noise floor)
#   - meaningful_failure weight 0.1  (informative dead-end: attested + non-
#                                     trivial diff + coherent rationale +
#                                     val_bpb within 2x the noise band)
#   - plain_failure     weight 0.0   (didn't beat king and uninformative)
KING_CHANGE_WEIGHT = 1.0
MEANINGFUL_FAILURE_WEIGHT = 0.1
PLAIN_FAILURE_WEIGHT = 0.0
NOISE_FLOOR_MARGIN_2X_MULTIPLIER = 2.0
RATIONALE_MIN_NON_WS_CHARS = 200
RATIONALE_MIN_PARAGRAPHS = 2
DIFF_MIN_CHANGED_LINES = 1  # >1 means ≥2 changed lines — admits single-scalar hypothesis tests

RALPH_ROOT = Path(__file__).resolve().parent.parent
SHUTDOWN = False


def _signal_handler(sig, frame):
    global SHUTDOWN
    log_info("shutdown signal received, finishing current epoch...")
    SHUTDOWN = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _submission_time_key(d: Path) -> tuple:
    """Sort key for first-come-first-validate ordering.

    Orders by the submission's HF PR creation time (from .hf_pr.json), falling
    back to the directory mtime for local/legacy bundles without PR metadata.
    Previously bundles were sorted by directory name (= bundle hash), which is
    arbitrary w.r.t. submission time and so unfair.
    """
    info_path = d / ".hf_pr.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            created = info.get("created_at")
            if created:
                ts = datetime.fromisoformat(created).timestamp()
                return (ts, info.get("pr_num") or 0, d.name)
        except Exception:
            pass
    try:
        ts = d.stat().st_mtime
    except OSError:
        ts = 0.0
    return (ts, 0, d.name)


def poll_queue(queue_dir: Path) -> list[Path]:
    """Return paths to pending submission bundles, oldest (first-submitted) first."""
    pending = queue_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    bundles = []
    for d in pending.iterdir():
        if d.is_dir() and (d / "submission.json").exists():
            bundles.append(d)
    bundles.sort(key=_submission_time_key)
    return bundles


def _safe_current_block(chain) -> int:
    """Current block height, or 0 if it can't be read (best-effort)."""
    try:
        return int(chain.get_current_block())
    except Exception:
        return 0


def archive_bundle(bundle_dir: Path, queue_dir: Path, dest: str) -> None:
    """Move a processed bundle to scored/ or rejected/.

    Rejected bundles are terminal — their hash is recorded in hf_state.json so
    they are never re-downloaded or re-evaluated — so the heavy training/
    checkpoint (~0.5 GB each) is GC'd on archive to keep queue/rejected/ from
    filling the disk. The lightweight manifests are kept as an audit trail.
    Opt out with RALPH_QUEUE_GC_REJECTED=0.
    """
    target = queue_dir / dest / bundle_dir.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(bundle_dir), str(target))
    if dest == "rejected" and os.environ.get(
        "RALPH_QUEUE_GC_REJECTED", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}:
        ckpt_dir = target / "training"
        if ckpt_dir.exists():
            try:
                shutil.rmtree(ckpt_dir)
                log_info(f"  queue GC: removed rejected checkpoint {target.name}/training")
            except OSError as e:
                log_warn(f"queue GC: could not remove {ckpt_dir}: {e}")


def _close_losing_prs(bundle_dir: Path, reason: str) -> None:
    """Close the HF (and GitHub recipe) PRs for a non-crowned submission so the
    repos don't accumulate open losing PRs. A king change MERGES its PRs instead
    and never reaches here. Call BEFORE archive_bundle (reads files in-place).

    Opt out with RALPH_CLOSE_LOSING_PRS=0. Best-effort — never raises, never
    affects scoring or weights.
    """
    if os.environ.get("RALPH_CLOSE_LOSING_PRS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    note = f"Closed by Ralph validator — not crowned: {reason}."
    # HF dataset PR (mapping written into .hf_pr.json at download time).
    try:
        hf_path = bundle_dir / ".hf_pr.json"
        if hf_path.exists():
            info = json.loads(hf_path.read_text(encoding="utf-8", errors="replace"))
            tok = os.environ.get("RALPH_BOT_HF_TOKEN") or os.environ.get("HF_TOKEN", "")
            if tok and info.get("pr_num"):
                from validator.hf_bot import close_pr as _hf_close
                ok, detail = _hf_close(
                    repo_id=info.get("repo_id", "RalphLabsAI/proof-bundles"),
                    pr_num=info["pr_num"], token=tok, comment=note,
                )
                log_info(f"  HF PR #{info['pr_num']} {'closed' if ok else 'close failed'}: {detail}")
    except Exception as e:
        log_warn(f"HF PR close skipped for {bundle_dir.name}: {e}")
    # GitHub recipe PR (from submission.json pr_url; often empty → no-op).
    try:
        sub_path = bundle_dir / "submission.json"
        if sub_path.exists():
            pr_url = json.loads(sub_path.read_text(encoding="utf-8", errors="replace")).get("pr_url", "")
            tok = os.environ.get("RALPH_BOT_GH_TOKEN", "")
            if tok and pr_url:
                from validator.github_bot import close_pr as _gh_close
                ok, detail = _gh_close(pr_url, tok, comment=note)
                log_info(f"  GitHub PR {pr_url} {'closed' if ok else 'close failed'}: {detail}")
    except Exception as e:
        log_warn(f"GitHub PR close skipped for {bundle_dir.name}: {e}")


def _require_gh_pr() -> bool:
    """Policy: every submission must carry a GitHub recipe PR. Opt out (e.g.
    testnet) with RALPH_REQUIRE_GH_PR=0."""
    return os.environ.get("RALPH_REQUIRE_GH_PR", "1").strip().lower() not in {"0", "false", "no", "off"}


def _verify_pr_if_required(result, bundle_dir: Path) -> tuple[bool, str]:
    """A submission MUST carry a valid GitHub recipe PR:
      - empty pr_url  → REJECT (no recipe PR), unless RALPH_REQUIRE_GH_PR=0.
      - pr_url + bot token → the PR's diff must be byte-equal to the bundle's
        patch.diff (verified), else REJECT.
      - pr_url but no bot token → can't verify the diff; allowed (operator must
        set RALPH_BOT_GH_TOKEN to enforce the match).
    Returns (ok, detail).
    """
    if not result.pr_url:
        if _require_gh_pr():
            return False, "no GitHub recipe PR (pr_url empty) — a recipe PR is required"
        return True, ""
    token = os.environ.get("RALPH_BOT_GH_TOKEN", "")
    if not token:
        return True, "pr_url present but no bot token to verify the diff — skipping match"
    patch_path = bundle_dir / "patch.diff"
    if not patch_path.exists():
        return True, "no patch.diff in bundle (baseline?) — skipping PR match"
    patch_text = patch_path.read_text()
    if not patch_text.strip():
        return True, "empty patch (baseline) — skipping PR match"
    from validator.github_bot import verify_pr_matches_bundle
    v = verify_pr_matches_bundle(result.pr_url, patch_text, token)
    return v.ok, v.detail


# Paths that count as "training-relevant" for the meaningful_failure gate.
# Anything under these directories OR matching these suffixes qualifies — a
# patch touching at least one of them passes the touches_training half of the
# _diff_is_nontrivial check.
#
# Includes `model/` so structural patches (attention variants, init schemes,
# residual scaling, etc.) earn credit on the attention_variant / init_scheme /
# structural axes. Without `model/`, a clean QK-Norm patch can beat the king
# on val_bpb and still be classified plain_failure for "not touching training."
_TRAINING_RELEVANT_PATH_TOKENS = (
    "recipe/", "training", "/optim", "configs/", "model/", "data/",
    ".yaml", ".yml", ".json", ".toml",
)


def _diff_is_nontrivial(patch_path: Path) -> bool:
    """A diff is non-trivial if it changes more than DIFF_MIN_CHANGED_LINES
    non-whitespace, non-comment lines AND at least one of the touched files
    looks like it actually affects training (model code, recipe code,
    training config, or data pipeline)."""
    if not patch_path.exists():
        return False
    try:
        text = patch_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    changed = 0
    touches_training = False
    for line in text.splitlines():
        if line.startswith("diff --git ") or line.startswith("+++ ") or line.startswith("--- "):
            ll = line.lower()
            if any(t in ll for t in _TRAINING_RELEVANT_PATH_TOKENS):
                touches_training = True
            continue
        if line.startswith("+") or line.startswith("-"):
            stripped = line[1:].strip()
            if stripped and not stripped.startswith("#"):
                changed += 1
    return changed > DIFF_MIN_CHANGED_LINES and touches_training


def _rationale_is_coherent(rationale_path: Path) -> bool:
    """Heuristic: rationale.md is coherent if it has at least
    RATIONALE_MIN_NON_WS_CHARS non-whitespace chars, at least
    RATIONALE_MIN_PARAGRAPHS non-empty paragraphs, and at least 4 distinct
    sentences (catches template / boilerplate / pure repetition).

    First cut is structural heuristic only; an LLM-judge pass is a documented
    followup — see RUN_PLAN_meaningful_failure.md."""
    if not rationale_path.exists():
        return False
    try:
        text = rationale_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    non_ws = "".join(text.split())
    if len(non_ws) < RATIONALE_MIN_NON_WS_CHARS:
        return False
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < RATIONALE_MIN_PARAGRAPHS:
        return False
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    if len(sentences) < 4:
        return False
    if len(set(sentences)) < len(sentences) * 0.6:
        return False
    return True


def _classify_outcome(
    *,
    decisively: bool,
    val_bpb: float,
    king_bpb: float | None,
    noise_floor_margin: float,
    bundle_dir: Path,
) -> tuple[str, float]:
    """Classify a submission that has already passed all four cheap ops.

    Returns (classification, weight_credit) — one of:
      - ("king_change",        KING_CHANGE_WEIGHT)        — beats king past noise floor
      - ("meaningful_failure", MEANINGFUL_FAILURE_WEIGHT) — didn't beat king, but
            attested + non-trivial diff + coherent rationale + val_bpb landed
            within 2x the 2sigma noise band. Rationale ships to the corpus.
      - ("plain_failure",      PLAIN_FAILURE_WEIGHT)      — anything else
    """
    if decisively:
        return "king_change", KING_CHANGE_WEIGHT

    if king_bpb is None:
        # No king yet — meaningful_failure is only definable vs an existing king.
        return "plain_failure", PLAIN_FAILURE_WEIGHT

    # Bar 1: val_bpb must land within 2x the noise band.
    # delta > 0 means challenger is worse than king. Inside-noise-band cases
    # (delta <= noise_floor_margin in either direction) are still candidates.
    delta = val_bpb - king_bpb
    if delta > NOISE_FLOOR_MARGIN_2X_MULTIPLIER * noise_floor_margin:
        return "plain_failure", PLAIN_FAILURE_WEIGHT

    # Bar 2: the diff has to change a training-relevant file with more than
    # DIFF_MIN_CHANGED_LINES non-trivial lines (not whitespace, not comments).
    if not _diff_is_nontrivial(bundle_dir / "patch.diff"):
        return "plain_failure", PLAIN_FAILURE_WEIGHT

    # Bar 3: rationale must be present and structurally coherent.
    if not _rationale_is_coherent(bundle_dir / "rationale.md"):
        return "plain_failure", PLAIN_FAILURE_WEIGHT

    return "meaningful_failure", MEANINGFUL_FAILURE_WEIGHT


def score_and_decide(
    chain,
    bundle_dir: Path,
    noise_floor_margin: float,
) -> dict:
    """Score one submission against the current king. Returns a result dict."""
    result = judge_submission(RALPH_ROOT, bundle_dir, chain=chain)

    if result.rejected:
        return {
            "status": "rejected",
            "miner_hotkey": result.miner_hotkey,
            "miner_github": result.miner_github,
            "pr_url": result.pr_url,
            "reason": result.rejected.reason,
            "detail": result.rejected.detail,
        }

    # Defense-in-depth: a non-rejected result MUST carry a hidden_eval. If op4
    # ever returns ok with no eval, reject here instead of crashing on
    # result.hidden_eval.val_bpb below — a single bad checkpoint would otherwise
    # take down the whole epoch loop.
    if result.hidden_eval is None:
        return {
            "status": "rejected",
            "miner_hotkey": result.miner_hotkey,
            "miner_github": result.miner_github,
            "pr_url": result.pr_url,
            "reason": "op4_hidden_eval",
            "detail": "hidden_eval missing (checkpoint unloadable / re-eval failed)",
        }

    pr_ok, pr_detail = _verify_pr_if_required(result, bundle_dir)
    if not pr_ok:
        return {
            "status": "rejected",
            "miner_hotkey": result.miner_hotkey,
            "miner_github": result.miner_github,
            "pr_url": result.pr_url,
            "reason": "invalid_recipe_pr",
            "detail": pr_detail,
        }

    king = chain.get_king()
    king_bpb = king.val_bpb if king else None
    king_bench = king.benchmark_accuracy if king else None
    # High-water mark: the reigning king's quality bar, persisted so it SURVIVES a
    # transiently-empty throne (clear_king from the no-merged-PR reinstate path, or
    # a missing/corrupt king.json). When the live king record is gone, a challenger
    # must STILL decisively beat this bar — not get a free crown at gain 0. is_first
    # (the only legitimate no-bar crown) is true ONLY at genuine genesis: no king
    # now AND no king ever (no high-water mark). This closes the gap where any
    # submission could grab the crown during an empty-throne window.
    hwm = chain.get_high_water_mark()
    bar_bpb = king_bpb if king else (hwm.get("val_bpb") if hwm else None)
    bar_bench = king_bench if king else (hwm.get("benchmark_accuracy") if hwm else None)
    # v1.2: single attested-execution tier. op2 either passed (tier="verified")
    # or rejected the submission outright. Treat the field defensively for any
    # legacy bundle that slips through.
    tier = result.operations.get("op2_attestation", {}).get("tier", "verified")

    score = score_bundle(
        val_bpb=result.hidden_eval.val_bpb,
        benchmark_accuracy=result.hidden_eval.benchmark_accuracy,
        king_val_bpb=bar_bpb,
        king_benchmark=bar_bench,
        noise_floor_margin=noise_floor_margin,
        matmul_ms=result.calibration["matmul_ms"],
        wall_clock_s=result.training_summary["wall_clock_s"],
        tier=tier,
    )

    is_first = king is None and hwm is None
    decisively = score.decisively_beats_king or is_first

    classification, weight_credit = _classify_outcome(
        decisively=decisively,
        val_bpb=result.hidden_eval.val_bpb,
        king_bpb=king_bpb,
        noise_floor_margin=noise_floor_margin,
        bundle_dir=bundle_dir,
    )
    accepted = classification == "king_change"
    if classification == "king_change":
        status = "accepted"
    elif classification == "meaningful_failure":
        status = "meaningful_failure"
    else:
        status = "below_threshold"

    return {
        "status": status,
        "classification": classification,
        "weight_credit": weight_credit,
        "miner_hotkey": result.miner_hotkey,
        "miner_github": result.miner_github,
        "pr_url": result.pr_url,
        "bundle_hash": result.bundle_hash,
        "val_bpb": result.hidden_eval.val_bpb,
        "benchmark_accuracy": result.hidden_eval.benchmark_accuracy,
        # validation-v2 Phase 1 audit-reproducibility fields, surfaced from the
        # hidden-eval result so the per-epoch audit report can pin the exact
        # eval an auditor must reproduce. tail_val_bpb is recorded only — the
        # scorer does not consume it yet.
        "val_seq_len": result.hidden_eval.val_seq_len,
        "sealed_stream_manifest_hash": result.hidden_eval.sealed_stream_manifest_hash,
        "tail_val_bpb": result.hidden_eval.tail_val_bpb,
        "quality_gain": score.quality_gain,
        "score": score.score,
        "tier": score.tier,
        "decisive": score.decisively_beats_king,
        "accepted": accepted,
        "is_first": is_first,
        "result": result,
        "score_report": score,
    }


def _pending_weights_path(chain) -> Path | None:
    """Return chain_dir/pending_weights.json or None if the chain has no dir."""
    chain_dir = getattr(chain, "chain_dir", None)
    if chain_dir is None:
        return None
    return Path(chain_dir) / "pending_weights.json"


def _load_pending_weights(chain) -> dict[str, float]:
    """Recover round_scores that were computed but never made it through
    set_weights (e.g. rate-limited mid-epoch). Returns empty dict if none."""
    p = _pending_weights_path(chain)
    if p is None or not p.exists():
        return {}
    try:
        return {k: float(v) for k, v in json.loads(p.read_text()).items()}
    except (json.JSONDecodeError, ValueError):
        return {}


def _save_pending_weights(chain, weights: dict[str, float]) -> None:
    p = _pending_weights_path(chain)
    if p is None:
        return
    p.write_text(json.dumps(weights, indent=2, sort_keys=True))


def _burn_fallback_enabled() -> bool:
    """Burn-to-uid-0 when nothing is scoreable this epoch. On by default
    (standard 'burn to owner' so the validator always sets weights + keeps
    vTrust). Disable with RALPH_BURN_FALLBACK in {0,false,no,off}."""
    return os.environ.get("RALPH_BURN_FALLBACK", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _log_validator_standing(chain) -> None:
    """Log this validator's own on-chain standing (uid / stake / vTrust / blocks
    since last weight-set) so operators can SEE they're registered + active.
    Best-effort, never raises; no-op on LocalChain (no metagraph)."""
    try:
        mg = getattr(chain, "metagraph", None)
        wallet = getattr(chain, "wallet", None)
        if mg is None or wallet is None:
            return
        ss58 = wallet.hotkey.ss58_address
        hotkeys = list(getattr(mg, "hotkeys", []))
        if ss58 not in hotkeys:
            log_warn(f"STANDING: hotkey {ss58[:12]}… is NOT registered on the subnet")
            return
        uid = hotkeys.index(ss58)

        def _f(attr):
            try:
                return float(getattr(mg, attr)[uid])
            except Exception:
                return None

        stake = _f("S")
        vtrust = _f("Tv")
        if vtrust is None:
            vtrust = _f("validator_trust")
        try:
            vpermit = bool(mg.validator_permit[uid])
        except Exception:
            vpermit = None
        parts = [f"uid={uid}"]
        if stake is not None:
            parts.append(f"stake={stake:.1f}")
        if vtrust is not None:
            parts.append(f"vtrust={vtrust:.4f}")
        if vpermit is not None:
            parts.append(f"vpermit={vpermit}")
        log_info("STANDING: " + " ".join(parts))
    except Exception as e:
        log_debug(f"standing log failed: {e}")


def _clear_pending_weights(chain) -> None:
    p = _pending_weights_path(chain)
    if p is not None and p.exists():
        p.unlink()


# §5.6: 10% of the per-epoch reward pool is allocated to meaningful_failure
# miners (equal split — until we have a real informativeness ranker), with
# the remaining 90% to the king. If there are zero meaningful_failures, the
# whole 100% goes to the king.
KING_POOL_FRACTION = 0.9
MEANINGFUL_FAILURE_POOL_FRACTION = 0.1


def _merge_recovered_weights(
    round_scores: dict[str, float],
    recovered: dict[str, float],
    current_king_hotkey: str | None,
) -> dict[str, float]:
    """Merge weights recovered from a rate-limited previous epoch into this
    epoch's round_scores WITHOUT resurrecting a stale king.

    `_apply_pool_split` already weights the CURRENT king authoritatively every
    epoch, so a recovered king-level weight (>= KING_POOL_FRACTION) for any
    hotkey OTHER than the current king is a previous reign whose set_weights got
    rate-limited. Merging it would split emission across two kings — the
    dethroned king keeps earning until a set_weights finally lands and clears the
    pending file. We drop those, and skip the current king (its authoritative
    weight is already in round_scores). Only sub-king (meaningful_failure)
    credits that never landed are recovered, max-by-hotkey.
    """
    for hk, w in recovered.items():
        if hk == current_king_hotkey:
            continue  # current king already weighted this epoch — don't override
        if w >= KING_POOL_FRACTION:
            continue  # stale king from a prior reign — never resurrect
        round_scores[hk] = max(round_scores.get(hk, 0.0), w)
    return round_scores


def _require_merged_king_pr() -> bool:
    """Opt-in policy: when setting weights, only emit to a king whose GitHub
    recipe PR is MERGED.

    The crown-time PR check (_verify_pr_if_required) is NOT re-applied at
    weight-set time, so a king crowned under a relaxed/transient config (e.g.
    RALPH_REQUIRE_GH_PR=0, or a crown whose auto-merge never landed) keeps
    earning emission forever. Enable on mainnet with
    RALPH_REQUIRE_MERGED_KING_PR=1 (mirrors RALPH_REQUIRE_GH_PR / RALPH_ALLOW_*).
    """
    return os.environ.get("RALPH_REQUIRE_MERGED_KING_PR", "0").strip().lower() in {"1", "true", "yes", "on"}


# A merged PR is immutable, so confirmed-merged pr_urls are cached and never
# re-hit the GitHub API. Only TRUE is cached: a not-yet-merged PR is re-checked,
# while a transient GitHub outage on an ALREADY-merged king returns the cached
# True instead of a fail-closed False (which would burn a legitimate king).
_MERGED_PR_URLS: set[str] = set()


def _king_record_pr_merged(king) -> bool:
    """True iff the given king record's recipe PR is merged.

    Reads pr_url from the king's proof_dir/submission.json and checks GitHub
    with RALPH_BOT_GH_TOKEN. Fail closed: a missing king / proof_dir / pr_url /
    token, or any API error, returns False so an unverifiable king is never
    treated as valid. Confirmed-merged pr_urls are cached (merge is immutable),
    so GitHub flakiness can't burn a king that was already verified merged.
    """
    if king is None:
        return False
    proof_dir = getattr(king, "proof_dir", "") or ""
    if not proof_dir:
        return False  # guard: never read a relative ./submission.json
    try:
        sub = json.loads((Path(proof_dir) / "submission.json").read_text(encoding="utf-8", errors="replace"))
        pr_url = sub.get("pr_url", "")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pr_url = ""
    if not pr_url:
        return False
    if pr_url in _MERGED_PR_URLS:
        return True
    from validator.github_bot import pr_is_merged
    merged = pr_is_merged(pr_url, os.environ.get("RALPH_BOT_GH_TOKEN", ""))
    if merged:
        _MERGED_PR_URLS.add(pr_url)
    return merged


def _king_pr_merged(chain) -> bool:
    """True iff the sitting king's recipe PR is merged (see _king_record_pr_merged)."""
    return _king_record_pr_merged(chain.get_king())


def _reinstate_valid_king(chain) -> None:
    """Repair a throne already occupied by a king with no MERGED recipe PR.

    Walks the king lineage newest -> oldest (each record's previous_king) and
    reinstates the first ancestor whose PR IS merged. If none qualifies, clears
    the throne so a fresh legitimately-merged submission can be crowned. The
    weight gate alone only stops emission — without this an unbeatable squatter
    (e.g. one sitting at the eval floor) blocks every challenger forever. No-op
    unless RALPH_REQUIRE_MERGED_KING_PR is set; best-effort (never raises, a
    failure here must not break the epoch).
    """
    if not _require_merged_king_pr():
        return
    try:
        king = chain.get_king()
        if king is None or _king_record_pr_merged(king):
            return
        from chain_layer.interface import KingRecord
        node = king
        while isinstance(node.previous_king, dict):
            # KingRecord.from_dict (not **) so a record serialized with
            # compute_cost_h100h or unknown keys reconstructs instead of raising
            # TypeError — which previously cleared the throne even when a valid
            # merged ancestor existed below.
            node = KingRecord.from_dict(node.previous_king)
            if _king_record_pr_merged(node):
                log_warn(
                    f"sitting king {king.miner_hotkey[:12]}… has no merged recipe PR "
                    f"— demoting to last valid king {node.miner_hotkey[:12]}…"
                )
                chain.set_king(node)
                return
        # No ancestor has a merged PR: clear the throne so the next legitimately
        # merged submission is crowned (until then run_epoch burns).
        chain.clear_king()
        log_warn(
            f"sitting king {king.miner_hotkey[:12]}… has no merged recipe PR and no "
            f"valid ancestor — throne cleared (burning until a valid king)"
        )
    except Exception as e:
        log_warn(f"king reinstatement check failed (throne unchanged): {e}")


def _apply_pool_split(
    chain,
    king_change_hotkey: str | None,
    meaningful_failure_hotkeys: list[str],
) -> dict[str, float]:
    """Compute the §5.6 90/10 pool split.

    - If a new king was crowned this epoch, that miner gets the king share.
      Otherwise the current sitting king gets the king share.
    - Meaningful failures split the 10% pool equally.
    - If no meaningful failures: king gets the full 100%.

    Returns {hotkey: weight} ready to feed chain.set_weights.
    """
    weights: dict[str, float] = {}
    king_hotkey = king_change_hotkey
    if king_hotkey is None:
        king = chain.get_king()
        if king is not None:
            king_hotkey = king.miner_hotkey
    # Weight-time integrity gate: a king must carry a MERGED recipe PR to keep
    # earning emission. The crown-time check is not re-applied here, so without
    # this a king crowned under a relaxed/transient config keeps the throne
    # forever. When it trips we withhold ALL weight this epoch -> run_epoch's
    # burn fallback emits nothing to the unvalidated king until a king with a
    # merged PR is crowned. Opt-in via RALPH_REQUIRE_MERGED_KING_PR.
    if king_hotkey and _require_merged_king_pr() and not _king_pr_merged(chain):
        log_warn(
            f"king {king_hotkey[:12]}… has no MERGED recipe PR — withholding all "
            f"weight this epoch (RALPH_REQUIRE_MERGED_KING_PR); burning until a "
            f"king with a merged PR is crowned"
        )
        return {}
    if not meaningful_failure_hotkeys:
        if king_hotkey:
            weights[king_hotkey] = 1.0
        return weights
    if king_hotkey:
        weights[king_hotkey] = KING_POOL_FRACTION
    per_mf = MEANINGFUL_FAILURE_POOL_FRACTION / len(meaningful_failure_hotkeys)
    for hk in meaningful_failure_hotkeys:
        # If a hotkey somehow appears as both king and meaningful_failure
        # (e.g. king resubmitted a near-miss in the same epoch), max the
        # two shares rather than overwrite.
        weights[hk] = max(weights.get(hk, 0.0), per_mf)
    return weights


def _submission_hotkey(bundle_dir: Path) -> str | None:
    """Cheap read of the claimed miner hotkey from a pending bundle (pre-op4)."""
    try:
        sub = json.loads((bundle_dir / "submission.json").read_text())
        hk = sub.get("miner_hotkey")
        return str(hk) if hk else None
    except Exception:
        return None


def run_epoch(
    chain,
    queue_dir: Path,
    noise_floor_margin: float,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_limit: int = 10,
    audit_random_rate: float = 0.10,
    audit_reports_enabled: bool = True,
    netuid: int = 40,
    eval_seed: int = 0,
    hf_publish_enabled: bool = False,
    hf_audit_repo: str | None = None,
) -> dict:
    """Process all pending submissions in one epoch.

    Weight allocation follows whitepaper §5.6:
      - king (new or sitting) gets 90% of the pool when meaningful_failures
        exist this epoch, 100% otherwise
      - meaningful_failures split the 10% pool equally

    Audit dispatcher: every accepted submission or close-margin king-change
    is probabilistically enqueued for re-audit (validator.audit_scheduler).
    Audit fails → chain.blacklist() the miner → next set_weights zeros them.

    Round-scores are persisted to chain_dir/pending_weights.json before
    chain.set_weights() so a rate-limited or crashed validator can recover
    the weights on the next epoch without re-scoring the (now-archived)
    bundles.
    """
    # Repair the throne BEFORE scoring/weighting: if the sitting king has no
    # merged recipe PR, demote to the last valid ancestor (or clear it) so this
    # epoch's challengers don't compare against an unbeatable invalid squatter.
    # Opt-in via RALPH_REQUIRE_MERGED_KING_PR; no-op otherwise.
    _reinstate_valid_king(chain)

    if hf_repo:
        try:
            new = poll_hub(queue_dir, repo_id=hf_repo, token=hf_token, limit=hf_limit)
            if new:
                log_info(f"pulled {len(new)} bundle(s) from HF Hub: {[b[:8] for b in new]}")
        except Exception as e:
            log_warn(f"HF Hub poll failed: {e}")

    bundles = poll_queue(queue_dir)
    # Recovery: pick up any weights left over from a previous epoch that
    # rate-limited or crashed before set_weights landed. They merge into the
    # current round_scores additively (max-by-hotkey) so a king that won
    # last epoch + got 0.1 meaningful_failure credit this epoch still gets
    # the larger share.
    recovered = _load_pending_weights(chain)
    if recovered:
        log_info(f"recovered {len(recovered)} pending weights from previous epoch")

    # NOTE: do NOT early-burn on an empty epoch. A king-based subnet must keep
    # weighting the SITTING king until someone dethrones it — burning idle
    # epochs would starve the reigning king and is exactly the behavior miners /
    # validators distrust. We fall through to the normal end-of-epoch weight
    # block: with zero bundles the scoring loop no-ops, _apply_pool_split()
    # re-asserts the sitting king (king → 1.0), and the BURN fallback there
    # fires ONLY when there is genuinely nothing to weight — i.e. NO king at all
    # (genesis, or right after the king is reset).
    if not bundles and not recovered:
        king = chain.get_king()
        if king is not None:
            log_info(f"no new submissions — holding king {king.miner_hotkey[:12]}… (weights below)")
        else:
            log_info("no submissions and no king yet — burning to uid 0 (genesis)")

    log_info(f"found {len(bundles)} pending submission(s)")
    epoch_results = {"submissions": len(bundles), "accepted": 0, "rejected": 0}

    # Block height at epoch start — recorded into the audit report's block
    # range. Best-effort; never fatal.
    try:
        epoch_start_block = chain.get_current_block()
    except Exception:
        epoch_start_block = 0

    # Track classifications per-hotkey so the pool split at end of epoch is
    # correct (and so two king_changes in one epoch — rare but legal —
    # don't double-share the king pool).
    king_change_hotkey: str | None = None
    meaningful_failure_hotkeys: list[str] = []
    # Accumulate every scored (non-rejected) submission's result dict for the
    # end-of-epoch audit report (validation-v2 Phase 1). Strictly additive —
    # this list is only consumed by the audit-report block, which is wrapped
    # in try/except so it can never affect scoring or weight-setting.
    scored_results: list[dict] = []
    # Audit dispatcher needs chain_dir to enqueue jobs. Both BittensorChain
    # and LocalChain expose .chain_dir.
    chain_dir = getattr(chain, "chain_dir", None)

    # Dynamic per-hotkey cooldown ("one-in-flight"): at most one bundle per
    # hotkey is evaluated per epoch; extras are DEFERRED (left queued) and retried
    # once the in-flight one has been scored. NOT Ninja's one-shot rule — a hotkey
    # can submit again next epoch. Opt-in via RALPH_ONE_IN_FLIGHT=1 (default off so
    # pulling this never changes live behavior until enabled). No explicit release:
    # a scored bundle leaves `pending`, so the next epoch's reconcile() frees the
    # hotkey; reconcile also clears any claim orphaned by a mid-epoch crash.
    inflight = None
    if os.environ.get("RALPH_ONE_IN_FLIGHT", "0") == "1" and chain_dir:
        from validator.inflight import InFlightGuard
        inflight = InFlightGuard(Path(chain_dir) / "inflight.json")
        inflight.reconcile({b.name for b in bundles})

    for bundle_dir in bundles:
        bundle_id = bundle_dir.name
        if inflight is not None:
            hk = _submission_hotkey(bundle_dir)
            if hk:
                ok, why = inflight.claim(hk, bundle_id)
                if not ok:
                    log_info(f"deferring {bundle_id} (one-in-flight): {why}")
                    continue
        log_info(f"scoring {bundle_id}...")

        try:
            result = score_and_decide(chain, bundle_dir, noise_floor_margin)
        except Exception as e:
            log_err(f"error scoring {bundle_id}: {e}")
            log_debug(traceback.format_exc())
            _close_losing_prs(bundle_dir, f"scoring error: {e}")
            archive_bundle(bundle_dir, queue_dir, "rejected")
            epoch_results["rejected"] += 1
            chain.append_event({
                "type": "submission_error",
                "timestamp": time.time(),
                "bundle_id": bundle_id,
                "error": str(e),
            })
            continue

        if result["status"] == "rejected":
            log_warn(f"rejected {bundle_id}: {result['reason']}")
            _close_losing_prs(bundle_dir, result["reason"])
            archive_bundle(bundle_dir, queue_dir, "rejected")
            epoch_results["rejected"] += 1
            chain.append_event({
                "type": "submission_rejected",
                "timestamp": time.time(),
                "miner_hotkey": result["miner_hotkey"],
                "miner_github": result.get("miner_github", ""),
                "reason": result["reason"],
            })
            continue

        miner_hotkey = result["miner_hotkey"]
        # Hotkey-registered gate: don't burn audit / king-update work on
        # an unregistered hotkey. The chain.set_weights filter would zero
        # them anyway, but rejecting here keeps the public corpus clean
        # and stops the §5.7 audit dispatcher from wasting time.
        if not chain.is_hotkey_registered(miner_hotkey):
            log_warn(f"rejected {bundle_id}: miner hotkey {miner_hotkey[:16]}... not registered")
            _close_losing_prs(bundle_dir, "miner hotkey not registered on subnet")
            archive_bundle(bundle_dir, queue_dir, "rejected")
            epoch_results["rejected"] += 1
            chain.append_event({
                "type": "submission_rejected",
                "timestamp": time.time(),
                "miner_hotkey": miner_hotkey,
                "miner_github": result.get("miner_github", ""),
                "reason": "hotkey_not_registered",
            })
            continue

        # Defensive NaN/Inf check — score_bundle returns non-decisive on
        # non-finite metrics but we also reject outright so the corpus
        # doesn't end up with NaN val_bpb rows that break dashboards.
        import math as _math
        val_bpb = result.get("val_bpb")
        bench_acc = result.get("benchmark_accuracy")
        if (val_bpb is None or not _math.isfinite(val_bpb)
                or bench_acc is None or not _math.isfinite(bench_acc)):
            log_warn(f"rejected {bundle_id}: non-finite metrics (val_bpb={val_bpb}, bench={bench_acc})")
            _close_losing_prs(bundle_dir, "non-finite metrics")
            archive_bundle(bundle_dir, queue_dir, "rejected")
            epoch_results["rejected"] += 1
            chain.append_event({
                "type": "submission_rejected",
                "timestamp": time.time(),
                "miner_hotkey": miner_hotkey,
                "miner_github": result.get("miner_github", ""),
                "reason": "non_finite_metrics",
            })
            continue

        # weight_credit is informational here; the actual epoch weights come
        # from _apply_pool_split() at end of epoch (§5.6 90/10 split).
        weight_credit = result.get("weight_credit", 0.0)

        chain.append_event({
            "type": "submission_scored",
            "timestamp": time.time(),
            "miner_hotkey": miner_hotkey,
            "miner_github": result.get("miner_github", ""),
            "val_bpb": result["val_bpb"],
            "quality_gain": result["quality_gain"],
            "score": result["score"],
            "weight_credit": weight_credit,
            "classification": result.get("classification", "unknown"),
            "tier": result["tier"],
            "decisive": result["decisive"],
            "accepted": result["accepted"],
        })

        # Accumulate a JSON-safe view of this scored submission for the
        # end-of-epoch audit report. We strip the non-serializable `result`
        # (ValidatorResult) and `score_report` (ScoreReport) objects and pull
        # the lineage/attestation fields the report wants off the underlying
        # ValidatorResult before discarding it.
        try:
            vr = result.get("result")
            parent_hash = None
            attestation_hash = None
            if vr is not None:
                ops = getattr(vr, "operations", {}) or {}
                # parent lineage hash is carried in the submission_received
                # preflight; the ValidatorResult doesn't surface it directly,
                # so leave None unless a future scorer threads it through.
                attestation_hash = (ops.get("op2_attestation") or {}).get("attestation_hash")
            scored_view = {
                k: v for k, v in result.items()
                if k not in ("result", "score_report")
            }
            scored_view.setdefault("parent_king_attestation_hash", parent_hash)
            scored_view.setdefault("attestation_hash", attestation_hash)
            scored_results.append(scored_view)
        except Exception as e:  # never let report bookkeeping affect scoring
            log_debug(f"audit-report accumulation skipped for {bundle_id}: {e}")

        # Meaningful-failure branch: didn't crown a new king, but the work was
        # informative. Track for the 90/10 pool split; archive the bundle.
        if result["status"] == "meaningful_failure":
            gh = result.get("miner_github", "")
            who = f"{gh} ({miner_hotkey[:12]}...)" if gh else f"{miner_hotkey[:20]}..."
            king_now = chain.get_king()
            king_bpb_str = f"{king_now.val_bpb:.4f}" if king_now else "—"
            log_info(
                f"MEANINGFUL FAILURE: {who} val_bpb={result['val_bpb']:.4f} "
                f"(king {king_bpb_str}); rationale archived to corpus, "
                f"will get equal share of 10% pool"
            )
            _close_losing_prs(
                bundle_dir,
                f"meaningful failure — credited (10% pool), not crowned "
                f"(val_bpb {result['val_bpb']:.4f} vs king {king_bpb_str})",
            )
            archive_bundle(bundle_dir, queue_dir, "meaningful_failure")
            epoch_results.setdefault("meaningful_failures", 0)
            epoch_results["meaningful_failures"] += 1
            if miner_hotkey not in meaningful_failure_hotkeys:
                meaningful_failure_hotkeys.append(miner_hotkey)
            chain.append_event({
                "type": "meaningful_failure_archived",
                "timestamp": time.time(),
                "miner_hotkey": miner_hotkey,
                "miner_github": result.get("miner_github", ""),
                "val_bpb": result["val_bpb"],
                "bundle_hash": result["bundle_hash"],
            })
            # Audit dispatcher: even close-margin meaningful_failures get a
            # probabilistic re-audit (helps detect "I almost won" gaming).
            if chain_dir is not None:
                try:
                    from validator.audit_scheduler import maybe_enqueue_audit
                    archived_path = queue_dir / "meaningful_failure" / bundle_id
                    job = maybe_enqueue_audit(
                        chain_dir=Path(chain_dir),
                        bundle_id=bundle_id,
                        miner_hotkey=miner_hotkey,
                        miner_github=result.get("miner_github", ""),
                        bundle_hash=result["bundle_hash"],
                        val_bpb=result["val_bpb"],
                        king_val_bpb=king_now.val_bpb if king_now else None,
                        quality_gain=result["quality_gain"],
                        classification="meaningful_failure",
                        proof_dir=archived_path,
                        noise_floor_margin=noise_floor_margin,
                        random_audit_rate=audit_random_rate,
                    )
                    if job is not None:
                        log_info(f"  audit enqueued: reason={job.reason}")
                except Exception as e:
                    log_warn(f"audit enqueue failed for {bundle_id}: {e}")
            continue

        if result["accepted"]:
            from chain_layer.interface import KingRecord
            king_before = chain.get_king()

            # KING TENURE GUARD: a freshly-crowned king must reign a minimum
            # number of blocks (RALPH_KING_MIN_TENURE_BLOCKS, default 300 — just
            # under sn40's ~360-block weight-set window) so it earns at least
            # one weight cycle before it can be dethroned. A challenger arriving
            # inside the incumbent's tenure is DEFERRED: left in queue/pending
            # (not archived, not crowned) and re-evaluated in a later epoch once
            # tenure lapses. crowned_at_block == 0 means legacy/unknown → no
            # protection (the bundle is crowned normally).
            if king_before is not None and getattr(king_before, "crowned_at_block", 0):
                tenure = int(os.environ.get("RALPH_KING_MIN_TENURE_BLOCKS", "300"))
                try:
                    age = chain.get_current_block() - king_before.crowned_at_block
                except Exception:
                    age = tenure  # can't read block → don't block dethroning
                if age < tenure:
                    log_info(
                        f"king {king_before.miner_hotkey[:12]}… in protected tenure "
                        f"({age}/{tenure} blocks) — deferring challenger "
                        f"{miner_hotkey[:12]}… (left pending, re-evaluated later)"
                    )
                    continue  # do NOT crown, do NOT archive — re-eval next epoch

            gh = result.get("miner_github", "")
            who = f"{gh} ({miner_hotkey[:12]}...)" if gh else f"{miner_hotkey[:20]}..."
            log_info(f"NEW KING: {who} val_bpb={result['val_bpb']:.4f}")
            new_king = KingRecord(
                miner_hotkey=miner_hotkey,
                bundle_hash=result["bundle_hash"],
                val_bpb=result["val_bpb"],
                benchmark_accuracy=result["benchmark_accuracy"],
                compute_cost=result["score_report"].compute_cost,
                crowned_at=time.time(),
                crowned_at_block=_safe_current_block(chain),
                # proof_dir is updated AFTER archive_bundle moves the
                # bundle to scored/, so the king pointer survives the move.
                proof_dir=str(queue_dir / "scored" / bundle_dir.name),
            )
            if king_before:
                import dataclasses
                new_king.previous_king = dataclasses.asdict(king_before)
            chain.set_king(new_king)
            # Persist the reigning king's quality bar so it survives a later
            # clear_king — a challenger arriving on an empty throne must still
            # beat it (no gain-0 free crown). Not updated on reinstate-demotion.
            chain.set_high_water_mark(new_king.val_bpb, new_king.benchmark_accuracy)
            epoch_results["accepted"] += 1
            king_change_hotkey = miner_hotkey
            # Audit dispatcher: this king_change gets a probabilistic
            # re-audit, always re-audited if margin was close to noise floor.
            if chain_dir is not None:
                try:
                    from validator.audit_scheduler import maybe_enqueue_audit
                    job = maybe_enqueue_audit(
                        chain_dir=Path(chain_dir),
                        bundle_id=bundle_id,
                        miner_hotkey=miner_hotkey,
                        miner_github=result.get("miner_github", ""),
                        bundle_hash=result["bundle_hash"],
                        val_bpb=result["val_bpb"],
                        king_val_bpb=king_before.val_bpb if king_before else None,
                        quality_gain=result["quality_gain"],
                        classification="king_change",
                        proof_dir=queue_dir / "scored" / bundle_id,
                        noise_floor_margin=noise_floor_margin,
                        random_audit_rate=audit_random_rate,
                    )
                    if job is not None:
                        log_info(f"  audit enqueued: reason={job.reason}")
                except Exception as e:
                    log_warn(f"audit enqueue failed for {bundle_id}: {e}")

            # Auto-merge the winning PR + tag + release on RalphLabsAI/recipe.
            # Requires RALPH_BOT_GH_TOKEN. Failures here don't unwind the king
            # — the on-chain crown already happened.
            bot_token = os.environ.get("RALPH_BOT_GH_TOKEN", "")
            pr_url = result.get("pr_url", "")
            # Read .hf_pr.json once up front so we can use the actual bundle
            # repo_id for the release-notes link (not just whatever the
            # validator's RALPH_HF_REPO env var happens to say). Falls back to
            # the env var if the bundle didn't ship via an HF PR.
            hf_pr_info: dict | None = None
            hf_pr_path = bundle_dir / ".hf_pr.json"
            if hf_pr_path.exists():
                try:
                    import json as _json
                    hf_pr_info = _json.loads(
                        hf_pr_path.read_text(encoding="utf-8", errors="replace")
                    )
                except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                    log_warn(f"failed to read .hf_pr.json for {bundle_id}: {e}")
            hf_bundle_repo = (
                (hf_pr_info or {}).get("repo_id")
                or os.environ.get("RALPH_HF_REPO", "RalphLabsAI/proof-bundles")
            )
            if bot_token and pr_url:
                # Pull the hypothesis (short) from submission.json and the full
                # rationale.md (if present) so the release notes can lead with
                # the miner's reasoning.
                #
                # NOTE: `hypothesis` is currently NOT covered by the miner's
                # signature on submission.json (followup: fold it into the
                # signed payload). For now we treat it as informational; the
                # rendering side (validator/github_bot.py) prefixes it with
                # "_Miner's claim..._" so readers know it's unverified.
                hypothesis = ""
                rationale_md = ""
                sub_path = bundle_dir / "submission.json"
                try:
                    import json as _json
                    sub = _json.loads(sub_path.read_text(encoding="utf-8", errors="replace"))
                    hypothesis = sub.get("hypothesis", "")
                except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                    log_warn(f"failed to load submission.json for {bundle_id}: {e}")
                rm = bundle_dir / "rationale.md"
                if rm.exists():
                    try:
                        # Defense-in-depth size cap: miner-side caps rationale
                        # at 64KB; we allow some slack (200KB) but anything
                        # larger is treated as adversarial and dropped.
                        if rm.stat().st_size > 200_000:
                            log_warn(
                                f"rationale.md for {bundle_id} too large "
                                f"({rm.stat().st_size} bytes) — skipping"
                            )
                        else:
                            rationale_md = rm.read_text(encoding="utf-8", errors="replace")
                    except (FileNotFoundError, UnicodeDecodeError, OSError) as e:
                        log_warn(f"failed to load rationale.md for {bundle_id}: {e}")
                # Clamp the rationale we hand off to merge_and_release to 30KB
                # to keep release notes manageable even if it slipped past
                # earlier caps.
                rationale_summary = rationale_md[:30_000] if rationale_md else ""
                try:
                    from validator.github_bot import merge_and_release
                    rel = merge_and_release(
                        pr_url=pr_url,
                        metrics={
                            "val_bpb": result["val_bpb"],
                            "quality_gain": result["quality_gain"],
                            "compute_cost_h100h": result["score_report"].compute_cost,
                            "benchmark_accuracy": result["benchmark_accuracy"],
                            "miner_hotkey": miner_hotkey,
                            "miner_github": result.get("miner_github", ""),
                            "bundle_hash": result["bundle_hash"],
                            "hypothesis": hypothesis,
                            "rationale_md": rationale_summary,
                            "hf_bundle_url": (
                                f"https://huggingface.co/datasets/{hf_bundle_repo}"
                                f"/tree/main/submissions/{result['bundle_hash'][:16]}"
                            ),
                        },
                        token=bot_token,
                    )
                    log_info(f"recipe released: {rel.tag} ({rel.release_url})")
                    chain.append_event({
                        "type": "recipe_released",
                        "timestamp": time.time(),
                        "tag": rel.tag,
                        "release_url": rel.release_url,
                        "merge_sha": rel.merge_sha,
                        "miner_hotkey": miner_hotkey,
                        "miner_github": result.get("miner_github", ""),
                    })
                except Exception as e:
                    log_warn(f"recipe release failed: {e}")
            elif pr_url and not bot_token:
                log_warn(f"king changed with PR {pr_url} but RALPH_BOT_GH_TOKEN unset — manual merge needed")

            # Merge the corresponding HF PR (if the bundle came from one).
            # Reuse hf_pr_info read at the top of the king-change block.
            if hf_pr_info is not None:
                hf_pr = hf_pr_info
                # Distinct local — must NOT shadow run_epoch's `hf_token` param,
                # which the end-of-epoch audit-report HF publish reuses below.
                hf_bot_token = os.environ.get("RALPH_BOT_HF_TOKEN") or os.environ.get("HF_TOKEN", "")
                if hf_bot_token:
                    from validator.hf_bot import merge_pr as hf_merge_pr
                    res = hf_merge_pr(
                        repo_id=hf_pr["repo_id"],
                        pr_num=hf_pr["pr_num"],
                        token=hf_bot_token,
                        comment=(
                            f"Crowned king. val_bpb={result['val_bpb']:.4f}, "
                            f"quality_gain={result['quality_gain']:+.4f}, "
                            f"miner={result.get('miner_github') or miner_hotkey[:12]}."
                        ),
                    )
                    if res.merged:
                        log_info(f"HF PR #{res.pr_num} merged on {hf_pr['repo_id']}")
                    else:
                        log_warn(f"HF PR #{res.pr_num} merge failed: {res.detail}")
                    chain.append_event({
                        "type": "hf_pr_merged" if res.merged else "hf_pr_merge_failed",
                        "timestamp": time.time(),
                        "repo_id": hf_pr["repo_id"],
                        "pr_num": res.pr_num,
                        "miner_hotkey": miner_hotkey,
                        "detail": res.detail,
                    })
                else:
                    log_warn(f"king changed but no HF token to merge PR #{hf_pr['pr_num']}")
        else:
            log_info(f"below threshold: {miner_hotkey[:20]}... gain={result['quality_gain']:+.4f}")
            _close_losing_prs(bundle_dir, f"below threshold (gain {result['quality_gain']:+.4f})")

        archive_bundle(bundle_dir, queue_dir, "scored")

    # §5.6 pool split: 90% king, 10% to meaningful_failures (equal split).
    # If no meaningful_failures this epoch, king gets 100%.
    round_scores = _apply_pool_split(chain, king_change_hotkey, meaningful_failure_hotkeys)

    # Merge any weights recovered from a previous rate-limited epoch so we don't
    # lose an unlanded meaningful_failure credit — but NEVER resurrect a king the
    # throne has since moved off of (that splits emission across two kings).
    current_king_hotkey = king_change_hotkey
    if current_king_hotkey is None:
        _sitting = chain.get_king()
        current_king_hotkey = _sitting.miner_hotkey if _sitting is not None else None
    round_scores = _merge_recovered_weights(round_scores, recovered, current_king_hotkey)

    # weights_set records whether the weight extrinsic actually landed this
    # epoch. It is INDEPENDENT of whether we build the audit report: the report
    # documents what the validator DECIDED (round_scores), and set_weights can
    # return early on rate-limit. We capture the bool here and stamp it into the
    # report envelope so an auditor can see decision-vs-landed separately.
    weights_set = False
    if round_scores:
        # Persist BEFORE attempting set_weights so a crash / rate-limit
        # mid-flight doesn't lose this epoch's credits.
        _save_pending_weights(chain, round_scores)
        # Show exactly what's being set (top entries) so validators can SEE the
        # weight decision in the logs, not just "setting weights".
        preview = ", ".join(
            f"{hk[:12]}…={w:.3f}"
            for hk, w in sorted(round_scores.items(), key=lambda kv: -kv[1])[:5]
        )
        log_info(f"WEIGHTS: setting {len(round_scores)} miner(s) [king 90% / mf 10%]: {preview}")
        ok = chain.set_weights(round_scores)
        weights_set = ok
        if ok:
            _clear_pending_weights(chain)
            log_info("WEIGHTS: ✓ set on-chain")
        else:
            log_warn(
                "WEIGHTS: deferred (rate-limited / failed) — pending_weights kept, "
                "retry next epoch (no credit lost)"
            )
    elif _burn_fallback_enabled():
        # BURN FALLBACK: nothing scoreable this epoch (no king, all rejected /
        # zero submissions). Still set weights every epoch so the validator
        # keeps its vTrust alive — burn the epoch's incentive to the owner uid
        # (default 0). Disable with RALPH_BURN_FALLBACK=0.
        log_info("WEIGHTS: no scoreable submissions — setting BURN weights (100% → uid 0)")
        weights_set = chain.set_burn_weights()
        log_info(
            "WEIGHTS: ✓ burn set on-chain" if weights_set
            else "WEIGHTS: burn deferred (rate-limited) — retry next epoch"
        )

    # validation-v2 Phase 1: validator audit report + on-chain anchor.
    # Build report_json from this epoch's scored results, hash + sign, anchor
    # the hash on-chain, and write the signed envelope locally.
    #
    # ORDERING (the fix): this block runs UNCONDITIONALLY after round_scores /
    # weight_snapshot are computed, regardless of whether set_weights() above
    # succeeded or rate-limited. A rate-limited epoch still produces a full
    # audit report (the validator's DECISION is what auditors replay; whether
    # the extrinsic landed is recorded separately via weights_set). set_weights
    # has no early-return that can reach here — we never return between the
    # weight block and this block.
    #
    # CRITICAL: the entire block is wrapped in try/except — an audit-report
    # failure (commit rate-limit, signing error, disk full, ...) must NEVER
    # break scoring or weight-setting, both of which have already completed
    # above. We log and continue. Gated behind `audit_reports_enabled`.
    if audit_reports_enabled and scored_results:
        try:
            _generate_audit_report(
                chain,
                scored_results=scored_results,
                weight_snapshot=round_scores,
                epoch_start_block=epoch_start_block,
                netuid=netuid,
                eval_seed=eval_seed,
                weights_set=weights_set,
                hf_publish_enabled=hf_publish_enabled,
                hf_audit_repo=hf_audit_repo,
                hf_token=hf_token,
            )
        except Exception as e:
            log_warn(f"audit-report generation failed (scoring/weights unaffected): {e}")
            log_debug(traceback.format_exc())

    return epoch_results


def _generate_audit_report(
    chain,
    *,
    scored_results: list[dict],
    weight_snapshot: dict[str, float],
    epoch_start_block: int,
    netuid: int,
    eval_seed: int,
    weights_set: bool = False,
    hf_publish_enabled: bool = False,
    hf_audit_repo: str | None = None,
    hf_token: str | None = None,
) -> None:
    """Build, hash, sign, on-chain-anchor, and persist the per-epoch audit
    report (validation-v2 Phase 1).

    Order (per the design's audit-anchor section): hash + sign the canonical
    report, anchor the hash on-chain BEFORE serving the report, record the
    commitment block into the envelope, then write the signed envelope locally.

    `weights_set` records whether the weight extrinsic landed this epoch; it is
    stamped into the envelope (NOT the signed report_json) so the report stays a
    pure record of the validator's decision while auditors can still see whether
    the on-chain weights matched it.

    Raises on any failure — the single caller wraps this in try/except so a
    failure here can't affect the already-completed scoring/weight-setting.
    """
    from validator.audit_report import (
        build_envelope,
        build_report_json,
        canonical_json,
        report_sha256,
        sign_report,
    )

    end_block = chain.get_current_block()
    epoch_id = f"{netuid}-{end_block}"
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    report_json = build_report_json(
        epoch_id=epoch_id,
        netuid=netuid,
        start_block=epoch_start_block,
        end_block=end_block,
        generated_at=generated_at,
        scored=scored_results,
        weight_snapshot=weight_snapshot,
        seed=eval_seed,
    )

    sha = report_sha256(report_json)

    # Sign with the validator hotkey Keypair when available (BittensorChain).
    # LocalChain has no wallet → signature is empty (local-write parity).
    signature = ""
    signer_hotkey = ""
    wallet = getattr(chain, "wallet", None)
    if wallet is not None:
        try:
            keypair = wallet.hotkey
            signature = sign_report(canonical_json(report_json), keypair)
            signer_hotkey = keypair.ss58_address
        except Exception as e:
            log_warn(f"audit-report signing failed (continuing unsigned): {e}")

    # Anchor on-chain BEFORE writing/serving the report (design: the commitment
    # is the trust anchor). commit_audit_root raises on failure.
    commitment_block = chain.commit_audit_root(sha)

    envelope = build_envelope(
        report_json,
        signature=signature,
        signer_hotkey=signer_hotkey,
        chain_commitment_block=commitment_block,
        weights_set=weights_set,
    )

    from validator.audit_report import write_report
    out_dir = getattr(chain, "chain_dir", None) or RALPH_ROOT
    report_path = write_report(
        envelope,
        Path(out_dir),
        hf_publish_enabled=hf_publish_enabled,
        hf_repo=hf_audit_repo,
        hf_token=hf_token,
    )
    if hf_publish_enabled:
        log_info(f"audit report HF-publish requested: {hf_audit_repo or 'RalphLabsAI/audit-reports'}")
    log_info(
        f"audit report committed: epoch={epoch_id} sha={sha[:16]}... "
        f"block={commitment_block} ({len(scored_results)} submissions) -> {report_path}"
    )


def _benchmark_selftest(ralph_root: Path) -> None:
    """Startup blind-forgeability check for the deployed benchmark (anti-gaming P0).

    If active_benchmark.json is content-forgeable (a blind score=+/-token_id sweep
    beats chance) AND the benchmark crown is enabled (RALPH_BENCHMARK_CROWN=1),
    REFUSE to start — running a gameable crown is worse than not starting; the
    operator must regenerate a content-whitened benchmark first. If the crown is off
    (the default), the scoring neutralization already protects the crown, so just
    warn. A self-test *error* (bad file / missing dep) never blocks startup.
    """
    try:
        from eval.benchmark import benchmark_blind_forgeable

        bp = ralph_root / "eval" / "private" / "active_benchmark.json"
        if not bp.exists():
            return
        forge, why = benchmark_blind_forgeable(json.loads(bp.read_text()))
        if forge and os.environ.get("RALPH_BENCHMARK_CROWN") == "1":
            raise SystemExit(
                f"REFUSING TO START: benchmark self-test failed — {why}. "
                "RALPH_BENCHMARK_CROWN=1 with a forgeable active_benchmark.json is "
                "exploitable; regenerate a content-whitened benchmark or unset "
                "RALPH_BENCHMARK_CROWN."
            )
        (log_warn if forge else log_info)(f"benchmark self-test: {why}")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — a self-test *error* must never block startup
        log_warn(f"benchmark self-test skipped: {e}")


def _run_idle_audit(chain, ralph_root: Path, noise_floor_margin: float):
    """Drain ONE pending audit job on an idle epoch (no submissions scored).

    Was dead code (run_pending_audits called nowhere -> 0 fraud/blacklist events).
    On a re-derivation mismatch run_pending_audits appends submission_fraud + blacklists
    the miner. Off: RALPH_AUDIT_LOOP_OFF=1 (default on). NOTE the audit is a proxy re-run
    until canonical-data / checkpoint-bound re-derivation land (plan P2/P4). Returns the
    run_pending_audits result dict, or None if disabled/errored.
    """
    if os.environ.get("RALPH_AUDIT_LOOP_OFF") == "1":
        return None
    try:
        from validator.audit_scheduler import run_pending_audits

        a = run_pending_audits(
            chain, ralph_root, ralph_root / "chain",
            noise_floor_margin=noise_floor_margin, limit=1,
        )
        if a.get("processed"):
            log_info(
                f"audit: processed {a['processed']} "
                f"(passed {a.get('passed', 0)}, failed {a.get('failed', 0)})"
            )
        return a
    except Exception as e:  # noqa: BLE001 — audit must never crash the epoch loop
        log_warn(f"audit loop error: {e}")
        return None


def main():
    p = argparse.ArgumentParser(description="Ralph continuous validator service")
    p.add_argument("--queue-dir", type=Path, default=RALPH_ROOT / "queue")
    p.add_argument("--epoch-seconds", type=int, default=120,
                   help="Seconds between epochs (default: 120, ~10 blocks)")
    p.add_argument("--noise-floor", type=float, default=0.013,
                   help="val_bpb margin for 'decisively beats king' (default: 0.013 from H100 calibration)")
    p.add_argument(
        "--hf-repo", default=os.environ.get("RALPH_HF_REPO", DEFAULT_HF_REPO),
        help=(
            f"HuggingFace dataset repo to poll (default: {DEFAULT_HF_REPO}). "
            "Set to empty string to disable."
        ),
    )
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace API token (defaults to $HF_TOKEN)")
    p.add_argument("--hf-limit", type=int, default=10,
                   help="Max bundles to download per epoch (default: 10)")
    p.add_argument(
        "--hf-publish-audit",
        action="store_true",
        default=os.environ.get("RALPH_HF_PUBLISH_AUDIT", "").strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Publish per-epoch audit reports to the HF dataset repo "
            "(prod only; default off). Also enabled via RALPH_HF_PUBLISH_AUDIT=1."
        ),
    )
    p.add_argument(
        "--hf-audit-repo",
        default=os.environ.get("RALPH_HF_AUDIT_REPO", "RalphLabsAI/audit-reports"),
        help="HF dataset repo for audit reports (default: RalphLabsAI/audit-reports)",
    )
    p.add_argument("--once", action="store_true", help="Run one epoch then exit")
    args = p.parse_args()

    log_info("=" * 60)
    log_info("  Ralph Validator Service")
    log_info("=" * 60)

    chain = get_chain(RALPH_ROOT)
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    (args.queue_dir / "pending").mkdir(exist_ok=True)

    king = chain.get_king()
    if king:
        log_info(f"current king: {king.miner_hotkey[:20]}... val_bpb={king.val_bpb:.4f}")
    else:
        log_info("no king yet — first submission will be crowned")

    log_info(f"queue: {args.queue_dir}")
    log_info(f"epoch interval: {args.epoch_seconds}s")
    log_info(f"noise floor margin: {args.noise_floor}")
    if args.hf_repo:
        log_info(f"HF Hub poll: {args.hf_repo} (limit {args.hf_limit}/epoch)")
    else:
        log_info("HF Hub poll: disabled")
    log_info(f"submit bundles to: {args.queue_dir / 'pending' / '<bundle_id>/'}")
    # Benchmark blind-forgeability self-test (anti-gaming P0). Warns if the deployed
    # active_benchmark.json is content-forgeable; REFUSES to start if it's forgeable
    # AND the crown is enabled (RALPH_BENCHMARK_CROWN=1) — see _benchmark_selftest.
    _benchmark_selftest(RALPH_ROOT)
    log_info("")

    epoch = 0
    while not SHUTDOWN:
        epoch += 1
        log_info(f"--- epoch {epoch} ---")

        try:
            result = run_epoch(
                chain,
                args.queue_dir,
                args.noise_floor,
                hf_repo=args.hf_repo or None,
                hf_token=args.hf_token,
                hf_limit=args.hf_limit,
                hf_publish_enabled=args.hf_publish_audit,
                hf_audit_repo=args.hf_audit_repo,
            )
            if result["submissions"] > 0:
                mf = result.get("meaningful_failures", 0)
                mf_str = f", {mf} meaningful failures" if mf else ""
                log_info(f"epoch {epoch}: {result['submissions']} submissions, "
                         f"{result['accepted']} accepted, {result['rejected']} rejected{mf_str}")
            else:
                log_info(f"epoch {epoch}: no pending submissions")
                # Idle: drain ONE pending audit job (was dead code -> 0 fraud events).
                # See _run_idle_audit; off via RALPH_AUDIT_LOOP_OFF=1.
                _run_idle_audit(chain, RALPH_ROOT, args.noise_floor)
        except Exception as e:
            log_err(f"epoch {epoch} failed: {e}")
            log_debug(traceback.format_exc())

        # Periodic on-chain standing so operators see they're registered + active
        # (addresses validators worried by quiet logs). Cheap: one metagraph sync.
        if epoch == 1 or epoch % 10 == 0:
            try:
                chain.sync()
            except Exception:
                pass
            _log_validator_standing(chain)

        if args.once:
            break

        log_info(f"sleeping {args.epoch_seconds}s until next epoch...")
        for _ in range(args.epoch_seconds):
            if SHUTDOWN:
                break
            time.sleep(1)

    log_info("validator service stopped")


if __name__ == "__main__":
    main()
