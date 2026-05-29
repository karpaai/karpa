#!/usr/bin/env python3
"""
End-to-end miner script — runs on a remote H100 to participate in Karpa.

Flow:
  1. Hash the patch file (or empty patch for baseline)
  2. Request handshake nonce — commits (hotkey, patch_hash, nonce) on-chain
  3. Run the canonical proof test (training in the official Docker image, or
     direct Python with --no-docker for testing)
  4. Assemble + sign the submission bundle
  5. Upload the bundle to HuggingFace Hub for validators to pick up

After this finishes, validators worldwide will find the bundle by polling
the HF dataset repo and score it on their side.

Usage on a remote H100:
    # Setup .env with BT_NETWORK=test BT_NETUID=16 BT_WALLET=... BT_HOTKEY=... HF_TOKEN=...
    python scripts/miner_run.py --patch patches/raise_lr.diff --label round1
    python scripts/miner_run.py --baseline --label baseline   # empty patch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401  — injects KARPA_RECIPE_DIR onto sys.path

from chain_layer.config import get_chain
from proof.runner import run_proof_test
from miner.submit import sign_submission
from miner.hub import upload_bundle


KARPA_ROOT = Path(__file__).resolve().parent.parent


def _get_hotkey_ss58(wallet_name: str, hotkey_name: str) -> str:
    import bittensor as bt
    w = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    return w.hotkey.ss58_address


def run_miner(
    patch_path: Path | None,
    label: str,
    config_path: str,
    tier: str,
    hf_repo: str,
    hf_token: str | None,
    seed: int,
    skip_upload: bool,
) -> dict:
    import os

    wallet_name = os.environ.get("BT_WALLET", "default")
    hotkey_name = os.environ.get("BT_HOTKEY", "default")
    miner_gh = os.environ.get("KARPA_MINER_GH", "")
    miner_hotkey = _get_hotkey_ss58(wallet_name, hotkey_name)

    print(f"\n{'='*60}")
    print(f"  KARPA MINER — {label}")
    print(f"{'='*60}")
    print(f"  wallet: {wallet_name}/{hotkey_name}")
    print(f"  hotkey: {miner_hotkey}")
    if miner_gh:
        print(f"  gh:     {miner_gh}")
    print(f"  config: {config_path}")
    print(f"  tier:   {tier}")
    print(f"  hf:     {hf_repo}")

    chain = get_chain(KARPA_ROOT)
    if not chain.is_hotkey_registered(miner_hotkey):
        raise RuntimeError(
            f"hotkey {miner_hotkey} is NOT registered on netuid "
            f"{getattr(chain, 'netuid', '?')}. Register first via btcli."
        )

    # ---- 1. Prepare submission directory -----------------------------------
    sub_dir = KARPA_ROOT / f"runs/miner/{label}_sub"
    proof_dir = KARPA_ROOT / f"runs/miner/{label}_proof"
    for d in [sub_dir, proof_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    target_patch = sub_dir / "patch.diff"
    if patch_path is None:
        target_patch.write_text("")
        patch_text = ""
    else:
        patch_text = patch_path.read_text()
        target_patch.write_text(patch_text)
    patch_hash = hashlib.sha256(patch_text.encode()).hexdigest()
    print(f"  patch_hash: {patch_hash[:24]}...  ({len(patch_text)} bytes)")

    # ---- 2. Handshake — commit on-chain ------------------------------------
    print(f"\n[1/5] handshake — committing (hotkey, patch_hash, nonce) on-chain...")
    nonce = chain.request_handshake_nonce(miner_hotkey, patch_hash)
    print(f"      nonce: {nonce[:32]}...")

    (sub_dir / "proof_request.json").write_text(json.dumps({
        "handshake_nonce": nonce,
        "seed": seed,
        "config_path": config_path,
        "miner_hotkey": miner_hotkey,
    }, indent=2))

    # ---- 3. Run the proof test ---------------------------------------------
    print(f"\n[2/5] proof test — running canonical training...")
    t0 = time.time()
    bundle = run_proof_test(
        karpa_root=KARPA_ROOT,
        submission_dir=sub_dir,
        out_dir=proof_dir,
        tier=tier,
    )
    elapsed = time.time() - t0
    print(f"      bundle_hash: {bundle.bundle_hash[:24]}...")
    print(f"      elapsed:     {elapsed:.1f}s")

    # ---- 4. Sign submission ------------------------------------------------
    print(f"\n[3/6] signing submission...")
    sig = sign_submission(KARPA_ROOT, miner_hotkey, bundle.bundle_hash, nonce)
    print(f"      signed by {sig['public_key_hex'][:24]}...")

    # ---- 5. Open PR against karpaai/recipe (before HF upload so it ends up
    #         in the HF PR's submission.json) -------------------------------
    pr_url = ""
    fork_url = os.environ.get("KARPA_RECIPE_FORK", "")
    gh_token = os.environ.get("KARPA_MINER_GH_TOKEN", "")
    upstream = os.environ.get("KARPA_RECIPE_UPSTREAM", "karpaai/recipe")
    if not patch_text.strip():
        print(f"\n[4/6] skipping recipe PR (baseline submission, empty patch)")
    elif skip_upload:
        print(f"\n[4/6] skipping recipe PR (--skip-upload also implies no PR)")
    elif not fork_url or not gh_token:
        print(f"\n[4/6] WARNING: KARPA_RECIPE_FORK or KARPA_MINER_GH_TOKEN missing — not opening recipe PR")
    else:
        print(f"\n[4/6] opening recipe PR against {upstream}...")
        from miner.github_pr import open_recipe_pr
        try:
            pr_url = open_recipe_pr(
                patch_text=patch_text,
                bundle_hash=bundle.bundle_hash,
                miner_hotkey=miner_hotkey,
                miner_github=miner_gh,
                hf_bundle_url="",  # not known yet; HF PR is opened next
                signature_hex=sig["signature_hex"],
                fork_url=fork_url,
                token=gh_token,
                upstream=upstream,
            )
            print(f"      recipe PR: {pr_url}")
        except Exception as e:
            print(f"      WARNING: recipe PR open failed ({e}). Submission still uploaded to HF.")

    submission = {
        "miner_hotkey": miner_hotkey,
        "miner_github": miner_gh,
        "handshake_nonce": nonce,
        "patch_path": str(target_patch),
        "proof_dir": str(proof_dir),
        "bundle_hash": bundle.bundle_hash,
        "signature_hex": sig["signature_hex"],
        "public_key_hex": sig["public_key_hex"],
        "submitted_at": time.time(),
        "label": label,
        "pr_url": pr_url,
        "hf_bundle_url": "",  # filled by validator/log only; the PR itself IS the bundle on HF
    }
    (proof_dir / "submission.json").write_text(json.dumps(submission, indent=2, sort_keys=True))

    # ---- 6. Upload bundle as a single HF PR (includes submission.json) -----
    if skip_upload:
        print(f"\n[5/6] skipping HF upload (--skip-upload)")
        url = None
    else:
        print(f"\n[5/6] uploading bundle to HF Hub {hf_repo} as PR...")
        url = upload_bundle(proof_dir, repo_id=hf_repo, token=hf_token)

    # ---- 7. Done -----------------------------------------------------------
    print(f"\n[6/6] DONE")
    print(f"  bundle_hash: {bundle.bundle_hash}")
    print(f"  proof_dir:   {proof_dir}")
    if url:
        print(f"  hf url:      {url}")
    if pr_url:
        print(f"  pr:          {pr_url}")
    print(f"\nValidators will now find this on HF Hub and score it.")
    print(f"Track status: tail -f {KARPA_ROOT}/chain*/events.jsonl  (on validator host)")

    return {
        "miner_hotkey": miner_hotkey,
        "bundle_hash": bundle.bundle_hash,
        "patch_hash": patch_hash,
        "nonce": nonce,
        "pr_url": pr_url,
        "proof_dir": str(proof_dir),
        "hf_url": url,
        "elapsed_s": elapsed,
    }


def main() -> None:
    import os

    p = argparse.ArgumentParser(description="Karpa end-to-end miner")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--patch", type=Path, help="Path to patch file to submit")
    g.add_argument("--baseline", action="store_true", help="Submit empty patch (baseline)")
    p.add_argument("--label", required=True, help="Human label for this run (used in paths)")
    p.add_argument("--config", default="configs/proxy_cpu_smoke.json",
                   help="Recipe config (default: proxy_cpu_smoke.json — use proxy_h100.json on H100)")
    p.add_argument("--tier", default="unverified", choices=["verified", "unverified"],
                   help="Attestation tier (verified requires CC; default: unverified)")
    p.add_argument("--hf-repo", default=os.environ.get("KARPA_HF_REPO", "karpaai/proof-bundles"),
                   help="HF dataset repo to upload to")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HF API token (defaults to $HF_TOKEN)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-upload", action="store_true",
                   help="Run locally but skip HF upload (for testing)")
    args = p.parse_args()

    result = run_miner(
        patch_path=args.patch,
        label=args.label,
        config_path=args.config,
        tier=args.tier,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        seed=args.seed,
        skip_upload=args.skip_upload,
    )
    print(f"\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
