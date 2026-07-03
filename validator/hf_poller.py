"""
HuggingFace Hub poller — fetches new submission bundles into the local queue.

The validator service calls poll_hub() each epoch to discover bundles that
miners have uploaded to the public dataset repo. New ones get downloaded
into `queue/pending/<bundle_hash>/` where the existing local-queue logic
picks them up.

State is tracked in queue/hf_state.json so we don't re-download already-
processed bundles after restarts. Each processed bundle is stamped with the
VALIDATOR_VERSION it was judged under; on a version bump, older-version entries
are reprocessed (re-downloaded + re-validated) so evaluation stays fair across
a logic upgrade.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from validator.version import VALIDATOR_VERSION

DEFAULT_REPO = "RalphLabsAI/proof-bundles"

# --- DoS intake caps (single-validator crash protection) ----------------------
# A miner bundle is downloaded, decrypted, and tar.gz-extracted into the ONE
# validator. An unbounded extract is a crash-DoS (a tiny .gz can decompress to
# terabytes — a decompression bomb — or ship one giant member). Hard-cap the
# encrypted size, the decrypted blob, and — streamed, so it never expands in RAM
# — the total decompressed bytes, BEFORE proof.runner unpacks it. Env-tunable.
_MAX_ENC_BYTES = int(os.environ.get("RALPH_MAX_ENC_BYTES", str(2 << 30)))          # 2 GiB
_MAX_BLOB_BYTES = int(os.environ.get("RALPH_MAX_BLOB_BYTES", str(2 << 30)))        # 2 GiB
_MAX_DECOMPRESSED_BYTES = int(os.environ.get("RALPH_MAX_DECOMPRESSED_BYTES", str(3 << 30)))  # 3 GiB

# --- HF fetch timeout (hang / slowloris protection) ---------------------------
# huggingface_hub has no total-download timeout: a fetch that stalls mid-stream
# (the socket lingers with no data) blocks the ENTIRE epoch loop indefinitely — a
# single dead/slow bundle froze the validator for ~35min, and a miner could upload
# a deliberately-slow bundle to freeze scoring (DoS). A SIGALRM wall-clock cap
# bounds each fetch: on expiry it's treated as transient (retried), and a bundle
# that stalls _MAX_STALLS times in a row is marked done so it can't churn. The
# default is generous so large legit bundles (checkpoints) still finish.
_HF_FETCH_TIMEOUT_S = int(os.environ.get("RALPH_HF_FETCH_TIMEOUT_S", "300"))  # 5 min
_MAX_STALLS = int(os.environ.get("RALPH_HF_MAX_STALLS", "3"))
_STALL_COUNTS: dict[str, int] = {}  # bundle_id -> consecutive stalls (in-process)


class _FetchTimeout(Exception):
    """A single HF fetch (list/download) exceeded _HF_FETCH_TIMEOUT_S."""


def _with_timeout(seconds: int, fn, *args, **kwargs):
    """Run fn under a hard SIGALRM wall-clock cap and restore the prior handler.

    Main-thread only (signals can't be armed off it); off the main thread it runs
    uncapped — the service loop that calls poll_hub IS the main thread, so the cap
    applies in production. A signal interrupts a blocked socket read (EINTR), which
    is exactly the stalled-download case.
    """
    import signal
    import threading

    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        return fn(*args, **kwargs)

    def _fire(signum, frame):
        raise _FetchTimeout(f"HF fetch exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _assert_blob_safe(blob: bytes) -> None:
    """Reject an oversized / decompression-bomb decrypted blob before unpack.
    Streams the gunzip with a hard byte cap so a bomb never expands in memory.
    Raises ValueError on violation (the caller treats it as a decrypt/unpack fail)."""
    import gzip
    import io

    if len(blob) > _MAX_BLOB_BYTES:
        raise ValueError(f"decrypted blob {len(blob)} B > {_MAX_BLOB_BYTES} B cap")
    total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(blob)) as g:
            while True:
                chunk = g.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_DECOMPRESSED_BYTES:
                    raise ValueError(
                        f"decompressed size exceeds {_MAX_DECOMPRESSED_BYTES} B cap "
                        f"(decompression bomb)"
                    )
    except OSError as e:
        raise ValueError(f"bad gzip stream: {e}") from e


# miner.hub titles every submission PR "Submit proof bundle <hash[:12]>".
_TITLE_HASH_RE = re.compile(r"Submit proof bundle ([0-9a-f]{6,})", re.IGNORECASE)


def _submission_dirs(files: list[str]) -> set[str]:
    """Set of submissions/<bundle_id>/ directory ids present in a file listing."""
    return {
        f.split("/")[1]
        for f in files
        if f.startswith("submissions/") and len(f.split("/")) >= 3
    }


def _pr_own_bundles(title: str, pr_dirs: set[str], main_dirs: set[str]) -> list[str]:
    """Identify the bundle dir(s) a PR actually introduces.

    list_repo_files(revision="refs/pr/N") returns the *cumulative* tree (the
    base main at PR-open time + the PR's added files), so the raw dir set is not
    the PR's own bundle. Primary signal: the PR title (`Submit proof bundle
    <hash12>`, set by miner.hub) — match its hash prefix to a dir at the PR ref.
    Fallback (non-standard title): dirs present at the ref but not on main.
    The fallback is avoided when the title parses, so a bundle deleted from main
    (e.g. a reverted mock) can't be mis-attributed to a later PR.
    """
    m = _TITLE_HASH_RE.search(title or "")
    if m:
        prefix = m.group(1).lower()
        matches = sorted(d for d in pr_dirs if d.lower().startswith(prefix))
        if matches:
            return matches
    return sorted(pr_dirs - main_dirs)


def _state_path(queue_dir: Path) -> Path:
    return queue_dir / "hf_state.json"


def _migrate_state(raw: dict) -> dict:
    """Normalise to {"validator_version": str, "processed": {bundle_id: version}}.

    Legacy state stored `processed` as a flat list of bundle_ids with no version;
    those are mapped to "legacy" so they re-process under the current version.
    """
    processed = raw.get("processed", {})
    if isinstance(processed, list):
        processed = {bid: "legacy" for bid in processed}
    elif not isinstance(processed, dict):
        processed = {}
    return {
        "validator_version": raw.get("validator_version", "legacy"),
        "processed": processed,
    }


def _load_state(queue_dir: Path) -> dict:
    p = _state_path(queue_dir)
    if not p.exists():
        return {"validator_version": VALIDATOR_VERSION, "processed": {}}
    try:
        return _migrate_state(json.loads(p.read_text()))
    except Exception:
        return {"validator_version": VALIDATOR_VERSION, "processed": {}}


def _save_state(queue_dir: Path, state: dict) -> None:
    _state_path(queue_dir).write_text(json.dumps(state, indent=2))


def list_remote_submissions(repo_id: str, token: Optional[str] = None) -> list[dict]:
    """Return the open HF PRs against the dataset, oldest-first.

    Each entry is a dict with bundle_id (= directory prefix under submissions/),
    pr_num, git_ref, and created_at (ISO-8601 PR creation time). Only the bundle
    a PR actually *adds* is attributed to it (see _pr_own_bundles), not the whole
    cumulative tree the PR ref exposes. Ordering is by created_at then pr_num so
    validation is first-come-first-served (fair), not bundle-hash lexical order.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        discussions = api.get_repo_discussions(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"[hf_poller] get_repo_discussions failed: {e}")
        return []

    # Bundles already on main are the baseline a PR's tree is layered on top of;
    # used only as the fallback when a PR title doesn't carry the bundle hash.
    try:
        main_dirs = _submission_dirs(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception as e:
        print(f"[hf_poller] list main files failed: {e}")
        main_dirs = set()

    pending = []
    for d in discussions:
        if not d.is_pull_request:
            continue
        if d.status != "open":
            continue
        ref = d.git_reference  # e.g. "refs/pr/3"
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset", revision=ref)
        except Exception as e:
            print(f"[hf_poller] list PR #{d.num} files failed: {e}")
            continue
        created_at = d.created_at.isoformat() if getattr(d, "created_at", None) else None
        own = _pr_own_bundles(d.title, _submission_dirs(files), main_dirs)
        if not own:
            print(f"[hf_poller] PR #{d.num}: no own bundle identified (title={d.title!r}); skipping")
            continue
        if len(own) > 1:
            print(f"[hf_poller] PR #{d.num}: multiple candidate bundles {own}; taking all")
        for bid in own:
            pending.append(
                {"bundle_id": bid, "pr_num": d.num, "git_ref": ref, "created_at": created_at}
            )

    # First-come-first-validate: oldest PR first. created_at is ISO-8601 UTC so
    # lexical order is chronological; pr_num breaks ties / covers a missing time.
    pending.sort(key=lambda p: (p["created_at"] or "", p["pr_num"]))
    return pending


def _close_pr_wrong_key(
    repo_id: str, pr_num: Optional[int], token: Optional[str], bundle_id: str, err: str,
) -> None:
    """Comment on the HF PR and close it when a bundle can't be decrypted.

    A decrypt failure means the bundle was sealed to an outdated validator public
    key (the seal key was rotated), so it can never be processed. Instead of the PR
    churning silently — re-downloaded and re-failed every epoch — give the miner
    actionable feedback (the current pubkey to re-seal to) and close the PR.

    Best-effort: any failure here (missing pr_num, perms, network) is swallowed so
    it never blocks marking the bundle done, which is what actually stops the churn.
    """
    if pr_num is None:
        return
    try:
        from huggingface_hub import HfApi

        from proof import bundle_crypto
        msg = (
            "🔒 **Bundle decryption failed — re-seal required**\n\n"
            "The validator could not decrypt this proof bundle. This means it was "
            "sealed to an **outdated validator public key** (the seal key was "
            "rotated).\n\n"
            "Please re-seal to the **current** validator public key and open a new "
            "submission:\n\n"
            f"```\n{bundle_crypto.DEFAULT_VALIDATOR_PUBKEY}\n```\n\n"
            "(`DEFAULT_VALIDATOR_PUBKEY` in `proof/bundle_crypto.py`.) Closing this "
            f"PR — resubmit once re-sealed.\n\n<sub>decrypt error: {err[:200]}</sub>"
        )
        HfApi(token=token).change_discussion_status(
            repo_id=repo_id, discussion_num=pr_num, new_status="closed",
            repo_type="dataset", comment=msg,
        )
        print(
            f"[hf_poller] closed PR #{pr_num} ({bundle_id[:8]}): "
            "decrypt failed, re-seal instructions posted"
        )
    except Exception as e:  # noqa: BLE001 — PR-close is best-effort; never block dedup
        print(f"[hf_poller] WARN: could not close PR #{pr_num} ({bundle_id[:8]}): {e}")


def download_one(
    bundle_id: str,
    repo_id: str,
    dest_dir: Path,
    token: Optional[str] = None,
    git_ref: str = "main",
    pr_num: int | None = None,
    created_at: str | None = None,
) -> str:
    """Download all files for one bundle into dest_dir/<bundle_id>/.

    git_ref is the revision to read from — `main` for legacy direct-commit
    flows, `refs/pr/N` for PR-based submissions (the default since miners
    aren't org members on RalphLabsAI).

    Returns a status string: "ok" (bundle materialised), "decrypt_failed" (blob
    downloaded but could not be decrypted/unpacked — permanent, wrong/rotated seal
    key; the PR is closed with re-seal instructions), or "unavailable" (transient —
    list/download/network failure, safe to retry next epoch).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    from proof import bundle_crypto

    out = dest_dir / bundle_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    training_dir = out / "training"
    training_dir.mkdir()

    training_files = {
        "checkpoint.pt", "training_log.jsonl", "final_state.json",
        "wandb_metrics.json", "wandb_run_url.txt",
    }

    try:
        all_files = _with_timeout(
            _HF_FETCH_TIMEOUT_S, list_repo_files,
            repo_id, repo_type="dataset", token=token, revision=git_ref,
        )
        prefix = f"submissions/{bundle_id}/"
        bundle_files = [f for f in all_files if f.startswith(prefix)]
    except _FetchTimeout as e:
        print(f"[hf_poller] list TIMEOUT for {bundle_id} @ {git_ref}: {e}")
        return "stalled"
    except Exception as e:
        print(f"[hf_poller] list files failed for {bundle_id} @ {git_ref}: {e}")
        return "unavailable"

    if not bundle_files:
        print(f"[hf_poller] no files found for {bundle_id} @ {git_ref}")
        return "unavailable"

    cache = out / "_hf_cache"
    decrypt_err: Optional[str] = None  # set iff an enc blob downloaded but wouldn't decrypt/unpack
    stalled = False  # set iff a fetch timed out (transient — retry, never close the PR)
    enc_remote = f"{prefix}{bundle_crypto.ENC_FILENAME}"
    if enc_remote in bundle_files:
        # Encrypted submission: download the blob, decrypt with the validator
        # key, and unpack — reproduces the same dir a plaintext bundle would.
        try:
            local = _with_timeout(
                _HF_FETCH_TIMEOUT_S, hf_hub_download,
                repo_id=repo_id, filename=enc_remote, repo_type="dataset",
                local_dir=str(cache), token=token, revision=git_ref,
            )
            enc_bytes = Path(local).read_bytes()
            if len(enc_bytes) > _MAX_ENC_BYTES:
                raise ValueError(f"encrypted bundle {len(enc_bytes)} B > {_MAX_ENC_BYTES} B cap")
            blob = bundle_crypto.decrypt(enc_bytes)
            _assert_blob_safe(blob)  # DoS: cap decrypted + decompressed size before unpack
            bundle_crypto.unpack_bundle(blob, out)
            success = 1
        except _FetchTimeout as e:
            print(f"[hf_poller] download TIMEOUT for {bundle_id}: {e}")
            success = 0
            stalled = True
        except Exception as e:
            print(f"[hf_poller] decrypt/unpack failed for {bundle_id}: {e}")
            success = 0
            decrypt_err = str(e)
    else:
        # Legacy plaintext: download each file into the bundle dir.
        success = 0
        for remote_path in bundle_files:
            filename = remote_path.split("/")[-1]
            try:
                local = _with_timeout(
                    _HF_FETCH_TIMEOUT_S, hf_hub_download,
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=str(cache),
                    token=token,
                    revision=git_ref,
                )
                dest = (training_dir / filename) if filename in training_files else (out / filename)
                shutil.copy2(local, dest)
                success += 1
            except _FetchTimeout as e:
                print(f"[hf_poller] download TIMEOUT for {bundle_id} ({filename}): {e}")
                stalled = True
                break
            except Exception as e:
                print(f"[hf_poller] download {filename} failed: {e}")

    if cache.exists():
        shutil.rmtree(cache)
    training_dir.mkdir(exist_ok=True)  # encrypted bundles without training/ still get the dir

    if success == 0:
        shutil.rmtree(out)
        if decrypt_err is not None:
            # Wrong/rotated seal key — permanent. Tell the miner how to re-seal and
            # close the PR; the caller stamps it done so it stops churning every epoch.
            _close_pr_wrong_key(repo_id, pr_num, token, bundle_id, decrypt_err)
            return "decrypt_failed"
        if stalled:
            return "stalled"  # fetch timed out — transient; caller counts + eventually skips
        return "unavailable"  # transient (network / mid-upload) — safe to retry

    # Annotate which PR this came from so the validator can merge later.
    if pr_num is not None:
        (out / ".hf_pr.json").write_text(json.dumps(
            {
                "pr_num": pr_num,
                "git_ref": git_ref,
                "repo_id": "RalphLabsAI/proof-bundles",
                "created_at": created_at,
            },
            indent=2,
        ))
    return "ok"


def poll_hub(
    queue_dir: Path,
    repo_id: str = DEFAULT_REPO,
    token: Optional[str] = None,
    limit: int = 10,
) -> list[str]:
    """Fetch new submission bundles from HF into queue/pending/.

    Returns the list of newly-downloaded bundle IDs.
    """
    queue_dir = Path(queue_dir)
    pending = queue_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    state = _load_state(queue_dir)
    processed = state.get("processed", {})  # {bundle_id: validator_version}

    # A bundle counts as done only if it was judged by the CURRENT validator
    # version. Entries from an older version are reprocessed: not in `done` →
    # re-downloaded → re-judged → re-stamped below.
    done = {bid for bid, ver in processed.items() if ver == VALIDATOR_VERSION}

    remote_prs = list_remote_submissions(repo_id, token=token)  # oldest-first
    if not remote_prs:
        return []

    new = [p for p in remote_prs if p["bundle_id"] not in done]
    if not new:
        return []

    summary = [(p["bundle_id"][:8], f"PR#{p['pr_num']}") for p in new[:limit]]
    print(f"[hf_poller] found {len(new)} new PR-bundle(s) on HF Hub: {summary}")

    downloaded = []
    for sub in new[:limit]:
        bid = sub["bundle_id"]
        print(f"[hf_poller] downloading {bid} from PR #{sub['pr_num']} ({sub['git_ref']})...")
        result = download_one(bid, repo_id, pending, token=token,
                              git_ref=sub["git_ref"], pr_num=sub["pr_num"],
                              created_at=sub.get("created_at"))
        if result == "ok":
            downloaded.append(bid)
            processed[bid] = VALIDATOR_VERSION
            _STALL_COUNTS.pop(bid, None)
        elif result == "decrypt_failed":
            # Permanent (wrong/rotated seal key). Stamp done so we don't re-poll +
            # re-close it every epoch — the PR was closed with re-seal instructions.
            processed[bid] = VALIDATOR_VERSION
            _STALL_COUNTS.pop(bid, None)
            print(f"[hf_poller] {bid[:8]}: decrypt failed — marked done (PR closed; re-seal to current pubkey)")
        elif result == "stalled":
            # Fetch timed out (the cap already fired, so no hang). Retry a few times
            # in case it's a transient network blip; if it keeps stalling, mark it
            # done so one unfetchable bundle can't re-stall the loop every epoch.
            _STALL_COUNTS[bid] = _STALL_COUNTS.get(bid, 0) + 1
            if _STALL_COUNTS[bid] >= _MAX_STALLS:
                processed[bid] = VALIDATOR_VERSION
                _STALL_COUNTS.pop(bid, None)
                print(f"[hf_poller] {bid[:8]}: download stalled {_MAX_STALLS}x — marked done (unfetchable, skipping)")
            else:
                print(f"[hf_poller] {bid[:8]}: download stalled ({_STALL_COUNTS[bid]}/{_MAX_STALLS}) — will retry")
        else:  # "unavailable" — transient; leave un-stamped so it retries next epoch
            print(f"[hf_poller] skipped {bid} (unavailable — will retry)")

    state["validator_version"] = VALIDATOR_VERSION
    state["processed"] = processed
    _save_state(queue_dir, state)
    return downloaded


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--queue-dir", type=Path, default=Path("queue"))
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    new = poll_hub(args.queue_dir, args.repo, args.token, args.limit)
    print(f"\nDownloaded {len(new)} bundle(s): {new}")


if __name__ == "__main__":
    main()
