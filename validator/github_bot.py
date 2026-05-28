"""Validator-side GitHub operations on karpaai/recipe.

Two responsibilities:
  1. verify_pr_matches_bundle — confirm the open PR's diff is byte-equal to
     the bundle's patch.diff. If a miner opened a PR with a different diff
     than what the proof test actually ran on, reject the submission.
  2. merge_and_release — when a submission is crowned king, squash-merge
     the PR, tag the merge commit `recipe-vX.Y.Z`, and publish a release
     with the metrics in the body.

Requires env var KARPA_BOT_GH_TOKEN — a PAT with `public_repo` scope on
karpaai/recipe (the merge actor — recommend a dedicated karpa-bot account).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

GH_API = "https://api.github.com"
RECIPE_REPO = "karpaai/recipe"


def _gh(method: str, path: str, token: str, body: dict | None = None, accept: str = "application/vnd.github+json") -> dict | str:
    url = f"{GH_API}{path}" if path.startswith("/") else path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", accept)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            if "diff" in accept or "patch" in accept:
                return raw.decode()
            return json.loads(raw.decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"github {method} {path} → {e.code}: {detail}") from None


def _parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """https://github.com/OWNER/REPO/pull/N → (OWNER, REPO, N)."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url.strip())
    if not m:
        raise ValueError(f"not a PR URL: {pr_url}")
    return m.group(1), m.group(2), int(m.group(3))


@dataclass
class PrVerifyResult:
    ok: bool
    detail: str


def verify_pr_matches_bundle(
    pr_url: str,
    bundle_patch_text: str,
    token: str,
    expected_repo: str = RECIPE_REPO,
) -> PrVerifyResult:
    """Verify the PR exists, is open against expected_repo:main, and that its
    diff is byte-equal to the bundle's patch.diff."""
    if not pr_url:
        return PrVerifyResult(False, "submission has no pr_url")
    try:
        owner, repo, num = _parse_pr_url(pr_url)
    except ValueError as e:
        return PrVerifyResult(False, str(e))
    if f"{owner}/{repo}".lower() != expected_repo.lower():
        return PrVerifyResult(False, f"PR points at {owner}/{repo}, expected {expected_repo}")

    try:
        pr = _gh("GET", f"/repos/{owner}/{repo}/pulls/{num}", token)
    except RuntimeError as e:
        return PrVerifyResult(False, f"PR fetch failed: {e}")

    if pr.get("state") != "open":
        return PrVerifyResult(False, f"PR state={pr.get('state')} (must be open)")
    if pr.get("base", {}).get("ref") != "main":
        return PrVerifyResult(False, f"PR base={pr.get('base', {}).get('ref')} (must be main)")

    # Fetch the diff
    try:
        pr_diff = _gh("GET", f"/repos/{owner}/{repo}/pulls/{num}", token, accept="application/vnd.github.diff")
    except RuntimeError as e:
        return PrVerifyResult(False, f"diff fetch failed: {e}")

    # Byte-equal against the bundle's patch. We compare sha256 of the
    # normalized whitespace-stripped diffs, because GitHub may add metadata
    # lines (index, mode bits) that aren't in the bundle's patch.
    def _normalize(d: str) -> bytes:
        # Drop git metadata lines that the bundle patch doesn't carry.
        kept = []
        for line in d.splitlines():
            if line.startswith(("index ", "new file mode ", "deleted file mode ", "old mode ", "new mode ")):
                continue
            kept.append(line.rstrip())
        return ("\n".join(kept) + "\n").encode()

    pr_sha = hashlib.sha256(_normalize(pr_diff)).hexdigest()
    bundle_sha = hashlib.sha256(_normalize(bundle_patch_text)).hexdigest()
    if pr_sha != bundle_sha:
        return PrVerifyResult(
            False,
            f"PR diff sha256 != bundle patch sha256 (pr={pr_sha[:12]}..., bundle={bundle_sha[:12]}...)",
        )
    return PrVerifyResult(True, f"PR #{num} matches bundle patch (sha={bundle_sha[:12]}...)")


def _latest_recipe_tag(token: str, repo: str = RECIPE_REPO) -> tuple[int, int, int]:
    """Return (major, minor, patch) of the latest recipe-vX.Y.Z tag, or (0,0,0)."""
    try:
        tags = _gh("GET", f"/repos/{repo}/tags?per_page=100", token)
    except RuntimeError:
        return (0, 0, 0)
    best = (0, 0, 0)
    for t in tags or []:
        m = re.match(r"recipe-v(\d+)\.(\d+)\.(\d+)$", t.get("name", ""))
        if not m:
            continue
        v = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if v > best:
            best = v
    return best


def _next_recipe_version(token: str, repo: str = RECIPE_REPO) -> str:
    major, minor, patch = _latest_recipe_tag(token, repo)
    if (major, minor, patch) == (0, 0, 0):
        return "recipe-v0.1.0"
    return f"recipe-v{major}.{minor}.{patch + 1}"


@dataclass
class ReleaseResult:
    tag: str
    release_url: str
    merge_sha: str


def merge_and_release(
    pr_url: str,
    metrics: dict,
    token: str,
    repo: str = RECIPE_REPO,
) -> ReleaseResult:
    """Squash-merge the PR, tag the merge commit, publish a release.

    metrics keys used in the release body:
      val_bpb, quality_gain, compute_cost_h100h, miner_hotkey,
      miner_github, bundle_hash, hf_bundle_url, wandb_url (optional).
    """
    owner, pr_repo, num = _parse_pr_url(pr_url)
    if f"{owner}/{pr_repo}".lower() != repo.lower():
        raise RuntimeError(f"PR points at {owner}/{pr_repo}, refusing to merge into {repo}")

    # 1. Merge the PR (squash). Include a Co-authored-by trailer so the
    # miner's GitHub account shows up on the recipe's Contributors graph
    # and gets contribution credit on their own profile.
    miner_gh = metrics.get("miner_github", "")
    body_with_credit = _release_body(metrics, pr_url)
    if miner_gh:
        # GitHub recognises the `username@users.noreply.github.com` form as
        # the canonical "no-reply" address for any user; commits credited to
        # it count toward the user's contribution graph.
        body_with_credit += (
            f"\n\nCo-authored-by: {miner_gh} <{miner_gh}@users.noreply.github.com>"
        )
    merge_resp = _gh(
        "PUT",
        f"/repos/{repo}/pulls/{num}/merge",
        token,
        {
            "merge_method": "squash",
            "commit_title": f"recipe submission #{num} — {miner_gh or metrics.get('miner_hotkey', '')[:12]}",
            "commit_message": body_with_credit,
        },
    )
    merge_sha = merge_resp.get("sha", "")

    # 2. Compute next version and create the tag on the merge commit
    version = _next_recipe_version(token, repo)
    _gh(
        "POST",
        f"/repos/{repo}/git/refs",
        token,
        {"ref": f"refs/tags/{version}", "sha": merge_sha},
    )

    # 3. Publish a release
    release = _gh(
        "POST",
        f"/repos/{repo}/releases",
        token,
        {
            "tag_name": version,
            "name": f"{version} — {metrics.get('miner_github') or metrics.get('miner_hotkey', '')[:12]}",
            "body": _release_body(metrics, pr_url),
            "draft": False,
            "prerelease": False,
        },
    )

    return ReleaseResult(
        tag=version,
        release_url=release.get("html_url", ""),
        merge_sha=merge_sha,
    )


def _release_body(m: dict, pr_url: str) -> str:
    lines = ["## Metrics", ""]
    if "val_bpb" in m:
        lines.append(f"- **val_bpb:** `{m['val_bpb']:.4f}`")
    if "quality_gain" in m:
        lines.append(f"- **quality_gain vs previous king:** `{m['quality_gain']:+.4f}`")
    if "compute_cost_h100h" in m:
        lines.append(f"- **compute_cost (H100-hours):** `{m['compute_cost_h100h']:.4f}`")
    if "benchmark_accuracy" in m:
        lines.append(f"- **benchmark_accuracy:** `{m['benchmark_accuracy']:.3f}`")

    lines += ["", "## Attribution", ""]
    if "miner_github" in m and m["miner_github"]:
        lines.append(f"- **GitHub:** @{m['miner_github']}")
    if "miner_hotkey" in m:
        lines.append(f"- **hotkey:** `{m['miner_hotkey']}`")
    if "bundle_hash" in m:
        lines.append(f"- **bundle_hash:** `{m['bundle_hash']}`")

    lines += ["", "## Links", ""]
    lines.append(f"- **PR:** {pr_url}")
    if m.get("hf_bundle_url"):
        lines.append(f"- **HF proof bundle:** {m['hf_bundle_url']}")
    if m.get("wandb_url"):
        lines.append(f"- **wandb run:** {m['wandb_url']}")

    return "\n".join(lines)
