"""Open a karpaai/recipe PR from the miner's fork as part of submission.

The PR carries (a) the patch the miner is submitting, (b) the on-chain
metadata that ties it to the proof bundle on HuggingFace. Validators verify
the PR exists, is open, and that its diff byte-matches the bundle's
patch.diff before accepting the submission.

Required env vars on the miner box:
  KARPA_MINER_GH         — miner's GitHub username (e.g. "karpa1-gh")
  KARPA_MINER_GH_TOKEN   — PAT with `public_repo` scope; pushes to the
                           miner's fork, opens PR upstream.
  KARPA_RECIPE_FORK      — full URL of the miner's fork
                           (e.g. https://github.com/karpa1-gh/recipe.git)

Optional:
  KARPA_RECIPE_UPSTREAM  — defaults to "karpaai/recipe"
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

GH_API = "https://api.github.com"


def _gh_request(method: str, path: str, token: str, body: dict | None = None) -> dict:
    url = f"{GH_API}{path}" if path.startswith("/") else path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"github {method} {path} → {e.code}: {detail}") from None


def _redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Scrub Authorization headers and any explicit secret strings from text.

    Used to keep PATs out of exception messages that the broader script may
    forward to logs / bittensor metrics on push failure.
    """
    if not text:
        return text
    out = text
    # Redact `Authorization: bearer <token>` regardless of casing / quoting.
    import re
    out = re.sub(
        r"(?i)(authorization\s*[:=]\s*['\"]?\s*bearer\s+)\S+",
        r"\1<redacted>",
        out,
    )
    # Redact `http.extraheader=...bearer <token>...` style git -c values.
    out = re.sub(
        r"(?i)(http\.extraheader\s*=\s*authorization\s*:\s*bearer\s+)\S+",
        r"\1<redacted>",
        out,
    )
    # Redact any token-in-URL form `https://<token>@host` defensively.
    out = re.sub(r"(https?://)[^/@\s]+@", r"\1<redacted>@", out)
    for s in secrets:
        if s:
            out = out.replace(s, "<redacted>")
    return out


def _run(cmd: list[str], cwd: Path, secrets: tuple[str, ...] = ()) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        safe_cmd = _redact(" ".join(cmd), secrets)
        safe_stdout = _redact(r.stdout, secrets)
        safe_stderr = _redact(r.stderr, secrets)
        raise RuntimeError(
            f"git command failed: {safe_cmd}\nstdout: {safe_stdout}\nstderr: {safe_stderr}"
        )
    return r.stdout


def open_recipe_pr(
    patch_text: str,
    bundle_hash: str,
    miner_hotkey: str,
    miner_github: str,
    hf_bundle_url: str | None,
    signature_hex: str,
    fork_url: str,
    token: str,
    upstream: str = "karpaai/recipe",
    rationale_text: str = "",
) -> str:
    """Push a branch with the patch applied to the miner's fork, then open
    a PR upstream. Returns the PR URL.

    The branch name is deterministic: `submit/<bundle_hash[:12]>` so the
    same submission cannot accidentally collide with another, and so the
    validator can resolve the PR by bundle_hash alone if it has to.

    When `rationale_text` is provided, it goes above the bundle metadata so
    the human reviewer reads the hypothesis before the proof identifiers.
    """
    if not token:
        raise RuntimeError("KARPA_MINER_GH_TOKEN is not set — cannot open PR")
    if not patch_text.strip():
        # Baseline submissions with an empty patch can't be a PR; skip.
        return ""

    short_hash = bundle_hash[:12]
    branch = f"submit/{short_hash}"
    title = f"[submit] {short_hash} — val by hotkey {miner_hotkey[:12]}…"

    bundle_block = "\n".join(
        line
        for line in (
            f"**bundle_hash:** `{bundle_hash}`",
            f"**miner_hotkey:** `{miner_hotkey}`",
            f"**miner_github:** @{miner_github}" if miner_github else None,
            f"**hf_bundle:** {hf_bundle_url}" if hf_bundle_url else None,
            f"**signature:** `{signature_hex[:32]}…`",
            "",
            "Submitted via `scripts/miner_run.py`. The validator will compare",
            "this PR's diff against the bundle's `patch.diff` byte-for-byte.",
        )
        if line is not None
    )

    if rationale_text.strip():
        body = rationale_text.rstrip() + "\n\n---\n\n## Submission identifiers\n\n" + bundle_block
    else:
        body = bundle_block

    # GitHub rejects PR bodies > 65,536 chars with 422 *after* we push. Cap
    # well under the limit (leaving room for the truncation marker itself).
    # Apply the same cap to the commit message body so the squashed merge
    # commit doesn't carry a 60KB blob either.
    _BODY_CAP = 60_000
    _TRUNC_MARK = "\n\n_…rationale truncated for GitHub's 65KB body limit…_"
    if len(body) > _BODY_CAP:
        keep = _BODY_CAP - len(_TRUNC_MARK)
        body = body[:keep] + _TRUNC_MARK
    commit_body = body
    if len(commit_body) > _BODY_CAP:
        keep = _BODY_CAP - len(_TRUNC_MARK)
        commit_body = commit_body[:keep] + _TRUNC_MARK

    workdir = Path(tempfile.mkdtemp(prefix="karpa_pr_"))
    try:
        # 1. Clone the fork
        _run(["git", "clone", "--depth=1", fork_url, str(workdir)], cwd=Path("/tmp"))

        # 2. Make sure we have upstream main + base off it (in case the fork is stale)
        upstream_url = f"https://github.com/{upstream}.git"
        _run(["git", "remote", "add", "upstream", upstream_url], cwd=workdir)
        _run(["git", "fetch", "--depth=1", "upstream", "main"], cwd=workdir)
        _run(["git", "checkout", "-B", branch, "upstream/main"], cwd=workdir)

        # 3. Apply the patch
        patch_path = workdir / ".karpa_submission.patch"
        patch_path.write_text(patch_text)
        _run(["git", "apply", "--whitespace=nowarn", str(patch_path)], cwd=workdir)
        patch_path.unlink()

        # 4. Commit
        _run(["git", "add", "-A"], cwd=workdir)
        _run(
            [
                "git",
                "-c", f"user.name={miner_github or 'karpa-miner'}",
                "-c", f"user.email={miner_github or 'miner'}@karpa.local",
                "commit",
                "-m", title,
                "-m", commit_body,
            ],
            cwd=workdir,
        )

        # 5. Push to the miner's fork.
        #
        # Auth: pass the PAT through `git -c http.extraheader=...` so the
        # token never appears in argv (ps aux). The plain fork_url is the
        # only URL on the command line. We DO NOT use --force-with-lease
        # here: after a shallow clone there is no remote-tracking ref for
        # `submit/<hash>`, so lease checks reject the second push of the
        # same bundle. The branch name is fully namespaced by bundle_hash
        # and lives on the miner's own fork, so an unconditional force is
        # safe and idempotent.
        # Auth: embed the PAT in the push URL. This is observable via
        # `ps aux` for the push's duration — known limitation tracked as a
        # followup (proper fix is a GIT_ASKPASS helper). _run() redacts the
        # token from any exception messages via the secrets= tuple, and the
        # token-bearing URL never lands in .git/config.
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(fork_url)
        # Strip any embedded creds before re-injecting our token.
        plain_netloc = parsed.hostname or parsed.netloc
        if parsed.port:
            plain_netloc = f"{plain_netloc}:{parsed.port}"
        push_netloc = f"{token}@{plain_netloc}"
        push_url = urlunparse(parsed._replace(netloc=push_netloc))
        _run(
            ["git", "push", "--force", push_url, branch],
            cwd=workdir,
            secrets=(token,),
        )

        # 6. Open PR via REST API
        head_owner = parsed.path.lstrip("/").split("/")[0]
        pr = _gh_request(
            "POST",
            f"/repos/{upstream}/pulls",
            token,
            {
                "title": title,
                "body": body,
                "head": f"{head_owner}:{branch}",
                "base": "main",
                "maintainer_can_modify": True,
            },
        )
        return pr.get("html_url") or pr.get("url", "")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
