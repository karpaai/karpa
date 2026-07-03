"""Hopper-only GPU hardware bind for the sn40 "1x H100-class" contest.

op2 already ES384-verifies the NRAS GPU EAT against NVIDIA's live JWKS and binds
its eat_nonce to this run's handshake nonce. That proves "a genuine NVIDIA CC GPU
attested THIS run" but says nothing about H100-vs-Blackwell. The signed die class
rides the SAME chain: the verified wrapper EAT commits, via `submods.GPU-0 =
['DIGEST', ['SHA-256', <hex>]]`, to a detached GPU EAT whose sha256 equals that
hex and whose `hwmodel` claim is the NVIDIA-signed die — "GH100" for BOTH H100 and
H200 (one Hopper die, cryptographically indistinguishable), "GB20X"/"GB100"/
"GB200" for Blackwell. Because the hwmodel EAT is digest-committed inside the
signature op2 already checked, a Blackwell miner cannot swap in a GH100 hwmodel
without breaking the committed digest, and cannot replay a foreign GH100
attestation because op2 binds eat_nonce to this run. We therefore read the die
straight from the digest-committed hwmodel and never trust the free-text
gpu_evidence['arch'] or attestation gpu_name (both miner-controlled).
"""
from __future__ import annotations

import base64
import hashlib
import json


def _b64json(seg: str) -> dict:
    seg += "=" * (-len(seg) % 4)
    return json.loads(base64.urlsafe_b64decode(seg))


def _all_jwts(o, acc: list) -> None:
    if isinstance(o, str) and o.count(".") == 2 and len(o) > 80:
        acc.append(o)
    elif isinstance(o, list):
        for x in o:
            _all_jwts(x, acc)
    elif isinstance(o, dict):
        for v in o.values():
            _all_jwts(v, acc)


def hwmodel_from_gpu_token(token) -> tuple[str | None, str]:
    """Return (hwmodel, reason). hwmodel is the NVIDIA-signed die class from the
    digest-committed detached GPU EAT ('GH100'/'GB20X'/...), or None if it cannot
    be safely established. Assumes op2 has ES384-verified the wrapper EAT and bound
    its eat_nonce to the handshake nonce."""
    if not token:
        return None, "no gpu_token"
    try:
        obj = json.loads(token) if isinstance(token, str) and token.strip()[:1] in "[{" else token
    except Exception as e:  # noqa: BLE001
        return None, f"gpu_token not JSON: {e}"
    jwts: list = []
    _all_jwts(obj, jwts)
    wrapper_digest = None
    detached = None  # (raw_jwt, claims)
    for j in jwts:
        try:
            cl = _b64json(j.split(".")[1])
        except Exception:  # noqa: BLE001 — not a JWT segment we can read
            continue
        sm = cl.get("submods")
        if isinstance(sm, dict):
            g0 = sm.get("GPU-0")
            if isinstance(g0, list) and g0 and g0[0] == "DIGEST" and isinstance(g0[1], list) and len(g0[1]) == 2:
                wrapper_digest = g0[1]  # ['SHA-256', '<hex>']
        if "hwmodel" in cl:
            detached = (j, cl)
    if wrapper_digest is None:
        return None, "no submods.GPU-0 DIGEST in signed wrapper EAT"
    if detached is None:
        return None, "no detached GPU EAT carrying hwmodel"
    alg, expect = wrapper_digest[0], str(wrapper_digest[1]).lower()
    if alg.upper().replace("-", "") != "SHA256":
        return None, f"unexpected submods digest alg {alg!r}"
    got = hashlib.sha256(detached[0].encode()).hexdigest()
    if got != expect:
        return None, f"hwmodel EAT digest mismatch (got {got[:16]} != committed {expect[:16]})"
    hw = str(detached[1].get("hwmodel") or "").upper()
    if not hw:
        return None, "empty hwmodel in committed EAT"
    return hw, "ok"


def verify_gpu_arch_allowed(attestation_dict, allow=("GH100",)) -> tuple[bool, str, str | None]:
    """Hopper-only bind over ALL attestation epochs. Reads the signed, digest-
    committed hwmodel from each epoch's gpu_token; allows only dies in `allow`
    (GH100 = H100 + H200, one Hopper die). GB20X/GB1xx/GB2xx / unknown / missing
    => deny. Returns (ok, reason, die)."""
    allow = {a.upper() for a in allow}
    eps = (attestation_dict or {}).get("epochs", [])
    if not eps:
        return False, "no attestation epochs to bind GPU arch", None
    seen = set()
    for i, ep in enumerate(eps):
        hw, why = hwmodel_from_gpu_token(ep.get("gpu_token"))
        if hw is None:
            return False, f"epoch {i}: cannot establish signed GPU die ({why})", None
        if hw not in allow:
            return False, (
                f"epoch {i}: GPU die {hw!r} not allowed (allow={sorted(allow)}; "
                f"Hopper-only 1x-H100 contest)"
            ), hw
        seen.add(hw)
    if len(seen) != 1:
        return False, f"inconsistent GPU die across epochs: {sorted(seen)}", None
    die = next(iter(seen))
    return True, f"GPU arch OK ({die} signed+digest-bound, {len(eps)} epochs)", die
