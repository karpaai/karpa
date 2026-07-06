"""Rolling meaningful-failure reward pool (§5.6 fix).

The 10% "informative dead-end" pool used to be split among the CURRENT epoch's
meaningful failures only, so a contributor earned for one (rate-limited) weight
window and then the 10% reverted to the king every quiet epoch — effectively no
sustained incentive. This keeps a persisted rolling window of recent meaningful
failures so recent contributors keep sharing the 10% until they age out.

SYBIL-RESISTANT: the split is deduped by COLDKEY (one share per operator, the most
recent hotkey per coldkey). A single operator running many hotkeys (e.g. the
star-dust9023 6-hotkey fleet) therefore gets ONE share, not one-per-hotkey, so it
cannot farm the pool by spamming meaningful failures across hotkeys.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def window_blocks() -> int:
    # ~1 day at 12s/block. A meaningful failure keeps earning for this long.
    return _envi("RALPH_MF_WINDOW_BLOCKS", 7200)


def window_max() -> int:
    return _envi("RALPH_MF_WINDOW_MAX", 128)


def _path(chain_dir) -> Path | None:
    # chain_dir is chain.chain_dir (the `chain/` dir holding king.json etc.); may be
    # None on a LocalChain — then the pool is stateless (empty each call).
    return (Path(chain_dir) / "meaningful_pool.json") if chain_dir else None


def load_pool(chain_dir) -> list[dict]:
    p = _path(chain_dir)
    if p is None:
        return []
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, list) else []
    except (OSError, ValueError):
        return []


def save_pool(chain_dir, pool: list[dict]) -> None:
    p = _path(chain_dir)
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(pool, indent=1))
        tmp.replace(p)
    except OSError:
        pass


def update_pool(pool: list[dict], new_hotkeys, block: int, val_bpbs: dict | None = None) -> list[dict]:
    """Append this epoch's meaningful-failure hotkeys stamped with the current block,
    prune to the rolling window (age + max size), return newest-first."""
    val_bpbs = val_bpbs or {}
    out = list(pool)
    for hk in new_hotkeys:
        out.append({"hotkey": hk, "block": int(block), "val_bpb": val_bpbs.get(hk)})
    cutoff = int(block) - window_blocks()
    out = [e for e in out if int(e.get("block", 0)) >= cutoff]
    out.sort(key=lambda e: -int(e.get("block", 0)))  # newest first
    return out[: window_max()]


def active_hotkeys(pool: list[dict], hotkey_to_coldkey: dict | None = None,
                   registered: set | None = None) -> list[str]:
    """Hotkeys currently sharing the 10% pool. Deduped by coldkey (one per operator,
    most-recent hotkey wins) when a hotkey->coldkey map is supplied. If `registered`
    is given, only hotkeys still on the subnet are returned (weighting a deregistered
    hotkey wastes the share)."""
    by_key: dict = {}
    for e in pool:  # newest-first: first-seen per key is the most recent
        hk = e.get("hotkey")
        if not hk:
            continue
        if registered is not None and hk not in registered:
            continue
        key = (hotkey_to_coldkey or {}).get(hk, hk)  # unknown coldkey -> own operator
        if key not in by_key:
            by_key[key] = hk
    return list(by_key.values())
