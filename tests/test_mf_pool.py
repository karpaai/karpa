"""Rolling meaningful-failure pool: recent contributors keep sharing the 10% until
they age out, and a sybil (many hotkeys / one coldkey) gets ONE share."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validator.mf_pool import active_hotkeys, update_pool  # noqa: E402


def test_recent_failure_persists_across_quiet_epochs():
    # epoch at block 1000: hkA fails. Quiet epochs follow (no new failures).
    pool = update_pool([], ["hkA"], block=1000)
    # 5 blocks later, no new failures -> hkA still in the window
    pool = update_pool(pool, [], block=1005)
    assert active_hotkeys(pool) == ["hkA"]


def test_ages_out_after_window():
    pool = update_pool([], ["hkA"], block=1000)
    # far beyond RALPH_MF_WINDOW_BLOCKS (default 7200)
    pool = update_pool(pool, [], block=1000 + 7201)
    assert active_hotkeys(pool) == []


def test_sybil_deduped_by_coldkey_one_share():
    # star-dust9023-style: 3 hotkeys, ONE coldkey, all fail -> 1 share
    pool = update_pool([], ["hkA", "hkB", "hkC"], block=1000)
    pool = update_pool(pool, ["hkD"], block=1001)  # different operator
    h2c = {"hkA": "ckX", "hkB": "ckX", "hkC": "ckX", "hkD": "ckY"}
    act = active_hotkeys(pool, h2c)
    assert len(act) == 2  # one per coldkey, NOT 4
    assert "hkD" in act
    # the ckX share goes to exactly one of its hotkeys
    assert len([h for h in act if h in ("hkA", "hkB", "hkC")]) == 1


def test_no_coldkey_map_dedups_by_hotkey():
    pool = update_pool([], ["hkA", "hkB"], block=1000)
    assert sorted(active_hotkeys(pool)) == ["hkA", "hkB"]


def test_deregistered_hotkeys_excluded():
    pool = update_pool([], ["hkA", "hkB"], block=1000)
    assert active_hotkeys(pool, registered={"hkA"}) == ["hkA"]


def test_window_max_caps_size():
    pool = []
    for b in range(200):
        pool = update_pool(pool, [f"hk{b}"], block=1000 + b)
    assert len(pool) <= 128  # RALPH_MF_WINDOW_MAX default


def test_most_recent_hotkey_wins_per_coldkey():
    pool = update_pool([], ["hkOld"], block=1000)
    pool = update_pool(pool, ["hkNew"], block=1050)  # same operator, newer
    h2c = {"hkOld": "ck1", "hkNew": "ck1"}
    assert active_hotkeys(pool, h2c) == ["hkNew"]
