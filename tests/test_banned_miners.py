"""Banned-submitter gate: the validator closes the PR (HF + linked GitHub recipe PR) for a
blocklisted identity instead of scoring it. The ban list is data (chain/banned_miners.json)
so identities are added/removed without a code change. Motivated by Kaizen0304 (5H3xirPk)
spamming 10 exploit bundles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401,E402
from validator.hf_poller import (  # noqa: E402
    _hf_user_ban_reason,
    _load_banned_miners,
    _submission_ban_reason,
)

BANNED = {"hotkeys": ["5H3xirPk"], "github": ["kaizen0304"], "hf_users": ["kaizen-hf"]}


def test_bans_kaizen_by_hotkey_prefix():
    sub = {"miner_hotkey": "5H3xirPkNrwRRedZYZfTCvUaVaJ9tN945zcEeaAmBNvWa9Dv", "miner_github": "x"}
    assert _submission_ban_reason(sub, BANNED).startswith("hotkey 5H3xirPk")


def test_bans_kaizen_by_github_case_insensitive():
    sub = {"miner_hotkey": "5OTHER", "miner_github": "Kaizen0304"}
    assert _submission_ban_reason(sub, BANNED) == "github kaizen0304"


def test_clean_miner_not_banned():
    sub = {"miner_hotkey": "5CqhtHE7BE8HgZ", "miner_github": "danielortega-dev"}
    assert _submission_ban_reason(sub, BANNED) is None


def test_bans_hf_account():
    assert _hf_user_ban_reason("Kaizen-HF", BANNED) == "HF account Kaizen-HF"
    assert _hf_user_ban_reason("someone-else", BANNED) is None
    assert _hf_user_ban_reason(None, BANNED) is None


def test_load_banned_miners_from_file(tmp_path, monkeypatch):
    (tmp_path / "chain").mkdir()
    (tmp_path / "chain" / "banned_miners.json").write_text(json.dumps(
        {"hotkeys": ["5H3xirPk"], "github": ["Kaizen0304"], "hf_users": ["Kaizen-HF"]}))
    monkeypatch.setenv("RALPH_ROOT", str(tmp_path))
    b = _load_banned_miners()
    assert b["hotkeys"] == ["5H3xirPk"]
    assert b["github"] == ["kaizen0304"]   # lowercased
    assert b["hf_users"] == ["kaizen-hf"]  # lowercased


def test_load_banned_miners_missing_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("RALPH_ROOT", str(tmp_path))  # no banned_miners.json
    b = _load_banned_miners()
    assert b == {"hotkeys": [], "github": [], "hf_users": []}
