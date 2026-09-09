"""The real-profile attach daemon is scoped per hermes home.

Every hermes process used to name its agent-browser attach daemon ``hermes-real-profile``.
Under a multiplex gateway plus resident specialist sessions, that is one socket dir shared
by processes driving DIFFERENT profile copies: the second process saw the first one's daemon
answer for a foreign data dir, closed it (leaving the first process's browser unreachable),
re-snapshotted its own copy, and left behind a daemon holding a dead CDP port for the next
caller to hang on. Reproduced 2026-09-09 (assistant vs. marketer resident sessions).
"""

import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_constants
import tools.browser_tool as bt
from tools import browser_tool_lifecycle
from tools import browser_tool_real_profile as rp


@pytest.fixture
def homes(tmp_path):
    """Two profile homes under one ``profiles/`` root plus the default home, all fake."""
    root = tmp_path / "hermes"
    default = root
    profiles = root / "profiles"
    assistant = profiles / "assistant"
    marketer = profiles / "marketer"
    for d in (default, assistant, marketer):
        d.mkdir(parents=True)
    with patch("hermes_cli.profiles._get_default_hermes_home", return_value=default), \
         patch("hermes_cli.profiles._get_profiles_root", return_value=profiles):
        yield {"default": default, "assistant": assistant, "marketer": marketer,
               "custom": tmp_path / "elsewhere"}


def _under(home: Path):
    return patch.dict(os.environ, {"HERMES_HOME": str(home)})


def test_session_name_is_per_hermes_home(homes):
    with _under(homes["default"]):
        assert bt._real_profile_session() == "hermes-real-profile"
    with _under(homes["assistant"]):
        assert bt._real_profile_session() == "hermes-real-profile-assistant"
    with _under(homes["marketer"]):
        assert bt._real_profile_session() == "hermes-real-profile-marketer"
    homes["custom"].mkdir()
    with _under(homes["custom"]):
        custom = bt._real_profile_session()
    assert custom.startswith("hermes-real-profile-") and len(custom) == len("hermes-real-profile-") + 8
    assert custom not in {"hermes-real-profile-assistant", "hermes-real-profile-marketer"}


def test_session_name_follows_context_override(homes):
    """The multiplex gateway scopes a turn with the context-local override, not the env var."""
    with _under(homes["assistant"]):
        token = hermes_constants.set_hermes_home_override(homes["marketer"])
        try:
            assert bt._real_profile_session() == "hermes-real-profile-marketer"
        finally:
            hermes_constants.reset_hermes_home_override(token)
        assert bt._real_profile_session() == "hermes-real-profile-assistant"


def test_cdp_cache_and_daemon_are_keyed_by_profile(homes, monkeypatch):
    """A cache hit for one profile must not be served to another, and each profile's daemon
    is addressed by its own name — the second profile never touches the first one's session."""
    monkeypatch.setattr(bt, "_real_profile_cdp_cache", {})
    monkeypatch.setattr(bt, "_real_profile_cdp_lock", threading.Lock())
    monkeypatch.setattr(rp._cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(rp._lp, "_using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(rp, "_cdp_http_ready", lambda cdp, *a, **k: True)
    monkeypatch.setattr(rp._session, "_prepare_session_socket_dir", lambda name: f"/tmp/agent-browser-{name}")

    get_cdp_calls, close_calls = [], []

    def _get_cdp(session_name, *a, **k):
        get_cdp_calls.append(session_name)
        return None  # no live daemon for this name

    monkeypatch.setattr(rp, "_agent_browser_get_cdp", _get_cdp)
    monkeypatch.setattr(rp, "_agent_browser_close_session",
                        lambda session_name, *a, **k: close_calls.append(session_name))
    monkeypatch.setattr(rp, "_surviving_chrome_cdp", lambda copy_dir, *a, **k: None)
    monkeypatch.setattr("hermes_cli.browser_connect.detect_default_chromium", lambda: "brave")
    monkeypatch.setattr("hermes_cli.browser_connect.real_profile_copy_dir",
                        lambda browser: str(homes["marketer"] / "browser-profile" / browser))
    monkeypatch.setattr("hermes_cli.browser_connect.snapshot_real_profile",
                        lambda browser, *a, **k: (None, "stop here"))

    # Assistant already holds a validated endpoint.
    bt._real_profile_cdp_cache["hermes-real-profile-assistant"] = "http://127.0.0.1:59612"

    with _under(homes["assistant"]):
        assert rp._real_profile_cdp() == ("http://127.0.0.1:59612", None)
    assert get_cdp_calls == []  # cache hit, no daemon round-trip

    with _under(homes["marketer"]):
        cdp, err = rp._real_profile_cdp()
    assert cdp is None and err.endswith("stop here")
    # Marketer resolved its OWN daemon name and never closed anybody's session.
    assert get_cdp_calls == ["hermes-real-profile-marketer"]
    assert close_calls == []
    # Assistant's entry is untouched by the marketer's cold path.
    assert bt._real_profile_cdp_cache == {"hermes-real-profile-assistant": "http://127.0.0.1:59612"}


def test_reaper_exempts_every_profiles_live_daemon(tmp_path, monkeypatch):
    """A live-owned daemon of ANOTHER profile is that process's business: not reaped, not
    treated as an untracked leak. A dead-owned one is still reaped like any other lane."""
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt, "_active_sessions", {})

    def mk(name, pid, owner):
        d = tmp_path / f"agent-browser-{name}"
        d.mkdir()
        (d / f"{name}.pid").write_text(str(pid))
        (d / f"{name}.owner_pid").write_text(str(owner))
        return d

    live = mk("hermes-real-profile-marketer", pid=4242, owner=os.getpid())  # this pid is alive
    dead = mk("hermes-real-profile-assistant", pid=4343, owner=99999)
    terminated = []

    def _pid_exists(pid):
        return pid in (4242, 4343, os.getpid())

    with patch("gateway.status._pid_exists", side_effect=_pid_exists), \
         patch("gateway.status.get_process_start_time", return_value=777), \
         patch("tools.browser_tool_lifecycle._verify_reapable_browser_daemon", return_value=True), \
         patch("tools.browser_tool_lifecycle._socket_dir_idle_seconds", return_value=10 ** 6), \
         patch("tools.process_registry.ProcessRegistry._terminate_host_pid",
               side_effect=lambda pid, expected_start=None: terminated.append(pid)):
        browser_tool_lifecycle._reap_orphaned_browser_sessions()

    assert terminated == [4343]
    assert live.exists() and not dead.exists()
