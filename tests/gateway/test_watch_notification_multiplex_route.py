"""Regression: completion injection must resolve a multiplex SECONDARY
profile's adapter, not just ``self.adapters``.

Incident 2026-09-02 (assistant profile served as a secondary of a
``gateway.multiplex_profiles`` gateway hosted by ``default``): a terminal
``background=true, notify_on_complete=true`` process finished, the watcher
drained the completion, and nothing reached the session — no injection log,
no drop warning. Root cause: ``_inject_watch_notification`` resolved its
adapter through ``resolve_delivery_transport(..., self.adapters)`` plus a
literal scan of ``self.adapters``. On a multiplexed gateway ``self.adapters``
only holds the DEFAULT profile's adapters; the assistant's Telegram adapter
lives in ``self._profile_adapters["assistant"]``. ``adapter`` resolved to
``None`` and the method returned ``None`` silently.

The ``/handoff`` path (``_replay_handoff``) and the reply path
(``_adapter_for_source``) were already profile-aware; the injection path was
not. ``_parse_session_key`` also rejected ``agent:<profile>:...`` keys, so a
rebuilt source lost its namespace and async-delegation routing enrichment
skipped secondary sessions entirely.

Contract under test:
- a completion keyed ``agent:<profile>:...`` is delivered through that
  profile's adapter;
- a stamped secondary profile with no live adapter is dropped fail-closed
  (never through the default profile's bot) and logged;
- the default-profile path is byte-for-byte unchanged (``agent:main:...``
  still resolves through ``self.adapters``);
- ``_parse_session_key`` accepts profile namespaces and reports ``profile``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, _parse_session_key


class _PushAdapter:
    """Stub push-capable platform adapter."""

    def __init__(self, name: str):
        self.name = name
        self.handled = []
        self.handle_message = AsyncMock(side_effect=self.handled.append)


def _runner(*, default_adapters=None, profile_adapters=None):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = default_adapters or {}
    runner._profile_adapters = profile_adapters or {}
    runner.config = SimpleNamespace(platforms={}, multiplex_profiles=True)
    # ``GatewayRunner.__init__`` captures this from ``_active_profile_name()``
    # at construction, i.e. before any turn enters ``_profile_runtime_scope``,
    # so on a multiplexed gateway it names the HOST profile. These tests build
    # the runner through ``object.__new__``, so stamp it the same way — a
    # runner without it falls back to the per-turn name and reproduces the
    # bug in the fixture rather than in the code under test.
    runner._primary_profile_name = "default"
    return runner


def _process_event(session_key: str, **overrides):
    evt = {
        "type": "process_completed",
        "session_id": "proc_0805a7af2be7",
        "session_key": session_key,
        "exit_code": 0,
    }
    evt.update(overrides)
    return evt


def test_parse_session_key_accepts_profile_namespace():
    parsed = _parse_session_key("agent:assistant:telegram:dm:6875916731:20478")
    assert parsed == {
        "profile": "assistant",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "6875916731",
        "thread_id": "20478",
    }
    legacy = _parse_session_key("agent:main:telegram:dm:6875916731:20478")
    assert legacy["profile"] == "default"
    assert {k: v for k, v in legacy.items() if k != "profile"} == {
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "6875916731",
        "thread_id": "20478",
    }
    assert _parse_session_key("not:a:key") is None
    assert _parse_session_key("agent::telegram:dm:1") is None


@pytest.mark.asyncio
async def test_injection_routes_secondary_profile_through_its_own_adapter():
    """The assistant-as-secondary case: default has no Telegram adapter; the
    assistant's lives in ``_profile_adapters``. The completion must reach it."""
    assistant_tg = _PushAdapter("assistant-telegram")
    runner = _runner(profile_adapters={"assistant": {Platform.TELEGRAM: assistant_tg}})

    result = await runner._inject_watch_notification(
        "[IMPORTANT: Background process proc_0805a7af2be7 completed normally]",
        _process_event("agent:assistant:telegram:dm:6875916731:20478"),
    )

    assert result is True, (
        f"injection returned {result!r} — the secondary profile's completion "
        "was dropped exactly as on 2026-09-02 (adapter resolved only from "
        "self.adapters)"
    )
    assert assistant_tg.handle_message.await_count == 1
    injected = assistant_tg.handled[0]
    assert injected.source.profile == "assistant"
    assert injected.source.chat_id == "6875916731"
    assert injected.source.thread_id == "20478"
    assert injected.internal is True


@pytest.mark.asyncio
async def test_injection_inside_secondary_runtime_scope_still_routes(monkeypatch, tmp_path):
    """Second incident (2026-09-02 12:13 / 14:36 / 14:37): the process watcher
    task is created at the end of the assistant's turn, INSIDE
    ``_profile_runtime_scope(assistant)``, and inherits that contextvar. The
    adapter resolver then took ``_active_profile_name()`` == "assistant" as
    "this is the host profile", consulted ``self.adapters`` (no Telegram
    there) and dropped the completion as "no live adapter" although the
    assistant's adapter was connected. Runs the real scope + real profile
    name resolution against a fake profile tree.

    ``_authorization_adapter`` now consults ``_profile_adapters`` first and
    compares against the launch-time ``_primary_profile_name`` instead of the
    per-turn name, so the trap no longer bites. This test pins that contract
    behaviourally: it must keep passing however the host identity is
    resolved."""
    from gateway.run import _profile_runtime_scope

    hermes_root = tmp_path / ".hermes"
    assistant_home = hermes_root / "profiles" / "assistant"
    assistant_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: hermes_root)
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: hermes_root / "profiles")

    assistant_tg = _PushAdapter("assistant-telegram")
    runner = _runner(profile_adapters={"assistant": {Platform.TELEGRAM: assistant_tg}})

    with _profile_runtime_scope(assistant_home):
        assert runner._active_profile_name() == "assistant"  # the trap
        result = await runner._inject_watch_notification(
            "[IMPORTANT: Background process proc_x completed normally]",
            _process_event("agent:assistant:telegram:dm:6875916731:20478"),
        )

    assert result is True, "completion dropped when injected under the secondary's scope"
    assert assistant_tg.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_injection_never_falls_back_to_default_bot_for_secondary(caplog):
    """A stamped secondary profile with no live adapter is dropped
    fail-closed and logged — never delivered through the default profile's
    same-platform adapter (that would post out the wrong bot)."""
    default_tg = _PushAdapter("default-telegram")
    runner = _runner(default_adapters={Platform.TELEGRAM: default_tg})

    with caplog.at_level("WARNING", logger="gateway.run"):
        result = await runner._inject_watch_notification(
            "[x]", _process_event("agent:engineer:telegram:dm:42:7"),
        )

    assert result is None
    assert default_tg.handle_message.await_count == 0
    assert any(
        "Dropping watch notification for profile engineer" in rec.getMessage()
        for rec in caplog.records
    ), "fail-closed drop must be visible in the log, not silent"


@pytest.mark.asyncio
async def test_injection_default_profile_path_unchanged():
    """Control: a legacy ``agent:main`` key still resolves through
    ``self.adapters`` and ignores ``_profile_adapters``."""
    default_tg = _PushAdapter("default-telegram")
    other = _PushAdapter("other-telegram")
    runner = _runner(
        default_adapters={Platform.TELEGRAM: default_tg},
        profile_adapters={"assistant": {Platform.TELEGRAM: other}},
    )

    result = await runner._inject_watch_notification(
        "[x]", _process_event("agent:main:telegram:dm:6875916731:20478"),
    )

    assert result is True
    assert default_tg.handle_message.await_count == 1
    assert other.handle_message.await_count == 0
    assert default_tg.handled[0].source.profile is None
