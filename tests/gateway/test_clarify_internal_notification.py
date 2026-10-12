"""A synthetic notification arriving while a clarify is pending must leave the prompt alone.

Process completions and other gateway wakes are ``internal`` MessageEvents that keep the default
``allow_gateway_control=True``. Routed into the clarify text-intercept they used to resolve an
open-ended (or "Other") prompt with the notification text, or cancel a choice prompt as unrelated
prose; dispatched inline they also skipped the admission receipt, so the notifier delivered them a
second time. They now take the busy path: queued once, delivered after the turn, prompt untouched.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="private", user_id="user1")


def _event(text, *, internal):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source(),
                        message_id="msg1", internal=internal)


NOTIFICATION = "[IMPORTANT: Background process proc_1 completed normally (exit code 0).]"


@pytest.fixture(autouse=True)
def _clean_clarify_state():
    from tools import clarify_gateway as cm

    def clear():
        with cm._lock:
            cm._entries.clear()
            cm._session_index.clear()
            cm._notify_cbs.clear()
    clear()
    yield
    clear()


def _register(session_key, shape):
    from tools import clarify_gateway as cm

    choices = None if shape == "open" else ["A", "B"]
    cm.register("clarify-1", session_key, "Pick one", choices)
    if shape == "other":
        cm.mark_awaiting_text("clarify-1")


def _key(adapter, event):
    return build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["open", "choices", "other"])
async def test_internal_event_skips_clarify_bypass_and_takes_busy_path(shape):
    from tools import clarify_gateway as cm

    adapter = _Adapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event(NOTIFICATION, internal=True)
    key = _key(adapter, event)
    adapter._active_sessions[key] = asyncio.Event()
    _register(key, shape)

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    adapter._busy_session_handler.assert_awaited_once_with(event, key)
    assert cm.get_pending_for_session(key, include_choice_prompts=True) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["open", "choices", "other"])
async def test_runner_intercept_never_answers_or_cancels_with_internal_event(shape):
    from tools import clarify_gateway as cm

    runner = GatewayRunner(GatewayConfig())
    adapter = _Adapter()
    event = _event(NOTIFICATION, internal=True)
    key = _key(adapter, event)
    _register(key, shape)

    assert await runner._hm_pending_reply_intercepts(event, event.source, key) is None
    entry = cm.get_pending_for_session(key, include_choice_prompts=True)
    assert entry is not None
    assert cm.wait_for_response("clarify-1", timeout=0.01) is None


@pytest.mark.asyncio
async def test_internal_event_is_queued_once_with_an_admission_receipt():
    """Through the real busy handler: the notification is queued (receipt set, so the notifier
    does not retry it) and the pending clarify stays answerable by the user."""
    from tools import clarify_gateway as cm

    runner = GatewayRunner(GatewayConfig())
    adapter = _Adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter._message_handler = AsyncMock(return_value="")
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    event = _event(NOTIFICATION, internal=True)
    key = _key(adapter, event)
    adapter._active_sessions[key] = asyncio.Event()
    _register(key, "open")

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert event._gateway_accepted is True
    queued = adapter._pending_messages.get(key)
    if queued is None:
        queued = runner._promote_queued_event(key, adapter, None)
    assert queued is not None and NOTIFICATION in (queued.text or "")
    assert cm.get_pending_for_session(key, include_choice_prompts=True) is not None

    # The human's reply still resolves the prompt.
    human = _event("my answer", internal=False)
    adapter._message_handler = AsyncMock(return_value="")
    await adapter.handle_message(human)
    adapter._message_handler.assert_awaited_once_with(human)
