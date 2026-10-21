"""A clarify card goes out only after the material it asks about has verifiably arrived.

Real ``TurnRunner._clarify_callback_sync`` + real ``tools.clarify_gateway`` + a real gateway loop
thread. The adapter records every outbound step in order, so the tests assert what the user would
see first.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run_turn_runner_clarify_delivery import UNDELIVERED_CONTEXT
from tools import clarify_gateway as cm
from tools.clarify_tool import clarify_tool

PLAN = [{"question": "Proceed with this plan?", "context": "## Plan\n1. research\n2. draft",
         "choices": ["go", "stop"]}]


class _Adapter(BasePlatformAdapter):
    def __init__(self, context_result=None, context_delay=0.0):
        self.log: list = []
        self._context_result = context_result or SendResult(success=True, message_id="c1")
        self._context_delay = context_delay

    def pause_typing_for_chat(self, chat_id):
        return None

    def resume_typing_for_chat(self, chat_id):
        return None

    async def send_clarify_context(self, chat_id, content, *, source, session_key, metadata=None):
        await asyncio.sleep(self._context_delay)
        self.log.append(("context", content))
        return self._context_result

    async def send_clarify(self, **kwargs):
        self.log.append(("card", kwargs["question"]))
        cm.resolve_gateway_clarify(kwargs["clarify_id"], "go")
        return SendResult(success=True, message_id="k1")

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.log.append(("text", content))
        return SendResult(success=True, message_id="t1")

    async def retire_clarify_card(self, clarify_id, notice):
        return None

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}


class _DuckAdapter:
    """A status surface that is not a BasePlatformAdapter and has no context send."""

    def __init__(self):
        self.log: list = []

    def pause_typing_for_chat(self, chat_id):
        return None

    async def send_clarify(self, **kwargs):
        self.log.append(("card", kwargs["question"]))
        cm.resolve_gateway_clarify(kwargs["clarify_id"], "go")
        return SendResult(success=True, message_id="k1")


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _clean_clarify_state():
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()
    yield


def _runner(adapter, loop, monkeypatch):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="42", _status_thread_metadata=None, source=None,
        session_key="sk-context", stream_consumer_holder=[None], _loop_for_step=loop)
    runner._close_native_stream_boundary = lambda *a, **k: None
    monkeypatch.setattr(cm, "get_clarify_timeout", lambda: 5)
    return runner


def _ask(runner):
    import json
    return json.loads(clarify_tool(PLAN, callback=runner._clarify_callback_sync))


def test_context_is_delivered_before_the_card(loop, monkeypatch):
    adapter = _Adapter()
    result = _ask(_runner(adapter, loop, monkeypatch))

    assert [kind for kind, _ in adapter.log] == ["context", "card"]
    assert adapter.log[0][1] == PLAN[0]["context"]
    # The card carries only the question; the plan is not repeated inside it.
    assert "## Plan" not in adapter.log[1][1]
    assert result["outcome"] == "submitted"
    assert result["responses"][0]["user_response"] == "go"


def test_failed_context_withholds_the_card(loop, monkeypatch):
    adapter = _Adapter(context_result=SendResult(success=False, error="Bad Request: message is too long"))
    result = _ask(_runner(adapter, loop, monkeypatch))

    assert [kind for kind, _ in adapter.log] == ["context"]
    assert result["outcome"] == "undelivered"
    assert result["notice"] == UNDELIVERED_CONTEXT
    assert cm.get_pending_for_session("sk-context", include_choice_prompts=True) is None


def test_context_slower_than_its_window_withholds_the_card(loop, monkeypatch):
    from gateway import run_turn_runner_clarify_delivery as delivery

    monkeypatch.setattr(delivery, "CONTEXT_ACK_WINDOW", 0.1)
    adapter = _Adapter(context_delay=0.5)
    result = _ask(_runner(adapter, loop, monkeypatch))

    assert result["outcome"] == "undelivered"
    assert result["notice"] == UNDELIVERED_CONTEXT
    assert all(kind != "card" for kind, _ in adapter.log)


def test_a_run_superseded_while_its_context_uploads_never_posts_the_card(loop, monkeypatch):
    """/stop or /new during the upload: the stale card must not reach the new conversation, and
    the prompt was never registered, so nothing of the successor's is torn down."""
    from gateway import run_turn_runner_clarify_delivery as delivery

    monkeypatch.setattr(delivery, "CONTEXT_POLL", 0.05)
    adapter = _Adapter(context_delay=0.5)
    runner = _runner(adapter, loop, monkeypatch)
    current = {"value": True}
    runner._ctx._run_still_current = lambda: current["value"]
    cm.register("successor", "sk-context", "Successor's question?", None)
    threading.Timer(0.1, lambda: current.update(value=False)).start()

    result = _ask(runner)

    assert result["outcome"] == "undelivered"
    assert result["notice"] == delivery.ABANDONED_CONTEXT
    assert all(kind != "card" for kind, _ in adapter.log)
    pending = cm.get_pending_for_session("sk-context", include_choice_prompts=True)
    assert pending is not None and pending.clarify_id == "successor"


def test_a_stop_after_the_context_arrived_still_withholds_the_card(loop, monkeypatch):
    adapter = _Adapter()
    runner = _runner(adapter, loop, monkeypatch)
    runner._ctx._run_still_current = lambda: not adapter.log  # superseded once the context landed

    result = _ask(runner)

    assert result["notice"].startswith("[clarify prompt not shown")
    assert [kind for kind, _ in adapter.log] == ["context"]


def test_surface_without_a_context_send_shows_it_inside_the_card(loop, monkeypatch):
    adapter = _DuckAdapter()
    result = _ask(_runner(adapter, loop, monkeypatch))

    assert [kind for kind, _ in adapter.log] == ["card"]
    assert adapter.log[0][1].startswith(PLAN[0]["context"])
    assert result["responses"][0]["user_response"] == "go"


def test_local_viewer_shows_the_context_inside_the_card(loop, monkeypatch):
    """A local viewer (TUI / Desktop / CLI) renders only the prompt; its send() reports success
    without showing anything, so a context sent ahead would count as delivered and be lost."""
    from gateway.config import Platform

    adapter = _Adapter()
    runner = _runner(adapter, loop, monkeypatch)
    runner._ctx.source = SimpleNamespace(platform=Platform.LOCAL)
    result = _ask(runner)

    assert [kind for kind, _ in adapter.log] == ["card"]
    assert adapter.log[0][1].startswith(PLAN[0]["context"])
    assert result["responses"][0]["user_response"] == "go"


def test_batch_panel_gets_each_question_with_its_context(loop, monkeypatch):
    """The CLI batch panel carries question text only: the context rides inside each question."""
    import gateway.session_cli_turn as cli_turn
    import gateway.session_clarify_batch as batch

    asked = []

    def ask(turn, questions):
        asked.extend(questions)
        return {"answers": {q["qid"]: "go" for q in questions}, "outcome": "submitted"}

    monkeypatch.setattr(cli_turn, "clarify_batch_turn", lambda turn: True)
    monkeypatch.setattr(batch, "ask_clarify_batch", ask)
    adapter = _Adapter()
    result = _ask(_runner(adapter, loop, monkeypatch))

    assert len(asked) == 1 and "context" not in asked[0]
    assert asked[0]["question"].startswith(PLAN[0]["context"])
    assert asked[0]["question"].endswith(PLAN[0]["question"])
    assert adapter.log == []
    assert result["responses"][0]["user_response"] == "go"
