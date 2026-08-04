"""End-to-end recovery tests for malformed Anthropic OAuth tool markup."""

import copy
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


MALFORMED = (
    "court\n"
    '<invoke name="mcp__web_search">\n'
    '<parameter name="query">test</parameter>\n'
    "</invoke>"
)


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _usage():
    return SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def _text_response(text, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model="claude-sonnet-4-6",
        usage=_usage(),
    )


def _tool_response():
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="tool_1",
                name="web_search",
                input={"query": "test"},
            )
        ],
        stop_reason="tool_use",
        model="claude-sonnet-4-6",
        usage=_usage(),
    )


def _openai_text_response(text):
    message = SimpleNamespace(
        content=text,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        refusal=None,
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                message=message,
                finish_reason="stop",
            )
        ],
        model="fallback-model",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )


@pytest.fixture()
def oauth_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_tool_defs()),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent.client = MagicMock()
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4-6"
    agent.base_url = "https://api.anthropic.com"
    agent._anthropic_base_url = agent.base_url
    agent._is_anthropic_oauth = True
    agent._disable_streaming = True
    agent.reasoning_config = {"enabled": True, "effort": "medium"}
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._fallback_chain = []
    agent._fallback_index = 0
    return agent


def test_malformed_response_retries_as_required_without_thinking(oauth_agent):
    responses = iter(
        [
            _text_response(MALFORMED, stop_reason="tool_use"),
            _tool_response(),
            _text_response("Done"),
        ]
    )
    requests = []

    def _call(kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)

    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch("model_tools.handle_function_call", return_value="search result") as tool,
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation("search")

    assert result["completed"] is True
    assert result["final_response"] == "Done"
    assert tool.call_count == 1
    assert len(requests) == 3
    assert requests[0]["tool_choice"] == {"type": "auto"}
    assert "thinking" in requests[0]
    assert requests[1]["tool_choice"] == {"type": "any"}
    assert "thinking" not in requests[1]
    assert "output_config" not in requests[1]
    assert requests[2]["tool_choice"] == {"type": "auto"}
    assert "thinking" in requests[2]
    assert "<invoke" not in repr(result["messages"])
    assert oauth_agent.session_api_calls == 3
    assert oauth_agent.session_input_tokens == 300
    assert oauth_agent.session_output_tokens == 60


def test_resume_drops_malformed_assistant_before_request_shaping(oauth_agent):
    requests = []

    def _call(kwargs):
        requests.append(copy.deepcopy(kwargs))
        return _text_response("Done")

    history = [
        {"role": "user", "content": "make a task"},
        {
            "role": "assistant",
            "content": MALFORMED,
            "finish_reason": "tool_calls",
        },
    ]
    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation(
            "continue", conversation_history=history
        )

    assert result["completed"] is True
    assert len(requests) == 1
    assert "<invoke" not in repr(requests[0]["messages"])
    assert "make a task" in repr(requests[0]["messages"])
    assert "continue" in repr(requests[0]["messages"])


def test_context_engine_replacement_reattaches_replay_provenance(oauth_agent):
    requests = []

    def _call(kwargs):
        requests.append(copy.deepcopy(kwargs))
        return _text_response("Done")

    def _replace_request(request_messages, **_kwargs):
        return [
            {
                key: value
                for key, value in message.items()
                if key != "_anthropic_oauth_replay_payloads"
            }
            for message in request_messages
        ]

    oauth_agent.context_compressor.select_context = _replace_request
    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation(
            "continue",
            conversation_history=[
                {"role": "user", "content": "make a task"},
                {
                    "role": "assistant",
                    "content": MALFORMED,
                    "finish_reason": "tool_calls",
                },
            ],
        )

    assert result["completed"] is True
    assert "<invoke" not in repr(requests[0]["messages"])


def test_context_engine_replay_normalizes_surrogate_payloads(oauth_agent):
    requests = []

    def _call(kwargs):
        requests.append(copy.deepcopy(kwargs))
        return _text_response("Done")

    def _replace_request(request_messages, **_kwargs):
        return [
            {
                key: value
                for key, value in message.items()
                if key != "_anthropic_oauth_replay_payloads"
            }
            for message in request_messages
        ]

    oauth_agent.context_compressor.select_context = _replace_request
    malformed = MALFORMED.replace("test", "bad\ud800query")
    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation(
            "continue",
            conversation_history=[
                {"role": "user", "content": "make a task"},
                {
                    "role": "assistant",
                    "content": malformed,
                    "finish_reason": "tool_calls",
                },
            ],
        )

    wire = repr(requests[0]["messages"])
    assert result["completed"] is True
    assert "<invoke" not in wire
    assert "\\ud800" not in wire


def test_replay_adds_text_when_invoke_removal_leaves_only_reasoning(oauth_agent):
    requests = []

    def _call(kwargs):
        requests.append(copy.deepcopy(kwargs))
        return _text_response("Done")

    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation(
            "continue",
            conversation_history=[
                {"role": "user", "content": "make a task"},
                {
                    "role": "assistant",
                    "content": '<invoke name="mcp__web_search"></invoke>',
                    "finish_reason": "tool_calls",
                    "reasoning_details": [
                        {"type": "reasoning", "text": "signed thought"}
                    ],
                },
            ],
        )

    wire = repr(requests[0]["messages"])
    assert result["completed"] is True
    assert "<invoke" not in wire
    assert "Malformed OAuth tool markup omitted" in wire


def test_resume_preserves_assistant_merged_after_malformed_turn(oauth_agent):
    requests = []

    def _call(kwargs):
        requests.append(copy.deepcopy(kwargs))
        return _text_response("Done")

    history = [
        {"role": "user", "content": "make a task"},
        {
            "role": "assistant",
            "content": MALFORMED,
            "finish_reason": "tool_calls",
        },
        {
            "role": "assistant",
            "content": "Preserve this assistant context.",
            "tool_calls": [
                {
                    "id": "call_keep",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"kept"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_keep",
            "name": "web_search",
            "content": "kept result",
        },
    ]
    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation(
            "continue", conversation_history=history
        )

    wire = repr(requests[0]["messages"])
    assert result["completed"] is True
    assert "<invoke" not in wire
    assert "Preserve this assistant context." in wire
    assert "call_keep" in wire


def test_fallback_redecoration_sanitizes_for_current_oauth_destination(oauth_agent):
    from agent.agent_runtime_helpers import repair_message_sequence
    from agent.anthropic_adapter import anthropic_oauth_message_invoke_payloads
    from agent.anthropic_oauth_replay import _ANTHROPIC_OAUTH_REPLAY_MARKER
    from agent.conversation_loop import _redecorate_prompt_cache_for_provider

    malformed_with_whitespace = f"  {MALFORMED}\n"
    messages = [
        {"role": "user", "content": "make a task"},
        {
            "role": "assistant",
            "content": malformed_with_whitespace,
            "finish_reason": "tool_calls",
        },
        {"role": "assistant", "content": "Preserve this context."},
        {"role": "user", "content": "continue"},
    ]
    payloads = anthropic_oauth_message_invoke_payloads(messages[1])
    repair_message_sequence(oauth_agent, messages)
    messages[1][_ANTHROPIC_OAUTH_REPLAY_MARKER] = 0
    entries = {
        0: {
            "payloads": payloads,
            "original": {"content": messages[1]["content"]},
        }
    }

    oauth_agent.api_mode = "chat_completions"
    oauth_agent.provider = "openrouter"
    oauth_agent._is_anthropic_oauth = False
    primary, _, _ = _redecorate_prompt_cache_for_provider(
        oauth_agent,
        messages,
        tools_for_api=[],
        anthropic_oauth_replay_entries=entries,
    )
    assert "<invoke" in repr(primary)

    oauth_agent.api_mode = "anthropic_messages"
    oauth_agent.provider = "anthropic"
    oauth_agent._is_anthropic_oauth = True
    fallback, _, _ = _redecorate_prompt_cache_for_provider(
        oauth_agent,
        messages,
        tools_for_api=[],
        anthropic_oauth_replay_entries=entries,
    )

    wire = repr(fallback)
    assert "<invoke" not in wire
    assert "Preserve this context." in wire

    oauth_agent.api_mode = "chat_completions"
    oauth_agent.provider = "openrouter"
    oauth_agent._is_anthropic_oauth = False
    restored, _, _ = _redecorate_prompt_cache_for_provider(
        oauth_agent,
        fallback,
        tools_for_api=[],
        anthropic_oauth_replay_entries=entries,
    )
    assert "<invoke" in repr(restored)


def test_oauth_redecoration_preserves_unrelated_assistant_blocks(oauth_agent):
    from agent.anthropic_adapter import anthropic_oauth_message_invoke_payloads
    from agent.anthropic_oauth_replay import _ANTHROPIC_OAUTH_REPLAY_MARKER
    from agent.conversation_loop import _redecorate_prompt_cache_for_provider

    candidate = {
        "role": "assistant",
        "content": MALFORMED,
        "finish_reason": "tool_calls",
    }
    payloads = anthropic_oauth_message_invoke_payloads(candidate)
    unrelated_blocks = [{"type": "text", "text": " leading text "}]
    messages = [
        {"role": "user", "content": "make a task"},
        {
            "role": "assistant",
            "content": MALFORMED,
            _ANTHROPIC_OAUTH_REPLAY_MARKER: 0,
        },
        {"role": "user", "content": "another question"},
        {
            "role": "assistant",
            "content": " leading text ",
            "anthropic_content_blocks": unrelated_blocks,
        },
    ]

    sanitized, _, _ = _redecorate_prompt_cache_for_provider(
        oauth_agent,
        messages,
        tools_for_api=[],
        anthropic_oauth_replay_entries={
            0: {
                "payloads": payloads,
                "original": {"content": MALFORMED},
            }
        },
    )

    preserved = next(
        message
        for message in sanitized
        if message.get("anthropic_content_blocks") == unrelated_blocks
    )
    assert preserved["content"] == " leading text "
    assert preserved["anthropic_content_blocks"] == unrelated_blocks


def test_non_oauth_request_keeps_content_without_internal_marker(oauth_agent):
    requests = []

    def _call(kwargs):
        requests.append(copy.deepcopy(kwargs))
        return _openai_text_response("Done")

    oauth_agent.api_mode = "chat_completions"
    oauth_agent.provider = "openrouter"
    oauth_agent._is_anthropic_oauth = False
    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation(
            "continue",
            conversation_history=[
                {"role": "user", "content": "make a task"},
                {
                    "role": "assistant",
                    "content": MALFORMED,
                    "finish_reason": "tool_calls",
                },
            ],
        )

    assert result["completed"] is True
    assert "<invoke" in repr(requests[0]["messages"])
    assert "_anthropic_oauth_replay_payloads" not in repr(requests[0]["messages"])


def test_recovery_exhaustion_fails_without_persisting_markup(oauth_agent):
    malformed = _text_response(MALFORMED, stop_reason="tool_use")
    requests = []

    def _call(kwargs):
        requests.append(copy.deepcopy(kwargs))
        return malformed

    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch("model_tools.handle_function_call") as tool,
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation("search")

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["partial"] is True
    assert result["error"] == "malformed_anthropic_oauth_tool_markup"
    assert "No tool was executed" in result["final_response"]
    assert tool.call_count == 0
    assert len(requests) == 4
    assert all(
        request["tool_choice"] == {"type": "any"}
        and "thinking" not in request
        for request in requests[1:]
    )
    assert "<invoke" not in repr(result["messages"])
    assert oauth_agent.session_api_calls == 4
    assert oauth_agent.session_input_tokens == 400
    assert oauth_agent.session_output_tokens == 80


def test_recovery_exhaustion_uses_configured_fallback(oauth_agent):
    responses = iter(
        [
            _text_response(MALFORMED, stop_reason="tool_use"),
            _text_response(MALFORMED, stop_reason="tool_use"),
            _text_response(MALFORMED, stop_reason="tool_use"),
            _text_response(MALFORMED, stop_reason="tool_use"),
            _openai_text_response("Recovered on fallback"),
        ]
    )
    requests = []

    def _call(kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(responses)

    def _activate_fallback():
        oauth_agent.api_mode = "chat_completions"
        oauth_agent.provider = "openrouter"
        oauth_agent.model = "fallback-model"
        oauth_agent.base_url = "https://openrouter.ai/api/v1"
        oauth_agent._base_url_lower = oauth_agent.base_url.lower()
        oauth_agent._base_url_hostname = "openrouter.ai"
        oauth_agent._is_anthropic_oauth = False
        return True

    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch.object(
            oauth_agent,
            "_try_activate_fallback",
            side_effect=_activate_fallback,
        ) as fallback,
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation("search")

    assert result["completed"] is True
    assert result["final_response"] == "Recovered on fallback"
    assert fallback.call_count == 1
    assert len(requests) == 5
    assert oauth_agent.session_api_calls == 5
    assert "<invoke" not in repr(result["messages"])


def test_execution_middleware_never_observes_raw_malformed_response(oauth_agent):
    responses = iter(
        [
            _text_response(MALFORMED, stop_reason="tool_use"),
            _tool_response(),
            _text_response("Done"),
        ]
    )
    observed = []

    def _call(_kwargs):
        return next(responses)

    def _middleware(payload, executor, **_kwargs):
        try:
            result = executor(payload)
        except Exception as exc:
            observed.append(exc)
            raise
        else:
            observed.append(result)
            return result

    with (
        patch.object(oauth_agent, "_interruptible_api_call", side_effect=_call),
        patch(
            "hermes_cli.middleware.run_llm_execution_middleware",
            side_effect=_middleware,
        ),
        patch("model_tools.handle_function_call", return_value="search result"),
        patch.object(oauth_agent, "_persist_session"),
        patch.object(oauth_agent, "_save_trajectory"),
        patch.object(oauth_agent, "_cleanup_task_resources"),
    ):
        result = oauth_agent.run_conversation("search")

    assert result["completed"] is True
    assert len(observed) == 3
    quarantined = observed[0]
    assert not hasattr(quarantined, "response")
    assert MALFORMED not in repr(vars(quarantined))
    assert quarantined.usage_snapshot == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "request_count": 1,
    }


def _stream(events, final_message):
    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.__iter__ = MagicMock(return_value=iter(events))
    stream.get_final_message.return_value = final_message
    return stream


def test_oauth_stream_releases_normal_text_before_final_validation(
    oauth_agent, monkeypatch
):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    oauth_agent._disable_streaming = False
    received = []
    oauth_agent._fire_stream_delta = received.append
    final_message = _text_response("Hello")
    stream = _stream(
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="Hello"),
            )
        ],
        final_message,
    )

    def _final_message():
        assert received == ["Hello"]
        return final_message

    stream.get_final_message.side_effect = _final_message
    oauth_agent._anthropic_client = MagicMock()
    oauth_agent._anthropic_client.messages.stream.return_value = stream
    oauth_agent._create_request_anthropic_client = (
        lambda *args, **kwargs: oauth_agent._anthropic_client
    )

    result = oauth_agent._interruptible_streaming_api_call({})

    assert result is final_message
    assert received == ["Hello"]


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_oauth_stream_preserves_fenced_invoke_examples_across_deltas(
    oauth_agent, monkeypatch, fence
):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    oauth_agent._disable_streaming = False
    received = []
    oauth_agent._fire_stream_delta = received.append
    text = f'{fence}xml\n<invoke name="mcp__web_search">\n{fence}\n'
    final_message = _text_response(text)
    stream = _stream(
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text=fence[:1]),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(
                    type="text_delta", text=fence[1:] + "xml\n"
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="<inv"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(
                    type="text_delta", text='oke name="mcp__web_search">\n'
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text=fence[:2]),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text=fence[2:] + "\n"),
            ),
        ],
        final_message,
    )

    def _final_message():
        assert "".join(received) == text
        return final_message

    stream.get_final_message.side_effect = _final_message
    oauth_agent._anthropic_client = MagicMock()
    oauth_agent._anthropic_client.messages.stream.return_value = stream
    oauth_agent._create_request_anthropic_client = (
        lambda *args, **kwargs: oauth_agent._anthropic_client
    )

    result = oauth_agent._interruptible_streaming_api_call({})

    assert result is final_message
    assert "".join(received) == text


def test_oauth_stream_discards_all_deltas_when_final_response_is_malformed(
    oauth_agent, monkeypatch
):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    oauth_agent._disable_streaming = False
    received = []
    oauth_agent._fire_stream_delta = received.append
    final_message = _text_response(MALFORMED, stop_reason="tool_use")
    stream = _stream(
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="court\n"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(
                    type="text_delta",
                    text='<invoke name="mcp__web_search">',
                ),
            ),
        ],
        final_message,
    )
    oauth_agent._anthropic_client = MagicMock()
    oauth_agent._anthropic_client.messages.stream.return_value = stream
    oauth_agent._create_request_anthropic_client = (
        lambda *args, **kwargs: oauth_agent._anthropic_client
    )

    result = oauth_agent._interruptible_streaming_api_call({})

    assert result is final_message
    assert received == ["court\n"]


def test_oauth_stream_discards_quarantine_when_writer_changes_before_release(
    oauth_agent, monkeypatch
):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    oauth_agent._disable_streaming = False
    generated_tools = []
    oauth_agent._fire_tool_gen_started = generated_tools.append
    final_message = _tool_response()
    stream = _stream(
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(
                    type="text_delta",
                    text='<invoke name="mcp__web_search">',
                ),
            ),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(
                    type="tool_use",
                    name="web_search",
                ),
            )
        ],
        final_message,
    )

    def _final_message_after_replacement_claim():
        oauth_agent._claim_stream_writer()
        return final_message

    stream.get_final_message.side_effect = _final_message_after_replacement_claim
    oauth_agent._anthropic_client = MagicMock()
    oauth_agent._anthropic_client.messages.stream.return_value = stream
    oauth_agent._create_request_anthropic_client = (
        lambda *args, **kwargs: oauth_agent._anthropic_client
    )

    result = oauth_agent._interruptible_streaming_api_call({})

    assert result is final_message
    assert generated_tools == []


def test_oauth_stream_stops_releasing_quarantine_when_writer_changes_mid_drain(
    oauth_agent, monkeypatch
):
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    oauth_agent._disable_streaming = False
    generated_tools = []
    oauth_agent.tool_gen_callback = generated_tools.append
    final_message = _tool_response()
    stream = _stream(
        [
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="thinking_delta", thinking="checking"),
            ),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(
                    type="tool_use",
                    name="web_search",
                ),
            ),
        ],
        final_message,
    )

    def _supersede_after_first_released_delta(_text):
        replacement = threading.Thread(target=oauth_agent._claim_stream_writer)
        replacement.start()
        replacement.join(timeout=2)

    oauth_agent._fire_reasoning_delta = _supersede_after_first_released_delta
    oauth_agent._anthropic_client = MagicMock()
    oauth_agent._anthropic_client.messages.stream.return_value = stream
    oauth_agent._create_request_anthropic_client = (
        lambda *args, **kwargs: oauth_agent._anthropic_client
    )

    result = oauth_agent._interruptible_streaming_api_call({})

    assert result is final_message
    assert generated_tools == []
