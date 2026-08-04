"""Regression tests for text-serialized Anthropic OAuth tool calls."""

from types import SimpleNamespace

from agent.anthropic_adapter import (
    anthropic_oauth_message_has_invoke_markup,
    anthropic_oauth_response_has_invoke_markup,
    build_anthropic_kwargs,
)


MALFORMED = (
    "court\n"
    '<invoke name="mcp__kanban_create">\n'
    '<parameter name="title">test</parameter>\n'
    "</invoke>"
)


def _tool():
    return {
        "type": "function",
        "function": {
            "name": "kanban_create",
            "description": "create a task",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_detects_raw_oauth_invoke_response():
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=MALFORMED)],
        stop_reason="tool_use",
    )

    assert anthropic_oauth_response_has_invoke_markup(response) is True


def test_detects_incomplete_oauth_invoke_tail():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text='court\n<invoke name="mcp__kanban_create">',
            )
        ],
        stop_reason="tool_use",
    )

    assert anthropic_oauth_response_has_invoke_markup(response) is True


def test_preserves_end_turn_examples_and_structured_tool_use():
    end_turn = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=MALFORMED)],
        stop_reason="end_turn",
    )
    structured = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text=MALFORMED),
            SimpleNamespace(
                type="tool_use",
                id="tool_1",
                name="mcp__kanban_create",
                input={"title": "test"},
            ),
        ],
        stop_reason="tool_use",
    )

    assert anthropic_oauth_response_has_invoke_markup(end_turn) is False
    assert anthropic_oauth_response_has_invoke_markup(structured) is False


def test_preserves_fenced_and_non_oauth_examples():
    fenced = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text=f"Example:\n```xml\n{MALFORMED}\n```",
            )
        ],
        stop_reason="end_turn",
    )
    bare_name = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text='<invoke name="kanban_create">\n</invoke>',
            )
        ],
        stop_reason="end_turn",
    )

    assert anthropic_oauth_response_has_invoke_markup(fenced) is False
    assert anthropic_oauth_response_has_invoke_markup(bare_name) is False


def test_inspects_all_assistant_replay_carriers_but_not_user_content():
    assert anthropic_oauth_message_has_invoke_markup(
        {"role": "assistant", "content": MALFORMED, "finish_reason": "tool_calls"}
    )
    assert anthropic_oauth_message_has_invoke_markup(
        {
            "role": "assistant",
            "content": "clean",
            "api_content": MALFORMED,
            "finish_reason": "tool_calls",
        }
    )
    assert anthropic_oauth_message_has_invoke_markup(
        {
            "role": "assistant",
            "content": "clean",
            "finish_reason": "tool_calls",
            "anthropic_content_blocks": [
                {"type": "text", "text": MALFORMED},
            ],
        }
    )
    assert not anthropic_oauth_message_has_invoke_markup(
        {"role": "user", "content": MALFORMED}
    )
    assert not anthropic_oauth_message_has_invoke_markup(
        {"role": "assistant", "content": MALFORMED, "finish_reason": "stop"}
    )


def test_oauth_replay_drops_malformed_assistant_and_keeps_user_message():
    kwargs = build_anthropic_kwargs(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "user", "content": "make a task"},
            {
                "role": "assistant",
                "content": MALFORMED,
                "finish_reason": "tool_calls",
            },
            {"role": "user", "content": "continue"},
        ],
        tools=[_tool()],
        max_tokens=4096,
        reasoning_config=None,
        is_oauth=True,
    )

    wire_text = repr(kwargs["messages"])
    assert "<invoke" not in wire_text
    assert "make a task" in wire_text
    assert "continue" in wire_text


def test_non_oauth_replay_is_unchanged():
    kwargs = build_anthropic_kwargs(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "user", "content": "make a task"},
            {"role": "assistant", "content": MALFORMED},
        ],
        tools=[_tool()],
        max_tokens=4096,
        reasoning_config=None,
        is_oauth=False,
    )

    assert "<invoke" in repr(kwargs["messages"])
