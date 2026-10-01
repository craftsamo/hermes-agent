"""Regression tests for text-serialized Anthropic OAuth tool calls."""

from types import SimpleNamespace

from agent.anthropic_oauth_markup import anthropic_oauth_response_has_invoke_markup


MALFORMED = (
    "court\n"
    '<invoke name="mcp__kanban_create">\n'
    '<parameter name="title">test</parameter>\n'
    "</invoke>"
)


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
        stop_reason="tool_use",
    )
    bare_name = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text='<invoke name="kanban_create">\n</invoke>',
            )
        ],
        stop_reason="tool_use",
    )

    assert anthropic_oauth_response_has_invoke_markup(fenced) is False
    assert anthropic_oauth_response_has_invoke_markup(bare_name) is False
