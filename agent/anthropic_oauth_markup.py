"""Detection of text-serialized tool calls leaked by Anthropic OAuth models.

Under the Claude Code OAuth identity a model occasionally ends a turn with
``stop_reason == "tool_use"`` but writes the call as ``<invoke name="mcp__...">`` text instead
of a native ``tool_use`` block. Hermes must never parse or execute that XML; these helpers only
recognize it (standalone lines, never inside Markdown fences) so the turn loop can discard the
response and request a native tool call instead.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_OAUTH_INVOKE_LINE_RE = re.compile(
    r"^[ \t]*<invoke\b(?=[^>\r\n]*\bname\s*=\s*"
    r"(?:\"mcp__[A-Za-z0-9_.:-]+\"|'mcp__[A-Za-z0-9_.:-]+'))[^>\r\n]*>",
    re.IGNORECASE,
)
_FENCE_MARKERS = ("```", "~~~")


def _oauth_invoke_markup_payloads(text: Any) -> list[str]:
    """Extract standalone leaked invoke blocks (an unterminated block runs to the end)."""
    if not isinstance(text, str) or "<invoke" not in text.lower():
        return []
    payloads: list[str] = []
    fence: Optional[str] = None
    offset = 0
    lower_text = text.lower()
    for line in text.splitlines(keepends=True):
        marker = line.lstrip()[:3]
        if marker in _FENCE_MARKERS:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            offset += len(line)
            continue
        match = _OAUTH_INVOKE_LINE_RE.match(line) if fence is None else None
        if match:
            close = lower_text.find("</invoke>", offset + match.end())
            end = close + len("</invoke>") if close >= 0 else len(text)
            payload = text[offset:end].strip()
            if payload:
                payloads.append(payload)
            if close < 0:
                break
        offset += len(line)
    return payloads


def anthropic_oauth_text_has_invoke_markup(text: Any) -> bool:
    """Line-level detector, safe to run on partial stream text."""
    return bool(_oauth_invoke_markup_payloads(text))


def anthropic_oauth_response_has_invoke_markup(response: Any) -> bool:
    """A native Anthropic response that stopped for tool use but carries the call only as text."""
    if getattr(response, "stop_reason", None) != "tool_use":
        return False
    blocks = getattr(response, "content", None)
    if not isinstance(blocks, list):
        return False
    if any(getattr(block, "type", None) == "tool_use" for block in blocks):
        return False
    return any(
        getattr(block, "type", None) == "text"
        and anthropic_oauth_text_has_invoke_markup(getattr(block, "text", None))
        for block in blocks
    )
