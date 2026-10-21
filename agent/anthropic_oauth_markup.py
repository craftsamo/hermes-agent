"""Detection and removal of text-serialized tool calls leaked by Anthropic OAuth models.

Under the Claude Code OAuth identity a model occasionally ends a turn with
``stop_reason == "tool_use"`` but writes the call as ``<invoke name="mcp__...">`` text instead
of a native ``tool_use`` block. Hermes must never parse or execute that XML; these helpers only
recognize it (standalone lines, never inside Markdown fences) so the turn loop can discard the
response and request a native tool call instead, and strip known payloads from history that
persisted before the recovery existed when it is replayed to an OAuth destination.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_OAUTH_INVOKE_LINE_RE = re.compile(
    r"^[ \t]*<invoke\b(?=[^>\r\n]*\bname\s*=\s*"
    r"(?:\"mcp__[A-Za-z0-9_.:-]+\"|'mcp__[A-Za-z0-9_.:-]+'))[^>\r\n]*>",
    re.IGNORECASE,
)
_FENCE_MARKERS = ("```", "~~~")
# Assistant fields that carry model-authored text; tool inputs and user rows are never inspected.
_TEXT_CARRIERS = ("content", "api_content", "anthropic_content_blocks")
_REMOVAL_CARRIERS = (*_TEXT_CARRIERS, "_anthropic_content_blocks")


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


def _content_oauth_invoke_payloads(content: Any) -> list[str]:
    if isinstance(content, str):
        return _oauth_invoke_markup_payloads(content)
    if not isinstance(content, list):
        return []
    payloads: list[str] = []
    for part in content:
        if isinstance(part, str):
            payloads.extend(_oauth_invoke_markup_payloads(part))
        elif isinstance(part, dict) and part.get("type") == "text":
            payloads.extend(_oauth_invoke_markup_payloads(part.get("text")))
    return payloads


def anthropic_oauth_message_text_invoke_payloads(message: Any) -> tuple[str, ...]:
    """Standalone invoke blocks in an assistant row's text carriers (finish metadata ignored)."""
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ()
    payloads = []
    for key in _TEXT_CARRIERS:
        payloads.extend(_content_oauth_invoke_payloads(message.get(key)))
    return tuple(dict.fromkeys(payload.strip() for payload in payloads if payload.strip()))


def anthropic_oauth_message_invoke_payloads(message: Any) -> tuple[str, ...]:
    """Malformed payloads of an unstructured tool-use turn, read while finish metadata exists."""
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ()
    if message.get("finish_reason") not in {"tool_calls", "tool_use"} or message.get("tool_calls"):
        return ()
    return anthropic_oauth_message_text_invoke_payloads(message)


def anthropic_oauth_message_has_invoke_markup(message: Any) -> bool:
    return bool(anthropic_oauth_message_invoke_payloads(message))


def _remove_oauth_invoke_payloads(content: Any, payloads: set[str]) -> Any:
    if isinstance(content, str):
        cleaned = content
        changed = False
        for payload in payloads:
            if payload in cleaned:
                cleaned = cleaned.replace(payload, "")
                changed = True
        return cleaned.strip() if changed else content
    if not isinstance(content, list):
        return content
    cleaned_parts = []
    for part in content:
        if isinstance(part, str):
            cleaned = _remove_oauth_invoke_payloads(part, payloads)
            if cleaned:
                cleaned_parts.append(cleaned)
        elif isinstance(part, dict) and part.get("type") == "text":
            cleaned = _remove_oauth_invoke_payloads(part.get("text"), payloads)
            if cleaned:
                cleaned_parts.append({**part, "text": cleaned})
        else:
            cleaned_parts.append(part)
    return cleaned_parts


def remove_anthropic_oauth_invoke_payloads(message: Dict[str, Any], payloads: set[str]) -> Optional[Dict[str, Any]]:
    """Remove known malformed payloads while preserving merged assistant data; ``None`` when
    nothing but the payload remained."""
    if message.get("role") != "assistant" or not payloads:
        return message
    cleaned = dict(message)
    for key in _REMOVAL_CARRIERS:
        if key in cleaned:
            cleaned[key] = _remove_oauth_invoke_payloads(cleaned[key], payloads)
    content = cleaned.get("content")
    has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
    has_payload = any(cleaned.get(key) for key in (
        "tool_calls", "reasoning", "reasoning_content", "reasoning_details",
        "anthropic_content_blocks", "_anthropic_content_blocks",
    ))
    return cleaned if has_content or has_payload else None


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
