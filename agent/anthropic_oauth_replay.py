"""Request-local provenance for malformed OAuth replay across role repair and failover.

History persisted before the leaked-markup recovery may hold assistant rows that are nothing
but text-serialized ``<invoke>`` calls. Replaying them to an Anthropic OAuth destination teaches
the model to repeat the shape, so the request copy drops those payloads — never the durable
history, and never for another destination. Ownership is tracked by a private integer marker on
the request dicts:

1. :func:`_anthropic_oauth_replay_targets` reads payloads BEFORE role repair (repair merges
   assistant runs and the merged row loses the ``finish_reason`` that identifies them);
2. ``build_api_messages`` stamps the marker on the request clone and opens an entry;
3. :func:`_reattach_anthropic_oauth_replay_markers` restores markers a ContextEngine dropped;
4. :func:`_snapshot_anthropic_oauth_replay_originals` records the fully-shaped carriers;
5. :func:`_sanitize_anthropic_oauth_replay_for_provider` restores the originals and strips the
   payloads only when the CURRENT destination is OAuth, so a fallback switch in either
   direction re-renders correctly;
6. :func:`strip_anthropic_oauth_replay_markers` removes the marker before kwargs are built.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List

from agent.message_sanitization import _sanitize_surrogates

logger = logging.getLogger("agent.conversation_loop")
_ANTHROPIC_OAUTH_REPLAY_MARKER = "_anthropic_oauth_replay_payloads"
_ANTHROPIC_OAUTH_REPLAY_CARRIERS = (
    "content", "api_content", "anthropic_content_blocks", "_anthropic_content_blocks",
)
_OMITTED_PLACEHOLDER = "[Malformed OAuth tool markup omitted]"
_HIDDEN_PART_TYPES = {"thinking", "reasoning", "redacted_thinking"}


def _assistant_runs(messages):
    run: List[Dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            run.append(message)
            continue
        if run:
            yield run
        run = []
    if run:
        yield run


def _anthropic_oauth_replay_targets(messages) -> Dict[int, set]:
    """Map every row of a malformed assistant run to the run's payloads, so whichever row
    survives role repair (the first, or a superseding one) carries them."""
    from agent.anthropic_oauth_markup import anthropic_oauth_message_invoke_payloads

    targets: Dict[int, set] = {}
    for run in _assistant_runs(messages):
        payloads = {
            _sanitize_surrogates(payload)
            for message in run for payload in anthropic_oauth_message_invoke_payloads(message)
        }
        if payloads:
            for message in run:
                targets[id(message)] = set(payloads)
    return targets


def _stamp_anthropic_oauth_replay_marker(source, api_msg, targets, entries) -> None:
    """Open an entry for a request clone whose source row was a malformed target."""
    payloads = (targets or {}).get(id(source))
    if payloads and entries is not None:
        marker = len(entries)
        api_msg[_ANTHROPIC_OAUTH_REPLAY_MARKER] = marker
        entries[marker] = {"payloads": tuple(sorted(payloads)), "original": None}


def _reattach_anthropic_oauth_replay_markers(messages, entries) -> None:
    """Restore provenance after a ContextEngine replaces request dicts."""
    if not entries:
        return
    from agent.anthropic_oauth_markup import anthropic_oauth_message_text_invoke_payloads

    used = {
        message.get(_ANTHROPIC_OAUTH_REPLAY_MARKER) for message in messages
        if isinstance(message, dict) and isinstance(message.get(_ANTHROPIC_OAUTH_REPLAY_MARKER), int)
    }
    for run in _assistant_runs(messages):
        payloads = {
            _sanitize_surrogates(payload)
            for message in run for payload in anthropic_oauth_message_text_invoke_payloads(message)
        }
        if not payloads:
            continue
        for marker, entry in entries.items():
            if marker in used or not payloads.intersection(entry["payloads"]):
                continue
            run[0][_ANTHROPIC_OAUTH_REPLAY_MARKER] = marker
            used.add(marker)
            break


def _snapshot_anthropic_oauth_replay_originals(messages, entries) -> Dict[int, Dict[str, Any]]:
    """Record each marked row's fully-shaped carriers; drop entries whose row did not survive."""
    observed = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        marker = message.get(_ANTHROPIC_OAUTH_REPLAY_MARKER)
        entry = entries.get(marker) if isinstance(marker, int) else None
        if not entry:
            continue
        entry["original"] = {
            key: deepcopy(message[key]) for key in _ANTHROPIC_OAUTH_REPLAY_CARRIERS if key in message
        }
        observed[marker] = entry
    return observed


def _has_visible_content(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    return any(
        (isinstance(part, str) and bool(part.strip()))
        or (isinstance(part, dict)
            and part.get("type") not in _HIDDEN_PART_TYPES
            and (part.get("type") != "text" or bool(str(part.get("text") or "").strip())))
        for part in content
    )


def _sanitize_anthropic_oauth_replay_for_provider(agent, messages, entries):
    """Sanitize known malformed replay only for the current OAuth destination."""
    if not entries:
        return messages
    from agent.anthropic_oauth_markup import remove_anthropic_oauth_invoke_payloads

    sanitized_messages = []
    changed = 0
    is_oauth_destination = agent.api_mode == "anthropic_messages" and agent._is_anthropic_oauth
    for message in messages:
        marker = message.get(_ANTHROPIC_OAUTH_REPLAY_MARKER) if isinstance(message, dict) else None
        entry = entries.get(marker) if isinstance(marker, int) else None
        if not entry or entry["original"] is None:
            sanitized_messages.append(message)
            continue
        restored = dict(message)
        original = entry["original"]
        for key in _ANTHROPIC_OAUTH_REPLAY_CARRIERS:
            if key in original:
                restored[key] = deepcopy(original[key])
            else:
                restored.pop(key, None)
        if is_oauth_destination:
            cleaned = remove_anthropic_oauth_invoke_payloads(restored, set(entry["payloads"]))
            if cleaned is None:
                cleaned = {
                    key: value for key, value in restored.items() if key not in _ANTHROPIC_OAUTH_REPLAY_CARRIERS
                }
                cleaned["content"] = _OMITTED_PLACEHOLDER
            elif (cleaned != restored and not _has_visible_content(cleaned.get("content"))
                  and not cleaned.get("tool_calls")):
                # Removal left only reasoning: an empty assistant turn is rejected on the wire.
                cleaned["content"] = _OMITTED_PLACEHOLDER
            restored = cleaned
            if restored != message:
                changed += 1
        sanitized_messages.append(restored)
    if changed:
        logger.warning("Sanitized %d malformed Anthropic OAuth assistant message(s) from request replay", changed)
    return sanitized_messages


def strip_anthropic_oauth_replay_markers(messages):
    """Copy of ``messages`` without the private marker (never reaches a provider)."""
    return [
        {key: value for key, value in message.items() if key != _ANTHROPIC_OAUTH_REPLAY_MARKER}
        if isinstance(message, dict) and _ANTHROPIC_OAUTH_REPLAY_MARKER in message else message
        for message in messages
    ]
