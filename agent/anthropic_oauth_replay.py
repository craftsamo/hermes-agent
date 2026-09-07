"""Request-local provenance for malformed OAuth replay across role repair and failover."""

from copy import deepcopy
import logging

from agent.message_sanitization import _sanitize_surrogates

logger = logging.getLogger("agent.conversation_loop")
_ANTHROPIC_OAUTH_REPLAY_MARKER = "_anthropic_oauth_replay_payloads"
_ANTHROPIC_OAUTH_REPLAY_CARRIERS = (
    "content", "api_content", "anthropic_content_blocks", "_anthropic_content_blocks",
)


def _anthropic_oauth_replay_targets(messages):
    """Map each malformed assistant run to the message surviving role repair."""
    from agent.anthropic_adapter import anthropic_oauth_message_invoke_payloads
    targets = {}
    run_target = None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            run_target = None
            continue
        if run_target is None:
            run_target = message
        payloads = {_sanitize_surrogates(payload) for payload in anthropic_oauth_message_invoke_payloads(message)}
        if payloads:
            targets.setdefault(id(run_target), set()).update(payloads)
    return targets


def _reattach_anthropic_oauth_replay_markers(messages, entries):
    """Restore provenance after a ContextEngine replaces request dicts."""
    if not entries:
        return
    from agent.anthropic_adapter import anthropic_oauth_message_text_invoke_payloads
    used = {
        message.get(_ANTHROPIC_OAUTH_REPLAY_MARKER) for message in messages
        if isinstance(message, dict) and isinstance(message.get(_ANTHROPIC_OAUTH_REPLAY_MARKER), int)
    }
    run_target = None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            run_target = None
            continue
        if run_target is None:
            run_target = message
        payloads = {_sanitize_surrogates(payload) for payload in anthropic_oauth_message_text_invoke_payloads(message)}
        if not payloads:
            continue
        for marker, entry in entries.items():
            if marker in used or not payloads.intersection(entry["payloads"]):
                continue
            run_target[_ANTHROPIC_OAUTH_REPLAY_MARKER] = marker
            used.add(marker)
            break


def _sanitize_anthropic_oauth_replay_for_provider(agent, messages, entries):
    """Sanitize known malformed replay only for the current OAuth destination."""
    if not entries:
        return messages
    from agent.anthropic_adapter import remove_anthropic_oauth_invoke_payloads
    sanitized_messages = []
    changed = 0
    is_oauth_destination = agent.api_mode == "anthropic_messages" and agent._is_anthropic_oauth
    for message in messages:
        if not isinstance(message, dict):
            sanitized_messages.append(message)
            continue
        marker = message.get(_ANTHROPIC_OAUTH_REPLAY_MARKER)
        entry = entries.get(marker) if isinstance(marker, int) else None
        if not entry:
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
            restored = remove_anthropic_oauth_invoke_payloads(restored, set(entry["payloads"]))
            if restored is None:
                restored = {**message, "content": "[Malformed OAuth tool markup omitted]"}
            content = restored.get("content")
            has_visible_content = (
                bool(content.strip()) if isinstance(content, str)
                else any(
                    (isinstance(part, str) and bool(part.strip()))
                    or (isinstance(part, dict)
                        and part.get("type") not in {"thinking", "reasoning", "redacted_thinking"}
                        and (part.get("type") != "text" or bool(str(part.get("text") or "").strip())))
                    for part in content
                ) if isinstance(content, list) else False
            )
            if not has_visible_content and not restored.get("tool_calls"):
                restored["content"] = "[Malformed OAuth tool markup omitted]"
            if restored != message:
                changed += 1
        sanitized_messages.append(restored)
    if changed:
        logger.warning("Sanitized %d malformed Anthropic OAuth assistant message(s) from request replay", changed)
    return sanitized_messages
