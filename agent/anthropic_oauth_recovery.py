"""Bounded recovery when an Anthropic OAuth response leaks its tool call as invoke text.

``perform_api_call`` raises :class:`_MalformedAnthropicOAuthToolMarkup` (carrying only usage
counters, never the malformed content) so execution middleware observes a failure; the retry
loop then hands it to :func:`handle_oauth_markup`, which accounts the billed-but-discarded
response, retries up to three times with native tool use forced where the model accepts it,
then falls back or ends the turn without executing anything. Logger name stays
``agent.conversation_loop`` for caplog parity.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, normalize_usage

logger = logging.getLogger("agent.conversation_loop")

_MAX_OAUTH_INVOKE_RECOVERIES = 3
_USAGE_SNAPSHOT_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "request_count",
)


class _MalformedAnthropicOAuthToolMarkup(RuntimeError):
    """Control signal carrying counters, never malformed response content."""

    def __init__(self, usage_snapshot):
        super().__init__("malformed Anthropic OAuth tool markup")
        self.usage_snapshot = usage_snapshot


def _snapshot_anthropic_oauth_usage(agent, response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    canonical = normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode)
    return {key: getattr(canonical, key) for key in _USAGE_SNAPSHOT_FIELDS}


def raise_if_malformed_oauth_response(agent, response) -> None:
    """Quarantine a native OAuth response whose tool call arrived only as invoke text."""
    if agent.api_mode != "anthropic_messages" or not agent._is_anthropic_oauth:
        return
    from agent.anthropic_oauth_markup import anthropic_oauth_response_has_invoke_markup
    if anthropic_oauth_response_has_invoke_markup(response):
        raise _MalformedAnthropicOAuthToolMarkup(_snapshot_anthropic_oauth_usage(agent, response))


def _record_discarded_anthropic_oauth_usage(agent, usage_snapshot, api_duration):
    """Account for a billed response without retaining its malformed content."""
    if usage_snapshot is None:
        return
    from agent.turn_usage import _agent_session_source

    canonical = CanonicalUsage(**usage_snapshot)
    usage_dict = {
        "prompt_tokens": canonical.prompt_tokens,
        "completion_tokens": canonical.output_tokens,
        "total_tokens": canonical.total_tokens,
        **{key: getattr(canonical, key) for key in _USAGE_SNAPSHOT_FIELDS if key != "request_count"},
    }
    agent.context_compressor.update_from_response(usage_dict)
    agent._last_turn_usage = dict(usage_dict)
    for key, value in usage_dict.items():
        setattr(agent, "session_" + key, getattr(agent, "session_" + key) + value)
    agent.session_api_calls += 1
    cost_result = estimate_usage_cost(
        agent.model, canonical, provider=agent.provider, base_url=agent.base_url,
        api_key=getattr(agent, "api_key", ""),
    )
    cost_delta = None
    if cost_result.amount_usd is not None:
        cost_delta = float(cost_result.amount_usd)
        agent.session_estimated_cost_usd += cost_delta
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source
    logger.info(
        "Discarded API call #%d accounted: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs",
        agent.session_api_calls, agent.model, agent.provider or "unknown", canonical.prompt_tokens,
        canonical.output_tokens, canonical.total_tokens, api_duration,
    )
    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.queue_token_counts(
                agent.session_id, source=_agent_session_source(agent),
                input_tokens=canonical.input_tokens, output_tokens=canonical.output_tokens,
                cache_read_tokens=canonical.cache_read_tokens, cache_write_tokens=canonical.cache_write_tokens,
                reasoning_tokens=canonical.reasoning_tokens, estimated_cost_usd=cost_delta,
                cost_status=cost_result.status, cost_source=cost_result.source,
                billing_provider=agent.provider, billing_base_url=agent.base_url,
                billing_mode="subscription_included" if cost_result.status == "included" else None,
                model=agent.model, api_call_count=1,
            )
        except Exception as exc:
            logger.debug("Discarded-response token persistence failed (session=%s): %s", agent.session_id, exc)


@dataclass
class OAuthRecoveryVerdict:
    """``action``: ``"continue"`` (resend with recovery armed), ``"break"`` (fallback armed on
    ``_retry``) or ``"return"`` (``result``: the failed turn). Other fields are loop locals."""

    action: str
    thinking_spinner: Any
    active_system_prompt: Any
    retry_count: int
    compression_attempts: int
    anthropic_oauth_invoke_retries: int
    anthropic_oauth_invoke_recovery: bool
    result: Any = None


def handle_oauth_markup(
    agent, *, malformed, thinking_spinner, api_start_time, anthropic_oauth_invoke_retries,
    anthropic_oauth_invoke_recovery, active_system_prompt, api_messages, _retry, retry_count,
    compression_attempts, messages, conversation_history, api_call_count,
) -> OAuthRecoveryVerdict:
    from agent.conversation_loop import _arm_fallback_restart
    from agent.turn_api_call import stop_thinking_spinner
    from agent.turn_failure_copy import stamp_failure

    thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)
    try:
        _record_discarded_anthropic_oauth_usage(agent, malformed.usage_snapshot, time.time() - api_start_time)
    except Exception:
        logger.warning("Failed to account discarded Anthropic OAuth response", exc_info=True)
    if anthropic_oauth_invoke_retries < _MAX_OAUTH_INVOKE_RECOVERIES and agent.tools:
        anthropic_oauth_invoke_retries += 1
        logger.warning(
            "Discarded malformed Anthropic OAuth tool markup (recovery %d/%d, session=%s)",
            anthropic_oauth_invoke_retries, _MAX_OAUTH_INVOKE_RECOVERIES, agent.session_id or "-",
        )
        agent._emit_wait_notice(
            "Model returned malformed tool markup; retrying the tool call "
            f"({anthropic_oauth_invoke_retries}/{_MAX_OAUTH_INVOKE_RECOVERIES})"
        )
        return OAuthRecoveryVerdict("continue", thinking_spinner, active_system_prompt, retry_count,
                                    compression_attempts, anthropic_oauth_invoke_retries, True)
    logger.error("Anthropic OAuth tool-markup recovery exhausted (session=%s)", agent.session_id or "-")
    if agent._try_activate_fallback():
        active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
        agent._buffer_status("Malformed Anthropic OAuth tool call; switched to fallback")
        return OAuthRecoveryVerdict("break", thinking_spinner, active_system_prompt, 0, 0, 0, False)
    agent._flush_status_buffer()
    # No assistant row: the failed-turn closer writes the display-only boundary, so the
    # failure text never replays to the model as something it said.
    agent._persist_session(messages, conversation_history)
    return OAuthRecoveryVerdict(
        "return", thinking_spinner, active_system_prompt, retry_count, compression_attempts,
        anthropic_oauth_invoke_retries, anthropic_oauth_invoke_recovery,
        stamp_failure({
            "final_response": (
                "Anthropic returned malformed tool-call markup that could not be recovered. "
                "No tool was executed."
            ),
            "messages": messages, "api_calls": api_call_count,
            "completed": False, "failed": True, "partial": True,
            "error": "malformed_anthropic_oauth_tool_markup",
        }, "invalid_response", True),
    )


def apply_oauth_recovery_tool_choice(agent, api_kwargs, *, anthropic_oauth_invoke_recovery: bool) -> None:
    """Arm the recovery resend: force native tool use (thinking off, as forcing requires).

    On a model that rejects forced tool use (Opus 5.5, Sonnet 5.5, Fable 5.1, Mythos 5.1) the
    recovery is a plain resend of the ordinary request: forcing would 400 the retry."""
    if not (anthropic_oauth_invoke_recovery and agent.api_mode == "anthropic_messages"
            and agent._is_anthropic_oauth and api_kwargs.get("tools")):
        return
    from agent.anthropic_adapter import _supports_forced_tool_choice

    if not _supports_forced_tool_choice(api_kwargs.get("model") or agent.model):
        return
    api_kwargs["tool_choice"] = {"type": "any"}
    for key in ("thinking", "output_config", "temperature"):
        api_kwargs.pop(key, None)
