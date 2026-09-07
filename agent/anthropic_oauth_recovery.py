"""Quarantine and bounded recovery of text-serialized OAuth tool calls."""

import logging
import time
from dataclasses import dataclass
from typing import Any

from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, normalize_usage

logger = logging.getLogger("agent.conversation_loop")


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
    return {key: getattr(canonical, key) for key in (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "reasoning_tokens", "request_count",
    )}


def _record_discarded_anthropic_oauth_usage(agent, usage_snapshot, api_duration):
    """Account for a billed response without retaining its malformed content."""
    if usage_snapshot is None:
        return
    canonical = CanonicalUsage(**usage_snapshot)
    usage_dict = {
        "prompt_tokens": canonical.prompt_tokens,
        "completion_tokens": canonical.output_tokens,
        "total_tokens": canonical.total_tokens,
        **{key: getattr(canonical, key) for key in (
            "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
        )},
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
            agent._session_db.update_token_counts(
                agent.session_id, input_tokens=canonical.input_tokens, output_tokens=canonical.output_tokens,
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
):
    from agent.turn_api_call import stop_thinking_spinner
    from agent.conversation_loop import _arm_fallback_restart
    from agent.message_metadata import append_message

    thinking_spinner = stop_thinking_spinner(agent, thinking_spinner)
    try:
        _record_discarded_anthropic_oauth_usage(agent, malformed.usage_snapshot, time.time() - api_start_time)
    except Exception:
        logger.warning("Failed to account discarded Anthropic OAuth response", exc_info=True)
    if anthropic_oauth_invoke_retries < 3 and agent.tools:
        anthropic_oauth_invoke_retries += 1
        logger.warning("Discarded malformed Anthropic OAuth tool markup (recovery %d/3, session=%s)",
                       anthropic_oauth_invoke_retries, agent.session_id or "-")
        agent._emit_wait_notice(
            "Model returned malformed tool markup; retrying the "
            f"tool call ({anthropic_oauth_invoke_retries}/3)"
        )
        return OAuthRecoveryVerdict("continue", thinking_spinner, active_system_prompt, retry_count,
                                    compression_attempts, anthropic_oauth_invoke_retries, True)
    logger.error("Anthropic OAuth tool-markup recovery exhausted (session=%s)", agent.session_id or "-")
    if agent._try_activate_fallback():
        active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
        agent._buffer_status("Malformed Anthropic OAuth tool call; switched to fallback")
        return OAuthRecoveryVerdict("break", thinking_spinner, active_system_prompt, 0, 0, 0, False)
    failure_message = (
        "Anthropic returned malformed tool-call markup that could not be recovered. No tool was executed."
    )
    append_message(messages, {"role": "assistant", "content": failure_message})
    agent._session_messages = messages
    agent._persist_session(messages, conversation_history)
    return OAuthRecoveryVerdict("return", thinking_spinner, active_system_prompt, retry_count,
                                compression_attempts, anthropic_oauth_invoke_retries,
                                anthropic_oauth_invoke_recovery, {
        "final_response": failure_message, "messages": messages, "api_calls": api_call_count,
        "completed": False, "failed": True, "partial": True,
        "error": "malformed_anthropic_oauth_tool_markup",
    })
