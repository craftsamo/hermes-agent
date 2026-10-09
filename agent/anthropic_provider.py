"""Anthropic provider identity: two billing lanes on one native backend.

``anthropic`` bills a Console API key (``ANTHROPIC_API_KEY``); ``anthropic-oauth`` bills a Claude
Pro/Max subscription (Hermes PKCE login, Claude Code credentials, ``ANTHROPIC_TOKEN``,
``CLAUDE_CODE_OAUTH_TOKEN``). They share the wire, model catalog, pricing and cache layout, so
backend checks ask :func:`is_anthropic_provider`; only credential discovery tells them apart.
Each lane refuses the other lane's token shape, so one never silently bills the other.
"""

from __future__ import annotations

from typing import Any

ANTHROPIC_API_PROVIDER = "anthropic"
ANTHROPIC_OAUTH_PROVIDER = "anthropic-oauth"
ANTHROPIC_PROVIDERS = frozenset({ANTHROPIC_API_PROVIDER, ANTHROPIC_OAUTH_PROVIDER})


def is_anthropic_provider(provider: Any) -> bool:
    """True for either native Anthropic lane."""
    return str(provider or "").strip().lower() in ANTHROPIC_PROVIDERS


def lane_accepts_token(provider: Any, token: Any) -> bool:
    """Whether *token* may bill *provider*'s lane. Only a token recognizably shaped for the OTHER lane
    is refused — an OAuth/setup token on ``anthropic``, a Console key (``sk-ant-api…`` /
    ``sk-ant-usr…``) on ``anthropic-oauth``. Every real Console key carries one of those prefixes,
    so an unrecognized shape cannot bill the other account; providers outside the Anthropic family
    accept any token."""
    normalized = str(provider or "").strip().lower()
    if normalized not in ANTHROPIC_PROVIDERS or not isinstance(token, str):
        return True
    from agent.anthropic_credentials import CONSOLE_KEY_PREFIXES, _is_oauth_token
    if normalized == ANTHROPIC_OAUTH_PROVIDER:
        return not token.startswith(CONSOLE_KEY_PREFIXES)
    return not _is_oauth_token(token)


def lane_mismatch_hint(provider: Any) -> str:
    """How to reach the other lane, for errors raised when a lane holds no usable credential."""
    if str(provider or "").strip().lower() == ANTHROPIC_OAUTH_PROVIDER:
        return ("No Claude Pro/Max subscription login found. Run 'hermes auth add anthropic-oauth' to sign in, "
                "or set ANTHROPIC_TOKEN. To bill a Console API key instead, use provider 'anthropic'.")
    return ("No Anthropic API key found. Set ANTHROPIC_API_KEY or run "
            "'hermes auth add anthropic --type api-key'. To bill a Claude Pro/Max subscription "
            "instead, use provider 'anthropic-oauth' ('hermes auth add anthropic-oauth').")
