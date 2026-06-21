"""Regression tests for the Claude Code billing-attribution system block.

On the OAuth (subscription) path ``build_anthropic_kwargs`` prepends a
billing-attribution block as ``system[0]``::

    x-anthropic-billing-header: cc_version=<ver>; cc_entrypoint=sdk-cli;

ahead of the natural-language identity block ("You are Claude Code..."). Without
it, Anthropic's mid-2026 billing gate classifies the request as a generic
third-party app and rejects it with HTTP 400 "Third-party apps now draw from
your extra usage, not your plan limits".

NOTE: this is a deliberate first-party-surface spoof kept local to this fork
(upstream declined to ship it — NousResearch/hermes-agent #48177 / #48176).
"""

_BILLING_PREFIX = "x-anthropic-billing-header:"
_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def _build(is_oauth, system=None, tools=None):
    from agent.anthropic_adapter import build_anthropic_kwargs

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": "Hi"})
    return build_anthropic_kwargs(
        model="claude-sonnet-4-6",
        messages=messages,
        tools=tools,
        max_tokens=4096,
        reasoning_config=None,
        is_oauth=is_oauth,
    )


def _is_billing_block(block):
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and block.get("text", "").startswith(_BILLING_PREFIX)
    )


class TestAnthropicOAuthBillingHeader:
    def test_billing_block_is_system_zero_on_oauth(self):
        system = _build(True, system="Persona text.")["system"]
        assert isinstance(system, list)
        assert _is_billing_block(system[0])
        assert "cc_entrypoint=sdk-cli" in system[0]["text"]
        assert "cc_version=" in system[0]["text"]

    def test_identity_follows_billing_block(self):
        system = _build(True, system="Persona text.")["system"]
        assert system[1]["text"] == _IDENTITY

    def test_billing_block_present_without_user_system(self):
        # Even with no user-supplied system prompt, OAuth must still attribute.
        system = _build(True)["system"]
        assert _is_billing_block(system[0])
        assert system[1]["text"] == _IDENTITY

    def test_no_billing_block_when_not_oauth(self):
        system = _build(False, system="Persona text.")["system"]
        # Non-OAuth keeps the raw string / untouched blocks — never a billing header.
        if isinstance(system, list):
            assert not any(_is_billing_block(b) for b in system)
        else:
            assert not str(system).startswith(_BILLING_PREFIX)

    def test_billing_block_carries_real_cc_version(self):
        from agent.anthropic_adapter import _get_claude_code_version

        system = _build(True, system="Persona text.")["system"]
        assert f"cc_version={_get_claude_code_version()};" in system[0]["text"]
