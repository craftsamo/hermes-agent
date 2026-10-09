"""Tests for Bug #12905 fix — stale OAuth token detection in hermes model flow.

Bug 3: `hermes model` with the Claude subscription provider (``anthropic-oauth``)
skips OAuth re-authentication when a stale ANTHROPIC_TOKEN exists in
~/.hermes/.env but no valid Claude Code credentials are available. The fast-path
silently proceeds to model selection with a broken token instead of offering re-auth.
"""

from hermes_cli.config import save_env_value

class TestStaleOAuthTokenDetection:
    """Bug 3: stale OAuth token must trigger needs_auth=True in _model_flow_anthropic_oauth."""

    def test_stale_oauth_token_triggers_reauth(self, tmp_path, monkeypatch, capsys):
        """
        Scenario: ANTHROPIC_TOKEN is an expired OAuth token and there are no
        valid Claude Code credentials anywhere. The flow MUST offer re-auth
        instead of silently skipping to model selection.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Pre-load .env with an expired OAuth token (sk-ant- prefix = OAuth)
        save_env_value("ANTHROPIC_TOKEN", "sk-ant-oat-ExpiredToken00000")
        save_env_value("ANTHROPIC_API_KEY", "")

        # No valid Claude Code credentials available (expired, no refresh token)
        monkeypatch.setattr(
            "agent.anthropic_credentials.read_claude_code_credentials",
            lambda: {
                "accessToken": "expired-cc-token",
                "refreshToken": "",          # No refresh — can't recover
                "expiresAt": 0,               # Already expired
                "source": "claude_code_credentials_file",
            },
        )
        monkeypatch.setattr(
            "agent.anthropic_credentials.is_claude_code_token_valid",
            lambda creds: False,             # Explicitly expired
        )
        monkeypatch.setattr(
            "agent.anthropic_credentials._is_oauth_token",
            lambda key: key.startswith("sk-ant-"),
        )
        # _resolve_claude_code_token_from_credentials has no valid path
        monkeypatch.setattr(
            "agent.anthropic_credentials._resolve_claude_code_token_from_credentials",
            lambda creds=None: None,
        )

        # Re-auth is the subscription login flow; record that it is reached and decline it.
        reauth_calls = []
        monkeypatch.setattr(
            "hermes_cli.main_provider_setup._run_anthropic_oauth_flow",
            lambda save_fn: reauth_calls.append(save_fn) or False,
        )

        from hermes_cli.model_setup_flows import _model_flow_anthropic_oauth

        _model_flow_anthropic_oauth({})

        assert reauth_calls, "a stale subscription token must route to re-authentication"
