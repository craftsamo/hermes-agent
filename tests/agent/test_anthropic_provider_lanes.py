"""Anthropic billing lanes: ``anthropic`` (Console API key) and ``anthropic-oauth`` (Claude
subscription) never resolve, lease or bill each other's credentials."""
import json

import pytest

from agent import anthropic_credentials as ac
from agent.anthropic_provider import lane_accepts_token

API_KEY = "sk-ant-api03-lane-fixture"
OAUTH_TOKEN = "sk-ant-oat01-lane-fixture"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ac, "read_claude_code_credentials", lambda: None)
    return tmp_path


def _pool(home, provider, *rows):
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {provider: [
        {"id": f"row{i}", "label": f"row{i}", "priority": i, "source": "manual", **row}
        for i, row in enumerate(rows)]}}))


@pytest.mark.parametrize("token", [API_KEY, OAUTH_TOKEN, "eyJ.jwt.fixture"])
def test_each_recognizable_anthropic_token_fits_exactly_one_lane(token):
    assert lane_accepts_token("anthropic", token) != lane_accepts_token("anthropic-oauth", token)
    assert lane_accepts_token("openrouter", token)


def test_unrecognized_token_shape_is_not_claimed_by_either_lane():
    # Neither lane can bill the other with it: every real Console key starts with ``sk-ant-api``.
    assert lane_accepts_token("anthropic", "proxy-key") and lane_accepts_token("anthropic-oauth", "proxy-key")


def test_env_tokens_resolve_only_on_their_own_lane(home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", OAUTH_TOKEN)
    monkeypatch.setenv("ANTHROPIC_TOKEN", API_KEY)
    assert ac.resolve_anthropic_token(provider="anthropic") is None
    assert ac.resolve_anthropic_token(provider="anthropic-oauth") is None

    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
    monkeypatch.setenv("ANTHROPIC_TOKEN", OAUTH_TOKEN)
    assert ac.resolve_anthropic_token(provider="anthropic") == API_KEY
    assert ac.resolve_anthropic_token(provider="anthropic-oauth") == OAUTH_TOKEN


@pytest.mark.parametrize("provider,misfiled,own", [
    ("anthropic", {"auth_type": "oauth", "access_token": OAUTH_TOKEN},
     {"auth_type": "api_key", "access_token": API_KEY}),
    ("anthropic-oauth", {"auth_type": "api_key", "access_token": API_KEY},
     {"auth_type": "oauth", "access_token": OAUTH_TOKEN}),
])
def test_pool_never_leases_the_other_lanes_token(home, provider, misfiled, own):
    from agent.credential_pool import load_pool

    _pool(home, provider, misfiled)
    assert load_pool(provider).select() is None
    persisted = json.loads((home / "auth.json").read_text())["credential_pool"][provider]
    assert [row["id"] for row in persisted] == ["row0"]  # skipped, not deleted

    _pool(home, provider, misfiled, own)
    assert load_pool(provider).select().access_token == own["access_token"]


@pytest.mark.parametrize("effective,key,lane", [
    ("anthropic-oauth", API_KEY, "anthropic-oauth"),  # the client's own lane wins
    ("", OAUTH_TOKEN, "anthropic-oauth"),              # unlabelled client: the key's shape decides
    ("", API_KEY, "anthropic"),
])
def test_auto_routed_auth_retry_stays_on_the_failed_clients_lane(effective, key, lane):
    from agent.auxiliary_client import _auth_refresh_provider_for_route

    assert _auth_refresh_provider_for_route("auto", "https://api.anthropic.com", effective, api_key=key) == lane


def test_misfiled_env_token_does_not_hide_the_lanes_own_credential(home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", OAUTH_TOKEN)
    _pool(home, "anthropic", {"auth_type": "api_key", "access_token": API_KEY})
    assert ac.resolve_anthropic_token(provider="anthropic") == API_KEY


def test_api_lane_without_a_key_points_at_the_subscription_lane(home, monkeypatch):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv("ANTHROPIC_TOKEN", OAUTH_TOKEN)
    with pytest.raises(AuthError, match="anthropic-oauth"):
        resolve_runtime_provider(requested="anthropic")
    runtime = resolve_runtime_provider(requested="anthropic-oauth")
    assert (runtime["provider"], runtime["api_key"]) == ("anthropic-oauth", OAUTH_TOKEN)
