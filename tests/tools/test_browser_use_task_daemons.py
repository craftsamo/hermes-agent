"""An unnamed browser_exec drives a harness daemon of its own task (tools/browser_use_cli.py).

The harness names a daemon by BU_NAME alone and reuses any live one. With every unnamed call on
``default``, a second task reused the daemon the first had latched to the first task's browser (the
packaged Chromium is one per task), or shared its tab on a browser both drive (the real profile).
"""
import json
import stat

import pytest

import tools.browser_use_cli as bu_cli
from tools import browser_tool_session as bt_session


@pytest.fixture(autouse=True)
def harness(tmp_path, monkeypatch):
    """A fake CLI like browser-harness: one daemon per BU_NAME latches the endpoint of the call that
    started it and ``--reload`` stops it; ``reloads`` lists the stopped names. Each task gets its own
    packaged Chromium."""
    for name in ("BU_NAME", "BU_AUTOSPAWN", "BROWSER_USE_API_KEY", "BROWSER_CDP_URL"):
        monkeypatch.delenv(name, raising=False)
    cli = tmp_path / "browser-harness"
    cli.write_text(f'''#!/bin/sh
name="${{BU_NAME:-default}}"; latch="{tmp_path}/daemon-$name"
if [ "${{1:-}}" = "--reload" ]; then echo "$name" >> "{tmp_path}/reloads"; rm -f "$latch"; exit 0; fi
cat > /dev/null
[ -f "$latch" ] || echo "${{BU_CDP_URL}}${{BU_CDP_WS}}" > "$latch"
printf "%s %s" "$name" "$(cat "$latch")"
''', encoding="utf-8")
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])
    monkeypatch.setattr(bu_cli, "_driven_daemons", {})
    monkeypatch.setattr(bu_cli, "_exec_route_is_sidecar", {})
    monkeypatch.setattr(bu_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: False)
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: "")
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override_raw", lambda: "")
    monkeypatch.setattr("tools.browser_tool_cloud._get_cloud_provider", lambda: None)
    monkeypatch.setattr(bt_session, "_run_browser_command",
                        lambda key, command, args=None, timeout=None, _engine_override=None:
                        {"success": True, "data": {"cdpUrl": f"ws://127.0.0.1:47000/devtools/browser/{key}"}})
    return tmp_path


def run(code="print(1)", **kw):
    """``(daemon name, endpoint its daemon latched)`` of one browser_exec."""
    result = json.loads(bu_cli.browser_exec(code, **kw))
    assert result.get("success"), result
    name, _, endpoint = result["output"].partition(" ")
    return name, endpoint


def test_two_unnamed_tasks_never_share_a_daemon():
    a, a_endpoint = run(task_id="task-a")
    b, b_endpoint = run(task_id="task-b")
    assert a_endpoint.endswith("/task-a")
    assert b_endpoint.endswith("/task-b")  # on one shared daemon, B drove A's browser
    assert a == bu_cli._task_daemon_name("task-a") != b == bu_cli._task_daemon_name("task-b")


def test_a_task_keeps_its_daemon_across_calls():
    first = run(task_id="task-a")
    assert run("print(2)", task_id="task-a") == first


def test_a_named_session_and_a_call_without_a_task_keep_their_daemons():
    assert run(task_id="task-a", session="r7k2")[0] == "r7k2"
    assert run()[0] == "default"


def test_an_operator_bu_name_still_wins(monkeypatch):
    monkeypatch.setenv("BU_NAME", "pinned")
    assert run(task_id="task-a")[0] == "pinned"


def test_each_tasks_private_url_sidecar_is_its_own(monkeypatch):
    monkeypatch.setattr(bu_cli, "_code_url_verdict", lambda code: bu_cli._CodeUrlVerdict(None, True))
    code = 'new_tab("http://192.168.0.1/")'
    a, _ = run(code, task_id="task-a")
    b, _ = run(code, task_id="task-b")
    assert a == bu_cli._sidecar_daemon_name(bu_cli._task_daemon_name("task-a")) != b
    assert len(a) <= 64
