"""An unnamed browser_exec drives a harness daemon of its own task (tools/browser_use_cli.py).

The harness names a daemon by BU_NAME alone and reuses any live one. With every unnamed call on
``default``, a second task reused the daemon the first had latched to the first task's browser (the
packaged Chromium is one per task), or shared its tab on a browser both drive (the real profile). Task
daemons end when idle instead of piling up until shutdown.
"""
import json
import stat
import threading

import pytest

import tools.browser_use_cli as bu_cli
from gateway.run import _profile_runtime_scope
from hermes_constants import hermes_home_key
from tools import browser_tool_session as bt_session

_reap_soon = bu_cli._reap_idle_task_daemons_soon


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
    monkeypatch.setattr(bu_cli, "_task_daemon_used", {})
    monkeypatch.setattr(bu_cli, "_task_daemon_busy", {})
    monkeypatch.setattr(bu_cli, "_task_daemon_reaping", set())
    monkeypatch.setattr(bu_cli, "_exec_route_is_sidecar", {})
    monkeypatch.setattr(bu_cli, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bu_cli, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(bu_cli, "_reap_idle_task_daemons_soon", lambda: None)
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


def reloads(harness):
    path = harness / "reloads"
    return path.read_text(encoding="utf-8-sig").split() if path.exists() else []


def age(name, seconds):
    for daemon in bu_cli._task_daemon_used:
        if daemon[-1] == name:
            bu_cli._task_daemon_used[daemon] -= seconds


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
    assert bu_cli._task_daemon_used == {}  # neither is a task daemon, so neither is idle-reaped


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


def test_an_idle_task_daemon_is_stopped_and_respawns_on_its_next_call(harness):
    a, _ = run(task_id="task-a")
    b, _ = run(task_id="task-b")
    age(a, 601)
    bu_cli.reap_idle_task_daemons(hermes_home_key(), idle_s=600)
    assert reloads(harness) == [a]
    assert [daemon[-1] for daemon in bu_cli._driven_daemons] == [b]
    assert run(task_id="task-a")[0] == a


def test_a_daemon_in_use_is_never_reaped(harness):
    a, _ = run(task_id="task-a")
    daemon = next(d for d in bu_cli._task_daemon_used if d[-1] == a)
    bu_cli._task_daemon_busy[daemon] = 1
    age(a, 10_000)
    bu_cli.reap_idle_task_daemons(hermes_home_key(), idle_s=600)
    assert reloads(harness) == []


def test_a_task_daemon_stopped_by_a_swap_is_not_reaped_again(harness):
    run(task_id="task-a")
    bu_cli.stop_harness_daemons()
    age(bu_cli._task_daemon_name("task-a"), 10_000)
    bu_cli.reap_idle_task_daemons(hermes_home_key(), idle_s=600)
    assert reloads(harness) == [bu_cli._task_daemon_name("task-a")]  # the swap's stop only
    assert bu_cli._task_daemon_used == {}


class _WatchedCondition(threading.Condition):
    """The daemon condition, announcing each wait for a reap to finish."""

    def __init__(self, lock):
        super().__init__(lock)
        self.waiting = threading.Event()

    def wait_for(self, predicate, timeout=None):
        if not predicate():
            self.waiting.set()
        return super().wait_for(predicate, timeout)


def test_a_call_waits_out_the_reap_of_its_own_daemon(harness, monkeypatch):
    a, _ = run(task_id="task-a")
    age(a, 601)
    cv = _WatchedCondition(bu_cli._driven_daemons_lock)
    monkeypatch.setattr(bu_cli, "_daemon_cv", cv)
    order, entered, release = [], threading.Event(), threading.Event()
    real_reload, real_run = bu_cli._reload_daemons, bu_cli._run_cli_killing_process_group

    def slow_reload(daemons):
        entered.set()
        release.wait(10)
        real_reload(daemons)
        order.append("reload")

    def tracked_run(cmd, code, env, timeout):
        if "--reload" not in cmd:
            order.append("call")
        return real_run(cmd, code, env, timeout)
    monkeypatch.setattr(bu_cli, "_reload_daemons", slow_reload)
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group", tracked_run)
    reaper = threading.Thread(target=bu_cli.reap_idle_task_daemons, args=(hermes_home_key(), 600))
    reaper.start()
    assert entered.wait(10)
    caller = threading.Thread(target=run, kwargs={"task_id": "task-a"})
    caller.start()
    assert cv.waiting.wait(10)  # the call is held while its daemon is being stopped
    release.set()
    reaper.join(10)
    caller.join(10)
    assert order == ["reload", "call"]
    assert reloads(harness) == [a]


def test_a_call_does_not_run_while_its_daemon_is_still_being_stopped(harness, monkeypatch):
    a, _ = run(task_id="task-a")
    bu_cli._task_daemon_reaping.add(next(d for d in bu_cli._task_daemon_used if d[-1] == a))
    monkeypatch.setattr(bu_cli, "_REAP_WAIT_S", 0.05)
    launched, real_run = [], bu_cli._run_cli_killing_process_group
    monkeypatch.setattr(bu_cli, "_run_cli_killing_process_group",
                        lambda cmd, code, env, timeout: launched.append(cmd) or real_run(cmd, code, env, timeout))
    result = json.loads(bu_cli.browser_exec("print(1)", task_id="task-a"))
    assert "still being stopped" in result["error"]
    assert launched == []  # never driven while the stop could still land on it


def test_a_shutdown_during_a_reap_still_stops_the_daemons_not_yet_reached(harness, monkeypatch):
    names = {run(task_id=task)[0] for task in ("task-a", "task-b")}
    for name in names:
        age(name, 10_000)
    entered, release = threading.Event(), threading.Event()
    real_reload = bu_cli._reload_daemons

    def first_reload_blocks(daemons):
        if not entered.is_set():
            entered.set()
            release.wait(10)
        real_reload(daemons)
    monkeypatch.setattr(bu_cli, "_reload_daemons", first_reload_blocks)
    reaper = threading.Thread(target=bu_cli.reap_idle_task_daemons, args=(hermes_home_key(), 600))
    reaper.start()
    assert entered.wait(10)
    bu_cli.stop_harness_daemons()  # what shutdown runs, while the reaper is on its first daemon
    stopped_by_shutdown = reloads(harness)
    release.set()
    reaper.join(10)
    assert len(stopped_by_shutdown) == 1
    assert set(reloads(harness)) == names and len(reloads(harness)) == 2


def test_a_call_reaps_other_idle_task_daemons_off_its_own_path(harness):
    a, _ = run(task_id="task-a")
    age(a, 10_000)
    reaper = _reap_soon()
    assert reaper is not None
    reaper.join(10)
    assert reloads(harness) == [a]


def test_a_profile_reaps_only_its_own_idle_daemons(harness, tmp_path, monkeypatch):
    """Building the harness env needs the owning profile's scope, so a call never stops another
    profile's daemons; that profile's own next call does."""
    monkeypatch.setenv("BH_RUNTIME_DIR", str(tmp_path / "harness-runtime"))
    homes = tmp_path / "a", tmp_path / "b"
    for home, task in zip(homes, ("task-a", "task-b")):
        home.mkdir()
        with _profile_runtime_scope(home, hydrate_secrets=False):
            run(task_id=task)
        age(bu_cli._task_daemon_name(task), 10_000)
    scopes, real_reload = [], bu_cli._reload_daemons
    monkeypatch.setattr(bu_cli, "_reload_daemons", lambda daemons: scopes.append(hermes_home_key())
                        or real_reload(daemons))
    with _profile_runtime_scope(homes[0], hydrate_secrets=False):
        reaper = _reap_soon()
    reaper.join(10)
    assert reloads(harness) == [bu_cli._task_daemon_name("task-a")]
    assert scopes == [hermes_home_key(homes[0])]  # the reaper thread kept the caller's profile scope
    assert list(bu_cli._driven_daemons.values()) == [hermes_home_key(homes[1])]
