"""Regression contracts for Hermes-owned browser-harness daemon selection.

The fake CLI models the installed harness's healthy-daemon reuse: an existing
daemon keeps its original CDP endpoint until stopped. No browser is launched.
"""

import json
import os
import sys
import subprocess
import shutil
from pathlib import Path

import pytest

from tools import browser_use_cli
from tools import browser_use_target as target
from tools.registry import registry

# These fake daemons deliberately outlive their CLI parent. Every kill is
# restricted by the production verifier to this fixture's private runtime.
pytestmark = pytest.mark.live_system_guard_bypass


@pytest.fixture(autouse=True)
def _runtime_cleanup(monkeypatch):
    """Remove only this test's runtimes, after its workers and daemons exit."""
    paths = set()
    original = target._runtime

    def runtime(key):
        path = original(key)
        paths.add(path)
        return path

    monkeypatch.setattr(target, "_runtime", runtime)
    yield paths
    for directory, key in list(target._USED):
        if original(key) in paths:
            target._USED.discard((directory, key))
            (Path(directory) / (key + ".json")).unlink(missing_ok=True)
    for path in paths:
        if path.exists():
            # Do not unlink a lease inode if a worker unexpectedly survived.
            with target._lock(path, 0):
                pass
            shutil.rmtree(path)
        assert not path.exists()


@pytest.fixture
def harness(tmp_path, monkeypatch, _runtime_cleanup):
    package = tmp_path / "browser_harness"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "daemon.py").write_text(
        """import json, os, socket, time, psutil, hashlib
from pathlib import Path
runtime = Path(os.environ['BH_RUNTIME_DIR'])
runtime.mkdir(parents=True, exist_ok=True)
with open(Path(os.environ['TEST_HARNESS_RUNTIME']) / 'owned.jsonl', 'a') as owned:
    owned.write(json.dumps({'runtime': str(runtime), 'name': os.environ['BU_NAME'], 'pid': os.getpid(), 'created': psutil.Process().create_time()}) + '\\n')
(runtime / 'bu.pid').write_text(str(os.getpid()), encoding='utf-8')
server = socket.socket(socket.AF_INET if os.name == 'nt' else socket.AF_UNIX)
server.bind(('127.0.0.1', 0) if os.name == 'nt' else str(runtime / 'bu.sock'))
if os.name == 'nt':
    (runtime / 'bu.port').write_text(json.dumps({'port': server.getsockname()[1], 'token': 'test'}), encoding='utf-8')
server.listen()
while True:
    connection, _ = server.accept()
    with connection:
        connection.recv(4096)
        response = {'pong': True, 'pid': os.getpid(), 'name': os.environ['BU_NAME'], 'endpoint': hashlib.sha256(os.environ['BU_CDP_WS'].encode()).hexdigest()}
        if (Path(os.environ['TEST_HARNESS_RUNTIME']) / 'bad-ipc').exists(): response['pong'] = False
        connection.sendall((json.dumps(response) + '\\n').encode())
""",
        encoding="utf-8",
    )
    cli = tmp_path / "harness.py"
    cli.write_text(
        """import json, os, sys, subprocess, time, socket
from pathlib import Path

root = Path(os.environ['TEST_HARNESS_RUNTIME'])
name = os.environ.get('BU_NAME', 'default')
runtime = Path(os.environ.get('BH_RUNTIME_DIR', str(root)))
runtime.mkdir(parents=True, exist_ok=True)
stem = 'bu' if os.environ.get('BH_RUNTIME_DIR') and os.environ.get('BH_RUNTIME_DIR_SHARED') != '1' else 'bu-' + name
state = runtime / (stem + '.json')
if sys.argv[1:] == ['--reload']:
    state.unlink(missing_ok=True)
    sys.exit(0)
code = sys.stdin.read()
if 'hermes-harness:' in code:
    if (root / 'fail-boot').exists():
        print('temporary bootstrap failure', file=sys.stderr)
        sys.exit(1)
    if (root / 'slow-boot').exists(): time.sleep(4)
if os.environ.get('BH_RUNTIME_DIR'):
    assert all(key not in os.environ for key in ('BU_CDP_URL', 'BU_AUTOSPAWN', 'BU_BROWSER_ID'))
    scratch = Path(os.environ['BH_TMP_DIR'])
    assert scratch == runtime / 'tmp' and scratch.is_dir()
    if os.name != 'nt': assert scratch.stat().st_mode & 0o777 == 0o700
    endpoint_file = runtime / ('bu.port' if os.name == 'nt' else 'bu.sock')
    if not endpoint_file.exists():
        subprocess.Popen([sys.executable, '-m', 'browser_harness.daemon'], cwd=Path(__file__).parent,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 5
        while not endpoint_file.exists():
            if time.monotonic() > deadline: raise RuntimeError('bootstrap timeout')
            time.sleep(0.01)
    if os.name == 'nt':
        port = json.loads(endpoint_file.read_text())['port']
        connection = socket.create_connection(('127.0.0.1', port))
    else:
        connection = socket.socket(socket.AF_UNIX)
        connection.connect(str(endpoint_file))
    with connection:
        connection.sendall(b'{"meta":"ping"}\\n')
        response = json.loads(connection.recv(4096))
    if 'BLOCK' in code:
        (root / 'entered').touch()
        while not (root / 'release').exists(): time.sleep(0.01)
        os.kill(response['pid'], 0)
    if 'hermes-harness:' in code:
        print('hermes-harness:' + ('unknown' if (root / 'missing-metadata').exists() else '9.9.9' if (root / 'future-version').exists() else '0.1.9'))
    elif 'FAIL_CODE' in code:
        print('partial https://example.com/page?item=42')
        print('Traceback (most recent call last):\\nNameError: undefined variable; endpoint=' + os.environ['BU_CDP_WS'], file=sys.stderr)
        sys.exit(1)
    else:
        (root / 'user-calls').touch()
        print(json.dumps(response))
    sys.exit(0)
if not state.exists():
    endpoint = os.environ.get('BU_CDP_WS') or os.environ.get('BU_CDP_URL')
    state.write_text(json.dumps({'endpoint': endpoint, 'name': name}), encoding='utf-8')
print(state.read_text(encoding='utf-8'))
""",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    selected = {
        "endpoint": "ws://127.0.0.1:41001/devtools/browser/11111111-1111-1111-1111-111111111111"
    }
    monkeypatch.setattr(
        browser_use_cli, "_find_cli", lambda: [sys.executable, str(cli)]
    )
    monkeypatch.setattr(
        browser_use_cli,
        "_base_subprocess_env",
        lambda: {
            "PATH": os.defpath,
            "TEST_HARNESS_RUNTIME": str(runtime),
            "BU_CDP_WS": selected["endpoint"],
        },
    )
    monkeypatch.setattr(browser_use_cli, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(browser_use_cli, "_read_browser_cfg", lambda: {})

    def execute(profile, session="", task_id="conversation", code="print(page_info())"):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / profile))
        result = json.loads(
            registry.get_entry("browser_exec").handler(
                {"code": code, "session": session}, task_id=task_id
            )
        )
        assert result["success"], result
        assert result.get("session", "") == session
        return json.loads(result["output"])

    yield selected, execute
    import psutil

    owned = runtime / "owned.jsonl"
    for line in owned.read_text().splitlines() if owned.exists() else []:
        record = json.loads(line)
        _runtime_cleanup.add(Path(record["runtime"]))
        target._stop(Path(record["runtime"]), record["name"], record, unlink=False)
        assert (
            not psutil.pid_exists(record["pid"])
            or psutil.Process(record["pid"]).create_time() != record["created"]
        )


def test_healthy_daemon_follows_selected_endpoint(harness):
    selected, execute = harness
    first = execute("assistant")
    assert first["endpoint"] == target._digest(selected["endpoint"])
    assert execute("assistant") == first

    # Nothing invalidates the old daemon's health. Only the selected target changes.
    selected["endpoint"] = (
        "ws://127.0.0.1:41002/devtools/browser/22222222-2222-2222-2222-222222222222"
    )
    second = execute("assistant")
    assert second["endpoint"] == target._digest(selected["endpoint"])
    assert second["endpoint"] != first["endpoint"]
    assert first["pid"] != second["pid"]
    assert first["name"] == second["name"]


@pytest.mark.parametrize("session", ["", "research"])
def test_profiles_do_not_share_daemon_identity(harness, session):
    selected, execute = harness
    first = execute("assistant", session=session)
    selected["endpoint"] = "ws://127.0.0.1:41002/devtools/browser/second"
    second = execute("marketer", session=session)
    assert second["name"] != first["name"]
    assert second["endpoint"] == target._digest(selected["endpoint"])


def test_task_isolation_and_inherited_namespace(harness, monkeypatch):
    _, execute = harness
    base = browser_use_cli._base_subprocess_env
    monkeypatch.setattr(
        browser_use_cli,
        "_base_subprocess_env",
        lambda: {
            **base(),
            "BU_NAME": "default",
            "BH_RUNTIME_DIR": "/do-not-touch",
            "BH_RUNTIME_DIR_SHARED": "1",
            "BH_TMP_DIR": "/do-not-touch",
            "BU_CDP_URL": "http://obsolete.invalid",
            "BU_AUTOSPAWN": "1",
            "BU_BROWSER_ID": "foreign-browser",
        },
    )
    first = execute("assistant", task_id="one")
    second = execute("assistant", task_id="two")
    assert first["name"] != second["name"]
    assert first["name"].startswith("hermes-")
    named = execute("assistant", session="shared", task_id="one")
    assert execute("assistant", session="shared", task_id="two")["pid"] == named["pid"]


def test_http_discovery_browser_uuid_changes_on_same_port(harness, monkeypatch):
    selected, execute = harness
    import requests
    from types import SimpleNamespace

    selected["endpoint"] = "http://user:password@127.0.0.1:41001/?token=secret"
    identity = ["11111111-1111-1111-1111-111111111111"]
    seen = []

    def get(url, timeout):
        seen.append(url)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "webSocketDebuggerUrl": "ws://127.0.0.1:41001/devtools/browser/"
                + identity[0]
            },
        )

    monkeypatch.setattr(requests, "get", get)
    first = execute("assistant")
    assert execute("assistant")["pid"] == first["pid"]
    identity[0] = "22222222-2222-2222-2222-222222222222"
    second = execute("assistant")
    assert first["pid"] != second["pid"]
    assert first["name"] == second["name"]
    assert seen[0] == "http://user:password@127.0.0.1:41001/json/version?token=secret"


def test_unverifiable_target_fails_closed(harness, monkeypatch):
    selected, execute = harness
    execute("assistant")
    selected["endpoint"] = "http://user:password@invalid/?token=secret"
    import requests

    def fail(*args, **kwargs):
        raise requests.ConnectionError(selected["endpoint"])

    monkeypatch.setattr(requests, "get", fail)
    result = json.loads(
        browser_use_cli.browser_exec("print('must not run')", task_id="conversation")
    )
    assert "error" in result
    assert "password" not in str(result) and "secret" not in str(result)


def test_rollover_and_idle_cleanup_stay_bounded(harness, monkeypatch, tmp_path):
    selected, execute = harness
    pids = []
    for index in range(6):
        selected["endpoint"] = f"ws://127.0.0.1:41001/opaque/{index}"
        pids.append(execute("assistant")["pid"])
    import psutil

    assert all(not psutil.pid_exists(pid) for pid in pids[:-1])
    directory = tmp_path / "assistant/cache/browser-use/daemons"
    assert len(list(directory.glob("*.json"))) == 1
    monkeypatch.setattr(target, "_IDLE_SECONDS", -1)
    target._reap(directory)
    assert not psutil.pid_exists(pids[-1])
    assert not list(directory.glob("*.json"))


def test_cross_process_switch_waits_for_active_call(harness, monkeypatch, tmp_path):
    import time
    import psutil

    selected, execute = harness
    first = execute("assistant")
    directory = tmp_path / "assistant/cache/browser-use/daemons"
    root = tmp_path / "runtime"
    script = """import json, sys
from pathlib import Path
from tools.browser_use_target import run_targeted
Path(sys.argv[4]).touch()
result = run_targeted(json.loads(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3]), '', 'conversation', 15, {})
assert result.returncode == 0, result.stderr
print(result.stdout)
"""
    env = {
        "PATH": os.defpath,
        "HERMES_HOME": str(tmp_path / "assistant"),
        "PYTHONPATH": str(Path(browser_use_cli.__file__).resolve().parents[1]),
    }

    def worker(code, started):
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                json.dumps(browser_use_cli._find_cli()),
                code,
                json.dumps(browser_use_cli._base_subprocess_env()),
                str(root / started),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def wait_for(path):
        deadline = time.monotonic() + 10
        while not path.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)

    active = worker("BLOCK", "active-started")
    switching = None
    try:
        wait_for(root / "entered")
        selected["endpoint"] = "ws://127.0.0.1:41002/opaque/new"
        switching = worker("print(1)", "switch-started")
        wait_for(root / "switch-started")
        monkeypatch.setattr(target, "_IDLE_SECONDS", -1)
        target._reap(directory)
        assert psutil.pid_exists(first["pid"])
        assert switching.poll() is None
        (root / "release").touch()
        old_out, old_err = active.communicate(timeout=15)
        new_out, new_err = switching.communicate(timeout=15)
        assert active.returncode == 0, old_err
        assert switching.returncode == 0, new_err
        assert json.loads(old_out)["endpoint"] == first["endpoint"]
        assert json.loads(new_out)["endpoint"] == target._digest(selected["endpoint"])
    finally:
        (root / "release").touch()
        for process in (active, switching):
            if process and process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_unknown_process_and_unconfirmed_startup_are_not_replaced(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    assert target._process(runtime, "hermes-test", {"pending": True}) is None
    with subprocess.Popen([
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
    ]) as process:
        try:
            target._stop(runtime, "hermes-test", {"pid": process.pid, "created": 1})
            assert process.poll() is None
        finally:
            process.terminate()
            process.wait(timeout=5)


def test_unconfirmed_termination_prevents_endpoint_cleanup(tmp_path, monkeypatch):
    import psutil
    from types import SimpleNamespace

    events = []

    def wait(timeout):
        raise psutil.TimeoutExpired(timeout)

    monkeypatch.setattr(
        target,
        "_process",
        lambda *args: SimpleNamespace(
            terminate=lambda: events.append("term"),
            kill=lambda: events.append("kill"),
            wait=wait,
        ),
    )
    pid_file = tmp_path / "bu.pid"
    pid_file.write_text("123", encoding="utf-8")
    with pytest.raises(psutil.TimeoutExpired):
        target._stop(tmp_path, "hermes-test", {})
    assert events == ["term", "kill"]
    assert pid_file.exists()


def test_timeout_releases_call_lease_without_replacing_healthy_daemon(harness):
    _, execute = harness
    first = execute("assistant")
    with pytest.raises(subprocess.TimeoutExpired):
        target.run_targeted(
            browser_use_cli._find_cli(),
            "BLOCK",
            browser_use_cli._base_subprocess_env(),
            "",
            "conversation",
            2,
            {},
        )
    assert execute("assistant")["pid"] == first["pid"]


def test_cloud_direct_mode_does_not_enter_owned_cdp_lifecycle(harness, monkeypatch):
    _, execute = harness
    base = browser_use_cli._base_subprocess_env

    def env():
        value = base()
        value.pop("BU_CDP_WS")
        return value

    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", env)
    monkeypatch.setattr(
        "tools.browser_tool_cloud._get_cloud_provider",
        lambda: type("Provider", (), {"name": "browser-use"})(),
    )
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda: "")
    monkeypatch.setattr(
        target, "run_targeted", lambda *args: pytest.fail("CDP lifecycle entered")
    )
    assert execute("assistant", session="cloud-name")["name"] == "cloud-name"


@pytest.mark.parametrize("failure", ["fail-boot", "slow-boot"])
def test_failed_bootstrap_can_retry(harness, monkeypatch, tmp_path, failure):
    _, execute = harness
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "assistant"))
    flag = tmp_path / "runtime" / failure
    flag.touch()
    with pytest.raises((target.TargetError, subprocess.TimeoutExpired)):
        target.run_targeted(
            browser_use_cli._find_cli(),
            "print(1)",
            browser_use_cli._base_subprocess_env(),
            "",
            "conversation",
            2,
            {},
        )
    assert not (tmp_path / "runtime/user-calls").exists()
    flag.unlink()
    assert execute("assistant")["name"].startswith("hermes-")


def test_reused_pid_marker_recovers_without_signalling_foreign_process(
    harness, monkeypatch, tmp_path
):
    import psutil

    _, execute = harness
    home = tmp_path / "assistant"
    monkeypatch.setenv("HERMES_HOME", str(home))
    directory = home / "cache/browser-use/daemons"
    directory.mkdir(parents=True)
    key = target._scope_key(home, "", "conversation")
    (directory / (key + ".json")).write_text(
        json.dumps({
            "pid": os.getpid(),
            "created": 1,
            "pending": True,
            "used": 0,
        })
    )
    assert execute("assistant")["pid"] != os.getpid()
    assert psutil.pid_exists(os.getpid())
    stale = directory / (target._scope_key(home, "", "orphan") + ".json")
    stale.write_text(
        json.dumps({"pid": os.getpid(), "created": 1, "pending": True, "used": 0})
    )
    target._reap(directory)
    assert not stale.exists()


def test_live_launcher_lease_blocks_pending_recovery(tmp_path):
    import psutil

    with pytest.raises(target.TargetError, match="busy"):
        target._process(
            tmp_path,
            "hermes-test",
            {
                "pending": True,
                "launchers": [[os.getpid(), psutil.Process().create_time()]],
            },
        )


def test_traceback_and_page_data_survive_endpoint_redaction(
    harness, monkeypatch, tmp_path
):
    selected, _ = harness
    selected["endpoint"] = (
        "wss://username:private-password@127.0.0.1/opaque?token=private-query"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "assistant"))
    result = json.loads(
        browser_use_cli.browser_exec("FAIL_CODE", task_id="conversation")
    )
    assert result["success"] is False
    assert "Traceback" in result["stderr"] and "NameError" in result["stderr"]
    assert "partial https://example.com/page?item=42" in result["output"]
    assert "private-password" not in str(result) and "private-query" not in str(result)
    assert (
        target.redact("request failed: private-query", [selected["endpoint"]])
        == "request failed: <redacted-cdp>"
    )


def test_capability_mismatch_is_actionable_and_never_runs_user_code(
    harness, monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "assistant"))
    (tmp_path / "runtime/future-version").touch()
    (tmp_path / "runtime/bad-ipc").touch()
    result = json.loads(
        browser_use_cli.browser_exec("print(1)", task_id="conversation")
    )
    assert "capability:" in result["error"]
    assert "9.9.9" in result["error"] and "IPC ping/PID" in result["error"]
    assert "supported/tested" not in result["error"]
    assert not (tmp_path / "runtime/user-calls").exists()


@pytest.mark.parametrize("metadata", ["future-version", "missing-metadata"])
def test_version_metadata_does_not_gate_compatible_daemon(harness, tmp_path, metadata):
    _, execute = harness
    (tmp_path / "runtime" / metadata).touch()
    first = execute("assistant")
    assert execute("assistant")["pid"] == first["pid"]


def test_subprocesses_share_one_decreasing_budget(harness, monkeypatch):
    _, execute = harness
    budgets = []
    original = subprocess.Popen.communicate

    def communicate(self, *args, **kwargs):
        if "timeout" in kwargs:
            budgets.append(kwargs["timeout"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", communicate)
    execute("assistant")
    assert len(budgets) == 2
    assert 0 < budgets[1] < budgets[0] < 300


def test_busy_and_private_runtime_errors_are_actionable(tmp_path, _runtime_cleanup):
    runtime = target._runtime(target._digest(str(tmp_path)))
    with target._lock(runtime, 1):
        with pytest.raises(target.TargetError, match="busy:.*timeout_s"):
            with target._lock(runtime, 0):
                pytest.fail("lock bypassed")
    conflict = target._runtime(target._digest(str(tmp_path) + "conflict"))
    conflict.touch()
    try:
        with pytest.raises(target.TargetError, match="private-runtime:.*non-directory"):
            with target._lock(conflict, 0):
                pytest.fail("non-directory bypassed")
    finally:
        conflict.unlink()
    assert {runtime, conflict} <= _runtime_cleanup


def test_scratch_cleanup_cannot_remove_runtime_ownership_files(harness):
    import psutil

    _, execute = harness
    first = execute("assistant")
    env = psutil.Process(first["pid"]).environ()
    runtime = Path(env["BH_RUNTIME_DIR"])
    scratch = Path(env["BH_TMP_DIR"])
    assert scratch == runtime / "tmp"
    if os.name != "nt":
        assert scratch.stat().st_mode & 0o777 == 0o700
    inode = (runtime / "owner.lock").stat().st_ino
    shutil.rmtree(scratch)
    assert (runtime / "owner.lock").stat().st_ino == inode
    assert int((runtime / "bu.pid").read_text()) == first["pid"]
    assert execute("assistant")["pid"] == first["pid"]
    assert scratch.is_dir()


@pytest.mark.parametrize("pid_text", [None, "not-a-pid", "1"])
def test_pid_file_preflight_fails_before_model_code(harness, tmp_path, pid_text):
    _, execute = harness
    execute("assistant")
    runtime = target._runtime(
        target._scope_key(tmp_path / "assistant", "", "conversation")
    )
    pid_file = runtime / "bu.pid"
    if pid_text is None:
        pid_file.unlink()
    else:
        pid_file.write_text(pid_text, encoding="utf-8")
    user_calls = tmp_path / "runtime/user-calls"
    user_calls.unlink()
    result = json.loads(
        browser_use_cli.browser_exec("print(1)", task_id="conversation")
    )
    assert "capability:" in result["error"] and "bu.pid" in result["error"]
    assert "Traceback" not in result["error"]
    assert not user_calls.exists()


def test_process_scan_skips_inaccessible_foreign_user(monkeypatch, tmp_path):
    import psutil
    from types import SimpleNamespace

    username = psutil.Process().username()

    def denied():
        raise psutil.AccessDenied(123)

    foreign = SimpleNamespace(
        pid=123,
        info={"cmdline": ["python", "-m", "browser_harness.daemon"]},
        environ=denied,
        username=lambda: username + "-foreign",
    )
    owned = SimpleNamespace(
        pid=456,
        info=foreign.info,
        environ=lambda: {"BU_NAME": "hermes-test", "BH_RUNTIME_DIR": str(tmp_path)},
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *args: iter([foreign, owned]))
    assert target._process(tmp_path, "hermes-test", {}) is owned
    foreign.username = lambda: username
    with pytest.raises(target.TargetError, match="ownership:"):
        target._process(tmp_path, "hermes-test", {})


@pytest.mark.parametrize("race", ["parent", "descendant"])
def test_timeout_scan_tolerates_process_exit(harness, monkeypatch, tmp_path, race):
    import psutil
    from types import SimpleNamespace

    _, execute = harness
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "assistant"))
    flag = tmp_path / "runtime/slow-boot"
    flag.touch()

    def gone(*args, **kwargs):
        raise psutil.NoSuchProcess(123)

    monkeypatch.setattr(
        psutil.Process,
        "children",
        gone
        if race == "parent"
        else lambda *args, **kwargs: [SimpleNamespace(cmdline=gone)],
    )
    with pytest.raises(subprocess.TimeoutExpired):
        target.run_targeted(
            browser_use_cli._find_cli(),
            "print(1)",
            browser_use_cli._base_subprocess_env(),
            "",
            "conversation",
            2,
            {},
        )
    flag.unlink()
    assert execute("assistant")["name"].startswith("hermes-")


@pytest.mark.parametrize(
    "cdp_url", [123, True, {"broken": "value"}, ["url"], "http://[invalid"]
)
def test_malformed_cdp_config_does_not_break_error_redaction(
    harness, monkeypatch, cdp_url
):
    message = "Upstream failed. See https://example.com/help?topic=browser"
    monkeypatch.setattr(
        browser_use_cli, "_route_backend", lambda *args, **kwargs: message
    )
    monkeypatch.setattr(
        browser_use_cli, "_read_browser_cfg", lambda: {"cdp_url": cdp_url}
    )
    assert json.loads(browser_use_cli.browser_exec("print(1)"))["error"] == message


def test_own_tab_marker_uses_private_pid_and_retries_failed_switch(
    monkeypatch, tmp_path
):
    from types import ModuleType, SimpleNamespace

    monkeypatch.setenv("BU_NAME", "hermes-test")
    monkeypatch.setenv("BH_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("BH_RUNTIME_DIR_SHARED", "0")
    pid_file = tmp_path / "bu.pid"
    pid_file.write_text("111", encoding="utf-8")
    default_pid = tmp_path / "bu-hermes-test.pid"
    default_pid.write_text("999", encoding="utf-8")
    harness_module = ModuleType("browser_harness")
    harness_module._ipc = SimpleNamespace(pid_path=lambda name: default_pid)
    monkeypatch.setitem(sys.modules, "browser_harness", harness_module)
    switched = []
    scope = {
        "cdp": lambda *args, **kwargs: {"targetId": "tab"},
        "switch_tab": switched.append,
    }
    exec(browser_use_cli._OWN_TAB_PREAMBLE, scope)
    marker = next(tmp_path.glob("hermes-bu-owntab-*"))
    assert marker.read_text() == "111"
    exec(browser_use_cli._OWN_TAB_PREAMBLE, scope)
    assert switched == ["tab"]
    pid_file.write_text("222", encoding="utf-8")

    def fail(tab):
        raise RuntimeError("switch failed")

    scope["switch_tab"] = fail
    exec(browser_use_cli._OWN_TAB_PREAMBLE, scope)
    assert marker.read_text() == "111"
    scope["switch_tab"] = switched.append
    exec(browser_use_cli._OWN_TAB_PREAMBLE, scope)
    assert switched == ["tab", "tab"] and marker.read_text() == "222"
    assert len(list(tmp_path.glob("hermes-bu-owntab-*"))) == 1
    pid_file.write_text("0", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Cannot identify browser daemon"):
        exec(browser_use_cli._OWN_TAB_PREAMBLE, scope)
    assert marker.read_text() == "222"
