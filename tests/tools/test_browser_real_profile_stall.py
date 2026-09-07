"""Bounded real-profile recovery: no writes through live browsers or expired budgets."""

import contextlib
import json
import multiprocessing
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import browser_connect as bc
from tools import browser_tool as bt
from tools import browser_tool_cloud as cloud
from tools import browser_tool_lightpanda_fallback as lightpanda
from tools import browser_tool_real_profile as rp
from tools import browser_use_cli as cli
from tools.registry import registry


def _locked_backup_probe(root, contention):
    """Run the potentially unbounded C backup in an expendable test-owned process."""
    src, dst = root / "Cookies", root / "copy" / "Cookies"
    dst.parent.mkdir()
    for path, value in ((src, "new"), (dst, "old")):
        with contextlib.closing(sqlite3.connect(path)) as db, db:
            db.execute("create table cookies(value)")
            db.execute("insert into cookies values (?)", (value,))
    if contention == "wal":
        with contextlib.closing(sqlite3.connect(src)) as writer:
            writer.execute("pragma journal_mode=wal")
            writer.execute("update cookies set value='fresh'")
            writer.commit()
            assert bc._copy_auth_file(str(src), str(dst))
            with contextlib.closing(sqlite3.connect(dst)) as db:
                assert db.execute("select value from cookies").fetchone() == ("fresh",)
        return
    with contextlib.closing(
        sqlite3.connect(src if contention == "source" else dst)
    ) as locked:
        locked.execute("begin exclusive")
        if contention == "source":
            # Normal reads contend; the immutable fallback must still copy committed data.
            assert bc._copy_auth_file(str(src), str(dst), deadline=time.monotonic() + 2)
            with contextlib.closing(sqlite3.connect(dst)) as db:
                assert db.execute("select value from cookies").fetchone() == ("new",)
            return
        try:
            bc._copy_auth_file(str(src), str(dst), deadline=time.monotonic() + 0.2)
        except TimeoutError:
            pass
        else:
            raise AssertionError("locked destination did not time out")
        assert locked.execute("select value from cookies").fetchone() == ("old",)
        locked.rollback()
    # Timeout must close its handles; a later fresh launch still refreshes auth.
    assert bc._copy_auth_file(str(src), str(dst))
    with contextlib.closing(sqlite3.connect(dst)) as db:
        assert db.execute("select value from cookies").fetchone() == ("new",)


@pytest.mark.parametrize("contention", ["destination", "source", "wal"])
def test_sqlite_contention_is_bounded_without_losing_auth(tmp_path, contention):
    proc = multiprocessing.get_context("spawn").Process(
        target=_locked_backup_probe, args=(tmp_path, contention)
    )
    proc.start()
    try:
        proc.join(10)
        assert not proc.is_alive(), "SQLite backup exceeded its deadline"
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()  # Only the isolated test probe, never an existing browser.
        proc.join(5)
        proc.close()


@pytest.mark.parametrize(
    "metadata", ["matching", "wrong-instance", "missing", "missing-lock"]
)
def test_owned_browser_survives_agent_browser_metadata_loss(
    tmp_path, monkeypatch, metadata
):
    copy = tmp_path / "browser-profile" / "brave"
    copy.mkdir(parents=True)
    (copy / "SingletonLock").symlink_to("host-123")
    (copy / "Local State").write_text("do not overwrite", encoding="utf-8")
    source = tmp_path / "source"
    (source / "Default").mkdir(parents=True)
    (source / "Local State").write_text("{}", encoding="utf-8")

    class Discovery(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps({
                    "webSocketDebuggerUrl": f"ws://127.0.0.1:{self.server.server_port}/devtools/browser/owned"
                }).encode()
            )

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Discovery)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if metadata not in ("missing", "missing-lock"):
        instance = "owned" if metadata == "matching" else "stale"
        (copy / "DevToolsActivePort").write_text(
            f"{server.server_port}\n/devtools/browser/{instance}\n", encoding="utf-8"
        )
    if metadata == "missing-lock":
        (copy / "SingletonLock").unlink()
        monkeypatch.setattr(
            "psutil.process_iter",
            lambda: iter([
                SimpleNamespace(cmdline=Mock(side_effect=SystemError("sysctl race"))),
                SimpleNamespace(cmdline=lambda: ["brave", f"--user-data-dir={copy}"]),
            ]),
        )
        assert not bc._snapshot_in_use(str(copy) + "-other")
    monkeypatch.setattr(bt, "_real_profile_cdp_cache", {})
    monkeypatch.setattr(cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(lightpanda, "_using_lightpanda_engine", lambda: False)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(bc, "detect_default_chromium", lambda: "brave")
    # The attach daemon lost its session metadata: no existing session to reuse.
    monkeypatch.setattr(rp, "_agent_browser_get_cdp", Mock(return_value=None))
    surviving = f"http://127.0.0.1:{server.server_port}"
    attach = Mock(return_value=(surviving, None))
    monkeypatch.setattr(rp, "_attach_agent_browser_to_real_profile", attach)
    snapshot = Mock(wraps=bc.snapshot_real_profile)
    monkeypatch.setattr(bc, "snapshot_real_profile", snapshot)
    launch = Mock(side_effect=AssertionError("must not launch or close a browser"))
    monkeypatch.setattr(rp, "_launch_real_profile_chrome", launch)
    monkeypatch.setattr(rp, "_agent_browser_close_session", launch)
    try:
        cdp, error = rp._real_profile_cdp(deadline=time.monotonic() + 5)
        if metadata == "matching":
            # Verified survivor: recovery re-attaches to it instead of snapshotting.
            assert (cdp, error) == (surviving, None)
            attach.assert_called_once()
            assert attach.call_args.args[:2] == (server.server_port, str(copy))
            assert bt._real_profile_cdp_cache["cdp"] == surviving
        else:
            assert cdp is None and "Refusing to overwrite" in error
            attach.assert_not_called()
        snapshot.assert_not_called()
        launch.assert_not_called()
        # Direct callers of snapshot must honor the same destination safety guard.
        dst, error = bc.snapshot_real_profile("brave", src=str(source))
        assert dst is None and "refusing to overwrite" in error
        assert (copy / "Local State").read_text(encoding="utf-8") == "do not overwrite"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


@pytest.mark.parametrize("phase", ["remaining", "expired", "lock"])
def test_browser_exec_preparation_consumes_operation_budget(monkeypatch, phase):
    monkeypatch.setattr(cli, "_find_cli", lambda: ["unused-cli"])
    monkeypatch.setattr(cli, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(cli, "_workspace_dir", lambda _: None)
    monkeypatch.setattr(cli, "_read_browser_cfg", lambda: {})
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "ok", ""))
    monkeypatch.setattr(cli.subprocess, "run", run)
    if phase == "lock":
        monkeypatch.setattr(cli, "_real_profile_consented", lambda: True)
        monkeypatch.setattr(cloud, "_use_real_profile", lambda: True)
        monkeypatch.setattr(lightpanda, "_using_lightpanda_engine", lambda: False)
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override_raw", lambda: "")
        monkeypatch.setattr(bt, "_real_profile_cdp_lock", threading.Lock())
        monkeypatch.setattr(cli, "_clamp_timeout", lambda _: 0.1)
        bt._real_profile_cdp_lock.acquire()
        started = time.monotonic()
        try:
            result = json.loads(
                registry.get_entry("browser_exec").handler({
                    "code": "print(1)",
                    "local": True,
                })
            )
        finally:
            bt._real_profile_cdp_lock.release()
        assert time.monotonic() - started < 5
        assert "timed out" in result["error"]
        run.assert_not_called()
        return

    clock = [100.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])

    def prepare(env, session, task_id, local, deadline):
        assert deadline == 110.0
        clock[0] += 11 if phase == "expired" else 3

    monkeypatch.setattr(cli, "_route_backend", prepare)
    result = json.loads(
        registry.get_entry("browser_exec").handler({
            "code": "print(1)",
            "timeout_s": 10,
        })
    )
    if phase == "remaining":
        assert result["success"]
        # Preparation spent 3s of the 10s budget; the harness gets only what is left.
        assert run.call_args.kwargs["timeout"] == 7
    else:
        assert "timed out" in result["error"]
        run.assert_not_called()


@pytest.mark.parametrize(
    "native_host",
    [
        pytest.param("macos", marks=pytest.mark.macos_only),
        pytest.param("linux", marks=pytest.mark.linux_only),
    ],
)
@pytest.mark.parametrize(
    "owner",
    [
        "dead",
        "live",
        "foreign",
        "malformed",
        "unreadable",
        "socket-only",
        "dead-with-holder",
    ],
)
def test_snapshot_lock_liveness(tmp_path, monkeypatch, native_host, owner):
    copy = tmp_path / "browser-profile" / "brave"
    copy.mkdir(parents=True)
    source = tmp_path / "source"
    (source / "Default").mkdir(parents=True)
    (source / "Local State").write_text("{}", encoding="utf-8")
    (copy / "Local State").write_text("untouched", encoding="utf-8")
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path)
    with subprocess.Popen([sys.executable, "-c", "pass"]) as child:
        child.wait(timeout=5)
    import psutil

    assert not psutil.pid_exists(child.pid)
    targets = {
        "dead": f"{socket.gethostname()}-{child.pid}",
        "dead-with-holder": f"{socket.gethostname()}-{child.pid}",
        "live": f"{socket.gethostname()}-{os.getpid()}",
        "foreign": f"foreign-{socket.gethostname()}-{child.pid}",
        "malformed": "not-a-host-pid",
        "unreadable": f"{socket.gethostname()}-{child.pid}",
    }
    lock = copy / "SingletonLock"
    if owner != "socket-only":
        lock.symlink_to(targets[owner])
    for name in ("SingletonSocket", "SingletonCookie"):
        (copy / name).symlink_to("stale-target")
    if owner == "unreadable":
        monkeypatch.setattr(
            psutil, "pid_exists", Mock(side_effect=PermissionError("denied"))
        )
    if owner == "dead-with-holder":
        monkeypatch.setattr(
            psutil,
            "process_iter",
            lambda: iter([
                SimpleNamespace(cmdline=lambda: ["brave", "--user-data-dir", str(copy)])
            ]),
        )
    stale = owner in ("dead", "socket-only")
    assert bc._snapshot_in_use(str(copy)) is not stale
    # The guard itself is read-only, including for proven stale locks.
    assert os.path.lexists(lock) == (owner != "socket-only")
    dst, error = bc.snapshot_real_profile("brave", src=str(source))
    if stale:
        assert dst == str(copy) and error is None
        assert not any(
            os.path.lexists(copy / name)
            for name in ("SingletonLock", "SingletonSocket", "SingletonCookie")
        )
    else:
        assert dst is None and "refusing to overwrite" in error
        assert os.readlink(lock) == targets[owner]
        assert (copy / "Local State").read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize("helper", ["session", "attach"])
def test_expired_preparation_propagates_before_subprocess(
    monkeypatch, tmp_path, helper
):
    monkeypatch.setattr(
        rp._install, "_find_agent_browser", lambda: "unused-agent-browser"
    )
    run = Mock(side_effect=AssertionError("expired operation must not spawn"))
    monkeypatch.setattr(rp.subprocess, "run", run)
    with pytest.raises(TimeoutError, match="preparation timed out"):
        if helper == "session":
            rp._agent_browser_session_cmd(
                "owned",
                "get",
                "cdp-url",
                log_label="test",
                deadline=time.monotonic() - 1,
            )
        else:
            rp._attach_agent_browser_to_real_profile(
                12345, str(tmp_path), deadline=time.monotonic() - 1
            )
    run.assert_not_called()


@pytest.mark.parametrize("status", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED])
def test_source_busy_retries_sqlite_without_raw_copy(tmp_path, monkeypatch, status):
    src, dst = tmp_path / "Cookies", tmp_path / "out" / "Cookies"
    connect = sqlite3.connect
    with contextlib.closing(connect(src)) as db, db:
        db.execute("create table cookies(value)")
        db.execute("insert into cookies values ('auth')")
    calls = []

    def contend_once(database, *args, **kwargs):
        if kwargs.get("uri"):
            calls.append(database)
            if len(calls) == 1:
                error = sqlite3.OperationalError("source is busy")
                error.sqlite_errorcode = status
                raise error
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", contend_once)
    monkeypatch.setattr(
        bc.shutil, "copy2", Mock(side_effect=AssertionError("no raw fallback"))
    )
    assert bc._copy_auth_file(str(src), str(dst))
    with contextlib.closing(connect(dst)) as db:
        assert db.execute("select value from cookies").fetchone() == ("auth",)
