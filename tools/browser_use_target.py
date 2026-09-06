"""Owned browser-harness runtimes. Never inspect or stop the global bu-default.

One daemon per profile/logical session; endpoint rollover is serialized with exec.
Idle daemons are reaped on subsequent calls (not by a background service).

Structural harness contract: BU_NAME/BU_CDP_WS,
BH_RUNTIME_DIR and *_SHARED=0; bu.pid and AF_UNIX bu.sock (Windows bu.port
with port/token); newline JSON meta:ping returning pong/pid. Daemons run as
python -m browser_harness.daemon and must close the CLI's inherited lease FD.
BH_TMP_DIR is runtime/tmp, separate from the lease, PID and own-tab marker.
Compatibility is verified through process ownership, environment and IPC, not
package-version equality. Version metadata is diagnostic only; there is no
untargeted fallback when a contract check fails.
"""

import atexit
import contextlib
import errno
import hashlib
import json
import os
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.parse import parse_qsl, unquote

import psutil

from hermes_constants import get_hermes_home
from utils import atomic_json_write

_IDLE_SECONDS = 600
_USED = set()


class TargetError(RuntimeError):
    """Safe, actionable lifecycle diagnostic (never raw provider exceptions)."""


def redact(text, endpoints):
    """Remove only the known CDP endpoints and their credential values."""
    text = str(text)
    secrets = set()
    for endpoint in filter(None, endpoints):
        endpoint = str(endpoint)
        secrets.add(endpoint)
        try:
            parsed = urlsplit(endpoint)
        except ValueError:
            continue
        secrets.update(filter(None, (parsed.username, parsed.password)))
        secrets.update(value for _, value in parse_qsl(parsed.query) if value)
        secrets.update(
            part.partition("=")[2]
            for part in parsed.query.split("&")
            if part.partition("=")[2]
        )
    for value in sorted(secrets | {unquote(s) for s in secrets}, key=len, reverse=True):
        text = re.sub(
            r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])",
            "<redacted-cdp>",
            text,
        )
    return text


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TargetError(
            "timeout: browser call exhausted its shared timeout_s budget; retry with a larger budget"
        )
    return remaining


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _scope_key(home, session, task_id):
    return _digest(
        json.dumps([
            str(home.resolve()),
            "named" if session else "task",
            session or task_id or "default",
        ])
    )


def _runtime(key):
    return Path(tempfile.gettempdir()).resolve() / ("hbu-" + key[:24])


@contextlib.contextmanager
def _lock(runtime, timeout):
    try:
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise TargetError(
            f"private-runtime: cannot prepare {runtime}; use a writable per-user temporary directory and remove conflicting non-directory entries"
        ) from None
    if os.name != "nt" and len(os.fsencode(runtime / "bu.sock")) >= 104:
        raise TargetError(
            "private-runtime: TMPDIR is too long for the harness socket; use a shorter per-user temporary directory"
        )
    if runtime.is_symlink() or (
        os.name != "nt"
        and (runtime.stat().st_uid != os.getuid() or runtime.stat().st_mode & 0o077)
    ):
        raise TargetError(
            f"private-runtime: {runtime} must be owned by this user with mode 0700; repair its ownership/permissions before retrying"
        )
    # Never unlink this file: waiters must continue locking the same inode.
    with open(runtime / "owner.lock", "a+b") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise TargetError(
                        "busy: another browser call holds this session; wait for it to finish or increase timeout_s"
                    ) from None
                time.sleep(0.05)
        try:
            yield handle
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            # Closing, rather than LOCK_UN, preserves a POSIX child's inherited
            # lease if Hermes exits while that CLI is still executing.


def _process(runtime, name, record):
    for launcher in record.get("launchers", []):
        try:
            process = psutil.Process(launcher[0])
            if (
                process.create_time() == launcher[1]
                and process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                raise TargetError(
                    "busy: previous browser CLI is still alive; wait before retrying"
                )
        except psutil.NoSuchProcess:
            pass
    # A stale/reused PID is not ownership. Also find a daemon that outlived an
    # interrupted CLI before publishing its PID file. Never signal scan misses.
    for process in psutil.process_iter(["cmdline"]):
        try:
            if (process.info["cmdline"] or [])[1:] != ["-m", "browser_harness.daemon"]:
                continue
            if record.get("pid") and (
                process.pid != record["pid"]
                or (
                    record.get("created") and process.create_time() != record["created"]
                )
            ):
                continue
            env = process.environ()
            if env.get("BU_NAME") == name and env.get("BH_RUNTIME_DIR") == str(runtime):
                return process
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            if process.username() != psutil.Process().username():
                continue
            raise TargetError(
                "ownership: cannot inspect a same-user harness process; check process permissions before retrying"
            ) from None
    return None


def _ping(runtime, timeout=2):
    request = {"meta": "ping"}
    if os.name == "nt":
        port = json.loads((runtime / "bu.port").read_text(encoding="utf-8"))
        address = ("127.0.0.1", port["port"])
        connection = socket.socket(socket.AF_INET)
        request["token"] = port["token"]
    else:
        address = str(runtime / "bu.sock")
        connection = socket.socket(socket.AF_UNIX)
    with connection:
        connection.settimeout(timeout)
        connection.connect(address)
        connection.sendall((json.dumps(request) + "\n").encode())
        with connection.makefile("rb") as stream:
            response = json.loads(stream.readline(4096))
    if response.get("pong") is not True or type(response.get("pid")) is not int:
        raise RuntimeError("Browser IPC identity is unavailable")
    return response["pid"]


def _stop(runtime, name, record, *, unlink=True, deadline=None):
    process = _process(runtime, name, record)
    if process:
        process.terminate()  # psutil guards against PID reuse on both signals.
        try:
            process.wait(timeout=min(3, _remaining(deadline)) if deadline else 3)
        except psutil.TimeoutExpired:
            process.kill()
            process.wait(timeout=_remaining(deadline) if deadline else 3)
    if unlink:
        if _process(runtime, name, {}):
            raise RuntimeError(
                "Browser runtime still has a live process; refusing replacement"
            )
        for suffix in ("pid", "sock", "port"):
            (runtime / ("bu." + suffix)).unlink(missing_ok=True)


def _endpoint(env, timeout=10):
    import requests

    endpoint = env.get("BU_CDP_WS") or env["BU_CDP_URL"]
    parsed = urlsplit(endpoint)
    if parsed.scheme in {"http", "https"}:
        path = parsed.path.rstrip("/")
        if not path.endswith("/json/version"):
            path += "/json/version"
        discovery = urlunsplit(parsed._replace(path=path, fragment=""))
        # requests preserves URL basic auth and query credentials, as the
        # existing CDP resolver does. Never surface its raw exception text.
        response = requests.get(discovery, timeout=timeout)
        response.raise_for_status()
        endpoint = response.json()["webSocketDebuggerUrl"]
        parsed = urlsplit(endpoint)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise RuntimeError("Browser discovery returned no WebSocket endpoint")
    reusable = bool(
        re.fullmatch(
            r"/devtools/browser/[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            parsed.path,
        )
    )
    return endpoint, reusable


def _reap(directory, exiting=False, deadline=None):
    for marker in directory.glob("*.json"):
        if deadline and time.monotonic() >= deadline:
            break
        key = marker.stem
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            continue
        try:
            with _lock(_runtime(key), 0):
                record = json.loads(marker.read_text(encoding="utf-8"))
                if exiting:
                    if (str(directory), key) not in _USED or record.get(
                        "owner"
                    ) != os.getpid():
                        continue
                elif time.time() - record.get("used", time.time()) < _IDLE_SECONDS:
                    continue
                _stop(_runtime(key), "hermes-" + key[:24], record, deadline=deadline)
                marker.unlink()
                _USED.discard((str(directory), key))
        except (OSError, ValueError, RuntimeError, psutil.Error):
            continue  # Busy, unconfirmed, or unidentifiable owners are never killed.


def _invoke(cmd, code, options, marker, record, deadline):
    _remaining(deadline)
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **options,
    ) as child:
        record["launchers"] = [[child.pid, psutil.Process(child.pid).create_time()]]
        atomic_json_write(marker, record, mode=0o600)
        try:
            stdout, stderr = child.communicate(code, timeout=_remaining(deadline))
        except subprocess.TimeoutExpired:
            # A uv launcher can leave its CLI child running. Keep those leases
            # discoverable on Windows too, where pass_fds is unavailable.
            with contextlib.suppress(psutil.NoSuchProcess):
                for descendant in psutil.Process(child.pid).children(recursive=True):
                    with contextlib.suppress(psutil.NoSuchProcess):
                        if descendant.cmdline()[1:] != ["-m", "browser_harness.daemon"]:
                            record["launchers"].append([
                                descendant.pid,
                                descendant.create_time(),
                            ])
            child.kill()
            child.wait()
            raise
        finally:
            atomic_json_write(marker, record, mode=0o600)
    record.pop("launchers", None)
    atomic_json_write(marker, record, mode=0o600)
    return subprocess.CompletedProcess(cmd, child.returncode, stdout, stderr)


def run_targeted(cmd, code, env, session, task_id, timeout, popen_kwargs):
    """Run under an exclusive lease, publishing a binding only after bootstrap.

    Interrupted bootstraps recover once their launcher leases are dead. Opaque
    WS endpoints reconnect each call: their URL cannot prove browser-instance identity.
    """
    deadline = time.monotonic() + timeout
    endpoints = [env.get("BU_CDP_WS"), env.get("BU_CDP_URL")]
    home = Path(get_hermes_home()).resolve()
    key = _scope_key(home, session, task_id)
    name, runtime = "hermes-" + key[:24], _runtime(key)
    directory = home / "cache" / "browser-use" / "daemons"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker = directory / (key + ".json")
    with _lock(runtime, _remaining(deadline)) as lease:
        scratch = runtime / "tmp"
        scratch.mkdir(mode=0o700, exist_ok=True)
        if scratch.is_symlink() or (
            os.name != "nt"
            and (scratch.stat().st_uid != os.getuid() or scratch.stat().st_mode & 0o077)
        ):
            raise TargetError(
                "private-runtime: harness scratch directory must be owned by this user with mode 0700"
            )
        try:
            endpoint, reusable = _endpoint(env, min(10, _remaining(deadline)))
        except Exception as error:
            raise TargetError("discovery: " + redact(str(error), endpoints)) from None
        endpoints.append(endpoint)
        target = _digest(endpoint)
        env.update(
            BU_NAME=name,
            BH_RUNTIME_DIR=str(runtime),
            BH_RUNTIME_DIR_SHARED="0",
            BH_TMP_DIR=str(scratch),
            BH_TMP_DIR_SHARED="0",
            BU_CDP_WS=endpoint,
        )
        for obsolete in ("BU_CDP_URL", "BU_AUTOSPAWN", "BU_BROWSER_ID"):
            env.pop(obsolete, None)
        record = (
            json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
        )
        version = record.get("version", "unknown")
        _USED.add((str(directory), key))
        process = _process(runtime, name, record)
        if process and (
            not reusable
            or record.get("target") != target
            or record.get("pending")
            or process.environ().get("BH_TMP_DIR") != str(scratch)
        ):
            _stop(runtime, name, record, deadline=deadline)
            process = None
        options = dict(text=True, env=env, **popen_kwargs)
        if os.name != "nt":
            # The CLI inherits the lease across the spawn -> marker-write crash
            # window. Its daemon must close it (verified by real-harness E2E).
            options["pass_fds"] = (lease.fileno(),)
        if not process:
            _stop(runtime, name, {}, deadline=deadline)
            record = {"pending": True, "used": time.time(), "owner": os.getpid()}
            boot = _invoke(
                cmd,
                "from importlib.metadata import PackageNotFoundError, version\n"
                "try:\n    print('hermes-harness:' + version('browser-harness'))\n"
                "except PackageNotFoundError:\n    print('hermes-harness:unknown')\n",
                options,
                marker,
                record,
                deadline,
            )
            if boot.returncode:
                raise TargetError(
                    "bootstrap: "
                    + redact(
                        boot.stderr or "CLI exited before becoming ready; retry",
                        endpoints,
                    )
                )
            version = next(
                (
                    line.partition(":")[2]
                    for line in boot.stdout.splitlines()
                    if line.startswith("hermes-harness:")
                ),
                "unknown",
            )
            version = redact(version, endpoints)[:80] or "unknown"
            process = _process(runtime, name, {"pending": True})
            if not process or process.environ().get("BU_CDP_WS") != endpoint:
                raise TargetError(
                    f"capability: browser-harness {version} did not publish a daemon matching BU_NAME/BH_RUNTIME_DIR/BU_CDP_WS. Repair the harness integration; no user code was run"
                )
        elif process.environ().get("BU_CDP_WS") != endpoint:
            raise TargetError(
                "binding: owned daemon does not match the selected CDP target; no user code was run"
            )
        try:
            if _ping(runtime, min(2, _remaining(deadline))) != process.pid:
                raise ValueError("PID mismatch")
            if int((runtime / "bu.pid").read_text(encoding="utf-8")) != process.pid:
                raise ValueError("PID file mismatch")
            # The own-tab preamble still re-reads bu.pid after CLI ensure_daemon,
            # which may self-heal between this preflight and code execution.
        except Exception:
            raise TargetError(
                f"capability: browser-harness {version} failed the owned IPC ping/PID check (bu.pid and JSON pong/pid must match the verified process). Repair the harness integration; no user code was run"
            ) from None
        record = {
            "pid": process.pid,
            "created": process.create_time(),
            "target": target,
            "used": time.time(),
            "owner": os.getpid(),
            "version": version,
        }
        atomic_json_write(marker, record, mode=0o600)
        try:
            result = _invoke(cmd, code, options, marker, record, deadline)
        finally:
            # The harness can self-restart during ensure_daemon. Capture that
            # process only after the CLI exits, still holding the owning lease.
            current = _process(runtime, name, {})
            if current:
                if current.pid != record["pid"]:
                    # The harness may have self-healed. Confirm the previous
                    # process is gone without unlinking the replacement's IPC.
                    _stop(runtime, name, record, unlink=False, deadline=deadline)
                record.update(pid=current.pid, created=current.create_time())
            record["used"] = time.time()
            atomic_json_write(marker, record, mode=0o600)
        result.stdout = redact(result.stdout, endpoints)
        result.stderr = redact(result.stderr, endpoints)
    # Cleanup uses only the unused portion of this call's budget.
    if time.monotonic() < deadline:
        _reap(directory, deadline=deadline)
    return result


def _cleanup():
    for directory in {directory for directory, _ in _USED}:
        _reap(Path(directory), exiting=True)


atexit.register(_cleanup)
