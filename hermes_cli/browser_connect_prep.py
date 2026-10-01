"""Real-profile preparation budget and the held-copy guard.

Resolving the real-profile CDP (lock wait, snapshot, launch, attach) runs BEFORE the harness
inside one ``browser_exec`` call, while holding the process-wide real-profile lock. Each step has
its own cap, but the steps must also spend from the CALL's budget: preparation that outlives the
tool call leaves its thread (and the lock) behind after the executor abandons it, and every later
call queues on that lock. The budget is carried in a context variable so the steps keep their
plain signatures; a scope can only narrow it, never widen it.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import os
import socket
import time
from collections.abc import Iterator

_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "real_profile_prep_deadline", default=None)


class PreparationTimeout(TimeoutError):
    """The shared preparation budget ran out. Being a TimeoutError it is also an OSError, so a
    handler that turns OSError into a per-file failure must re-raise this ahead of it."""


def remaining(limit: float = math.inf) -> float:
    """Seconds a step may spend: ``limit`` capped by the active budget; raises once it is spent
    (call it bare as a checkpoint)."""
    deadline = _DEADLINE.get()
    left = limit if deadline is None else min(limit, deadline - time.monotonic())
    if left <= 0:
        raise PreparationTimeout(
            "real-profile browser preparation timed out; retry with a larger timeout_s.")
    return left


@contextlib.contextmanager
def budget(seconds: float | None = None, *, deadline: float | None = None) -> Iterator[None]:
    """Narrow the active budget to ``seconds`` from now and/or the monotonic ``deadline``."""
    bounds = [b for b in (_DEADLINE.get(), deadline,
                          None if seconds is None else time.monotonic() + seconds) if b is not None]
    token = _DEADLINE.set(min(bounds) if bounds else None)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def _user_data_dir_arg(argv: list[str]) -> Iterator[str]:
    for i, arg in enumerate(argv):
        if arg.startswith("--user-data-dir="):
            yield arg.partition("=")[2]
        elif arg == "--user-data-dir" and i + 1 < len(argv):
            yield argv[i + 1]


def snapshot_in_use(dst: str) -> bool:
    """True when a Chromium may still own the profile copy ``dst``, so overlaying its databases is
    unsafe (and a SQLite backup into a held file is exactly what wedged preparation). Conservative:
    a SingletonLock is stale only when it names THIS host and a pid that no longer exists; then an
    exact ``--user-data-dir`` scan must also come back clear (socket/cookie symlinks survive
    crashes, and a lock can be missing while the browser lives). Read-only."""
    import psutil

    lock = os.path.join(dst, "SingletonLock")
    if os.path.lexists(lock):
        if os.name != "posix":
            return True
        try:
            host, _, pid = os.readlink(lock).rpartition("-")
            if (host != socket.gethostname() or not pid.isascii() or not pid.isdecimal()
                    or int(pid) <= 0 or psutil.pid_exists(int(pid))):
                return True
        except (OSError, ValueError, OverflowError, psutil.Error):
            return True  # foreign, malformed, or unreadable ownership is no proof of staleness
    target = os.path.normcase(os.path.realpath(dst))
    for proc in psutil.process_iter():
        remaining(30.0)
        try:
            argv = proc.cmdline()
        except (psutil.Error, OSError, SystemError):
            # Gone or unreadable. macOS proc_cmdline can surface a sysctl race as SystemError;
            # the SingletonLock check above covers a holder whose cmdline cannot be read.
            continue
        if any(os.path.normcase(os.path.realpath(v)) == target for v in _user_data_dir_arg(argv)):
            return True
    return False
