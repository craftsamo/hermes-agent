"""Real-profile local browsing: snapshot the user's default Chromium profile into a
hermes-owned copy, launch the real browser binary on it, and attach agent-browser.

State (``_REAL_PROFILE_SESSION``, ``_real_profile_cdp_lock``, ``_real_profile_cdp_cache``,
``_real_profile_chrome_procs``) lives in ``tools.browser_tool``; it is read
through ``_bt`` (resolved per call — never import ``tools.browser_tool`` at import time).
"""

import os
import re
import shutil
import subprocess
import sys
import time
import json
import urllib.request
from urllib.parse import urlsplit
from typing import Optional, Tuple
from hermes_cli.browser_connect import _profile_remaining
from tools.browser_tool_origin import origin_module as _origin
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_install as _install
from tools import browser_tool_lightpanda_fallback as _lp
from tools import browser_tool_session as _session

_RP = "browser.use_real_profile is on, but "


def _terminate_real_profile_chrome() -> None:
    """Terminate real-browser processes launched for real-profile sessions (idempotent, atexit-safe);
    agent-browser only ATTACHED to them, so its own session cleanup never kills them."""
    from tools.browser_lightpanda import _terminate
    _bt = _origin()
    while _bt._real_profile_chrome_procs:
        _terminate(_bt._real_profile_chrome_procs.pop(), what="real-profile chrome")


def _cdp_http_ready(http_cdp: str, deadline=None) -> bool:
    """True when an ``http://host:port`` CDP discovery root answers."""
    from tools.browser_lightpanda import _cdp_ready

    return _cdp_ready(http_cdp, timeout=_profile_remaining(deadline, 1.0))


def _real_profile_daemon_env() -> dict:
    """Reaper-visible socket dir + ``owner_pid`` claim like every other lane (agent-browser's
    default dir is invisible to the reaper — #100855). The daemon-side idle timeout is dropped:
    Chrome is launched by Hermes, not the daemon, so a self-exiting daemon would leave Chrome
    holding the copy dir under the next snapshot overlay."""
    _bt = _origin()
    socket_dir = _session._prepare_session_socket_dir(_bt._real_profile_session())
    env = _session._agent_browser_command_env(socket_dir)
    env.pop("AGENT_BROWSER_IDLE_TIMEOUT_MS", None)
    return env


def _agent_browser_session_cmd(
    session_name: str, *cmd: str, log_label: str, deadline=None
) -> Optional[subprocess.CompletedProcess]:
    """Run ``agent-browser --session <name> <cmd...>``; None when agent-browser is missing or the run fails."""
    _bt = _origin()
    try:
        browser_cmd = _install._find_agent_browser()
    except FileNotFoundError:
        return None
    try:
        return subprocess.run([*_session._agent_browser_argv(browser_cmd), "--session", session_name, *cmd],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=_profile_remaining(deadline, 15),
                              env=_real_profile_daemon_env(), stdin=subprocess.DEVNULL)
    except TimeoutError:
        raise
    except (subprocess.SubprocessError, OSError) as e:
        _bt.logger.debug("real-profile %s failed: %s", log_label, e)
        return None


def _agent_browser_get_cdp(session_name: str, deadline=None) -> Optional[str]:
    """HTTP CDP discovery root of an agent-browser session (from its ``ws://`` cdp-url), or None."""
    proc = _agent_browser_session_cmd(
        session_name, "get", "cdp-url", log_label="get cdp-url", deadline=deadline
    )
    m = re.search(r"ws://127\.0\.0\.1:(\d+)/", (proc.stdout or "").strip()) if proc is not None else None
    return f"http://127.0.0.1:{m.group(1)}" if m else None


def _read_devtools_port(data_dir: str) -> Optional[str]:
    """First line of Chrome's ``DevToolsActivePort`` in ``data_dir`` (None when unreadable)."""
    try:
        with open(os.path.join(data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            return fh.readline().strip()
    except OSError:
        return None


def _surviving_chrome_cdp(data_dir: str, deadline=None) -> Optional[str]:
    """HTTP CDP root of a Chrome still running on ``data_dir``, or None. ``DevToolsActivePort``
    outlives a crashed Chrome and its port can be recycled by another local CDP server, so the
    file's browser id (line 2) must match what ``/json/version`` reports before it is trusted."""
    try:
        with open(os.path.join(data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            port, browser_path = fh.readline().strip(), fh.readline().strip()
    except OSError:
        return None
    if not port.isdigit() or not 0 < int(port) < 65536 or not browser_path.startswith("/devtools/browser/"):
        return None
    http_cdp = f"http://127.0.0.1:{port}"
    try:
        # Loopback discovery must not go through an operator's HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{http_cdp}/json/version", timeout=_profile_remaining(deadline, 2.0)) as response:
            version = json.load(response)
        ws_url = version.get("webSocketDebuggerUrl") if isinstance(version, dict) else None
    except TimeoutError:
        raise
    except (OSError, ValueError):
        return None
    if not isinstance(ws_url, str):
        return None
    ws = urlsplit(ws_url)
    return http_cdp if ws.port == int(port) and ws.path == browser_path else None


def _cdp_on_data_dir(http_cdp: str, data_dir: str) -> bool:
    """True when the CDP endpoint's browser runs on ``data_dir`` (DevToolsActivePort match proves it
    is our profile copy, not a throwaway temp dir a raced/stale launch fell back to)."""
    m = re.search(r":(\d+)", http_cdp or "")
    return bool(m) and _read_devtools_port(data_dir) == m.group(1)


def _agent_browser_close_session(session_name: str, deadline=None) -> None:
    """Best-effort close of an agent-browser session (stale/wrong-dir cleanup)."""
    _agent_browser_session_cmd(
        session_name, "close", log_label="session close", deadline=deadline
    )


_REAL_PROFILE_CHROME_FLAGS = (
    "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check",
    "--disable-background-networking", "--disable-component-update", "--disable-default-apps",
    "--disable-hang-monitor", "--disable-popup-blocking", "--disable-prompt-on-repost",
    "--disable-sync", "--disable-features=Translate", "--no-startup-window",
)


def _real_profile_unsupported_reason(browser) -> Optional[str]:
    """Fail-closed message when the default browser can't be used, else None.

    A pre-release channel lives in a profile dir we don't resolve; normalizing to the stable
    family would drive a DIFFERENT profile/account (wrong-principal bug), so refuse rather than guess.
    """
    from hermes_cli.browser_connect import UNSUPPORTED_CHANNEL
    if browser is None:
        return (_RP + "your default browser is not a supported Chromium browser (Chrome, Edge, Brave, "
                "Brave Origin, Chromium). Real-profile browsing requires a Chromium default; set one or turn the toggle off.")
    if browser == UNSUPPORTED_CHANNEL:
        return (_RP + "your default browser is a pre-release Chromium channel (Beta / Dev / Canary), which "
                "real-profile browsing does not support. Set your default to a "
                "stable Chrome / Edge / Brave / Brave Origin / Chromium, or turn the toggle off.")
    return None


def _real_profile_snapshot_error(err: str) -> str:
    """User-facing message for a failed profile snapshot; a locked profile adds the approved-close
    command, which the agent must ASK the user about first (it quits their browser)."""
    from hermes_cli.browser_connect import _PROFILE_LOCKED_PREFIX
    if err and err.startswith(_PROFILE_LOCKED_PREFIX):
        return (err[len(_PROFILE_LOCKED_PREFIX):] + " To close it (only after the user approves — it "
                "quits their browser and loses unsaved tabs), run: `hermes browser close-profile`, then retry.")
    return f"{_RP}{err}"


_UA_PLATFORMS = {
    "darwin": "Macintosh; Intel Mac OS X 10_15_7",
    "win32": "Windows NT 10.0; Win64; x64",
}


def _chrome_major_version(real_binary: str) -> Optional[str]:
    """Major version the binary reports (``Brave Browser 152.1.94.121`` -> ``152``), else None."""
    try:
        proc = subprocess.run([real_binary, "--version"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=10,
                              stdin=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError):
        return None
    m = re.search(r"(\d+)\.\d+\.\d+", f"{proc.stdout or ''} {proc.stderr or ''}")
    return m.group(1) if m else None


def _headless_user_agent(real_binary: str) -> Optional[str]:
    """Ordinary-Chrome UA for the headless launch; None when the version cannot be resolved.

    New headless advertises ``HeadlessChrome/<v>``, and sites that gate on the UA STRING refuse it
    even though the engine is perfectly current: WhatsApp Web answers a fully up-to-date headless
    Chromium 152 with its "WhatsApp works with Google Chrome 100+ / update Chrome" page, which no
    relaunch can clear. Advertise the same version under the ordinary ``Chrome/`` token instead.
    The version is read from the binary itself, so a browser update never leaves a stale UA behind;
    an unreadable version returns None and the launch keeps the native UA rather than lying about
    which engine is running. Headless only — a headed window already sends the ordinary UA.
    """
    major = _chrome_major_version(real_binary)
    if not major:
        return None
    platform = _UA_PLATFORMS.get(sys.platform, "X11; Linux x86_64")
    return (f"Mozilla/5.0 ({platform}) AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{major}.0.0.0 Safari/537.36")


def _purge_session_restore_state(copy_dir: str) -> None:
    """Drop Chromium's saved tab/window state so a launch starts with no restored tabs.

    The copy is a LONG-LIVED directory reused by every launch, so Chromium's ordinary session
    restore replays the previous run's tabs into the new one. Measured on Brave 152 headless:
    the tabs come back after a graceful SIGTERM (``exit_type`` = ``Normal``) just as they do
    after a SIGKILL, so this state — not the exit kind — is the only lever. Across relaunches
    it compounds: an assistant profile reached 128 pages / 79 workers / 7.5 GB RSS, which
    wedges ``Runtime.evaluate`` behind the renderer load and leaves the browser too busy to
    answer SIGTERM. Restoring tabs has no value here anyway — every ``browser_exec`` call
    drives the tab it opens or is attached to.

    Only ``<profile>/Sessions`` is removed. Logins are unaffected: cookies live in ``Cookies``,
    passwords in ``Login Data``, and site-side sessions (WhatsApp Web's linked device) in
    ``IndexedDB``. Best-effort — a failure here must never block a launch.
    """
    try:
        entries = os.listdir(copy_dir)
    except OSError:
        return
    for name in entries:
        sessions = os.path.join(copy_dir, name, "Sessions")
        if os.path.isdir(sessions):
            shutil.rmtree(sessions, ignore_errors=True)


def _launch_real_profile_chrome(
    real_binary: str, copy_dir: str, deadline=None
) -> Tuple[Optional[int], Optional[str]]:
    """Launch the user's REAL browser binary on the profile COPY; return (debug_port, error).

    agent-browser's own launch force-adds --use-mock-keychain / --password-store=basic, which makes
    macOS Chrome drop every keychain-encrypted cookie (signed-out copy); launching the real binary
    ourselves keeps the OS keychain path intact and agent-browser attaches via ``--cdp <port>``.
    Headless by default (a focus-stealing window defeats a background capability); Chrome's NEW
    headless shares the profile's cookie store (legacy --headless does not). browser.headed /
    AGENT_BROWSER_HEADED opts into a window, except on a display-less Linux host (launch would die).
    A headless launch also overrides the ``HeadlessChrome/<v>`` user agent (see
    ``_headless_user_agent``), which UA-gating sites reject outright.
    """
    _bt = _origin()
    deadline = time.monotonic() + _profile_remaining(deadline, 30.0)
    try:
        os.unlink(os.path.join(copy_dir, "DevToolsActivePort"))  # stale port confuses reuse probes
    except OSError:
        pass
    _purge_session_restore_state(copy_dir)
    chrome_argv = [real_binary, f"--user-data-dir={copy_dir}", *_REAL_PROFILE_CHROME_FLAGS]
    _has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not (_cloud._is_headed_mode() and (_has_display or not sys.platform.startswith("linux"))):
        chrome_argv.append("--headless=new")
        user_agent = _headless_user_agent(real_binary)
        if user_agent:
            chrome_argv.append(f"--user-agent={user_agent}")
    try:
        chrome_proc = subprocess.Popen(chrome_argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       stdin=subprocess.DEVNULL, start_new_session=True, env=_bt._build_browser_env())
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"{_RP}the launch failed: {e}"
    _bt._real_profile_chrome_procs.append(chrome_proc)

    while time.monotonic() < deadline:
        line = _read_devtools_port(copy_dir) or ""
        if line.isdigit():
            return int(line), None
        if chrome_proc.poll() is not None:
            _terminate_real_profile_chrome()
            return None, _RP + "Chrome exited during startup (another instance may hold the profile copy)."
        time.sleep(0.25)
    _terminate_real_profile_chrome()
    return None, _RP + "the real-profile browser did not expose a debug port in time. Retry, or turn the toggle off."


def _attach_agent_browser_to_real_profile(
    port: int, copy_dir: str, deadline=None
) -> Tuple[Optional[str], Optional[str]]:
    """Make agent-browser ATTACH to the running Chrome (never launch its own); returns ``(http_cdp, error)``.

    The daemon may answer with the endpoint of a browser IT spawned (throwaway temp profile);
    the DevToolsActivePort OUR Chrome wrote is authoritative on disagreement.
    """
    _bt = _origin()
    try:
        browser_cmd = _install._find_agent_browser()
    except FileNotFoundError as e:
        return None, f"{_RP}the local browser engine (agent-browser) is not installed: {e}"
    session_name = _bt._real_profile_session()
    argv = [*_session._agent_browser_argv(browser_cmd), "--session", session_name,
            "--cdp", str(port), "open", "about:blank"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=_profile_remaining(deadline, _bt._get_open_command_timeout(first_open=True)),
                              env=_real_profile_daemon_env(), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return (
            None,
            _RP
            + "the real-profile browser took too long to start. Retry, or turn the toggle off.",
        )
    except TimeoutError:
        raise
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"{_RP}the launch failed: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, f"{_RP}the real-profile browser failed to start: {tail[-1] if tail else f'exit {proc.returncode}'}"
    cdp = _agent_browser_get_cdp(session_name, deadline=deadline)
    our_port = _read_devtools_port(copy_dir)
    if our_port is not None and (m := re.search(r":(\d+)", cdp or "")) and m.group(1) != our_port:
        cdp = f"http://127.0.0.1:{our_port}"
    if not cdp:
        return None, _RP + "the real-profile browser started without exposing a devtools endpoint. Retry, or turn the toggle off."
    return cdp, None


def _real_profile_cdp(deadline=None) -> tuple:
    """Resolve ``(cdp_url, error)`` for consented real-profile browsing.

    Snapshot -> launch real binary on the copy -> return its HTTP CDP endpoint. The copy is a
    non-default dir, so it sidesteps the Chrome >=136 default-profile remote-debugging block and
    never contends with the user's running browser. One agent-browser session PER hermes home
    is reused across calls (cached, re-validated). ``(None, message)`` fail-closed; ``(None, None)``
    when consent is off.
    """
    _bt = _origin()
    session_name = _bt._real_profile_session()
    if not _cloud._use_real_profile():
        # Consent is off: delete any snapshot store (copies of cookies/logins) so
        # revoking consent actually removes the credential copies.
        try:
            from hermes_cli.browser_connect import cleanup_real_profile_snapshots
            cleanup_real_profile_snapshots()
        except Exception as e:
            _bt.logger.debug("real-profile cleanup-on-consent-off failed: %s", e)
        _bt._real_profile_cdp_cache.pop(session_name, None)
        return None, None

    # Lightpanda rejects ``--profile``; check BEFORE default-browser detection so a
    # host with no Chromium default still reports the actionable engine conflict.
    if _lp._using_lightpanda_engine():
        return None, (_RP + "browser.engine is set to 'lightpanda', which cannot load a real Chromium profile. "
                      "Set browser.engine to 'auto' or 'chrome' to use real-profile browsing, or turn the toggle off.")

    from hermes_cli.browser_connect import (detect_default_chromium, real_profile_copy_dir,
                                            real_profile_executable, snapshot_real_profile)

    if not _bt._real_profile_cdp_lock.acquire(
        timeout=_profile_remaining(deadline, 30.0)
    ):
        return (
            None,
            _RP
            + "timed out waiting for another real-profile browser preparation. Retry later.",
        )
    try:
        cached = _bt._real_profile_cdp_cache.get(session_name)
        if cached and _cdp_http_ready(cached, deadline=deadline):
            # Re-claim the daemon's socket dir so the orphan reaper's idle clock sees
            # this process still using it (a cache hit never runs a daemon command).
            _session._prepare_session_socket_dir(session_name)
            return cached, None
        _bt._real_profile_cdp_cache.pop(session_name, None)

        browser = detect_default_chromium()
        unsupported = _real_profile_unsupported_reason(browser)
        if unsupported:
            return None, unsupported

        # Reuse BEFORE writing anything. CRITICAL: the snapshot overlay (truncates/rewrites
        # Cookies / Login Data) must NOT run while a live copy-browser (maybe from a previous
        # hermes process) holds the user-data-dir open — that corrupts the databases.
        copy_dir = real_profile_copy_dir(browser)
        existing = _agent_browser_get_cdp(session_name, deadline=deadline)
        if (
            existing
            and _cdp_http_ready(existing, deadline=deadline)
            and _cdp_on_data_dir(existing, copy_dir)
        ):
            _bt._real_profile_cdp_cache[session_name] = existing
            return existing, None
        if existing:  # stale/wrong-dir session: close it so nothing holds the dir open
            _agent_browser_close_session(session_name, deadline=deadline)
        # A Chrome from an earlier hermes process can still hold the copy dir after its attach
        # daemon was reaped (that owner died). Re-attach to it rather than overlay a live profile;
        # if the daemon cannot attach, fail closed — never snapshot over an open profile. Not ours
        # to terminate (no Popen handle): it lives until the user closes it, by design.
        surviving = _surviving_chrome_cdp(copy_dir, deadline=deadline)
        if surviving:
            cdp, err = _attach_agent_browser_to_real_profile(
                int(surviving.rsplit(":", 1)[1]), copy_dir, deadline=deadline
            )
            if not cdp:
                return None, err
            _bt._real_profile_cdp_cache[session_name] = cdp
            _bt.logger.info("real-profile: re-attached to surviving Chrome at %s (%s)", cdp, copy_dir)
            return cdp, None
        # DevToolsActivePort can be missing or unverifiable while a browser still holds the copy
        # (crash mid-startup, recycled port). The snapshot overlay would corrupt its databases and
        # its sqlite backup blocks on the held file, so refuse rather than guess.
        from hermes_cli.browser_connect import _snapshot_in_use

        if _snapshot_in_use(copy_dir, deadline=deadline):
            return (
                None,
                _RP
                + "the profile copy is in use but its browser could not be verified. Refusing to overwrite login data or launch a fallback browser.",
            )

        # Resolve the binary BEFORE snapshotting: a bad browser.real_profile_binary fails closed
        # without copying a single cookie.
        real_binary, err = real_profile_executable(browser)
        if err:
            return None, f"{_RP}{err}"
        if real_binary is None:
            return None, f"{_RP}the real browser binary for '{browser}' could not be found. Reinstall it or turn the toggle off."
        _profile_remaining(deadline, 60.0)
        copy_dir, err = snapshot_real_profile(browser, deadline=deadline)
        if err or not copy_dir:
            return None, _real_profile_snapshot_error(err)
        port, err = _launch_real_profile_chrome(
            real_binary, copy_dir, deadline=deadline
        )
        if port is None:
            return None, err
        cdp, err = _attach_agent_browser_to_real_profile(
            port, copy_dir, deadline=deadline
        )
        if not cdp:
            return None, err
        _bt._real_profile_cdp_cache[session_name] = cdp
        _bt.logger.info("real-profile browser ready for %s at %s (%s)", browser, cdp, copy_dir)
        return cdp, None
    except TimeoutError as e:
        return None, str(e)
    finally:
        _bt._real_profile_cdp_lock.release()
