"""Tests for the real-profile headless user-agent override.

Chrome's new headless advertises ``HeadlessChrome/<version>``. Sites that gate on the UA
string then refuse a browser that is perfectly current: WhatsApp Web answers a headless
Chromium 152 with "WhatsApp works with Google Chrome 100+ / update Chrome" (reproduced
2026-09-10 on the Brave clone; the same launch with an ordinary ``Chrome/`` UA reaches the
QR login page). Relaunching cannot clear it, so the launch advertises the SAME version
under the ordinary token.

Invariants under test:
- headless launch -> ``--user-agent`` present, carrying the binary's own major version and
  no "Headless" anywhere
- headed launch -> no ``--user-agent`` (a window already sends the ordinary UA)
- version unreadable (probe fails / unparsable) -> NO flag, native UA kept; never a guessed
  version, and the launch still proceeds
"""
import subprocess
from unittest.mock import Mock

import pytest

from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_real_profile as bt_real_profile


def _argv_of(monkeypatch, tmp_path, *, headed=False, version_output="Brave Browser 152.1.94.121"):
    """Run the launch with everything stubbed; return the argv it would have spawned."""
    recorded = {}

    def fake_run(argv, **kwargs):
        assert argv[1] == "--version"
        if version_output is None:
            raise OSError("cannot execute")
        return subprocess.CompletedProcess(argv, 0, stdout=version_output, stderr="")

    def fake_popen(argv, **kwargs):
        recorded["argv"] = argv
        proc = Mock()
        proc.poll.return_value = None
        return proc

    monkeypatch.setattr(bt_real_profile.subprocess, "run", fake_run)
    monkeypatch.setattr(bt_real_profile.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bt_cloud, "_is_headed_mode", lambda: headed)
    monkeypatch.setattr(bt_real_profile, "_read_devtools_port", lambda copy_dir: "51234")
    monkeypatch.setattr(bt_real_profile, "_origin", lambda: Mock(_real_profile_chrome_procs=[]))

    port, err = bt_real_profile._launch_real_profile_chrome("/clone/Brave Agent", str(tmp_path))
    assert (port, err) == (51234, None)
    return recorded["argv"]


def _user_agent_flags(argv):
    return [a for a in argv if a.startswith("--user-agent=")]


class TestHeadlessUserAgent:
    def test_headless_launch_advertises_ordinary_chrome(self, tmp_path, monkeypatch):
        argv = _argv_of(monkeypatch, tmp_path)

        assert "--headless=new" in argv
        flags = _user_agent_flags(argv)
        assert len(flags) == 1
        ua = flags[0].split("=", 1)[1]
        assert "Headless" not in ua
        assert "Chrome/152.0.0.0" in ua  # the binary's OWN major version, not a pinned one

    def test_headed_launch_keeps_the_native_user_agent(self, tmp_path, monkeypatch):
        argv = _argv_of(monkeypatch, tmp_path, headed=True)

        assert "--headless=new" not in argv
        assert _user_agent_flags(argv) == []

    @pytest.mark.parametrize("version_output", [None, "", "Brave Browser (unknown)"])
    def test_unreadable_version_keeps_the_native_user_agent(
        self, tmp_path, monkeypatch, version_output
    ):
        """Fail OPEN: no flag rather than a guessed version, and the launch still happens."""
        argv = _argv_of(monkeypatch, tmp_path, version_output=version_output)

        assert "--headless=new" in argv
        assert _user_agent_flags(argv) == []


class TestChromeMajorVersion:
    @pytest.mark.parametrize(
        "output, expected",
        [
            ("Brave Browser 152.1.94.121", "152"),
            ("Google Chrome 152.0.7977.83", "152"),
            ("Chromium 141.0.7390.54 snap", "141"),
            ("Microsoft Edge 152.0.3485.66", "152"),
            ("", None),
            ("Brave Browser", None),
        ],
    )
    def test_parses_the_major_version(self, monkeypatch, output, expected):
        monkeypatch.setattr(
            bt_real_profile.subprocess, "run",
            lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout=output, stderr=""))

        assert bt_real_profile._chrome_major_version("/clone/browser") == expected

    def test_probe_failure_is_not_fatal(self, monkeypatch):
        def boom(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 10)

        monkeypatch.setattr(bt_real_profile.subprocess, "run", boom)

        assert bt_real_profile._chrome_major_version("/clone/browser") is None
        assert bt_real_profile._headless_user_agent("/clone/browser") is None

    def test_platform_token_follows_the_host(self, monkeypatch):
        monkeypatch.setattr(bt_real_profile, "_chrome_major_version", lambda binary: "152")

        monkeypatch.setattr(bt_real_profile.sys, "platform", "darwin")
        assert "Macintosh; Intel Mac OS X 10_15_7" in bt_real_profile._headless_user_agent("/b")

        monkeypatch.setattr(bt_real_profile.sys, "platform", "win32")
        assert "Windows NT 10.0; Win64; x64" in bt_real_profile._headless_user_agent("/b")

        monkeypatch.setattr(bt_real_profile.sys, "platform", "linux")
        assert "X11; Linux x86_64" in bt_real_profile._headless_user_agent("/b")
