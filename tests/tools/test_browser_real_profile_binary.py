"""Tests for browser.real_profile_binary (real-profile launch executable override).

Native behavior: the snapshot is launched with the detected default browser's own
binary (``chromium_executable``). On macOS that binary lives in the same app
bundle as the user's everyday browser, and LaunchServices treats every process
out of one bundle as ONE app: while the headless snapshot browser is alive, a
Dock / Spotlight / ``open`` launch of the bundle only activates it, so the
everyday browser cannot be opened. The override points the launch at a clone of
the bundle at another path instead.

Invariants under test:
- override set + executable -> that path is launched, native resolver ignored
- override set + not executable / missing -> FAIL CLOSED (error), never the real bundle
- override unset -> native chromium_executable, byte-for-byte
- the bad-override check happens BEFORE the snapshot (no cookies copied)
"""
import os
import stat
from unittest.mock import Mock, patch

from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_install as bt_install
from tools import browser_tool_real_profile as bt_real_profile


def _make_exec(path):
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class TestRealProfileExecutable:
    def test_override_wins_over_native(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc

        clone = _make_exec(tmp_path / "Brave Agent")
        monkeypatch.setattr(bc, "_browser_setting", lambda key: clone if key == "real_profile_binary" else None)
        monkeypatch.setattr(bc, "chromium_executable", lambda *a, **k: "/Applications/Real.app/Contents/MacOS/Real")

        assert bc.real_profile_executable("brave") == (clone, None)

    def test_override_expands_user(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc

        clone = _make_exec(tmp_path / "Brave Agent")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(bc, "_browser_setting", lambda key: "~/Brave Agent" if key == "real_profile_binary" else None)

        assert bc.real_profile_executable("brave") == (clone, None)

    def test_missing_override_fails_closed(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc

        missing = str(tmp_path / "gone")
        monkeypatch.setattr(bc, "_browser_setting", lambda key: missing if key == "real_profile_binary" else None)
        monkeypatch.setattr(bc, "chromium_executable", lambda *a, **k: "/Applications/Real.app/Contents/MacOS/Real")

        path, err = bc.real_profile_executable("brave")
        assert path is None
        assert err and "real_profile_binary" in err and missing in err

    def test_non_executable_override_fails_closed(self, tmp_path, monkeypatch):
        import hermes_cli.browser_connect as bc

        plain = tmp_path / "not-a-binary"
        plain.write_text("")
        plain.chmod(stat.S_IRUSR | stat.S_IWUSR)
        monkeypatch.setattr(bc, "_browser_setting", lambda key: str(plain) if key == "real_profile_binary" else None)

        path, err = bc.real_profile_executable("brave")
        assert path is None
        assert err and "real_profile_binary" in err

    def test_unset_keeps_native_resolver(self, monkeypatch):
        import hermes_cli.browser_connect as bc

        monkeypatch.setattr(bc, "_browser_setting", lambda key: None)
        monkeypatch.setattr(bc, "chromium_executable", lambda browser, system=None: f"/native/{browser}")

        assert bc.real_profile_executable("chrome") == ("/native/chrome", None)

    def test_blank_override_keeps_native_resolver(self, monkeypatch):
        import hermes_cli.browser_connect as bc

        monkeypatch.setattr(bc, "_browser_setting", lambda key: "   " if key == "real_profile_binary" else None)
        monkeypatch.setattr(bc, "chromium_executable", lambda browser, system=None: f"/native/{browser}")

        assert bc.real_profile_executable("chrome") == ("/native/chrome", None)


class TestRealProfileCdpUsesOverride:
    def _reset(self):
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.clear()
        bt._real_profile_chrome_procs.clear()

    def test_launch_uses_override_binary(self, tmp_path):
        import tools.browser_tool as bt
        self._reset()
        clone = _make_exec(tmp_path / "Brave Agent")
        launched = []

        class FakeChrome:
            def poll(self):
                return None

        def fake_popen(argv, **kw):
            launched.append(argv[0])
            (tmp_path / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
            return FakeChrome()

        proc = Mock(return_value=None, returncode=0, stdout="", stderr="")
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="brave"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(str(tmp_path), None)), \
             patch("hermes_cli.browser_connect._browser_setting",
                   side_effect=lambda key: clone if key == "real_profile_binary" else None), \
             patch("hermes_cli.browser_connect.chromium_executable",
                   return_value="/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"), \
             patch.object(bt.subprocess, "Popen", side_effect=fake_popen), \
             patch.object(bt_real_profile, "_agent_browser_get_cdp",
                          side_effect=[None, "http://127.0.0.1:41000"]), \
             patch.object(bt_install, "_find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch.object(bt.subprocess, "run", return_value=proc), \
             patch.object(bt_cloud, "_is_headed_mode", return_value=False):
            cdp, err = bt_real_profile._real_profile_cdp()
        assert err is None and cdp == "http://127.0.0.1:41000"
        assert launched == [clone], "the clone must be launched, never the real bundle"
        self._reset()

    def test_bad_override_fails_closed_before_snapshot(self, tmp_path):
        import tools.browser_tool as bt
        self._reset()
        snapshot = Mock(return_value=(str(tmp_path), None))
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="brave"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", snapshot), \
             patch("hermes_cli.browser_connect._browser_setting",
                   side_effect=lambda key: str(tmp_path / "gone") if key == "real_profile_binary" else None), \
             patch.object(bt_real_profile, "_agent_browser_get_cdp", return_value=None), \
             patch.object(bt.subprocess, "Popen") as popen:
            cdp, err = bt_real_profile._real_profile_cdp()
        assert cdp is None
        assert err and "real_profile_binary" in err
        snapshot.assert_not_called()
        popen.assert_not_called()
        assert bt._real_profile_chrome_procs == []
