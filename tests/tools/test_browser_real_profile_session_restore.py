"""The real-profile copy must not restore the previous run's tabs.

The profile copy is long-lived and reused by every launch, so Chromium's ordinary
session restore replays the last run's tabs into the new browser. Measured on Brave
152 headless, they come back after a graceful SIGTERM (``exit_type`` = ``Normal``)
exactly as they do after a SIGKILL, so the exit kind is not the lever —
``<profile>/Sessions`` is. Left alone it compounds across relaunches: an assistant
profile reached 128 pages / 79 workers / 7.5 GB RSS, at which point ``Runtime.evaluate``
timed out behind the renderer load and the browser was too busy to answer SIGTERM.
"""
import os
from unittest.mock import Mock, patch

from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_install as bt_install
from tools import browser_tool_real_profile as bt_real_profile


def _seed_profile_copy(root):
    """A copy that looks like a used Chromium profile: tab state + auth state."""
    default = root / "Default"
    (default / "Sessions").mkdir(parents=True)
    (default / "Sessions" / "Session_13433593351078273").write_bytes(b"SNSS-tab-state")
    (default / "Sessions" / "Tabs_13433593352388094").write_bytes(b"SNSS-tab-state")
    (default / "IndexedDB" / "https_web.whatsapp.com_0.indexeddb.leveldb").mkdir(parents=True)
    (default / "Cookies").write_bytes(b"cookie-db")
    (default / "Login Data").write_bytes(b"login-db")
    (root / "Local State").write_text("{}")
    return default


class TestPurgeSessionRestoreState:
    def test_removes_tab_state_but_keeps_auth(self, tmp_path):
        default = _seed_profile_copy(tmp_path)

        bt_real_profile._purge_session_restore_state(str(tmp_path))

        assert not (default / "Sessions").exists()
        # Logins must survive: cookies, passwords, and site-side sessions (WhatsApp
        # Web keeps its linked device in IndexedDB, not in a cookie).
        assert (default / "Cookies").read_bytes() == b"cookie-db"
        assert (default / "Login Data").read_bytes() == b"login-db"
        assert (default / "IndexedDB" / "https_web.whatsapp.com_0.indexeddb.leveldb").is_dir()
        assert (tmp_path / "Local State").exists()

    def test_covers_every_profile_directory_in_the_copy(self, tmp_path):
        """The copy normally carries only ``Default``, but a pin can land elsewhere."""
        for name in ("Default", "Profile 12"):
            (tmp_path / name / "Sessions").mkdir(parents=True)
            (tmp_path / name / "Sessions" / "Tabs_1").write_bytes(b"x")

        bt_real_profile._purge_session_restore_state(str(tmp_path))

        assert not (tmp_path / "Default" / "Sessions").exists()
        assert not (tmp_path / "Profile 12" / "Sessions").exists()

    def test_is_best_effort_and_never_raises(self, tmp_path):
        # Nothing to purge, and a copy_dir that does not exist at all: a launch must
        # not be blocked by this bookkeeping.
        bt_real_profile._purge_session_restore_state(str(tmp_path))
        bt_real_profile._purge_session_restore_state(str(tmp_path / "missing"))


class TestLaunchPurgesBeforeStartingChrome:
    def _reset(self):
        import tools.browser_tool as bt
        bt._real_profile_cdp_cache.clear()

    def test_sessions_are_gone_by_the_time_chrome_starts(self, tmp_path):
        """Ordering matters: purging after launch would let Chromium restore first."""
        import tools.browser_tool as bt
        self._reset()
        default = _seed_profile_copy(tmp_path)
        seen = {}

        class FakeChrome:
            def poll(self):
                return None

        def fake_popen(argv, **kw):
            seen["sessions_existed_at_launch"] = (default / "Sessions").exists()
            (tmp_path / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
            return FakeChrome()

        proc = Mock(return_value=None, returncode=0, stdout="", stderr="")
        with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
             patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"), \
             patch("hermes_cli.browser_connect.snapshot_real_profile", return_value=(str(tmp_path), None)), \
             patch("hermes_cli.browser_connect.chromium_executable", return_value="/usr/bin/chrome"), \
             patch.object(bt.subprocess, "Popen", side_effect=fake_popen), \
             patch.object(bt_real_profile, "_agent_browser_get_cdp",
                          side_effect=[None, "http://127.0.0.1:41000"]), \
             patch.object(bt_install, "_find_agent_browser", return_value="/usr/bin/agent-browser"), \
             patch.object(bt.subprocess, "run", return_value=proc), \
             patch.object(bt_cloud, "_is_headed_mode", return_value=False):
            cdp, err = bt_real_profile._real_profile_cdp()

        assert err is None and cdp == "http://127.0.0.1:41000"
        assert seen["sessions_existed_at_launch"] is False
        assert (default / "Cookies").exists()  # the launch kept the login state
        self._reset()
