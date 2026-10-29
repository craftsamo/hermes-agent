"""The thread a continuable cron delivery opens takes its name from the run's ``[TITLE: …]`` line,
else the job's ``thread_title``, else ``Hermes — <job name>``; the marker never reaches the chat."""

from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler_delivery import (
    _cron_thread_title, _deliver_result, _open_continuable_cron_thread, _split_title_marker,
)


class TestSplitTitleMarker:
    def test_the_first_marker_line_is_taken_and_removed(self):
        rest, title = _split_title_marker("Found it.\n[TITLE: Dependabot #12 critical: lodash]\nDetails follow.")
        assert title == "Dependabot #12 critical: lodash"
        assert rest == "Found it.\nDetails follow."

    def test_no_marker_leaves_content_alone(self):
        text = "Plain brief with [brackets] and a TITLE word."
        assert _split_title_marker(text) == (text, None)

    def test_marker_only_on_its_own_line(self):
        text = "inline [TITLE: not a marker] text"
        assert _split_title_marker(text) == (text, None)

    def test_an_empty_marker_is_dropped_without_a_title(self):
        assert _split_title_marker("[TITLE: ]\nbody") == ("body", None)


class TestCronThreadTitle:
    def test_run_title_wins_over_job_field_over_default(self):
        assert _cron_thread_title({"id": "j", "name": "dependabot-fix"}) == "Hermes — dependabot-fix"
        assert _cron_thread_title({"id": "j", "name": "x", "thread_title": "Dependabot fix"}) == "Dependabot fix"
        assert _cron_thread_title({"id": "j", "name": "x", "thread_title": "Dependabot fix",
                                   "_run_thread_title": "Dependabot #12"}) == "Dependabot #12"

    def test_title_is_one_line_and_capped(self):
        job = {"id": "j", "name": "x", "thread_title": "  multi\nline   title  "}
        assert _cron_thread_title(job) == "multi line title"
        job["thread_title"] = "y" * 200
        title = _cron_thread_title(job)
        assert len(title) == 120 and title.endswith("...")

    def test_open_thread_uses_the_resolved_title(self):
        adapter = MagicMock()
        adapter.create_handoff_thread = AsyncMock(return_value="4242")
        loop = MagicMock()

        def run(coro, _loop):
            import asyncio
            f = Future()
            f.set_result(asyncio.run(coro))
            return f

        with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=run):
            thread_id = _open_continuable_cron_thread(
                {"id": "j", "name": "dependabot-fix", "_run_thread_title": "Dependabot #12"}, adapter, "123", loop)
        assert thread_id == "4242"
        adapter.create_handoff_thread.assert_awaited_once_with("123", "Dependabot #12")


class TestDeliveryStripsTheMarker:
    """End to end through ``_deliver_result``: the chat gets the body without the marker, the
    thread gets the title, and the transient title does not outlive the delivery."""

    def _deliver(self, content: str, job_extra: dict | None = None):
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        pconfig.extra = {}
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True, message_id="m1", raw_response=None)
        adapter.create_handoff_thread = AsyncMock(return_value="777")
        adapter.supports_inchannel_continuable = False
        adapter.supports_inchannel_continuable_for_platform = None
        adapter._session_store = MagicMock()
        loop = MagicMock()
        loop.is_running.return_value = True

        def run(coro, _loop):
            import asyncio
            f = Future()
            try:
                f.set_result(asyncio.run(coro))
            except BaseException as e:
                f.set_exception(e)
            return f

        job = {"id": "job1", "name": "dependabot-fix", "deliver": "telegram:123", "attach_to_session": True,
               **(job_extra or {})}
        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=run), \
             patch("agent.async_utils.safe_schedule_threadsafe", side_effect=run), \
             patch("gateway.mirror.mirror_to_session", return_value=True) as mirror:
            error = _deliver_result(job, content, adapters={Platform.TELEGRAM: adapter}, loop=loop)
        return error, adapter, mirror, job

    def test_marker_names_the_thread_and_leaves_the_chat(self):
        error, adapter, mirror, job = self._deliver("[TITLE: Dependabot #12 critical: lodash]\nPR #40 is up.")
        assert error is None
        adapter.create_handoff_thread.assert_awaited_once_with("123", "Dependabot #12 critical: lodash")
        sent = adapter.send.call_args
        assert sent.args[1] == "PR #40 is up." and sent.kwargs["metadata"]["thread_id"] == "777"
        assert "[TITLE:" not in mirror.call_args.args[2]
        assert "_run_thread_title" not in job

    def test_job_thread_title_is_the_fallback(self):
        error, adapter, _mirror, _job = self._deliver("PR #40 is up.", {"thread_title": "Dependabot fix"})
        assert error is None
        adapter.create_handoff_thread.assert_awaited_once_with("123", "Dependabot fix")
        assert adapter.send.call_args.args[1] == "PR #40 is up."

    def test_default_name_is_unchanged(self):
        _error, adapter, _mirror, _job = self._deliver("PR #40 is up.")
        adapter.create_handoff_thread.assert_awaited_once_with("123", "Hermes — dependabot-fix")


def test_create_job_keeps_thread_title(tmp_path, monkeypatch):
    from cron import jobs as jobs_api
    with jobs_api.use_cron_store(tmp_path):
        job = jobs_api.create_job("p", "1h", name="j", thread_title="  Dependabot\nfix ")
        assert job["thread_title"] == "Dependabot fix"
        plain = jobs_api.create_job("p", "1h", name="k")
        assert "thread_title" not in plain


class TestCreateDmTopicRetries:
    """Transient connection/pool errors on ``_create_dm_topic`` retry up to 3 times before giving
    up (which would otherwise drop the report into the flat DM root); permanent BadRequests and
    ambiguous read timeouts do not retry."""

    def _adapter(self):
        from tests.gateway.conftest import _ensure_telegram_mock
        _ensure_telegram_mock()
        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
        adapter._bot = AsyncMock()
        return adapter

    @pytest.mark.asyncio
    async def test_transient_network_error_retries_and_succeeds(self):
        from types import SimpleNamespace
        adapter = self._adapter()
        NetErr, _, _ = adapter._telegram_error_types()
        adapter._bot.create_forum_topic = AsyncMock(side_effect=[
            NetErr("httpx.ConnectError"),
            NetErr("httpx.ConnectError"),
            SimpleNamespace(message_thread_id=888),
        ])
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
            thread_id = await adapter.create_handoff_thread("123", "Dependabot #121 high: systeminformation")
        assert thread_id == "888"
        assert adapter._bot.create_forum_topic.await_count == 3
        assert [c.args[0] for c in sleep.await_args_list] == [1, 2]

    @pytest.mark.asyncio
    async def test_pool_timeout_drains_connections_and_retries(self):
        from types import SimpleNamespace
        adapter = self._adapter()
        _, _, TimedOut = adapter._telegram_error_types()
        pool_err = TimedOut("Pool timeout: All connections in the connection pool are occupied. Request was *not* sent")
        adapter._bot.create_forum_topic = AsyncMock(side_effect=[
            pool_err,
            SimpleNamespace(message_thread_id=999),
        ])
        adapter._drain_general_connections_after_pool_timeout = AsyncMock()
        with patch("asyncio.sleep", new_callable=AsyncMock):
            thread_id = await adapter._create_dm_topic(123, "Dependabot #121")
        assert thread_id == 999
        adapter._drain_general_connections_after_pool_timeout.assert_awaited_once()
        assert adapter._bot.create_forum_topic.await_count == 2

    @pytest.mark.asyncio
    async def test_bad_request_and_ambiguous_timeout_do_not_retry(self):
        adapter = self._adapter()
        _, BadReq, TimedOut = adapter._telegram_error_types()
        for err in (BadReq("Bad Request: chat is not a forum"), TimedOut("Timed out")):
            adapter._bot.create_forum_topic = AsyncMock(side_effect=err)
            with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
                assert await adapter._create_dm_topic(123, "Dependabot #121") is None
            assert adapter._bot.create_forum_topic.await_count == 1
            sleep.assert_not_awaited()

