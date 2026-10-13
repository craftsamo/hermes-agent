"""``BasePlatformAdapter.send_clarify_context``: the material a clarify card asks about.

It goes out like a final reply (formatted text, then MEDIA attachments) and reports success only
when every part arrived; bare local paths in a plan stay text instead of becoming uploads.
"""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _Adapter(BasePlatformAdapter):
    def __init__(self, document_ok=True):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.log: list = []
        self._document_ok = document_ok

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.log.append(("text", chat_id, content))
        return SendResult(success=True, message_id="t1")

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None,
                            metadata=None, **kwargs):
        self.log.append(("document", chat_id, file_path))
        return SendResult(success=self._document_ok, message_id="d1", error=None if self._document_ok else "boom")

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0.0, **kwargs):
        # Base contract: success when at least one image of the batch arrived.
        self.log.append(("images", chat_id, [url for url, _ in images]))
        ok = [not url.endswith("broken.png") for url, _ in images]
        return SendResult(success=any(ok), message_id="i1", error=None if any(ok) else "image failed")

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.asyncio
async def test_text_then_media_and_bare_paths_stay_text(tmp_path):
    storyboard = tmp_path / "storyboard.pdf"
    storyboard.write_bytes(b"%PDF-1.4\n")
    adapter = _Adapter()
    content = f"Storyboard, 27 beats. Draft lives in {tmp_path}/notes.md\nMEDIA:{storyboard}"

    result = await adapter.send_clarify_context("42", content, source=None, session_key="sk")

    assert result.success
    assert [entry[0] for entry in adapter.log] == ["text", "document"]
    assert "notes.md" in adapter.log[0][2] and "MEDIA:" not in adapter.log[0][2]
    assert adapter.log[1] == ("document", "42", str(storyboard))


@pytest.mark.asyncio
async def test_a_failed_attachment_fails_the_whole_context(tmp_path):
    track = tmp_path / "track.pdf"
    track.write_bytes(b"%PDF-1.4\n")
    adapter = _Adapter(document_ok=False)

    result = await adapter.send_clarify_context("42", f"Two takes:\nMEDIA:{track}", source=None, session_key="sk")

    assert not result.success


@pytest.mark.asyncio
async def test_one_failed_image_of_several_fails_the_context(tmp_path):
    """A batch send succeeds when any image arrived; context images go one per send instead."""
    good, broken = tmp_path / "good.png", tmp_path / "broken.png"
    for path in (good, broken):
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = _Adapter()

    result = await adapter.send_clarify_context(
        "42", f"Compare:\nMEDIA:{good}\nMEDIA:{broken}", source=None, session_key="sk")

    assert not result.success
    assert [entry[0] for entry in adapter.log] == ["text", "images", "images"]


class _TextOnly(BasePlatformAdapter):
    """Inherits the base media senders, which answer with an "unavailable" notice."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.log: list = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.log.append(("text", chat_id, content))
        return SendResult(success=True, message_id="t1")

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.asyncio
async def test_a_surface_that_cannot_attach_files_fails_instead_of_sending_a_notice(tmp_path):
    draft = tmp_path / "draft.pdf"
    draft.write_bytes(b"%PDF-1.4\n")
    adapter = _TextOnly()

    result = await adapter.send_clarify_context("42", f"Review:\nMEDIA:{draft}", source=None, session_key="sk")

    assert not result.success and "files" in result.error
    assert adapter.log == []


@pytest.mark.asyncio
async def test_text_only_context_works_on_a_text_only_surface():
    adapter = _TextOnly()

    result = await adapter.send_clarify_context("42", "## Plan\n1. research", source=None, session_key="sk")

    assert result.success
    assert adapter.log == [("text", "42", "## Plan\n1. research")]


@pytest.mark.asyncio
async def test_a_media_path_that_cannot_be_delivered_fails_before_anything_is_sent(tmp_path):
    adapter = _Adapter()

    result = await adapter.send_clarify_context(
        "42", f"Review this draft:\nMEDIA:{tmp_path}/missing/draft.pdf", source=None, session_key="sk")

    assert not result.success
    assert adapter.log == []
