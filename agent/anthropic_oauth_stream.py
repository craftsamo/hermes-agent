"""Stream-side quarantine for leaked Anthropic OAuth tool markup.

Text is released line by line; once an unfenced ``<invoke name="mcp__...">`` line appears, that
line and every later delta (text, reasoning, tool start) is held until the final message is
known. A malformed final response discards the held deltas (the turn loop retries it); a valid
one releases them in order while this attempt still owns the stream writer. Markdown fence state
is tracked across deltas so documented invoke examples keep streaming incrementally.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.anthropic_oauth_markup import (
    anthropic_oauth_response_has_invoke_markup, anthropic_oauth_text_has_invoke_markup,
)

_FENCE_MARKERS = ("```", "~~~")
_INVOKE_PREFIX = "<invoke"


class AnthropicOAuthStreamQuarantine:
    """Route one stream attempt's display deltas; ``enabled=False`` delivers immediately."""

    def __init__(self, emitters: Dict[str, Callable[[str], None]], *, enabled: bool) -> None:
        self._emitters = emitters
        self._enabled = enabled
        self._line_buffer = ""
        self._fence: Optional[str] = None
        self._active = False
        self._pending: List[Tuple[str, str]] = []

    def deliver(self, kind: str, payload: str) -> None:
        if not self._enabled:
            self._emitters[kind](payload)
            return
        if self._active:
            self._pending.append((kind, payload))
            return
        if kind != "text":
            self._emitters[kind](payload)
            return
        self._feed_text(payload)

    def _quarantine(self, text: str) -> None:
        self._active = True
        self._pending.append(("text", text))
        self._line_buffer = ""

    def _feed_text(self, payload: str) -> None:
        emit_text = self._emitters["text"]
        self._line_buffer += payload
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            line += "\n"
            line_marker = line.lstrip()[:3]
            if line_marker in _FENCE_MARKERS:
                if self._fence is None:
                    self._fence = line_marker
                elif self._fence == line_marker:
                    self._fence = None
                emit_text(line)
            elif self._fence is None and anthropic_oauth_text_has_invoke_markup(line):
                self._quarantine(line + self._line_buffer)
                return
            else:
                emit_text(line)
        self._settle_partial_line()

    def _settle_partial_line(self) -> None:
        """Release the unterminated tail unless it may still become a fence or invoke line."""
        buffer = self._line_buffer
        stripped = buffer.lstrip().lower()
        if self._fence:
            if stripped.startswith(self._fence):
                self._fence = None
                self._release_line_buffer()
            elif stripped and not self._fence.startswith(stripped):
                self._release_line_buffer()
            return
        if stripped and any(
            candidate.startswith(stripped) or stripped.startswith(candidate) for candidate in _FENCE_MARKERS
        ):
            if len(stripped) >= 3:
                self._fence = stripped[:3]
                self._release_line_buffer()
        elif anthropic_oauth_text_has_invoke_markup(buffer):
            self._quarantine(buffer)
        elif stripped and not (_INVOKE_PREFIX.startswith(stripped) or stripped.startswith(_INVOKE_PREFIX)):
            self._release_line_buffer()

    def _release_line_buffer(self) -> None:
        self._emitters["text"](self._line_buffer)
        self._line_buffer = ""

    def finish(self, final_message: Any, writer_still_current: Callable[[], bool]) -> None:
        """Flush the trailing partial line, then release held deltas unless ``final_message``
        is the malformed shape the turn loop will discard."""
        if not self._enabled:
            return
        if self._line_buffer:
            self._settle_partial_line()
            if self._line_buffer and not self._active:
                self._release_line_buffer()
        if not self._active or not writer_still_current():
            return
        if anthropic_oauth_response_has_invoke_markup(final_message):
            return
        for kind, payload in self._pending:
            if not writer_still_current():
                break
            self._emitters[kind](payload)
