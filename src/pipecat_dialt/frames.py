"""Dialt-specific frames for information Pipecat cannot represent losslessly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pipecat.frames.frames import DataFrame, ErrorFrame, InterruptionFrame, SystemFrame


@dataclass
class DialtSpeechActivityFrame(SystemFrame):
    """Detected activity, not a semantic turn or permission to interrupt."""

    speaking: bool
    speech_id: str | None = None


@dataclass
class DialtConversationFrame(DataFrame):
    """Detached accepted SDK projection for the owned context aggregator."""

    items: list[dict[str, Any]]
    first_index: int


@dataclass
class DialtInterruptionFrame(InterruptionFrame):
    """Provider-originated output discard; echoes must not interrupt a new reply."""

    response_id: str


@dataclass
class DialtTranscriptCorrectionFrame(DataFrame):
    """Replace a previously projected transcript without creating a new turn.

    Pipecat 1.x has no standard transcript-revision frame. Consumers should
    replace the transcript identified by ``item_id`` at ``revision`` with
    ``text`` (including empty retractions). ``turn_id`` is segment metadata,
    not canonical identity. The owned pair updates context independently.
    """

    turn_id: str
    text: str
    speaker: Literal["user", "assistant"]
    revision: int | None = None
    result: dict[str, Any] | None = None
    item_id: str | None = None


@dataclass
class DialtErrorFrame(ErrorFrame):
    """A sanitized Dialt error retaining stable machine-readable metadata."""

    code: str | None = None
    retryable: bool | None = None
