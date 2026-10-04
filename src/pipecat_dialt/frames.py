"""Dialt-specific frames for information Pipecat cannot represent losslessly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pipecat.frames.frames import DataFrame, ErrorFrame


@dataclass
class DialtTranscriptCorrectionFrame(DataFrame):
    """Replace a previously projected transcript without creating a new turn.

    Pipecat 1.x has no standard transcript-revision frame. Consumers should
    replace the transcript identified by ``turn_id`` with ``text``.
    """

    turn_id: str
    text: str
    speaker: Literal["user", "assistant"]
    revision: int | None = None
    result: dict[str, Any] | None = None


@dataclass
class DialtErrorFrame(ErrorFrame):
    """A sanitized Dialt error retaining stable machine-readable metadata."""

    code: str | None = None
    retryable: bool | None = None
