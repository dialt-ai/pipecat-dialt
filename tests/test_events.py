from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from dialt import SessionEvent
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)

from pipecat_dialt import DialtTranscriptCorrectionFrame
from pipecat_dialt.frames import DialtInterruptionFrame

from .conftest import CapturingService, FakeSession


def event(event_type: str, **data: object) -> SessionEvent:
    if "turn_id" in data:
        data.setdefault("response_id", f"response-{data['turn_id']}")
    return SessionEvent(type=event_type, t_ms=0, data=dict(data))


@pytest.mark.asyncio
async def test_response_frames_are_ordered_and_bracketed_once(service: CapturingService) -> None:
    await service._handle_event(event("turn", turn_id="r1"))
    await service._handle_event(event("text_delta", turn_id="r1", delta="Hello "))
    await service._handle_event(
        SessionEvent(
            type="audio",
            t_ms=1,
            data={"response_id": "response-r1"},
            audio=np.full(160, 0.25, dtype=np.float32),
        )
    )
    await service._handle_event(event("utterance", turn_id="r1", text="Hello there"))
    await service._handle_event(event("done", turn_id="r1"))
    await service._handle_event(event("done", turn_id="r1"))

    types = [type(frame) for frame in service.frames]
    assert types == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TTSTextFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSStoppedFrame,
        LLMFullResponseEndFrame,
    ]
    audio = next(frame for frame in service.frames if isinstance(frame, TTSAudioRawFrame))
    assert audio.sample_rate == 16_000
    assert audio.num_channels == 1
    assert len(audio.audio) == 320


@pytest.mark.asyncio
async def test_late_events_cannot_reopen_closed_generation(service: CapturingService) -> None:
    await service._handle_event(event("turn", turn_id="old"))
    await service._handle_event(event("reconnecting", attempt=1))
    await service._handle_event(event("turn", turn_id="old"))
    await service._handle_event(event("text_delta", turn_id="old", delta="stale"))

    assert sum(isinstance(frame, LLMFullResponseStartFrame) for frame in service.frames) == 1
    assert sum(isinstance(frame, LLMFullResponseEndFrame) for frame in service.frames) == 1
    assert not any(isinstance(frame, LLMTextFrame) for frame in service.frames)
    assert service.broadcasts == [DialtInterruptionFrame]


@pytest.mark.asyncio
async def test_asr_boundary_order_and_correction_do_not_duplicate_turn(
    service: CapturingService,
) -> None:
    await service._handle_event(event("asr", turn_id="u1", text="book Thursday"))
    await service._handle_event(
        event("asr_correction", turn_id="u1", text="book Tuesday", revision=2)
    )

    assert service.broadcasts == [
        ProposedUserStartedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    assert sum(isinstance(frame, TranscriptionFrame) for frame in service.frames) == 1
    assert not any(isinstance(frame, DialtTranscriptCorrectionFrame) for frame in service.frames)


@pytest.mark.asyncio
async def test_interruption_reports_unplayed_audio_and_closes_output(
    service: CapturingService,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    await service._handle_event(event("turn", turn_id="r1"))
    await service._handle_event(
        SessionEvent(
            type="audio",
            t_ms=1,
            data={"response_id": "response-r1"},
            audio=np.zeros(16_000, dtype=np.float32),
        )
    )
    assert service._response is not None
    service._response.playback_started_at = time.perf_counter() - 0.25

    await service._handle_event(event("interrupted", turn_id="r1", barge_seq=9))

    assert service.broadcasts == [DialtInterruptionFrame]
    assert session.client_events[0][0] == "playback_stopped"
    assert session.client_events[0][1]["response_id"] == "response-r1"
    assert 700 <= session.client_events[0][1]["discarded_ms"] <= 800
    assert isinstance(service.frames[-2], TTSStoppedFrame)
    assert isinstance(service.frames[-1], LLMFullResponseEndFrame)


@pytest.mark.asyncio
async def test_machine_error_is_sanitized_and_keeps_stable_fields(
    service: CapturingService,
) -> None:
    await service._handle_event(
        event(
            "error",
            code="capacity",
            retryable=True,
            detail="private-provider/private-model unavailable",
        )
    )
    frame = service.frames[-1]
    assert frame.code == "capacity"
    assert frame.retryable is True
    assert "private-provider" not in frame.error
    assert "private-model" not in frame.error


@pytest.mark.asyncio
async def test_disconnect_is_idempotent_and_clears_session_state(
    service: CapturingService,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    service._closed_turns.add("old-turn")
    service._dispatched_tool_calls.add("call-1")
    service._completed_tool_calls.add("call-1")
    service._terminal_tool_results["call-1"] = ("result", "unknown")

    await service._disconnect()
    await service._disconnect()

    assert session.closed == 1
    assert service._session is None
    assert service._closed_turns == set()
    assert service._dispatched_tool_calls == set()
    assert service._completed_tool_calls == set()
    assert service._terminal_tool_results == {}


async def test_startup_receipt_tracks_latest_lifecycle_without_claiming_playback_drain(
    service: CapturingService,
) -> None:
    for status in ("queued", "started", "completed"):
        await service._handle_event(
            event(
                "reply_ack",
                operation_id=service._startup_operation_id,
                response_id="logical",
                accepted=True,
                status=status,
            )
        )
        assert service.startup_reply_ack["status"] == status
    detached = service.startup_reply_ack
    detached["status"] = "modified"
    assert service.startup_reply_ack["status"] == "completed"
    assert service._active_response_id == "logical"
    await service._handle_event(event("response_done", response_id="logical", status="completed"))
    assert service._active_response_id is None


async def test_canonical_observer_never_replays_stale_or_duplicate_revisions(
    service: CapturingService,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    item = {"item_id": "u", "kind": "message", "role": "user", "text": "Tuesday", "revision": 2}
    session.conversation = {"items": [item], "first_index": 0}
    old = {**item, "text": "Thursday", "revision": 1}
    await service._handle_event(event("conversation_item", index=0, first_index=0, item=old))
    await service._handle_event(event("conversation_item", index=0, first_index=0, item=item))
    corrections = [f for f in service.frames if isinstance(f, DialtTranscriptCorrectionFrame)]
    assert [(f.item_id, f.revision, f.text) for f in corrections] == [("u", 2, "Tuesday")]


@pytest.mark.parametrize("wait_ms,expected_discard_ms", [(10_000, 400), (150, 1250)])
async def test_final_discard_rebases_after_tool_wait_and_retains_bridge_residual(
    service: CapturingService,
    monkeypatch: Any,
    wait_ms: int,
    expected_discard_ms: int,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    now = [100.0]
    monkeypatch.setattr("pipecat_dialt.service.time.perf_counter", lambda: now[0])
    await service._handle_event(event("turn", response_id="logical", turn_id="bridge"))
    await service._handle_event(
        SessionEvent(
            type="audio",
            t_ms=0,
            data={"response_id": "logical", "turn_id": "bridge"},
            audio=np.zeros(16_000, dtype=np.float32),  # 1000 ms bridge
        )
    )
    await service._handle_event(event("done", response_id="logical", turn_id="bridge"))
    now[0] += wait_ms / 1000
    await service._handle_event(event("turn", response_id="logical", turn_id="final"))
    await service._handle_event(
        SessionEvent(
            type="audio",
            t_ms=wait_ms,
            data={"response_id": "logical", "turn_id": "final"},
            audio=np.zeros(6400, dtype=np.float32),  # 400 ms newly queued final audio
        )
    )
    await service._interrupt_local()
    assert session.client_events == [
        (
            "playback_stopped",
            {
                "response_id": "logical",
                "discarded_ms": expected_discard_ms,
            },
        )
    ]
    assert session.interrupts == ["logical"]
