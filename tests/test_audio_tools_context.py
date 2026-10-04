from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np
import pytest
from dialt import SessionEvent
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import InputAudioRawFrame, LLMSetToolsFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.types import NOT_GIVEN

from .conftest import CapturingService, FakeSession


@pytest.mark.asyncio
async def test_stereo_pcm_is_downmixed_before_sdk_send(service: CapturingService) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    stereo = np.array([[1000, -1000], [3000, 1000]], dtype="<i2")

    await service._send_audio(
        InputAudioRawFrame(stereo.tobytes(), sample_rate=16_000, num_channels=2)
    )

    assert np.frombuffer(session.audio[0], dtype="<i2").tolist() == [0, 2000]


@pytest.mark.asyncio
async def test_invalid_audio_alignment_is_rejected(service: CapturingService) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    await service._send_audio(
        InputAudioRawFrame(b"\x00\x01\x02", sample_rate=16_000, num_channels=1)
    )
    assert session.audio == []
    assert service.frames[-1].code == "invalid_audio_frame"


@pytest.mark.asyncio
async def test_tool_call_is_dispatched_exactly_once_by_id(service: CapturingService) -> None:
    calls: list[Any] = []

    async def capture(self: CapturingService, function_calls: Any) -> None:
        calls.extend(function_calls)

    service.run_function_calls = MethodType(capture, service)
    tool_event = SessionEvent(
        type="tool_call",
        t_ms=0,
        data={"id": "call-1", "name": "lookup", "args": {"key": "x"}},
    )
    await service._handle_event(tool_event)
    await service._handle_event(tool_event)

    assert len(calls) == 1
    assert calls[0].tool_call_id == "call-1"
    assert calls[0].arguments == {"key": "x"}


@pytest.mark.asyncio
async def test_not_given_tool_update_clears_manifest_and_auto_registered_handler(
    service: CapturingService,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)

    async def lookup(params: Any) -> None:
        await params.result_callback({"found": True})

    schema = FunctionSchema(
        name="lookup",
        description="Look up a value.",
        properties={"key": {"type": "string"}},
        required=["key"],
        handler=lookup,
    )
    await service.process_frame(
        LLMSetToolsFrame(ToolsSchema(standard_tools=[schema])), FrameDirection.DOWNSTREAM
    )
    assert service.has_function("lookup")

    await service.process_frame(LLMSetToolsFrame(NOT_GIVEN), FrameDirection.DOWNSTREAM)

    assert session.tools == [
        [
            {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            }
        ],
        [],
    ]
    assert not service.has_function("lookup")


@pytest.mark.asyncio
async def test_tool_result_keeps_encoded_string_and_is_sent_once(
    service: CapturingService,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    context = LLMContext()
    await service._handle_context(context)
    encoded = '{"status":"ok"}'
    context.add_message({"role": "tool", "tool_call_id": "call-1", "content": encoded})
    await service._handle_context(context)
    await service._handle_context(context)

    assert session.tool_results == [("call-1", encoded, "unknown")]


@pytest.mark.asyncio
async def test_live_tool_result_is_sent_when_it_arrives_with_first_context(
    service: CapturingService,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    service._dispatched_tool_calls.add("call-live")
    context = LLMContext(
        [{"role": "tool", "tool_call_id": "call-live", "content": '{"receipt":"ok"}'}]
    )

    await service._handle_context(context)

    assert session.tool_results == [("call-live", '{"receipt":"ok"}', "unknown")]


@pytest.mark.asyncio
async def test_resumed_tool_reannouncement_replays_result_without_execution(
    service: CapturingService,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    calls: list[Any] = []

    async def capture(self: CapturingService, function_calls: Any) -> None:
        calls.extend(function_calls)

    service.run_function_calls = MethodType(capture, service)
    tool_event = SessionEvent(
        type="tool_call",
        t_ms=0,
        data={"id": "call-resume", "name": "lookup", "args": {"key": "x"}},
    )
    await service._handle_event(tool_event)
    await service._send_tool_result("call-resume", '{"receipt":"ok"}')
    await service._handle_event(tool_event)

    assert len(calls) == 1
    assert session.tool_results == [
        ("call-resume", '{"receipt":"ok"}', "unknown"),
        ("call-resume", '{"receipt":"ok"}', "unknown"),
    ]


@pytest.mark.asyncio
async def test_provider_turn_marker_survives_intervening_non_user_context(
    service: CapturingService,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    context = LLMContext()
    await service._handle_context(context)
    service._provider_user_turns_pending = 1

    context.add_message({"role": "assistant", "content": "working"})
    context.add_message({"role": "tool", "tool_call_id": "c", "content": "done"})
    await service._handle_context(context)
    assert service._provider_user_turns_pending == 1

    context.add_message({"role": "user", "content": "provider transcript"})
    await service._handle_context(context)
    assert service._provider_user_turns_pending == 0
    assert session.injections == []

    context.add_message({"role": "user", "content": "typed follow-up"})
    await service._handle_context(context)
    assert session.injections == [("typed follow-up", "user", True)]


@pytest.mark.asyncio
async def test_tool_cancel_correlates_id_and_emits_terminal_outcome(
    service: CapturingService,
) -> None:
    session = service._session
    assert isinstance(session, FakeSession)
    cancelled: list[str] = []

    async def capture_cancel(self: CapturingService, tool_call_id: str) -> None:
        cancelled.append(tool_call_id)

    service._cancel_function_calls_by_tool_call_id = MethodType(capture_cancel, service)
    event = SessionEvent(type="tool_cancel", t_ms=0, data={"id": "call-9"})
    await service._handle_event(event)
    await service._handle_event(event)

    assert cancelled == ["call-9"]
    assert session.tool_results == [("call-9", None, "cancelled")]
