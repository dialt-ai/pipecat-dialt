"""Positive integration checks using Pipecat 1.12's real pipeline and aggregators."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from dialt import DialtMode, DialtSession, SessionEvent
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    DataFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_dialt import (
    DialtContextAggregatorPair,
    DialtLLMAdapter,
    DialtLLMService,
    DialtSpeechActivityFrame,
    DialtTranscriptCorrectionFrame,
)
from pipecat_dialt.frames import DialtInterruptionFrame

from .conftest import FakeSession


@dataclass
class ProviderEvent(DataFrame):
    event: SessionEvent


@dataclass
class AssertContext(DataFrame):
    messages: list[dict[str, Any]]


def event(kind: str, **data: Any) -> ProviderEvent:
    return ProviderEvent(SessionEvent(type=kind, t_ms=0, data=data))


def audio(response_id: str, turn_id: str) -> ProviderEvent:
    return ProviderEvent(
        SessionEvent(
            type="audio",
            t_ms=0,
            data={"response_id": response_id, "turn_id": turn_id},
            audio=np.zeros(1600, dtype=np.float32),
        )
    )


def item(item_id: str, role: str, text: str, revision: int = 1) -> dict[str, Any]:
    return {"item_id": item_id, "revision": revision, "kind": "message", "role": role, "text": text}


class ProbeService(DialtLLMService):
    def __init__(self, context: LLMContext | None, startup_run: bool) -> None:
        super().__init__(
            api_key="test",
            initial_context=context,
            startup_run=startup_run,
            mode=DialtMode(instructions="Help.", greeting="Hello"),
        )
        self.fake = FakeSession()
        self.connections: list[tuple[Any, Any]] = []
        self.emitted: list[tuple[Any, FrameDirection]] = []
        self.context_checks: list[tuple[Any, Any]] = []

    async def _connect(self) -> None:
        if self._session is not None:
            return
        history = (
            DialtLLMAdapter.initial_history(self._initial_context) if self._initial_context else []
        )
        self.connections.append((self._effective_mode(), history))
        self.fake.conversation = {"items": copy.deepcopy(history), "first_index": 0}
        self._session = self.fake

    async def push_frame(
        self, frame: Any, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        self.emitted.append((frame, direction))
        await super().push_frame(frame, direction)

    async def process_frame(self, frame: Any, direction: FrameDirection) -> None:
        if isinstance(frame, AssertContext):
            actual = self._context.get_messages() if self._context is not None else None
            self.context_checks.append((copy.deepcopy(actual), frame.messages))
        elif isinstance(frame, ProviderEvent):
            ev = frame.event
            if ev.type == "conversation_snapshot":
                self.fake.conversation = copy.deepcopy(ev.data)
            elif ev.type == "conversation_item":
                # Simulate the public SDK's detached, revision-filtered projection.
                values = self.fake.conversation["items"]
                offset = ev.data["index"]
                replacement = ev.data["item"]
                if offset == len(values):
                    values.append(copy.deepcopy(replacement))
                elif replacement["revision"] > values[offset]["revision"]:
                    values[offset] = copy.deepcopy(replacement)
            await self._handle_event(ev)
        else:
            await super().process_frame(frame, direction)


async def run(
    frames: list[Any],
    messages: list[Any] | None = None,
    *,
    startup_run: bool = True,
    eager_context: bool = False,
    framework_run: bool = True,
) -> tuple[Any, Any, Any]:
    context = LLMContext(messages or [])
    pair = DialtContextAggregatorPair(context)
    service = ProbeService(context if eager_context else None, startup_run)
    await run_test(
        Pipeline([pair.user(), service, pair.assistant()]),
        frames_to_send=[
            *([LLMRunFrame()] if framework_run else []),
            SleepFrame(),
            *frames,
            SleepFrame(),
        ],
    )
    assert len(service.context_checks) == sum(isinstance(f, AssertContext) for f in frames)
    for actual, expected in service.context_checks:
        assert actual == expected
    return context, service, pair


async def test_final_transcript_upstream_once_without_echo() -> None:
    ctx, svc, pair = await run(
        [
            event("asr", turn_id="u1", text="Thursday"),
            event("conversation_item", first_index=0, index=0, item=item("u1", "user", "Thursday")),
            event("turn", response_id="r1", turn_id="s1"),
            event("done", response_id="r1", turn_id="s1"),
            LLMRunFrame(),
        ]
    )
    assert ctx.get_messages() == [{"role": "user", "content": "Thursday"}]
    transcripts = [(f.text, d) for f, d in svc.emitted if isinstance(f, TranscriptionFrame)]
    assert transcripts == [("Thursday", FrameDirection.UPSTREAM)]
    assert pair.user().aggregation_string() == ""
    assert svc.fake.injections == []
    assert len(svc.fake.replies) == 1


async def test_initial_history_composed_once_and_repeated_run_deduplicated() -> None:
    messages = [
        {"role": "system", "content": "Framework system."},
        {"role": "developer", "content": "Framework developer."},
        {"role": "user", "content": "Find two orders"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c-red",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"order":"red","count":2}'},
                },
                {
                    "id": "c-blue",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"order":"blue","count":7}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c-blue", "content": {"amount": 91}},
        {"role": "tool", "tool_call_id": "c-red", "content": {"amount": 13}},
        {"role": "assistant", "content": "Red 13, blue 91"},
    ]
    original = copy.deepcopy(messages)
    ctx, svc, _ = await run([LLMRunFrame(), LLMRunFrame()], messages)
    mode, history = svc.connections[0]
    assert mode.greeting is False
    assert mode.instructions == "Help.\n\nFramework system.\n\nFramework developer."
    assert [
        (v["kind"], v.get("call_id"), v.get("arguments"), v.get("content")) for v in history
    ] == [
        ("message", None, None, None),
        ("tool_call", "c-red", {"order": "red", "count": 2}, None),
        ("tool_call", "c-blue", {"order": "blue", "count": 7}, None),
        ("tool_result", "c-blue", None, {"amount": 91}),
        ("tool_result", "c-red", None, {"amount": 13}),
        ("message", None, None, None),
    ]
    assert len({v["item_id"] for v in history}) == 6
    assert all(v["revision"] == 1 for v in history)
    assert svc.fake.tool_results == []
    assert svc.fake.injections == []
    assert len(svc.connections) == len(svc.fake.replies) == 1
    assert ctx.get_messages() == original


@pytest.mark.parametrize("eager_context", [False, True])
async def test_greeting_only_default_connects_without_microphone_deadlock(
    eager_context: bool,
) -> None:
    _, svc, _ = await run(
        [
            InputAudioRawFrame(b"\x00\x00" * 160, sample_rate=16000, num_channels=1),
        ],
        startup_run=False,
        eager_context=eager_context,
        framework_run=False,
    )
    assert svc.connections[0][0].greeting == "Hello"
    assert svc.fake.replies == []
    assert len(svc.fake.audio) == 1


async def test_local_interrupt_suppresses_late_output_and_does_not_touch_new_response() -> None:
    _, svc, _ = await run(
        [
            event("turn", response_id="old", turn_id="s1"),
            audio("old", "s1"),
            SleepFrame(),
            InterruptionFrame(),
            SleepFrame(),
            audio("old", "s1"),
            event("turn", response_id="new", turn_id="s2"),
            event("interrupted", response_id="old", turn_id="s1"),
            SleepFrame(),
            DialtInterruptionFrame(response_id="old"),
            audio("new", "s2"),
            event("done", response_id="new", turn_id="s2"),
        ]
    )
    assert svc.fake.interrupts == ["old"]
    assert [f.context_id for f, _ in svc.emitted if isinstance(f, TTSAudioRawFrame)] == ["s1", "s2"]
    assert [fields["response_id"] for _, fields in svc.fake.client_events] == ["old"]


async def test_provider_interrupt_echo_never_calls_sdk_and_activity_is_independent() -> None:
    _, svc, _ = await run(
        [
            event("turn", response_id="r", turn_id="s"),
            audio("r", "s"),
            event("speech_started", speech_id="speech-1"),
            SleepFrame(),
            event("speech_stopped", speech_id="speech-1"),
            audio("r", "s"),
            event("interrupted", response_id="r", turn_id="s"),
            DialtInterruptionFrame(response_id="r"),
            audio("r", "s"),
        ]
    )
    assert svc.fake.interrupts == []
    assert len([f for f, _ in svc.emitted if isinstance(f, TTSAudioRawFrame)]) == 2
    activity = [
        f.speaking
        for f, d in svc.emitted
        if isinstance(f, DialtSpeechActivityFrame) and d == FrameDirection.UPSTREAM
    ]
    assert activity == [True, False]
    assert len(svc.fake.client_events) == 1


@pytest.mark.parametrize("logical_done_first", [False, True])
async def test_segment_done_tool_wait_and_post_done_playback(logical_done_first: bool) -> None:
    fences = [
        event("done", response_id="r", turn_id="final"),
        event("response_done", response_id="r", status="completed"),
    ]
    if logical_done_first:
        fences.reverse()
    _, svc, _ = await run(
        [
            event("turn", response_id="r", turn_id="bridge"),
            audio("r", "bridge"),
            event("done", response_id="r", turn_id="bridge"),
            event("turn", response_id="r", turn_id="final"),
            audio("r", "final"),
            *fences,
            SleepFrame(),
            InterruptionFrame(),
        ]
    )
    assert len([f for f, _ in svc.emitted if isinstance(f, LLMFullResponseEndFrame)]) == 2
    assert svc.fake.interrupts == []  # Generation finished; host discard is separate.
    assert len(svc.fake.client_events) == 1
    assert svc.fake.client_events[0][1]["response_id"] == "r"


async def test_interrupt_during_tool_wait_targets_logical_response() -> None:
    _, svc, _ = await run(
        [
            event("turn", response_id="r", turn_id="bridge"),
            audio("r", "bridge"),
            event("done", response_id="r", turn_id="bridge"),
            SleepFrame(),
            InterruptionFrame(),
            SleepFrame(),
            event("turn", response_id="r", turn_id="final"),
            audio("r", "final"),
        ]
    )
    assert svc.fake.interrupts == ["r"]
    assert len([f for f, _ in svc.emitted if isinstance(f, TTSAudioRawFrame)]) == 1


@pytest.mark.parametrize("before_done", [False, True])
async def test_user_and_assistant_revisions_replace_and_empty_retract(before_done: bool) -> None:
    updates = [
        event("conversation_item", first_index=0, index=0, item=item("u", "user", "Tuesday", 2)),
        event(
            "conversation_item",
            first_index=0,
            index=1,
            item=item("a", "assistant", "Tuesday booked", 2),
        ),
    ]
    done = event("done", response_id="r", turn_id="s")
    ctx, svc, pair = await run(
        [
            event("asr", turn_id="u", text="Thursday"),
            event("conversation_item", first_index=0, index=0, item=item("u", "user", "Thursday")),
            event("turn", response_id="r", turn_id="s"),
            event("text_delta", response_id="r", turn_id="s", delta="Thursday booked"),
            event(
                "conversation_item",
                first_index=0,
                index=1,
                item=item("a", "assistant", "Thursday booked"),
            ),
            *([*updates, done] if before_done else [done, *updates]),
            SleepFrame(),
            AssertContext(
                [
                    {"role": "user", "content": "Tuesday"},
                    {"role": "assistant", "content": "Tuesday booked"},
                ]
            ),
            event("conversation_item", first_index=0, index=0, item=item("u", "user", "", 3)),
            event("conversation_item", first_index=0, index=1, item=item("a", "assistant", "", 3)),
            LLMRunFrame(),
        ]
    )
    assert ctx.get_messages() == []
    assert pair.user().aggregation_string() == pair.assistant().aggregation_string() == ""
    corrections = [f for f, _ in svc.emitted if isinstance(f, DialtTranscriptCorrectionFrame)]
    assert [(f.item_id, f.revision, f.text) for f in corrections] == [
        ("u", 2, "Tuesday"),
        ("a", 2, "Tuesday booked"),
        ("u", 3, ""),
        ("a", 3, ""),
    ]
    assert svc.fake.injections == []
    assert len(svc.fake.replies) == 1


async def test_projection_replaces_in_place_and_snapshot_reconciles_window() -> None:
    context = LLMContext([{"role": "system", "content": "Host instruction"}])
    pair = DialtContextAggregatorPair(context)
    from pipecat_dialt.frames import DialtConversationFrame

    await pair.user().process_frame(
        DialtConversationFrame(
            [item("u", "user", "Thursday"), item("a", "assistant", "Booked")], 0
        ),
        FrameDirection.UPSTREAM,
    )
    reference = context.get_messages()[1]
    await pair.user().process_frame(
        DialtConversationFrame(
            [item("u", "user", "Tuesday", 2), item("a", "assistant", "Booked")], 0
        ),
        FrameDirection.UPSTREAM,
    )
    assert context.get_messages()[1] is reference
    assert reference["content"] == "Tuesday"
    await pair.user().process_frame(
        DialtConversationFrame([item("a", "assistant", "", 2)], 1), FrameDirection.UPSTREAM
    )
    assert context.get_messages() == [{"role": "system", "content": "Host instruction"}]


@pytest.mark.parametrize("outcome", ["failed", "unknown"])
async def test_tool_outcome_and_business_content_survive_projection_history_round_trip(
    outcome: str,
) -> None:
    from pipecat_dialt.frames import DialtConversationFrame

    context = LLMContext()
    pair = DialtContextAggregatorPair(context)
    content = {"status": "customer-status", "outcome": "business-outcome", "amount": 37}
    items = [
        {
            "item_id": "call-item",
            "revision": 1,
            "kind": "tool_call",
            "call_id": "customer-call",
            "name": "lookup",
            "arguments": {"order": 7},
        },
        {
            "item_id": "result-item",
            "revision": 1,
            "kind": "tool_result",
            "call_id": "customer-call",
            "content": content,
            "outcome": outcome,
        },
    ]
    await pair.user().process_frame(DialtConversationFrame(items, 0), FrameDirection.UPSTREAM)
    assert context.get_messages()[1] == {
        "role": "tool",
        "tool_call_id": "customer-call",
        "content": content,
        "outcome": outcome,
    }
    history = DialtLLMAdapter.initial_history(context)
    assert history[0]["arguments"] == {"order": 7}
    assert history[1]["call_id"] == "customer-call"
    assert history[1]["outcome"] == outcome
    assert history[1]["content"] == content


async def test_connect_passes_top_level_history_not_mode_or_injection(monkeypatch: Any) -> None:
    captured: list[dict[str, Any]] = []
    fake = FakeSession()

    async def connect(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        return fake

    monkeypatch.setattr(DialtSession, "connect", connect)
    context = LLMContext([{"role": "user", "content": "Prior question"}])
    service = DialtLLMService(
        api_key="test",
        mode=DialtMode(instructions="Help", greeting="Hello"),
        initial_context=context,
        startup_run=True,
    )
    # Avoid a long-lived consumer in this boundary test; pipeline tests own lifecycle.
    monkeypatch.setattr(service, "create_task", lambda coro: coro.close())
    await service._connect()
    await service._handle_context(context)
    await service._handle_context(context)
    assert captured[0]["initial_history"] == [item("pipecat-history-0", "user", "Prior question")]
    assert captured[0]["mode"].greeting is False
    assert len(fake.replies) == 1
    assert fake.injections == []


@pytest.mark.parametrize(
    "messages,match",
    [
        (
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
            "media",
        ),
        ([{"role": "tool", "tool_call_id": "c", "content": "IN_PROGRESS"}], "pending"),
    ],
)
def test_unsupported_initial_history_fails_instead_of_truncating(messages: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        DialtLLMAdapter.initial_history(LLMContext(messages))


async def test_import_snapshot_replaces_initial_entries_without_duplicates() -> None:
    ctx, svc, _ = await run(
        [
            event(
                "conversation_snapshot",
                first_index=0,
                items=[
                    item("pipecat-history-1", "user", "Old question"),
                    item("pipecat-history-2", "assistant", "Old answer"),
                ],
            ),
            SleepFrame(),
            AssertContext(
                [
                    {"role": "system", "content": "Host instruction"},
                    {"role": "user", "content": "Old question"},
                    {"role": "assistant", "content": "Old answer"},
                ]
            ),
            event(
                "conversation_snapshot",
                first_index=1,
                items=[
                    item("pipecat-history-2", "assistant", "Corrected answer", 2),
                ],
            ),
        ],
        [
            {"role": "system", "content": "Host instruction"},
            {"role": "user", "content": "Old question"},
            {"role": "assistant", "content": "Old answer"},
        ],
    )
    assert ctx.get_messages() == [
        {"role": "system", "content": "Host instruction"},
        {"role": "assistant", "content": "Corrected answer"},
    ]
    assert svc.fake.injections == []


@pytest.mark.parametrize("with_turn", [False, True])
async def test_real_tool_survives_response_interrupt_and_preserves_business_result(
    with_turn: bool,
) -> None:
    context = LLMContext(
        tools=ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="lookup",
                    description="Look up an order",
                    properties={},
                    required=[],
                )
            ]
        )
    )
    pair = DialtContextAggregatorPair(context)
    service = ProbeService(context, False)
    started = asyncio.Event()
    calls: list[str] = []

    async def lookup(params: Any) -> None:
        calls.append(params.tool_call_id)
        started.set()
        await asyncio.sleep(0.4)
        await params.result_callback({"status": "business-status", "receipt": {"amount": 37}})

    service.register_function("lookup", lookup)
    await run_test(
        Pipeline([pair.user(), service, pair.assistant()]),
        frames_to_send=[
            LLMRunFrame(),
            SleepFrame(),
            *([event("turn", response_id="r", turn_id="s")] if with_turn else []),
            event("tool_call", response_id="r", id="customer-call", name="lookup", args={}),
            SleepFrame(0.2),
            InterruptionFrame(),
            SleepFrame(0.6),
        ],
    )
    assert started.is_set()
    assert calls == ["customer-call"]
    assert service.fake.interrupts == ["r"]
    assert service.fake.tool_results == [
        ("customer-call", '{"status": "business-status", "receipt": {"amount": 37}}', "unknown")
    ]


@pytest.mark.parametrize(
    "history,match",
    [
        ([item(str(i), "user", "x") for i in range(257)], "256"),
        ([item("oversized", "user", "é" * 8192)], "16 KiB"),
        ([item(str(i), "user", "x" * 16000) for i in range(9)], "128 KiB"),
        (
            [
                {
                    "item_id": "c",
                    "revision": 1,
                    "kind": "tool_call",
                    "call_id": "c",
                    "name": "lookup",
                    "arguments": {"order": 7},
                }
            ],
            "unresolved",
        ),
    ],
)
async def test_actual_sdk_rejects_atomic_history_before_connect(history: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        await DialtSession.connect(
            "ws://127.0.0.1:1",
            api_key="test",
            session_id="test",
            mode=DialtMode(instructions="Help", greeting=False),
            initial_history=history,
        )


async def test_immediately_buffered_snapshot_and_greeting_survive_pipeline_start(
    monkeypatch: Any,
) -> None:
    from collections.abc import AsyncIterator

    class BufferedSession(FakeSession):
        async def events(self) -> AsyncIterator[SessionEvent]:
            self.conversation = {
                "first_index": 0,
                "items": [item("pipecat-history-0", "user", "Prior question")],
            }
            yield SessionEvent(type="conversation_snapshot", t_ms=0, data=self.conversation)
            yield event("turn", response_id="greeting", turn_id="opener").event
            yield audio("greeting", "opener").event
            yield event("done", response_id="greeting", turn_id="opener").event

    session = BufferedSession()

    async def connect(*args: Any, **kwargs: Any) -> Any:
        return session

    monkeypatch.setattr(DialtSession, "connect", connect)
    context = LLMContext([{"role": "user", "content": "Prior question"}])
    pair = DialtContextAggregatorPair(context)
    service = DialtLLMService(
        api_key="test",
        initial_context=context,
        mode=DialtMode(instructions="Help", greeting="Hello"),
    )
    down, _ = await run_test(
        Pipeline([pair.user(), service, pair.assistant()]), frames_to_send=[SleepFrame(0.4)]
    )
    assert len([f for f in down if isinstance(f, TTSAudioRawFrame)]) == 1
    assert context.get_messages() == [{"role": "user", "content": "Prior question"}]
    assert session.replies == []
    assert session.closed == 1


async def test_started_receipt_is_interruptible_before_first_turn() -> None:
    _, svc, _ = await run(
        [
            event(
                "reply_ack",
                operation_id="host-run",
                response_id="pre-audio",
                accepted=True,
                status="started",
            ),
            SleepFrame(),
            InterruptionFrame(),
            SleepFrame(),
            event("turn", response_id="pre-audio", turn_id="late"),
            audio("pre-audio", "late"),
        ]
    )
    assert svc.fake.interrupts == ["pre-audio"]
    assert not any(isinstance(f, TTSAudioRawFrame) for f, _ in svc.emitted)
