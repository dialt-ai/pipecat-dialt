from __future__ import annotations

from typing import Any

import pytest
from dialt import DialtMode
from pipecat.processors.frame_processor import FrameDirection

from pipecat_dialt import DialtLLMService


class FakeSession:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.client_events: list[tuple[str, dict[str, Any]]] = []
        self.tool_results: list[tuple[str, Any, str]] = []
        self.tools: list[list[Any]] = []
        self.injections: list[tuple[str, str, bool]] = []
        self.replies: list[str] = []
        self.interrupts: list[str] = []
        self.conversation: dict[str, Any] = {"items": [], "first_index": 0}
        self.closed = 0

    async def send_audio(self, audio: bytes) -> None:
        self.audio.append(audio)

    async def request_reply(self, *, operation_id: str) -> dict[str, Any]:
        self.replies.append(operation_id)
        return {"accepted": True, "status": "queued", "operation_id": operation_id}

    async def interrupt(self, *, response_id: str) -> dict[str, Any]:
        self.interrupts.append(response_id)
        return {"accepted": True}

    async def playback_stopped(self, response_id: str, discarded_ms: float) -> None:
        await self.send_client_event(
            "playback_stopped", response_id=response_id, discarded_ms=discarded_ms
        )

    async def send_client_event(self, event: str, **fields: Any) -> None:
        self.client_events.append((event, fields))

    async def send_tool_result(
        self, call_id: str, content: Any, *, outcome: str = "unknown", verified: bool = False
    ) -> None:
        self.tool_results.append((call_id, content, outcome))

    async def set_tools(self, tools: list[Any]) -> None:
        self.tools.append(tools)

    async def set_instructions(self, instructions: str, *, new_speaker: bool = False) -> None:
        return None

    async def set_voice(self, voice: str) -> None:
        return None

    async def set_tool_choice(self, choice: Any) -> None:
        return None

    async def inject_context(
        self, text: str, *, role: str = "context", reply: bool = False
    ) -> dict[str, Any]:
        self.injections.append((text, role, reply))
        return {"accepted": True}

    async def finish_input_turn(self) -> None:
        return None

    async def close(self) -> None:
        self.closed += 1


class CapturingService(DialtLLMService):
    def __init__(self, *, turn_detection: str = "server") -> None:
        super().__init__(
            api_key="test",
            mode=DialtMode(
                instructions="Be a concise test assistant.",
                greeting=False,
                turn_detection=turn_detection,
            ),
        )
        self.frames: list[Any] = []
        self.directions: list[FrameDirection] = []
        self.broadcasts: list[type[Any]] = []
        self.interruptions = 0

    async def push_frame(
        self,
        frame: Any,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        self.frames.append(frame)
        self.directions.append(direction)

    async def broadcast_frame(self, frame_type: type[Any], **kwargs: Any) -> None:
        self.broadcasts.append(frame_type)

    async def broadcast_interruption(self) -> None:
        self.interruptions += 1

    async def start_processing_metrics(self) -> None:
        return None

    async def start_ttfb_metrics(self) -> None:
        return None

    async def stop_ttfb_metrics(self) -> None:
        return None

    async def stop_all_metrics(self) -> None:
        return None


@pytest.fixture
def service() -> CapturingService:
    instance = CapturingService()
    instance._session = FakeSession()
    return instance
