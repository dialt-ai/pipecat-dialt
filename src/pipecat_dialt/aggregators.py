"""Pipecat aggregators with a single accepted, identity-preserving Dialt projection."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from typing import Any, cast

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

from .frames import DialtConversationFrame


class _DialtUserAggregator(LLMUserAggregator):
    def __init__(self, context: LLMContext) -> None:
        super().__init__(
            context,
            params=LLMUserAggregatorParams(
                user_turn_strategies=ExternalUserTurnStrategies(enable_interruptions=False),
            ),
            _realtime_service_mode=True,
        )
        # Initial dialogue is replaced by its accepted import, not appended a second time.
        self._owned_messages: list[Any] = [
            message
            for message in context.get_messages()
            if isinstance(message, dict)
            and (
                message.get("role") in {"user", "assistant", "tool"}
                or async_tool_messages.parse_message(message) is not None
            )
        ]
        self.first_index = 0
        self._item_messages: dict[str, dict[str, Any]] = {}

    async def _handle_transcription(self, frame: TranscriptionFrame) -> None:
        if frame.user_id == "dialt-user":
            # ASR is an observer/turn signal. Only accepted conversation items own text.
            return
        await super()._handle_transcription(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, DialtConversationFrame):
            self._apply_projection(frame)
            await self.push_frame(frame, direction)
        else:
            await super().process_frame(frame, direction)

    def _apply_projection(self, frame: DialtConversationFrame) -> None:
        messages = self._context.get_messages()
        owned_ids = {id(message) for message in self._owned_messages}
        call_ids = {item["call_id"] for item in frame.items if "call_id" in item}
        # Keep host instructions and host-only messages. Native tool frames may already
        # have written a result; canonical evidence replaces those entries by call ID.
        retained = [
            message
            for message in messages
            if id(message) not in owned_ids
            and not (
                isinstance(message, dict)
                and (
                    message.get("tool_call_id") in call_ids
                    or (
                        (payload := async_tool_messages.parse_message(message)) is not None
                        and payload.tool_call_id in call_ids
                    )
                    or any(
                        call.get("id") in call_ids
                        for call in cast(Any, message).get("tool_calls", [])
                    )
                )
            )
        ]
        projected: list[dict[str, Any]] = []
        for item in frame.items:
            if item["kind"] == "message":
                if item["text"]:
                    projected.append({"role": item["role"], "content": item["text"]})
            elif item["kind"] == "tool_call":
                projected.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": item["call_id"],
                                "type": "function",
                                "function": {
                                    "name": item["name"],
                                    "arguments": json.dumps(item["arguments"]),
                                },
                            }
                        ],
                    }
                )
            elif item["kind"] == "tool_result":
                projected.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": copy.deepcopy(item["content"]),
                        "outcome": item["outcome"],
                    }
                )
            else:
                continue
            if item["kind"] == "message" and not item["text"]:
                continue
            replacement = projected[-1]
            existing = self._item_messages.get(item["item_id"])
            if existing is not None:
                existing.clear()
                existing.update(replacement)
                projected[-1] = existing
            self._item_messages[item["item_id"]] = projected[-1]
        retained_item_ids = {item["item_id"] for item in frame.items}
        self._item_messages = {
            key: value for key, value in self._item_messages.items() if key in retained_item_ids
        }
        self._owned_messages = projected
        self.first_index = frame.first_index
        messages[:] = cast(Any, [*retained, *projected])


class DialtContextAggregatorPair:
    """Real Pipecat aggregators, with accepted Dialt items as the transcript authority.

    Use this pair instead of ``LLMContextAggregatorPair``. Streaming assistant text
    and legacy ASR remain visible to observers but never create a competing buffered
    transcript. Revisions (including empty retractions) atomically replace the
    accepted projection at its original position; snapshots also reconcile resume gaps.
    """

    def __init__(self, context: LLMContext) -> None:
        self._user = _DialtUserAggregator(context)
        self._assistant = LLMAssistantAggregator(
            context,
            params=LLMAssistantAggregatorParams(),
            _realtime_service_mode=True,
            _paired_user_aggregator=self._user,
        )

    def user(self) -> LLMUserAggregator:
        """Return the upstream user aggregator."""
        return self._user

    def assistant(self) -> LLMAssistantAggregator:
        """Return the downstream assistant/tool aggregator."""
        return self._assistant

    def __iter__(self) -> Iterator[LLMUserAggregator | LLMAssistantAggregator]:
        return iter((self._user, self._assistant))
