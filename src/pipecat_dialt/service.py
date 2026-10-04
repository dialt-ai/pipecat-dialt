"""Dialt realtime voice service for Pipecat."""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from dialt import (
    DEFAULT_REALTIME_URL,
    OUTPUT_SR,
    DialtError,
    DialtMode,
    DialtSession,
    SessionEvent,
    ToolDefinition,
    float32_to_pcm16,
)
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMSetToolChoiceFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings, ServiceSettings
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.text.base_text_aggregator import AggregationType
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given

from .adapter import DialtLLMAdapter
from .frames import DialtErrorFrame, DialtTranscriptCorrectionFrame


@dataclass
class DialtLLMSettings(LLMSettings):
    """Runtime-updatable Dialt settings.

    ``model`` is deliberately unsupported: Dialt's private serving identity is
    not part of the public integration contract.
    """

    voice: str | NotGiven | None = field(default_factory=lambda: NOT_GIVEN)
    tools: ToolsSchema | list[ToolDefinition] | list[dict[str, Any]] | NotGiven | None = field(
        default_factory=lambda: NOT_GIVEN
    )
    tool_choice: str | dict[str, Any] | NotGiven | None = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class _ResponseState:
    turn_id: str
    text: str = ""
    audio_started: bool = False
    generated_audio_ms: int = 0
    playback_started_at: float | None = None
    barge_seq: int | None = None


class DialtLLMService(LLMService[DialtLLMAdapter]):
    """One full-duplex Dialt ASR/turn/LLM/tool/TTS service.

    The published ``dialt-sdk`` owns the wire protocol, authentication,
    session state, reconnect and resume. This service owns only Pipecat media
    conversion, standard frame projection and explicitly supplied Pipecat tool
    execution.
    """

    Settings = DialtLLMSettings
    adapter_class = DialtLLMAdapter
    _settings: DialtLLMSettings

    def __init__(
        self,
        *,
        api_key: str,
        mode: DialtMode,
        settings: DialtLLMSettings | None = None,
        base_url: str = DEFAULT_REALTIME_URL,
        user: str | None = None,
        timezone: str | None = None,
        session_id: str | None = None,
        connect_timeout_s: float = 15.0,
        auto_reconnect: bool = True,
        reconnect_base_s: float = 0.5,
        reconnect_max_s: float = 5.0,
        max_reconnect_attempts: int = 12,
        **kwargs: Any,
    ) -> None:
        """Create a Dialt service using a complete SDK ``DialtMode``."""
        defaults = DialtLLMSettings(
            model=None,
            system_instruction=mode.instructions,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            voice=mode.voice,
            tools=mode.tools,
            tool_choice=mode.tool_choice,
        )
        if settings is not None:
            defaults.apply_update(settings)
        if defaults.model is not None:
            raise ValueError("Dialt does not expose or accept a serving model identifier")
        instruction = assert_given(defaults.system_instruction)
        if not instruction:
            raise ValueError("Dialt system_instruction must be non-empty")

        super().__init__(settings=defaults, **kwargs)
        self._api_key = api_key
        self._base_mode = mode
        self._base_url = base_url
        self._user = user
        self._timezone = timezone
        self._session_id = session_id or f"pipecat-{uuid.uuid4()}"
        self._connect_timeout_s = connect_timeout_s
        self._auto_reconnect = auto_reconnect
        self._reconnect_base_s = reconnect_base_s
        self._reconnect_max_s = reconnect_max_s
        self._max_reconnect_attempts = max_reconnect_attempts

        self._session: DialtSession | None = None
        self._event_task: asyncio.Task[Any] | None = None
        self._disconnecting = False
        self._context: LLMContext | None = None
        self._last_context_message_count = 0
        self._provider_user_turns_pending = 0
        self._input_speech_active = False
        self._response: _ResponseState | None = None
        self._closed_turns: set[str] = set()
        self._dispatched_tool_calls: set[str] = set()
        self._completed_tool_calls: set[str] = set()
        self._terminal_tool_results: dict[str, tuple[Any, str]] = {}
        self._resampler: Any | None = None
        self._resampler_input_rate: int | None = None

    @property
    def session(self) -> DialtSession | None:
        """Return the active SDK session for diagnostics and advanced controls."""
        return self._session

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Advertise Dialt's server-driven turn boundaries when enabled."""
        server_turns = self._base_mode.turn_detection == "server"
        self._warn_if_realtime_service_emits_no_turn_frames(server_turns)
        return LLMServiceMetadataFrame(
            service_name=self.name,
            is_realtime_service=True,
            user_turn_strategies=ExternalUserTurnStrategies() if server_turns else None,
        )

    def _effective_mode(self) -> DialtMode:
        tools = assert_given(self._settings.tools)
        provider_tools = self._convert_tools(tools)
        return dataclasses.replace(
            self._base_mode,
            instructions=cast(str, assert_given(self._settings.system_instruction)),
            voice=assert_given(self._settings.voice),
            tools=provider_tools,
            tool_choice=assert_given(self._settings.tool_choice),
        )

    def _convert_tools(
        self,
        tools: ToolsSchema | list[ToolDefinition] | list[dict[str, Any]] | None,
    ) -> list[ToolDefinition] | None:
        if isinstance(tools, ToolsSchema):
            return self.get_llm_adapter().to_provider_tools_format(tools)
        return cast(list[ToolDefinition] | None, tools)

    def _service_tools(self) -> ToolsSchema | list[Any] | None:
        return assert_given(self._settings.tools)

    async def setup(self, setup: FrameProcessorSetup) -> None:  # type: ignore[override]
        """Set up Pipecat state and open the SDK-owned session."""
        await super().setup(setup)
        await self._connect()

    async def cleanup(self) -> None:
        """Release the SDK session and all local tasks idempotently."""
        await super().cleanup()  # type: ignore[no-untyped-call]
        await self._disconnect()

    async def stop(self, frame: EndFrame) -> None:
        """Stop the live session on an orderly pipeline end."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        """Stop the live session on pipeline cancellation."""
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self) -> None:
        if self._session is not None:
            return
        try:
            self._session = await DialtSession.connect(
                self._base_url,
                api_key=self._api_key,
                session_id=self._session_id,
                sr=OUTPUT_SR,
                mode=self._effective_mode(),
                user=self._user,
                timezone=self._timezone,
                connect_timeout_s=self._connect_timeout_s,
                auto_reconnect=self._auto_reconnect,
                reconnect_base_s=self._reconnect_base_s,
                reconnect_max_s=self._reconnect_max_s,
                max_reconnect_attempts=self._max_reconnect_attempts,
            )
            self._event_task = self.create_task(self._consume_events(self._session))
        except DialtError as error:
            await self._push_dialt_error(error.code, error.retryable)

    async def _disconnect(self) -> None:
        if self._disconnecting:
            return
        self._disconnecting = True
        session, self._session = self._session, None
        task, self._event_task = self._event_task, None
        try:
            if session is not None:
                await session.close()
            if task is not None and task is not asyncio.current_task():
                await self.cancel_task(task, timeout=1.0)
            await self.stop_all_metrics()  # type: ignore[no-untyped-call]
            if self._resampler is not None:
                await self._resampler.reset()
        finally:
            self._response = None
            self._context = None
            self._last_context_message_count = 0
            self._provider_user_turns_pending = 0
            self._input_speech_active = False
            self._closed_turns.clear()
            self._dispatched_tool_calls.clear()
            self._completed_tool_calls.clear()
            self._terminal_tool_results.clear()
            self._disconnecting = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Send transport media and application controls into the Dialt session."""
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._send_audio(frame)
        elif isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, LLMSetToolsFrame):
            self._sync_registered_tool_handlers(frame.tools)
            tools = frame.tools if is_given(frame.tools) else []
            await self._set_tools(cast(ToolsSchema | list[Any], tools))
        elif isinstance(frame, LLMSetToolChoiceFrame):
            await self._set_tool_choice(frame.tool_choice)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._base_mode.turn_detection == "client" and self._session is not None:
                await self._session.finish_input_turn()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._response is not None:
                self._response.playback_started_at = None

        await self.push_frame(frame, direction)

    async def _send_audio(self, frame: InputAudioRawFrame) -> None:
        session = self._session
        if session is None:
            return
        if len(frame.audio) % (2 * frame.num_channels):
            await self._push_dialt_error("invalid_audio_frame", False)
            return
        pcm = np.frombuffer(frame.audio, dtype="<i2")
        if frame.num_channels > 1:
            pcm = (
                pcm.reshape(-1, frame.num_channels)
                .astype(np.int32)
                .mean(axis=1)
                .round()
                .astype(np.int16)
            )
        mono = pcm.astype("<i2", copy=False).tobytes()
        if frame.sample_rate != OUTPUT_SR:
            if self._resampler_input_rate != frame.sample_rate:
                if self._resampler is not None:
                    await self._resampler.reset()
                self._resampler = create_stream_resampler(clear_after_secs=None)
                self._resampler_input_rate = frame.sample_rate
            resampler = self._resampler
            assert resampler is not None
            mono = await resampler.resample(mono, frame.sample_rate, OUTPUT_SR)
        if mono:
            await session.send_audio(mono)

    async def _consume_events(self, session: DialtSession) -> None:
        try:
            async for event in session.events():
                if session is not self._session:
                    return
                await self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._disconnecting:
                await self._push_dialt_error("event_consumer_failed", True)

    async def _handle_event(self, event: SessionEvent) -> None:
        handlers = {
            "turn": self._handle_turn,
            "text_delta": self._handle_text_delta,
            "audio": self._handle_audio,
            "utterance": self._handle_utterance,
            "done": self._handle_done,
            "interrupted": self._handle_interrupted,
            "canceled": self._handle_canceled,
            "asr": self._handle_asr,
            "asr_correction": self._handle_asr_correction,
            "tool_call": self._handle_tool_call,
            "tool_cancel": self._handle_tool_cancel,
            "reconnecting": self._handle_reconnecting,
            "resume_failed": self._handle_terminal_error,
            "error": self._handle_error,
        }
        handler = handlers.get(event.type)
        if handler is not None:
            await handler(event)

    async def _handle_turn(self, event: SessionEvent) -> None:
        turn_id = self._turn_id(event.data)
        if not turn_id or turn_id in self._closed_turns:
            return
        if self._response is not None:
            await self._close_response(interrupted=True, clear_output=True)
        self._response = _ResponseState(turn_id=turn_id)
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        await self.push_frame(LLMFullResponseStartFrame())

    async def _handle_text_delta(self, event: SessionEvent) -> None:
        response = self._response_for(event.data)
        delta = event.data.get("delta")
        if response is None or not isinstance(delta, str) or not delta:
            return
        response.text += delta
        await self.stop_ttfb_metrics()
        await self._push_text(delta)

    async def _handle_audio(self, event: SessionEvent) -> None:
        response = self._response
        if response is None or event.audio is None or not event.audio.size:
            return
        if not response.audio_started:
            response.audio_started = True
            response.playback_started_at = time.perf_counter()
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSStartedFrame(context_id=response.turn_id))
        pcm = float32_to_pcm16(event.audio)
        response.generated_audio_ms += round(len(pcm) / 2 * 1000 / OUTPUT_SR)
        await self.push_frame(
            TTSAudioRawFrame(
                pcm,
                sample_rate=OUTPUT_SR,
                num_channels=1,
                context_id=response.turn_id,
            )
        )

    async def _handle_utterance(self, event: SessionEvent) -> None:
        response = self._response_for(event.data)
        text = event.data.get("text")
        if response is None or not isinstance(text, str) or not text:
            return
        if not response.text:
            response.text = text
            await self._push_text(text)
        elif event.data.get("corrected") or text != response.text:
            response.text = text
            await self.push_frame(
                DialtTranscriptCorrectionFrame(
                    turn_id=response.turn_id,
                    text=text,
                    speaker="assistant",
                    revision=self._revision(event.data),
                    result=dict(event.data),
                )
            )

    async def _handle_done(self, event: SessionEvent) -> None:
        if self._response_for(event.data) is not None:
            await self._close_response(interrupted=False, clear_output=False)

    async def _handle_interrupted(self, event: SessionEvent) -> None:
        response = self._response_for(event.data)
        if response is None:
            return
        response.barge_seq = self._optional_int(event.data.get("barge_seq"))
        if not self._input_speech_active:
            self._input_speech_active = True
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        await self._report_playback_stopped(response)
        await self._close_response(interrupted=True, clear_output=False)

    async def _handle_canceled(self, event: SessionEvent) -> None:
        if self._response_for(event.data) is not None:
            await self._close_response(interrupted=True, clear_output=True)

    async def _handle_asr(self, event: SessionEvent) -> None:
        text = event.data.get("text")
        if not isinstance(text, str) or not text:
            return
        if not self._input_speech_active:
            self._input_speech_active = True
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        self._provider_user_turns_pending += 1
        await self.push_frame(
            TranscriptionFrame(
                text,
                user_id="dialt-user",
                timestamp=time_now_iso8601(),
                result=dict(event.data),
                finalized=True,
            )
        )
        self._input_speech_active = False
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _handle_asr_correction(self, event: SessionEvent) -> None:
        text = event.data.get("text")
        turn_id = self._turn_id(event.data)
        if isinstance(text, str) and text and turn_id:
            await self.push_frame(
                DialtTranscriptCorrectionFrame(
                    turn_id=turn_id,
                    text=text,
                    speaker="user",
                    revision=self._revision(event.data),
                    result=dict(event.data),
                )
            )

    async def _handle_tool_call(self, event: SessionEvent) -> None:
        call_id = event.data.get("id")
        name = event.data.get("name")
        arguments = event.data.get("args")
        if (
            not isinstance(call_id, str)
            or not isinstance(name, str)
            or not isinstance(arguments, Mapping)
        ):
            await self._push_dialt_error("invalid_tool_call", False)
            return
        cached = self._terminal_tool_results.get(call_id)
        if cached is not None:
            session = self._session
            if session is not None:
                result, outcome = cached
                await session.send_tool_result(call_id, result, outcome=outcome)
            return
        if call_id in self._dispatched_tool_calls:
            return
        self._dispatched_tool_calls.add(call_id)
        context = self._context or LLMContext()
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    context=context,
                    tool_call_id=call_id,
                    function_name=name,
                    arguments=dict(arguments),
                )
            ]
        )

    async def _handle_tool_cancel(self, event: SessionEvent) -> None:
        call_id = event.data.get("id")
        if not isinstance(call_id, str) or call_id in self._completed_tool_calls:
            return
        await self._cancel_function_calls_by_tool_call_id(call_id)
        session = self._session
        if session is not None:
            await session.send_tool_result(call_id, None, outcome="cancelled")
        self._completed_tool_calls.add(call_id)
        self._terminal_tool_results[call_id] = (None, "cancelled")

    async def _handle_reconnecting(self, event: SessionEvent) -> None:
        if self._response is not None:
            await self._close_response(interrupted=True, clear_output=True)

    async def _handle_terminal_error(self, event: SessionEvent) -> None:
        await self._handle_error(event)
        if self._response is not None:
            await self._close_response(interrupted=True, clear_output=True)

    async def _handle_error(self, event: SessionEvent) -> None:
        await self._push_dialt_error(
            cast(str | None, event.data.get("code")),
            cast(bool | None, event.data.get("retryable")),
        )

    async def _close_response(self, *, interrupted: bool, clear_output: bool) -> None:
        response, self._response = self._response, None
        if response is None:
            return
        self._closed_turns.add(response.turn_id)
        if clear_output:
            await self.broadcast_interruption()  # type: ignore[no-untyped-call]
        if response.audio_started:
            await self.push_frame(TTSStoppedFrame(context_id=response.turn_id))
        await self.push_frame(LLMFullResponseEndFrame())
        await self.stop_all_metrics()  # type: ignore[no-untyped-call]

    async def _report_playback_stopped(self, response: _ResponseState) -> None:
        session = self._session
        if session is None:
            return
        elapsed_ms = 0
        if response.playback_started_at is not None:
            elapsed_ms = round((time.perf_counter() - response.playback_started_at) * 1000)
        discarded_ms = max(
            0,
            response.generated_audio_ms - min(elapsed_ms, response.generated_audio_ms),
        )
        fields: dict[str, Any] = {"discarded_ms": discarded_ms}
        if response.barge_seq is not None:
            fields["barge_seq"] = response.barge_seq
        await session.send_client_event("playback_stopped", **fields)

    async def _push_text(self, text: str) -> None:
        llm_frame = LLMTextFrame(text)
        llm_frame.append_to_context = False
        await self.push_frame(llm_frame)
        tts_frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        tts_frame.includes_inter_frame_spaces = True
        await self.push_frame(tts_frame)

    async def _handle_context(self, context: LLMContext) -> None:
        first_context = self._context is None
        previous_count = self._last_context_message_count
        self._context = context
        messages = context.get_messages()
        self._last_context_message_count = len(messages)
        await self._sync_context_configuration(context)
        await self._process_completed_function_calls(send_results=not first_context)
        if first_context:
            return

        new_messages = messages[previous_count:]
        for message in new_messages:
            if isinstance(message, LLMSpecificMessage) or not isinstance(message, Mapping):
                continue
            if message.get("role") != "user":
                continue
            if self._provider_user_turns_pending:
                self._provider_user_turns_pending -= 1
                continue
            text = DialtLLMAdapter._text_content(message.get("content"))
            if text and self._session is not None:
                await self._session.inject_context(text, role="user", reply=True)

    async def _sync_context_configuration(self, context: LLMContext) -> None:
        invocation = self.get_llm_adapter().get_llm_invocation_params(
            context,
            system_instruction=cast(str, assert_given(self._settings.system_instruction)),
        )
        session = self._session
        if session is None:
            return
        instruction = invocation["system_instruction"]
        if instruction and instruction != assert_given(self._settings.system_instruction):
            await session.set_instructions(instruction)
        if is_given(context.tools):
            await session.set_tools(invocation["tools"])

    async def _process_completed_function_calls(self, *, send_results: bool) -> None:
        if self._context is None:
            return
        for message in self._context.get_messages():
            if isinstance(message, LLMSpecificMessage) or not isinstance(message, Mapping):
                continue
            async_payload = async_tool_messages.parse_message(message)
            if async_payload is not None:
                if async_payload.tool_call_id in self._completed_tool_calls:
                    continue
                if async_payload.kind == "intermediate":
                    await self._push_dialt_error("intermediate_tool_result_unsupported", False)
                elif async_payload.kind == "final":
                    if send_results or async_payload.tool_call_id in self._dispatched_tool_calls:
                        await self._send_tool_result(
                            async_payload.tool_call_id, async_payload.result
                        )
                    self._completed_tool_calls.add(async_payload.tool_call_id)
                continue
            if message.get("role") != "tool" or message.get("content") == "IN_PROGRESS":
                continue
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id in self._completed_tool_calls:
                continue
            if send_results or call_id in self._dispatched_tool_calls:
                await self._send_tool_result(call_id, message.get("content"))
            self._completed_tool_calls.add(call_id)

    async def _send_tool_result(self, call_id: str, result: Any) -> None:
        session = self._session
        if session is None:
            return
        await session.send_tool_result(call_id, result, outcome="unknown")
        self._terminal_tool_results[call_id] = (result, "unknown")

    async def _set_tools(self, tools: ToolsSchema | list[Any]) -> None:
        session = self._session
        if session is None:
            return
        if isinstance(tools, ToolsSchema):
            converted = self.get_llm_adapter().to_provider_tools_format(tools)
        else:
            converted = cast(list[ToolDefinition], tools)
        await session.set_tools(converted)

    async def _set_tool_choice(self, choice: str | dict[str, Any]) -> None:
        if self._session is not None:
            await self._session.set_tool_choice(choice)

    async def _update_settings(self, delta: ServiceSettings) -> dict[str, Any]:
        if not isinstance(delta, LLMSettings):
            raise TypeError("Dialt settings updates must use LLMSettings")
        if is_given(delta.model) and delta.model is not None:
            raise ValueError("Dialt does not expose or accept a serving model identifier")
        changed = await super()._update_settings(delta)
        session = self._session
        if session is None:
            return changed
        if "system_instruction" in changed:
            instruction = assert_given(self._settings.system_instruction)
            if not instruction:
                raise ValueError("Dialt system_instruction must be non-empty")
            await session.set_instructions(instruction)
        if "voice" in changed:
            voice = assert_given(self._settings.voice)
            if voice is not None:
                await session.set_voice(voice)
        if "tools" in changed:
            tools = self._convert_tools(assert_given(self._settings.tools)) or []
            await session.set_tools(tools)
        if "tool_choice" in changed:
            choice = assert_given(self._settings.tool_choice)
            if choice is not None:
                await session.set_tool_choice(choice)
        handled = {"system_instruction", "voice", "tools", "tool_choice"}
        self._warn_unhandled_updated_settings(  # type: ignore[no-untyped-call]
            changed.keys() - handled
        )
        return changed

    async def _push_dialt_error(self, code: str | None, retryable: bool | None) -> None:
        safe_code = code if isinstance(code, str) and code else "unknown"
        await self.push_frame(
            DialtErrorFrame(
                error=f"Dialt realtime error ({safe_code})",
                processor=self,
                code=code,
                retryable=retryable,
            )
        )

    def _response_for(self, data: Mapping[str, Any]) -> _ResponseState | None:
        response = self._response
        turn_id = self._turn_id(data)
        if response is None or (turn_id is not None and turn_id != response.turn_id):
            return None
        return response

    @staticmethod
    def _turn_id(data: Mapping[str, Any]) -> str | None:
        value = data.get("turn_id")
        return str(value) if value is not None and str(value) else None

    @staticmethod
    def _revision(data: Mapping[str, Any]) -> int | None:
        return DialtLLMService._optional_int(data.get("revision"))

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None
