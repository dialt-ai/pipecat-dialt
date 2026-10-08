"""Pipecat context and tool conversion for Dialt realtime sessions."""

from __future__ import annotations

import copy
import json
from typing import Any, TypedDict, cast

from dialt import ToolDefinition
from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage


class DialtLLMInvocationParams(TypedDict):
    """Dialt values derived from a universal Pipecat context."""

    system_instruction: str | None
    messages: list[dict[str, str]]
    tools: list[ToolDefinition]


class DialtLLMAdapter(BaseLLMAdapter[DialtLLMInvocationParams]):
    """Convert Pipecat contexts and standard tools into Dialt SDK values."""

    @property
    def id_for_llm_specific_messages(self) -> str:
        """Return the provider identifier for opaque context messages."""
        return "dialt-realtime"

    def get_llm_invocation_params(
        self,
        context: LLMContext,
        **kwargs: Any,
    ) -> DialtLLMInvocationParams:
        """Convert a context without mutating its shared messages or tools."""
        system_instruction = cast(str | None, kwargs.get("system_instruction"))
        source = [
            cast(dict[str, Any], copy.deepcopy(message))
            for message in self.get_messages(context)
            if not isinstance(message, LLMSpecificMessage)
            and async_tool_messages.parse_message(message) is None
        ]
        instructions = [system_instruction] if system_instruction else []
        instructions.extend(
            self._text_content(message.get("content"))
            for message in source
            if message.get("role") in {"system", "developer"}
        )
        effective_instruction = "\n\n".join(text for text in instructions if text) or None
        messages: list[dict[str, str]] = []
        for message in source:
            role = str(message.get("role", "context"))
            if role in {"system", "developer"}:
                continue
            content = self._text_content(message.get("content"))
            if content and role in {"user", "assistant", "context"}:
                messages.append({"role": role, "content": content})

        converted = self.from_standard_tools(context.tools)
        tools = cast(list[ToolDefinition], converted) if isinstance(converted, list) else []
        return {
            "system_instruction": effective_instruction,
            "messages": messages,
            "tools": tools,
        }

    def get_messages_for_logging(self, context: LLMContext) -> list[dict[str, Any]]:
        """Return a redacted, provider-filtered context representation."""
        return [
            cast(dict[str, Any], message)
            for message in self.get_messages(context, truncate_large_values=True)
            if not isinstance(message, LLMSpecificMessage)
        ]

    def to_provider_tools_format(self, tools_schema: ToolsSchema) -> list[ToolDefinition]:
        """Convert standard Pipecat function schemas to Dialt definitions."""
        return [self._function_to_tool(function) for function in tools_schema.standard_tools]

    @staticmethod
    def _function_to_tool(function: FunctionSchema) -> ToolDefinition:
        tool: ToolDefinition = {
            "name": function.name,
            "description": function.description,
            "parameters": {
                "type": "object",
                "properties": copy.deepcopy(function.properties),
                "required": list(function.required),
            },
        }
        return tool

    @staticmethod
    def initial_history(context: LLMContext) -> list[dict[str, Any]]:
        """Convert completed text/dialogue and tool evidence; never flatten tool identity.

        The SDK atomically validates pairing, JSON and byte/item limits before connecting.
        Unsupported media and pending work fail rather than silently losing evidence.
        """
        items: list[dict[str, Any]] = []
        completed_async = {
            payload.tool_call_id
            for source in context.get_messages()
            if not isinstance(source, LLMSpecificMessage)
            and (payload := async_tool_messages.parse_message(source)) is not None
            and payload.kind == "final"
        }
        for index, source_message in enumerate(context.get_messages()):
            if isinstance(source_message, LLMSpecificMessage):
                raise ValueError(
                    "Dialt initial history does not support provider-specific messages"
                )
            message = cast(dict[str, Any], source_message)
            base = {"item_id": f"pipecat-history-{index}", "revision": 1}
            payload = async_tool_messages.parse_message(source_message)
            if payload is not None:
                if payload.tool_call_id not in completed_async:
                    raise ValueError("Dialt initial history does not support pending tools")
                if payload.kind == "final":
                    items.append(
                        {
                            **base,
                            "kind": "tool_result",
                            "call_id": payload.tool_call_id,
                            "content": payload.result,
                            "outcome": "unknown",
                        }
                    )
                continue
            role = message.get("role")
            if role in {"system", "developer"}:
                continue
            content = message.get("content")
            if isinstance(content, list) and any(
                not isinstance(part, dict) or part.get("type") != "text" for part in content
            ):
                raise ValueError("Dialt initial history does not support media")
            if role in {"user", "assistant"}:
                text = DialtLLMAdapter._text_content(content)
                if text:
                    items.append({**base, "kind": "message", "role": role, "text": text})
                    if "interrupted" in message:
                        items[-1]["interrupted"] = message["interrupted"]
                for call in message.get("tool_calls", []):
                    function = call["function"]
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    items.append(
                        {
                            **base,
                            "item_id": f"{base['item_id']}-call-{call['id']}",
                            "kind": "tool_call",
                            "call_id": call["id"],
                            "name": function["name"],
                            "arguments": arguments,
                        }
                    )
                if not text and not message.get("tool_calls"):
                    raise ValueError("Dialt initial history requires completed text messages")
            elif role == "tool":
                if content == "IN_PROGRESS":
                    raise ValueError("Dialt initial history does not support pending tools")
                items.append(
                    {
                        **base,
                        "kind": "tool_result",
                        "call_id": message["tool_call_id"],
                        "content": copy.deepcopy(content),
                        "outcome": message.get("outcome", "unknown"),
                    }
                )
            else:
                raise ValueError(f"Unsupported Dialt initial history role: {role}")
        return items

    @staticmethod
    def _text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
        return ""
