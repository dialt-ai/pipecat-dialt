"""Pipecat context and tool conversion for Dialt realtime sessions."""

from __future__ import annotations

import copy
from typing import Any, TypedDict, cast

from dialt import ToolDefinition
from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
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
        ]
        context_instruction = self._extract_initial_system(
            source, system_instruction=system_instruction
        )
        effective_instruction = self._resolve_system_instruction(
            context_instruction,
            system_instruction,
            discard_context_system=True,
        )
        messages: list[dict[str, str]] = []
        for message in source:
            role = str(message.get("role", "context"))
            if role in {"system", "developer"}:
                role = "context"
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
