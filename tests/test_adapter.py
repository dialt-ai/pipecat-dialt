from __future__ import annotations

import copy

import pytest
from dialt import DialtMode
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext

from pipecat_dialt import DialtLLMAdapter, DialtLLMService, DialtLLMSettings


def test_adapter_converts_tools_without_mutating_context() -> None:
    function = FunctionSchema(
        name="lookup",
        description="Look up a value.",
        properties={"key": {"type": "string"}},
        required=["key"],
    )
    context = LLMContext(
        messages=[
            {"role": "system", "content": "Context instruction"},
            {"role": "user", "content": "Find it"},
        ],
        tools=ToolsSchema(standard_tools=[function]),
    )
    original = copy.deepcopy(context.get_messages())

    params = DialtLLMAdapter().get_llm_invocation_params(
        context, system_instruction="Service instruction"
    )

    assert params["system_instruction"] == "Service instruction\n\nContext instruction"
    assert params["messages"] == [{"role": "user", "content": "Find it"}]
    assert params["tools"] == [
        {
            "name": "lookup",
            "description": "Look up a value.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        }
    ]
    assert context.get_messages() == original


def test_private_serving_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="serving model"):
        DialtLLMService(
            api_key="test",
            mode=DialtMode(instructions="Test", greeting=False),
            settings=DialtLLMSettings(model="private-backend"),
        )


def test_server_turn_detection_advertises_external_strategies() -> None:
    service = DialtLLMService(
        api_key="test",
        mode=DialtMode(instructions="Test", greeting=False, turn_detection="server"),
    )
    metadata = service.service_metadata_frame()
    assert metadata.is_realtime_service is True
    assert metadata.user_turn_strategies is not None


def test_client_turn_detection_does_not_advertise_external_strategies() -> None:
    service = DialtLLMService(
        api_key="test",
        mode=DialtMode(instructions="Test", greeting=False, turn_detection="client"),
    )
    assert service.service_metadata_frame().user_turn_strategies is None


def test_completed_async_tool_is_history_evidence_not_developer_instructions() -> None:
    result = '{"status":"business-status","amount":37}'
    context = LLMContext(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"order":7}'},
                    }
                ],
            },
            async_tool_messages.build_started_message("c"),
            async_tool_messages.build_final_result_message("c", result),
        ]
    )
    invocation = DialtLLMAdapter().get_llm_invocation_params(context, system_instruction="Help")
    assert invocation["system_instruction"] == "Help"
    history = DialtLLMAdapter.initial_history(context)
    assert [(i["kind"], i["call_id"]) for i in history] == [
        ("tool_call", "c"),
        ("tool_result", "c"),
    ]
    assert history[0]["arguments"] == {"order": 7}
    assert history[1]["content"] == result
    assert history[1]["outcome"] == "unknown"
