# pipecat-dialt

`pipecat-dialt` is the author-maintained Dialt realtime voice provider for
[Pipecat](https://github.com/pipecat-ai/pipecat). One `DialtLLMService` handles the complete
speech-to-speech path: ASR, turn detection, endpointing, response generation, tool orchestration,
interruption, and TTS.

This is a community integration maintained by Dialt, not by the Pipecat maintainers. It follows
Pipecat's external integration policy and is independently versioned under Apache-2.0.

> **Release status:** source-complete alpha. This repository has not yet been published to PyPI or
> listed in the official Pipecat documentation.

## Install

Requires Python 3.11 or newer.

```bash
uv add pipecat-dialt
```

Until the first public release, install from a checkout:

```bash
uv add --editable /path/to/pipecat-dialt
```

For the local microphone/speaker example, also install Pipecat's local audio dependency:

```bash
uv sync --extra local
```

Set `DIALT_API_KEY` in the environment. Do not put it in source code.

## Foundational pipeline

Run [`examples/foundational.py`](examples/foundational.py):

```bash
uv run python examples/foundational.py
```

The important topology is:

```python
llm = DialtLLMService(
    api_key=os.environ["DIALT_API_KEY"],
    mode=DialtMode(
        instructions="You are a concise, helpful voice assistant.",
        greeting="Hello! How can I help?",
    ),
)

context = LLMContext()
user, assistant = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        user_turn_strategies=ExternalUserTurnStrategies(),
    ),
    realtime_service_mode=True,
)
pipeline = Pipeline(
    [
        transport.input(),
        user,
        llm,  # Dialt owns ASR + turns + LLM + tools + TTS
        transport.output(),
        assistant,
    ]
)
```

There is intentionally no Pipecat VAD, Smart Turn analyzer, STT, second LLM, or TTS service. The
example configures `ExternalUserTurnStrategies` explicitly so the aggregator never constructs a
default Smart Turn analyzer. `DialtLLMService` also advertises those strategies when
`DialtMode.turn_detection == "server"`; the universal user aggregator resolves Dialt's proposed
turn frames and clears transport output on interruption.

## Application tools

Supply Pipecat tools through `LLMContext`. Handler-carrying schemas and direct async functions are
auto-registered by Pipecat:

```python
from pipecat.services.llm_service import FunctionCallParams


async def lookup_order(params: FunctionCallParams, order_id: str):
    """Look up an order by ID."""
    await params.result_callback({"order_id": order_id, "status": "shipped"})


context = LLMContext(tools=[lookup_order])
```

Dialt dispatches the call, Pipecat executes the explicitly supplied application handler, and the
service returns one correlated terminal result through `dialt-sdk`. Duplicate call IDs and repeated
context frames do not execute or return a tool twice. Intermediate streaming tool results are not
supported by Pipecat realtime services; the service emits a machine-readable error and waits for
the final result.

## Ownership boundary

| Dialt / `dialt-sdk` owns | Pipecat / this integration owns |
| --- | --- |
| Protocol and authentication | Input/output media transport |
| Logical session lifecycle | PCM downmixing/resampling to 16 kHz mono |
| Reconnect, token rotation, and resume | Standard Pipecat frame projection |
| ASR and transcript revisions | Output-buffer interruption propagation |
| Turn detection and endpointing | Observers and context aggregation |
| Response and tool orchestration | Explicit Pipecat application tool execution |
| Interruption decisions and TTS | Best available playback discard accounting |

The integration never opens its own Dialt WebSocket and does not duplicate SDK retry or resume
logic. It also rejects `settings.model`: private serving provider/model identities are not part of
Dialt's public contract. Use the public `DialtMode.brain` setting (`"fast"` or `"smart"`) when a
different conversational tier is required.

## Configuration

Pass the published SDK's `DialtMode` to the service. `instructions` is required by
`dialt-sdk>=0.41.0`.

```python
mode = DialtMode(
    instructions="Help customers with orders. Keep spoken answers brief.",
    voice="<public Dialt roster key>",
    greeting=False,
    brain="fast",
    turn_detection="server",
)
llm = DialtLLMService(api_key=api_key, mode=mode)
```

Constructor-only connection options are `base_url`, `user`, `timezone`, `session_id`,
`connect_timeout_s`, `auto_reconnect`, `reconnect_base_s`, `reconnect_max_s`, and
`max_reconnect_attempts`. Runtime `DialtLLMSettings` updates support `system_instruction`, `voice`,
`tools`, and `tool_choice`; the service delegates each change to the corresponding SDK method.

`turn_detection="client"` is available for applications that deliberately provide their own turn
boundaries. In that mode the service does not advertise external strategies and commits input on
`UserStoppedSpeakingFrame`. The recommended and example configuration is server mode.

## Frames and observers

The service projects provider activity into standard frames:

- `TranscriptionFrame` for final user ASR;
- `ProposedUserStartedSpeakingFrame` / `ProposedUserStoppedSpeakingFrame` for server turns;
- one `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame` pair per Dialt `turn_id`;
- `LLMTextFrame`, `TTSTextFrame`, and 16 kHz mono `TTSAudioRawFrame` output;
- standard Pipecat function-call lifecycle frames from `LLMService`.

Pipecat 1.x has no standard correction or structured provider-error frame. This package exposes:

- `DialtTranscriptCorrectionFrame`: replace the transcript identified by `turn_id`; do not append
  a second turn;
- `DialtErrorFrame`: a sanitized message plus stable `code` and `retryable` fields.

## Interruption and playback accounting

On a Dialt `interrupted` event, the external turn strategy broadcasts Pipecat's interruption so
the output transport drops queued audio. The service reports `playback_stopped` through the SDK,
including `barge_seq` and estimated discarded milliseconds.

Pipecat 1.12 does not expose a generic device playback cursor or clear acknowledgement to upstream
processors. Like Pipecat's built-in realtime providers, this integration estimates heard audio as
the smaller of generated PCM duration and elapsed wall-clock playback time. A transport with a real
device/client playback cursor should eventually expose that receipt to replace this approximation.

## Compatibility

| pipecat-dialt | Python | Pipecat tested | dialt-sdk tested |
| --- | --- | --- | --- |
| 0.1.x | 3.11–3.13 | 1.12.0 | 0.41.0 |

Dependencies intentionally use bounded ranges (`pipecat-ai>=1.12,<2`,
`dialt-sdk>=0.41,<0.42`). Every release must update this table with the exact versions exercised.

## Development

```bash
uv sync --all-groups
uv lock --check
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
uv run twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [RELEASING.md](RELEASING.md), and
[CHANGELOG.md](CHANGELOG.md). Security reports should follow [SECURITY.md](SECURITY.md).
