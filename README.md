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
from pipecat_dialt import DialtContextAggregatorPair, DialtLLMService

context = LLMContext()
llm = DialtLLMService(
    api_key=os.environ["DIALT_API_KEY"],
    initial_context=context,
    mode=DialtMode(
        instructions="You are a concise, helpful voice assistant.",
        greeting="Hello! How can I help?",
    ),
)

user, assistant = DialtContextAggregatorPair(context)
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
owned pair configures external turn strategies without constructing a default Smart Turn analyzer.
Its semantic turn proposals do not authorize interruption: Dialt makes that decision separately.
Use `DialtContextAggregatorPair`, not the stock pair, to keep the shared context consistent with
accepted transcript revisions and avoid duplicate legacy/canonical appends.

## Startup and initial history

There are two explicit startup paths:

- **Greeting-only (default):** pass `initial_context=context` and optionally `mode.greeting`.
  `StartFrame` connects immediately after forwarding startup downstream; loading history is
  silent and no `LLMRunFrame` is needed. Early SDK output cannot race pre-start aggregators.
  This is the foundational microphone example and cannot wait on a transcript to connect.
- **Framework startup run:** set `startup_run=True` and queue `LLMRunFrame()` after pipeline
  start (for example in `on_pipeline_started`). The user aggregator supplies the initial
  `LLMContextFrame`; the service disables automatic greeting before connecting and requests
  one response after SDK readiness, using a stable operation ID. Repeated run/context frames
  do not request a second startup response. `reply_ack` is admission/lifecycle, not speaker drain.

With `startup_run=True`, `initial_context` is optional. If omitted, connection waits for the first
context frame because Pipecat's `StartFrame` precedes it. Queue the startup run explicitly; never
wait for microphone speech to produce that first context. Audio arriving before it is not sent.
Supplying `initial_context` allows early connection with history while still requesting the startup
response only when the framework runs it.

The initial context imports user/assistant text and completed tool exchanges, correlated by call
ID (not tool name or arrival order). Explicit service instructions come first, then framework
system/developer messages in order, composed once. They are not history or synthetic caller text.
Historical tools never execute. Pending tools, media and unsupported provider-specific messages
fail explicitly. The SDK atomically enforces 256 items, 128 KiB total and 16 KiB per item before
readiness or generation; the integration never truncates history.

Later context frames synchronize tools and return results for dispatched application calls only.
They never reimport the shared transcript or request responses. For a deliberately new typed caller
turn, use the public SDK `llm.session.inject_context(text, role="user", reply=True)`; for a new
host-requested reply without inventing a caller turn, use `request_reply(operation_id=...)`.

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

Pass the published SDK's `DialtMode` to the service. `instructions` is required.

```python
mode = DialtMode(
    instructions="Help customers with orders. Keep spoken answers brief.",
    voice="<public Dialt roster key>",
    greeting=False,
    brain="fast",
    turn_detection="server",
)
llm = DialtLLMService(api_key=api_key, mode=mode, initial_context=context)
```

Constructor-only connection options are `base_url`, `user`, `timezone`, `session_id`,
`connect_timeout_s`, `auto_reconnect`, `reconnect_base_s`, `reconnect_max_s`, and
`max_reconnect_attempts`, `initial_context`, and `startup_run`.
Runtime `DialtLLMSettings` updates support `system_instruction`, `voice`,
`tools`, and `tool_choice`; the service delegates each change to the corresponding SDK method.

`turn_detection="client"` is available for applications that deliberately provide their own turn
boundaries. In that mode the service does not advertise external strategies and commits input on
`UserStoppedSpeakingFrame`. The recommended and example configuration is server mode.

## Frames and observers

The service projects provider activity into standard frames:

- upstream `TranscriptionFrame` for final user ASR;
- `DialtSpeechActivityFrame(speaking, speech_id)` promptly on detected speech start/stop; this
  observer indication is not a Pipecat speaking proposal and cannot trigger local barge-in;
- `ProposedUserStartedSpeakingFrame` / `ProposedUserStoppedSpeakingFrame` for provider-confirmed
  semantic turns around final ASR (not speech-duration measurements), with interruptions disabled;
- one `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame` pair per segment `turn_id`;
- `LLMTextFrame`, `TTSTextFrame`, and 16 kHz mono `TTSAudioRawFrame` output;
- standard Pipecat function-call lifecycle frames from `LLMService`.

Pipecat 1.x has no standard correction or structured provider-error frame. This package exposes:

- `DialtTranscriptCorrectionFrame`: canonical observer revision with `item_id`, `revision`, role
  and text; replace the matching display item, including empty-text retraction, rather than append;
- `DialtErrorFrame`: a sanitized message plus stable `code` and `retryable` fields.

## Interruption and playback accounting

An incoming Pipecat `InterruptionFrame` immediately closes local output and suppresses late events,
then calls SDK `interrupt(response_id=...)` for the stable logical response. Provider interruption
broadcasts a marked standard-interruption subclass to clear transport output; delayed echoes never
interrupt a newer response. Response cancellation does not cancel application tools: only Dialt's
separate `tool_cancel` event does so by call ID.

Segment `done` does not end the logical target during tool waits. `response_done` ends generation
only; it may arrive before final segment `done` and never drops that segment's delivered content.
Playback tracking survives both until transport drain or a host discard. Host discard calls public
SDK `playback_stopped(response_id, discarded_ms)`, including after producer completion.

Pipecat 1.12 does not expose a generic device playback cursor or clear acknowledgement to upstream
processors. Like Pipecat's built-in realtime providers, this integration estimates heard audio as
the smaller of generated PCM duration and elapsed wall-clock playback time. At the first audio
chunk of each new segment, it carries forward only the estimated unplayed residual and resets the
clock, so a bridge/tool wait cannot consume newly queued final audio. This remains an estimate:
transport buffering, gaps within a segment and device latency are not measured. A transport with a
real device/client playback cursor should expose that receipt to replace this approximation.

## Accepted conversation and revisions

`session.conversation` is a detached read-only `{items, first_index}` SDK projection. The owned pair
maps accepted item IDs to shared-context messages in order, updates revisions in place for both
roles, and removes empty retractions. It does not buffer provisional Dialt text as accepted history,
so a late aggregation flush cannot restore a corrected transcript. Resume snapshots reconcile the
retained window; `first_index` makes eviction explicit. Observers can also inspect upstream
conversation frames. Host system/developer instructions remain separate; internal prompts, hidden
reasoning and managed tools are never imported from Dialt. Customer tool result business fields
remain intact. This projection does not edit broker history.

## Compatibility

| pipecat-dialt | Python | Pipecat tested | dialt-sdk tested |
| --- | --- | --- | --- |
| 0.1.x | 3.11–3.13 | 1.12.0 | 0.41.0 |
| 0.2.x | 3.11–3.13 | 1.12.0 | 0.45.0 |

Dependencies intentionally use bounded ranges (`pipecat-ai>=1.12,<2`,
`dialt-sdk>=0.45.0,<0.46.0`). Version 0.2 requires the coordinated broker capabilities
`conversation_v1`, `request_reply_v1`, `targeted_interrupt_v1`, and `speech_activity_v1`;
unsupported servers fail explicitly rather than silently falling back. Every release must update
this table with the exact versions exercised.

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
