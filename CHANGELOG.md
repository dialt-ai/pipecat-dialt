# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - Unreleased

### Fixed

- Project final user transcripts upstream without echoing accepted dialogue back to Dialt.
- Import structured completed dialogue/tool history before readiness; compose application and
  framework instructions once. Explicit framework startup runs disable automatic greeting and
  request exactly one logical reply; the default microphone/greeting path connects immediately.
- Target local interruption by stable response ID, suppress late output, protect newer responses
  from provider echoes, and account for host playback discard after producer completion.
- Rebase playback estimates for each new audio segment, retaining estimated residual without
  counting tool-wait time as playback of future final audio.
- Preserve terminal tool outcomes alongside customer result fields in shared-context projection
  and subsequent structured history imports.
- Separate detected speech activity from semantic turn proposals and interruption decisions.
- Keep user and assistant revisions (including empty retractions and resume snapshots) current in
  shared context through the supported `DialtContextAggregatorPair`, without provisional buffers
  or duplicate legacy/canonical appends. Keep tool cancellation separate from response interruption.

### Changed

- Require `dialt-sdk>=0.45.0,<0.46.0` and its coordinated broker capabilities.
- Use the owned real-Pipecat aggregator integration in the foundational example and README.
- Add real Pipecat 1.12 pipeline regression tests for all five issue #1 findings.

## [0.1.0] - 2026-10-02

### Added

- Initial `DialtLLMService`, adapter, typed settings, and public frame exports.
- Provider-driven external user turns, 16 kHz PCM media projection, transcript corrections,
  tool execution/cancellation, runtime settings, structured errors, and SDK-owned resume handling.
- Foundational no-VAD local-audio example and hermetic lifecycle/ordering regression tests.

[Unreleased]: https://github.com/dialt-ai/pipecat-dialt/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dialt-ai/pipecat-dialt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dialt-ai/pipecat-dialt/releases/tag/v0.1.0
