# Contributing

Contributions are welcome through issues and pull requests in this repository. By contributing,
you agree that your work is licensed under Apache-2.0.

## Development

1. Install Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
2. Run `uv sync --all-groups`.
3. Make a focused change with tests that distinguish the intended behavior from likely mistakes.
4. Run the full command set documented in the README.

Keep the protocol boundary strict: use the public `dialt-sdk`; do not reproduce Dialt wire frames,
session retry, reconnect, or resume logic in this package. Keep server-driven examples free of a
second VAD or endpointing policy. Never add private serving provider or model identities to logs,
frames, settings, fixtures, or documentation.

Do not use live credentials in unit tests. Live validation scripts and logs must avoid printing
secrets and must clean up disposable resources.
