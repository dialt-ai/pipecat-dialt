# Releasing

Releases are made by a Dialt maintainer. Nothing in CI publishes automatically from a pull request.

Before the first release, configure a PyPI pending trusted publisher for organization `dialt-ai`,
repository `pipecat-dialt`, and workflow `publish.yml`. Leave the optional environment name blank.
The workflow uses OIDC; do not add a PyPI API token to GitHub.

1. Update the version in `pyproject.toml`, the compatibility table, and `CHANGELOG.md`.
2. Test the exact supported Pipecat and `dialt-sdk` versions, including the hosted smoke tests.
3. Run `uv lock --check`, tests, Ruff, strict mypy, `uv build`, `twine check`, wheel content/import
   checks, and the disclosure audit.
4. Commit, push the clean main branch, and push an annotated `vX.Y.Z` tag.
5. Create a GitHub release for that tag and publish it once. The `Publish to PyPI` workflow builds
   from the release tag and publishes the distributions through PyPI Trusted Publishing.
6. Verify the workflow's attestations and a clean-environment installation from PyPI.
7. Only after a public package and demo exist, submit the separate official Pipecat docs listing.

If publication fails before uploading distributions, fix the workflow on main and use its manual
dispatch with the existing release tag. The workflow checks out that tag, preserving the reviewed
package source without moving a published tag or republishing the GitHub release.

Publishing, pushing tags, and submitting upstream documentation require explicit maintainer action.
