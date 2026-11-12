# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue. Use GitHub's
"Report a vulnerability" button on this repository's Security tab (private vulnerability
reporting). Include what you found, how to reproduce it, and the version or commit you tested.

We aim to acknowledge reports within 3 working days and to agree a fix and disclosure date
with you. Please give us a reasonable time to ship a fix before disclosing publicly.

## Scope

In scope: the `splat` CLI and its crates, the exported browser viewers (`crates/splat-webexport`), the
MCP server (`splat mcp`), the downloader (`crates/splat-fetch`: pinned URLs and sha256 checks),
and the release archives and install scripts.

Out of scope: third-party helpers that splat downloads or runs as separate programs (FFmpeg,
COLMAP, ONNX Runtime, Spirula, hosted models on fal.ai). Report those to their projects.

## Supported versions

Only the latest release gets security fixes.
