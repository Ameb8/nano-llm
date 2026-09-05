# v0.1 release qualification

This record is the release gate for the specification's definition of v0.1.
It must be updated only from a clean worktree with the commands and artifacts
named below. A failed or unrun item keeps the package version on its
development identity; a documented RSS miss is not hidden or converted into a
contract change.

## Current audit — not qualified

Date: 2026-09-05

The codebase is **not eligible to identify as v0.1**. `Cargo.toml` remains at
`0.1.0-dev` and `IS_DEV_BUILD` remains true.

The provider adapters have extensive injected-transport conformance coverage,
but the executable's default `build_provider` factory constructs
`TargetProvider`. That provider deliberately returns `InvalidResponse` for
both completion modes. Consequently no configured provider can complete an
actual request from the production binary. This fails the canonical requirement
that every provider family pass non-streaming, streaming, tool, safety, error,
and fallback scenarios in the executable release path.

No version change is permitted until a production outbound transport with
bundled-root HTTPS verification, disabled redirects, attempt timeouts,
incremental SSE reads, cancellation, and the corresponding end-to-end provider
matrix is implemented and passes.

## Evidence ledger

| Obligation | Existing objective evidence | Status |
|---|---|---|
| Strict config, secrets, routes, and validation | `tests/yaml_subset_tests.rs`, `tests/cli_tests.rs` | covered |
| HTTP limits, auth, response envelope, request IDs, concurrency | `tests/http_contract_tests.rs`, `tests/route_tests.rs`, `tests/server_runtime_tests.rs` | covered |
| Provider request/response, tools, safety, error, and streaming translation | adapter tests in `src/openai_compatible.rs`, `src/anthropic.rs`, and `src/gemini.rs` | covered at injected transport seam |
| Failover, commitment, cancellation, logging, and graceful shutdown | `tests/chat_dispatch_tests.rs`, `tests/server_runtime_tests.rs` | covered at router/runtime seams |
| Static artifacts and scratch image | `tests/packaging_tests.rs`, `scripts/verify-static-artifact.sh`; `task release-linux` was attempted | **externally blocked**: Docker socket access is denied in this environment |
| Secret non-disclosure | config/CLI redaction tests and request-log canary tests | covered at tested seams |
| Executable provider matrix | no production transport | **blocked** |
| Release-build idle RSS | local `cargo build --release --locked` succeeded; the helper was then unable to bind `127.0.0.1:40138` in this sandbox (`EPERM`), so no RSS value was fabricated | **externally blocked** |

The matrix intentionally treats OpenAI, Mistral, DeepSeek, and
`openai_compatible` as separately qualified presets even though they share one
adapter, and treats Anthropic and Gemini as distinct translation families. The
current adapter-level evidence does not promote that result to release
qualification because the process factory does not use those adapters.

## Required final commands

```text
task check
task release-linux
scripts/measure-idle-rss.sh dist/x86_64/nano-llm tests/fixtures/release-minimal.yaml
```

Record the measured KiB and whether it is below 20 MiB. Run an artifact smoke
test for each architecture and a secret scan over validation output, logs, HTTP
responses, and exported artifacts. Only after every ledger row is passing may
the version become `0.1.0` and the development-build indicator become false.
