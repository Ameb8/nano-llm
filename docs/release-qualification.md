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

The executable matrix is run by `task executable-matrix` and CI. It launches
the normal binary against a local Rustls HTTPS peer, using an additional CA
root only in a debug test build (`NANO_LLM_EXECUTABLE_TEST_CA_PEM`). This does
not disable certificate or hostname verification; release artifacts do not
compile the debug-only test-root path and continue to use bundled WebPKI roots.
The fixture leaf and CA are under `tests/fixtures/executable-tls/`.

Revision: working tree for issue #42 (2026-09-05). Command:

```text
cargo test --features executable-test-tls --test executable_matrix_tests -- --test-threads=1
```

Result: passed (two tests; bounded local sockets and subprocess cleanup).

## Evidence ledger

| Obligation | Existing objective evidence | Status |
|---|---|---|
| Strict config, secrets, routes, and validation | `tests/yaml_subset_tests.rs`, `tests/cli_tests.rs` | covered |
| HTTP limits, auth, response envelope, request IDs, concurrency | `tests/http_contract_tests.rs`, `tests/route_tests.rs`, `tests/server_runtime_tests.rs` | covered |
| Provider request/response, tools, safety, error, and streaming translation | adapter tests in `src/openai_compatible.rs`, `src/anthropic.rs`, and `src/gemini.rs` | covered at injected transport seam |
| Failover, commitment, cancellation, logging, and graceful shutdown | `tests/chat_dispatch_tests.rs`, `tests/server_runtime_tests.rs` | covered at router/runtime seams |
| Static artifacts and scratch image | `tests/packaging_tests.rs`, `scripts/verify-static-artifact.sh`; `task release-linux` was attempted | **externally blocked**: Docker socket access is denied in this environment |
| Secret non-disclosure | config/CLI redaction tests and request-log canary tests | covered at tested seams |
| Executable provider matrix: OpenAI, Mistral, DeepSeek, custom compatible, Anthropic, Gemini | `tests/executable_matrix_tests.rs`: normal binary, YAML/env resolution, verified loopback TLS, text/tool/refusal/error and public-model rewriting; paths and credential placement captured for every preset | covered for listed scenarios |
| Executable fallback / redirect | `binary_fallback_is_file_ordered_and_does_not_follow_redirects`: two requests prove primary restart, one attempt per entry, redirect does not escape the target, and secondary success | covered |
| Executable streaming, slow-body/saturation/disconnect and post-commit lifecycle | router/runtime seam coverage exists; process-matrix expansion remains required | **unqualified** |
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
