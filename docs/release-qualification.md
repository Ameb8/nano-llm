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

Revision: `d72cfc5` plus the uncommitted issue #43 qualification changes
(2026-09-05). Commands:

```text
cargo build --locked
scripts/smoke-release-artifact.sh aarch64 target/debug/nano-llm
cargo build --locked --release
scripts/measure-idle-rss.sh target/release/nano-llm tests/fixtures/release-minimal.yaml
cargo test --all-targets --all-features
cargo test --features executable-test-tls --test executable_matrix_tests -- --test-threads=1
```

Results: `task check` passed (189 tests) and the executable matrix passed (two
process-level tests). The host was Linux/aarch64; the controlled debug-binary
smoke passed health, authenticated models, keyless controlled provider I/O,
and incremental SSE natively. This is development-binary evidence only, not
release-artifact evidence.

The RSS helper sampled `target/release/nano-llm` on Linux/aarch64 after one
second of idle time with `tests/fixtures/release-minimal.yaml`, loopback bind,
and `--no-auth`: **2,592 KiB = 2.53 MiB**. The target wording is less than
**20 MiB**, which is 20,480 KiB (not 20 decimal MB); this sample is below that
threshold. It is deliberately not claimed as a final RSS result because the
required `dist/x86_64/nano-llm` release artifact was not exported.

## Evidence ledger

| Obligation | Existing objective evidence | Status |
|---|---|---|
| Strict config, secrets, routes, and validation | `tests/yaml_subset_tests.rs`, `tests/cli_tests.rs` | covered |
| HTTP limits, auth, response envelope, request IDs, concurrency | `tests/http_contract_tests.rs`, `tests/route_tests.rs`, `tests/server_runtime_tests.rs` | covered |
| Provider request/response, tools, safety, error, and streaming translation | adapter tests in `src/openai_compatible.rs`, `src/anthropic.rs`, and `src/gemini.rs` | covered at injected transport seam |
| Failover, commitment, cancellation, logging, and graceful shutdown | `tests/chat_dispatch_tests.rs`, `tests/server_runtime_tests.rs` | covered at router/runtime seams |
| Static artifacts and scratch image | `tests/packaging_tests.rs`; verifier now checks ELF architecture, static/static-PIE classification, no ELF interpreter, and no `DT_NEEDED`; workflow runs both-architecture artifact smoke and scratch checks | **unqualified**: `task release-linux` began downloading the Rust builder image on this host but did not complete before the local command window; no `dist/` artifact or checksum exists yet |
| Secret non-disclosure | config/CLI redaction tests and request-log canary tests | covered at tested seams |
| Executable provider matrix: OpenAI, Mistral, DeepSeek, custom compatible, Anthropic, Gemini | `tests/executable_matrix_tests.rs`: normal binary, YAML/env resolution, verified loopback TLS, text/tool/refusal/error and public-model rewriting; paths and credential placement captured for every preset | covered for listed debug-process scenarios; final release-artifact rerun remains required |
| Executable fallback / redirect | `binary_fallback_is_file_ordered_and_does_not_follow_redirects`: two requests prove primary restart, one attempt per entry, redirect does not escape the target, and secondary success | covered |
| Artifact public-surface smoke | `scripts/smoke-release-artifact.sh`: controlled provider, health, authenticated models, chat, incremental SSE, and canary scan; native or QEMU runner is mandatory | **unqualified** until it runs on both exported artifacts |
| Scratch startup and contents | workflow asserts UID `65532:65532`, only `nano-llm` in the exported filesystem, no config, missing secret, and valid mounted-config cases | **unqualified** pending workflow/authorized runner result |
| Production TLS and redirect security | Production uses Rustls WebPKI bundled roots, disables redirects/proxies, and the debug matrix adds a loopback CA only as an additional root while retaining chain/hostname checks | **unqualified**: a release artifact still needs a credential-bearing request to a publicly trusted controlled endpoint with no CA mount, plus chain/hostname rejection evidence |
| Secret/body canary scan | Release smoke scans its gateway log plus health, models, chat, and SSE outputs for master-key and request-body canaries; matrix asserts its subprocess output has no key canary | **unqualified** until run against exported artifacts and scratch runtime |
| Release-build idle RSS | Host release-profile sample: 2,592 KiB / 2.53 MiB on Linux/aarch64 after one idle second; helper reports the exact 20 MiB = 20,480 KiB comparison | **unqualified**: required `dist/x86_64` artifact sample not yet available |

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

`task release-linux` now calls `scripts/smoke-release-artifact.sh` for each
architecture. It requires native execution or `qemu-<arch>` and fails rather
than treating inspection-only output as executable evidence. Record each
artifact SHA-256, runner/emulation, measured KiB, and whether it is below 20
MiB. Run the credential-bearing public-root TLS method and a secret scan over
validation output, logs, HTTP responses, and exported artifacts. Only after
every ledger row is passing may the version become `0.1.0` and the development
build indicator become false.
