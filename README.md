# nano-llm

`nano-llm` is a single-binary, config-driven LLM routing gateway written in Rust. It provides one strict OpenAI-compatible chat endpoint in front of a small set of providers, with fixed-priority, per-request failover.

It is designed for a single operator or small trusted team: one process, one YAML configuration file, one shared inbound key, static routing, and troubleshooting-oriented logs. The target footprint is under 20 MB idle RSS.

The canonical v0.1 specification is [docs/specs/nano-llm.md](docs/specs/nano-llm.md).

## v0.1 at a glance

- `POST /v1/chat/completions` is the sole generation endpoint. It supports text chat, SSE streaming, and portable OpenAI-shaped function tools.
- Repeated `model_name` entries form an ordered fallback route. Each request tries the first target, then advances only after a target failure.
- Supported provider families are OpenAI, Mistral, DeepSeek, `openai_compatible`, Anthropic, and the Gemini Generative Language API.
- Configuration is validated and resolved at startup, then remains immutable for the lifetime of the process. There is no hot reload.
- `/health` is an unauthenticated liveness check; `/v1/models` lists configured public model names and requires gateway authentication.

v0.1 does not include embeddings, legacy completions, Vertex AI, inbound TLS, a web UI, cross-request circuit breaking, load balancing, multi-tenant keys, or provider-specific request extensions. Multimodal input and structured output are also outside the portable v0.1 interface and are rejected.

## Configuration

The configuration file is a strict, LiteLLM-shaped YAML subset. It contains one `model_list` and authenticated deployments also require `general_settings`. Only `master_key` and `api_key` accept environment references, and they must use the exact form `os.environ/VAR_NAME`; literal secrets are rejected.

```yaml
model_list:
  - model_name: fast
    litellm_params:
      model: mistral/mistral-small-latest
      api_key: os.environ/MISTRAL_API_KEY

  # Repeating the name adds the next fixed-priority fallback target.
  - model_name: fast
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  request_timeout: 30
  overall_timeout: 120
  max_in_flight: 64
```

File order is attempt order. There are no cross-model fallback references, retries, weights, load balancing, or adaptive health state. A successful first target always serves the request; each new request starts with that target.

`request_timeout` defaults to 30 seconds, `overall_timeout` to 120 seconds, and `max_in_flight` to 64. Per-target `litellm_params.timeout` overrides the request timeout. All timeout values are whole seconds.

The provider prefix in `litellm_params.model` selects the adapter:

| Prefix | Provider |
|---|---|
| `openai/` | OpenAI |
| `mistral/` | Mistral |
| `deepseek/` | DeepSeek |
| `openai_compatible/` | Custom OpenAI-compatible endpoint |
| `anthropic/` | Anthropic |
| `gemini/` | Google Generative Language API |

Branded providers require `api_key`. `openai_compatible/` requires an explicit `api_base` and may omit `api_key` for a keyless local or trusted-network endpoint. An `api_base` is a versioned base URL, not a complete operation URL; the adapter appends its operation path.

Unknown configuration keys, duplicate YAML keys, invalid environment references, and invalid values are fatal startup errors. The accepted YAML subset is one document with string mapping keys; aliases, anchors, merge keys, and custom tags are rejected.

## Running

```bash
# Run with the default loopback bind address, 127.0.0.1:4000.
cargo run -- --config config.yaml

# Or run the compiled binary.
./nano-llm --config config.yaml

# Validate configuration and print the resolved route table with secrets redacted.
./nano-llm --config config.yaml --validate
```

The complete CLI is:

```text
nano-llm --config <path> [--bind <address>] [--no-auth] [--validate]
```

`--no-auth` is intended only for local development and is rejected on a non-loopback bind address. In that mode, `general_settings` may be omitted. The listener serves plain HTTP; use a reverse proxy, tunnel, or load balancer to terminate TLS for remote deployments.

## Static Linux artifacts and container image

Release artifacts are statically linked Linux binaries for `x86_64` and
`aarch64`. Build and inspect both locally with Docker Buildx:

```bash
task release-linux
```

The resulting binaries are written to `dist/x86_64/nano-llm` and
`dist/aarch64/nano-llm`. `task release-linux` also starts each artifact (using
native execution or `qemu-aarch64`/`qemu-aarch64-static`), checks health and
models, sends a controlled chat request, and verifies incremental SSE plus
canary non-disclosure. The bundled-root transport policy verifies outbound
provider HTTPS without host CA files and never shells out to a provider CLI.

Build the runtime image with `task image`. It is `FROM scratch` and contains
only `/nano-llm`; mount YAML at `/etc/nano-llm/config.yaml` and provide the
environment variables referenced by its secret fields:

```bash
docker run --rm -p 4000:4000 \
  -v "$PWD/config.yaml:/etc/nano-llm/config.yaml:ro" \
  -e LITELLM_MASTER_KEY -e OPENAI_API_KEY nano-llm:local
```

There is deliberately no image `HEALTHCHECK`: a scratch image has no shell or
HTTP client. Probe its plain `http://…/health` listener externally. nano-llm
does not provide inbound TLS, certificates, or ACME; outbound provider HTTPS
uses bundled trust roots.

Before pushing a release-related change, reproduce every job in the
`release-artifacts` workflow locally:

```bash
docker buildx inspect --bootstrap
task ci-release
```

The Buildx platform list must include both `linux/amd64` and `linux/arm64`.
`task ci-release` runs the executable provider matrix, builds and smoke-tests
both static artifacts, then builds and smoke-tests the scratch runtime image.
It requires Docker Buildx plus native, QEMU, or `binfmt_misc` execution for
both architectures.

To collect an idle-memory qualification sample from a release artifact, run:

```bash
scripts/measure-idle-rss.sh dist/x86_64/nano-llm tests/fixtures/release-minimal.yaml
```

The helper reports KiB and MiB. The documented target is **less than 20 MiB**,
which is 20,480 KiB; it does not treat decimal MB as equivalent.

## Making a request

When authentication is enabled, `/v1/*` endpoints require the configured `master_key` as a bearer token. `/health` is always unauthenticated.

```bash
curl http://127.0.0.1:4000/v1/chat/completions \\
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "fast",
    "messages": [{"role": "user", "content": "Hello world!"}]
  }'
```

The supported request fields are `model`, `messages`, `stream`, `stream_options.include_usage`, `max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`, `stop`, `tools`, and `tool_choice`. Unknown fields and unsupported nested shapes are rejected before routing. If a route includes Anthropic, every request must specify exactly one of `max_tokens` or `max_completion_tokens`.

For streaming requests, nano-llm can fall back until it has sent the first canonical SSE chunk. After that point the response is committed, so a later upstream failure closes the stream without switching providers.

## HTTP endpoints

| Endpoint | Behavior |
|---|---|
| `POST /v1/chat/completions` | Text chat completions, including SSE and function tools. |
| `GET /v1/models` | Lists each configured public model name once, in first-seen order. |
| `GET /health` | Unauthenticated liveness check; returns `{"status":"ok"}` and does not probe providers. |

`/v1/completions` and `/v1/embeddings` are not implemented in v0.1. Other paths and unsupported methods return 404.

All `/v1/*` errors use an OpenAI-shaped JSON envelope. The gateway never forwards upstream error bodies or credentials to clients. Responses include an `x-request-id` header for correlation.

---

## Development & Quality Workflow

`nano-llm` uses [Task](https://taskfile.dev) as the canonical interface for formatting, linting, tests, and aggregate quality checks. Implementing coding agents and contributors must run relevant Taskfile targets (specifically `task check`) before handoff.

- `task format` — formats Rust source code (`cargo fmt --all`)
- `task lint` — runs Clippy with warnings treated as errors (`cargo clippy --all-targets --all-features -- -D warnings`)
- `task test` — executes the automated test suite (`cargo test --all-targets --all-features`)
- `task check` — runs format verification, linting, and tests; fails if any constituent check fails
- `task ci-release` — reproduces all release-artifact CI jobs locally
