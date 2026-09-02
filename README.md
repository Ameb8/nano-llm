# nano-llm

A lightweight, single-binary LLM routing gateway written in Rust.

`nano-llm` exposes a single OpenAI-compatible API surface in front of multiple upstream LLM providers, providing transparent per-request failovers and LiteLLM-compatible YAML configuration with minimal memory footprint (<20MB idle RSS).

---

## Features

- **OpenAI-Compatible Inbound API**: Clients send standard OpenAI requests (`/v1/chat/completions`, `/v1/embeddings`), regardless of upstream provider.
- **Transparent Failover**: Automatically retries failed requests (timeouts, 429s, 5xx errors) against fallback models/providers without returning errors to the client until all targets are exhausted.
- **Streaming Support**: Full SSE streaming with pre-first-chunk buffering so failover can occur even on streaming calls if an upstream fails immediately.
- **Multi-Provider Support**:
  - OpenAI, Mistral, DeepSeek (native / OpenAI-compatible passthrough)
  - Anthropic (translated)
  - Google Gemini & Vertex AI (translated)
- **Zero Heavy Dependencies**: No database, no web UI, no background worker. Simple static binary.

---

## Configuration

Configuration is defined via a YAML file (e.g., `config.yaml`). Environment variables can be referenced using `os.environ/VAR_NAME`.

```yaml
model_list:
  # Primary model target
  - model_name: fast-chat
    litellm_params:
      model: mistral/mistral-small-latest
      api_key: os.environ/MISTRAL_API_KEY
      timeout: 30

  # Reusing model_name adds an implicit fallback target to the group
  - model_name: fast-chat
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  # Additional model definitions
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gemini-flash
    litellm_params:
      model: vertex_ai/gemini-2.0-flash
      vertex_location: "global"
      vertex_project: os.environ/VERTEX_PROJECT_ID
      vertex_credentials: os.environ/VERTEX_CREDENTIALS_PATH

general_settings:
  master_key: os.environ/NANO_LLM_MASTER_KEY
  request_timeout: 30
  max_retries: 2
  # Optional explicit cross-model fallback chains
  fallbacks:
    claude: [fast-chat]
```

---

## Usage

### Running the Server

```bash
# Build and run with a config file
cargo run -- --config config.yaml

# Or run the compiled binary
./nano-llm --config config.yaml
```

### Making Requests

Send standard OpenAI API requests authenticated with your `master_key`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $NANO_LLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fast-chat",
    "messages": [
      {"role": "user", "content": "Hello world!"}
    ]
  }'
```

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completions (streaming & non-streaming) |
| `POST /v1/embeddings` | Text embeddings |
| `GET /health` | Gateway liveness check |
| `GET /v1/models` | List configured models |

---

## Development & Quality Workflow

`nano-llm` uses [Task](https://taskfile.dev) as the canonical interface for formatting, linting, tests, and aggregate quality checks. Implementing coding agents and contributors must run relevant Taskfile targets (specifically `task check`) before handoff.

- `task format` — formats Rust source code (`cargo fmt --all`)
- `task lint` — runs Clippy with warnings treated as errors (`cargo clippy --all-targets --all-features -- -D warnings`)
- `task test` — executes the automated test suite (`cargo test --all-targets --all-features`)
- `task check` — runs format verification, linting, and tests; fails if any constituent check fails

