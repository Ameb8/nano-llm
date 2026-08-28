# RALLM — Rust LLM Gateway

**Spec v0.1**

A single-binary, config-driven LLM routing gateway in Rust. LiteLLM-config-compatible
for the subset of features that matter: model list, provider params, and fallback
chains. No database, no built-in dashboard, no enterprise gate. Config is YAML,
loaded at startup, hot-reloadable via SIGHUP (stretch goal).

---

## 1. Goals / Non-Goals

### Goals
- Drop-in-*shaped* config: a `model_list` + `general_settings` YAML file, close enough
  to LiteLLM's schema that porting an existing config is a find/replace job, not a rewrite.
- Real transparent failover: client sends one request, gateway tries target 1, on
  failure (timeout, 5xx, 429, connection error) transparently retries against target 2,
  3, etc. Client never sees the first failure unless *all* targets are exhausted.
- Streaming (SSE) support end-to-end, including failover *before* the first byte
  of a streamed response has been sent to the client. (Once bytes have started
  streaming to the client, failover is no longer possible — see §6.4.)
- One inbound API surface: OpenAI-compatible `/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`. Clients speak OpenAI's wire format
  regardless of which upstream provider actually serves the request.
- Provider support: OpenAI, Anthropic, Gemini (both Generative Language API and
  Vertex AI), Mistral, DeepSeek.
- Minimal footprint: target <20MB idle RSS, single static binary, no runtime deps.
- Simple bearer-token auth on the gateway's own inbound endpoint (`master_key`).

### Non-goals (explicitly out of scope for v0.1)
- Circuit breaker / adaptive health tracking across requests (per your instruction —
  retry/failover is per-request only; no cross-request cooldown state).
- Budgets, spend tracking, per-key rate limiting, usage dashboards.
- Semantic caching, prompt guardrails, PII redaction.
- MCP gateway, agent tooling, tool-call translation beyond passthrough.
- Multi-tenant virtual keys / RBAC. One `master_key`, that's it.
- A web UI. Config is the UI.

---

## 2. Config Schema

### 2.1 Top-level shape

```yaml
model_list:
  - model_name: <string>          # the name clients request
    litellm_params:
      model: <provider>/<upstream-model-id>
      api_key: os.environ/<VAR>   # or literal string (discouraged)
      api_base: <string>          # optional override, e.g. self-hosted/proxy endpoints
      vertex_location: <string>   # vertex_ai only
      vertex_credentials: os.environ/<VAR>  # vertex_ai only; path to service-account JSON
      timeout: <seconds>          # optional, per-target override
      # ... provider-specific passthrough params (temperature defaults, etc.) — v0.2

  - model_name: <string>          # SAME model_name reused = additional fallback target
    litellm_params:
      ...

general_settings:
  master_key: os.environ/<VAR>
  request_timeout: <seconds>      # global default, per-target override wins
  max_retries: <int>               # global default retry count per target before
                                    # moving to next fallback target (default: 1, i.e.
                                    # no retry, just fail to next target)
```

### 2.2 Key design decision: how fallback groups are formed

LiteLLM's real schema uses a separate `fallbacks:` block mapping model name →
ordered list of other model names. Reusing the **same `model_name` across multiple
`model_list` entries** (as shown in your example — `vx-gemini-3.7-f` appearing
once is fine, but note `mistral-fast` and `codestrol-latest` are *different*
model names, i.e. not fallback groups of each other) is visually clean but
ambiguous: does the client ask for `"model": "mistral-fast"` and expect it to
silently fall back to `codestrol-latest`? Not unless they share a `model_name`.

**Decision for v0.1:** support *both*, because your pasted example implies the
first (implicit grouping by repeated `model_name`) but LiteLLM's actual docs use
the second (explicit `fallbacks:`). Concretely:

```yaml
model_list:
  - model_name: default
    litellm_params:
      model: mistral/mistral-small-latest
      api_key: os.environ/MISTRAL_API_KEY

  - model_name: default          # same model_name → 2nd target in same group
    litellm_params:
      model: vertex_ai/gemini-3.7-flash
      vertex_location: "global"
      vertex_credentials: os.environ/VERTEX_CREDENTIALS_PATH

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  # optional explicit fallback cross-references, for when you want
  # DIFFERENT model_names to fall back into each other:
  fallbacks:
    - mistral-fast: [vx-gemini-3.7-f, vx-gemini-3.6-f]
```

Rule: entries sharing a `model_name` are merged into one ordered target list
(order = file order). `general_settings.fallbacks` additionally appends targets
from *other* model_names onto a group, for cross-group fallback chains. Both are
optional; a `model_name` with a single entry and no `fallbacks` reference simply
has no failover — it either succeeds or returns an error.

### 2.3 `os.environ/VAR` resolution

Any string value of the exact form `os.environ/VAR_NAME` is resolved from the
process environment at config-load time. Missing env var → fail fast at startup
with a clear error naming the config path and var name. No silent empty-string
fallback.

### 2.4 Provider identification

The prefix before `/` in `litellm_params.model` selects the provider adapter:

| Prefix        | Provider                          | Wire format required          |
|---------------|------------------------------------|-------------------------------|
| `openai/`     | OpenAI                             | native (passthrough)          |
| `anthropic/`  | Anthropic                          | translate                     |
| `mistral/`    | Mistral                            | native (OpenAI-compatible)    |
| `deepseek/`   | DeepSeek                           | native (OpenAI-compatible)    |
| `gemini/`     | Google Generative Language API     | translate                     |
| `vertex_ai/`  | Google Vertex AI (Gemini via GCP)  | translate + GCP auth          |

Everything after the first `/` is passed through verbatim as the upstream
`model` field (or path component, for providers that put it in the URL).

### 2.5 Validation at load time

- Every `model_list` entry must have `model_name` and `litellm_params.model`.
- `litellm_params.model` must have a recognized provider prefix.
- `vertex_ai/*` entries must have `vertex_location` and `vertex_credentials`.
- All `os.environ/*` references must resolve.
- `general_settings.master_key` is required unless `--no-auth` flag is passed
  explicitly (for local dev only — should warn loudly on startup if unset).
- Config errors are fatal at startup, not runtime. Never start serving with a
  broken model group silently dropped.

---

## 3. Runtime Architecture

```
                    ┌─────────────────────────────────┐
                    │        HTTP Server (axum)        │
                    │  /v1/chat/completions            │
                    │  /v1/completions                 │
                    │  /v1/embeddings                  │
                    │  /health                          │
                    └────────────────┬──────────────────┘
                                     │  bearer auth check (master_key)
                                     ▼
                    ┌─────────────────────────────────┐
                    │           Router                 │
                    │  - look up model_name → target[] │
                    │  - iterate targets in order       │
                    └────────────────┬──────────────────┘
                                     │
                     ┌───────────────┼───────────────┐
                     ▼               ▼               ▼
              ┌───────────┐   ┌───────────┐   ┌───────────┐
              │ Provider  │   │ Provider  │   │ Provider  │
              │ Adapter   │   │ Adapter   │   │ Adapter   │
              │ (target1) │   │ (target2) │   │ (target3) │
              └─────┬─────┘   └───────────┘   └───────────┘
                    │  translate request → send → on error, return
                    │  Err(RetryableError) to Router, which advances
                    │  to next target
                    ▼
              upstream provider API
```

### 3.1 Crate layout (suggested)

```
rallm/
  Cargo.toml
  src/
    main.rs              # CLI entry, config load, server bootstrap
    config/
      mod.rs              # schema structs (serde), validation
      env_resolve.rs       # os.environ/VAR resolution
      fallback_groups.rs   # merge model_list entries into target lists
    server/
      mod.rs               # axum router setup
      auth.rs              # bearer token middleware
      handlers.rs          # /v1/chat/completions etc — thin, delegates to router
    router/
      mod.rs               # core dispatch loop: try target[i], on fail advance
      retry.rs             # retry policy (max_retries per target, backoff)
      error.rs             # RetryableError vs FatalError classification
    providers/
      mod.rs               # Provider trait
      openai.rs
      anthropic.rs
      gemini.rs             # Generative Language API
      vertex.rs             # Vertex AI (Gemini via GCP, separate auth path)
      mistral.rs
      deepseek.rs
    translate/
      mod.rs
      openai_wire.rs        # canonical request/response types (OpenAI shape)
      anthropic_wire.rs     # Anthropic ↔ OpenAI translation
      gemini_wire.rs        # Gemini ↔ OpenAI translation
    streaming/
      mod.rs                # SSE passthrough + re-framing helpers
```

### 3.2 The `Provider` trait

```rust
#[async_trait]
trait Provider: Send + Sync {
    /// Non-streaming call. Takes the canonical (OpenAI-shape) request,
    /// returns canonical response or a classified error.
    async fn complete(&self, req: &CanonicalRequest) -> Result<CanonicalResponse, ProviderError>;

    /// Streaming call. Returns a stream of canonical SSE chunks.
    async fn complete_stream(&self, req: &CanonicalRequest)
        -> Result<BoxStream<'static, Result<CanonicalChunk, ProviderError>>, ProviderError>;
}
```

Every provider adapter implements this against **one canonical internal
request/response shape** (OpenAI's, since 3 of 5 providers are already
OpenAI-wire-compatible and clients speak OpenAI anyway). Anthropic and
Gemini/Vertex adapters do request translation in, response translation out.
Mistral, DeepSeek, and OpenAI itself are near-passthrough (base URL + auth
header swap, maybe minor field renames).

### 3.3 `ProviderError` classification

This is the crux of correct failover: every error a provider adapter can
produce needs to be tagged retryable or not.

```rust
enum ProviderError {
    Retryable(RetryableKind),
    Fatal(String),   // e.g. malformed request — retrying won't help, but
                      // *should* still advance to next target in case it's a
                      // provider-specific quirk, not a genuinely bad request
}

enum RetryableKind {
    Timeout,
    ConnectionError,
    RateLimited { retry_after: Option<Duration> },
    ServerError(u16),      // 5xx
    Overloaded,             // Anthropic's 529, etc.
}
```

**Decision:** treat *all* upstream errors as advance-to-next-target, including
"fatal" ones — the only thing that should stop the fallback walk early is
exhausting the target list. A 400 from provider A might be a 200 from provider
B if the request happens to hit a provider-specific validation quirk (e.g. a
param name mismatch after translation). This is simpler than trying to
perfectly classify every error and matches "transparent failover" as you
described it. Rate-limit `retry_after` is honored only for same-target retries
within `max_retries`, not for the cross-target advance (advancing is immediate).

### 3.4 Retry vs Fallback — two distinct knobs

- **Retry**: same target, same provider, up to `max_retries` times (config:
  `general_settings.max_retries`, default 1 = no retry). Use for transient
  network blips. Respects `Retry-After` header if present, capped at some
  sane max wait (e.g. 5s) so a single request doesn't hang the client forever.
- **Fallback**: advance to the *next target* in the group's ordered list.
  Always happens after retries for the current target are exhausted. No
  configurable limit beyond "list is exhausted."

Total attempts for a request = sum of `max_retries` across all targets in the
group.

---

## 4. Streaming & Failover Interaction

This is the sharpest edge in the whole design, worth calling out explicitly.

### 4.1 Rule

**Failover is only possible before the first byte has been forwarded to the
client.** Once the gateway has started writing SSE chunks downstream, it has
committed to that upstream — a mid-stream provider failure becomes a stream
termination (with an SSE `error` event or an abrupt close), not a silent
retry, because:
- The client may have already rendered/acted on partial tokens.
- Re-issuing the same prompt against a different provider mid-stream would
  either duplicate content or produce an incoherent transcript.

### 4.2 Implementation approach

The gateway buffers the **first chunk** from the upstream stream before
forwarding anything to the client. This costs one chunk of latency (typically
tens of ms) but means:
- If target 1's stream fails/errors before yielding any chunk (including an
  immediate 4xx/5xx on the streaming request itself), the gateway can still
  transparently advance to target 2 — client never knows target 1 was tried.
- Once the first chunk is received and forwarded, the gateway is "locked in"
  to that upstream for the rest of the request.

```
connect to target[i] (streaming)
  ├─ error before first chunk? → advance to target[i+1], retry loop
  └─ first chunk received?
       → forward it, then pipe remaining chunks directly (locked in)
       → upstream fails mid-stream? → emit SSE error event, close. Do NOT
         advance to target[i+1].
```

### 4.3 Non-streaming requests

No such constraint — buffer the full response, and if the upstream call
fails at any point (including a failure while reading the body), advance to
the next target normally.

---

## 5. Provider Adapters — Notes Per Provider

- **OpenAI**: near passthrough. Base URL `api.openai.com/v1`, `Authorization:
  Bearer <key>`. Canonical format IS OpenAI's format, so this adapter is
  mostly "forward the request, forward the response."
- **Mistral**: OpenAI-compatible API. Base URL differs
  (`api.mistral.ai/v1`), auth header same shape. Minimal translation, if any.
- **DeepSeek**: OpenAI-compatible API. Same pattern as Mistral — different
  base URL, same wire shape.
- **Anthropic**: real translation required. Messages API has a distinct
  request shape (`system` as top-level field not a message role,
  `max_tokens` required, content blocks, different streaming event names —
  `message_start`, `content_block_delta`, etc. instead of OpenAI's
  `choices[].delta`). This is the first adapter to write after OpenAI, since
  it exercises the translation layer fully.
- **Gemini (Generative Language API)**: `generateContent` /
  `streamGenerateContent` endpoints, API key as query param (`?key=`), request
  shape uses `contents[].parts[]`, roles are `user`/`model` not
  `user`/`assistant`. Translation required.
- **Vertex AI (Gemini via GCP)**: same Gemini request/response shape as
  above, but auth is OAuth2 via service account JSON
  (`vertex_credentials`), not a static API key, and the endpoint is
  region/project-scoped (`vertex_location`, plus a GCP project ID — **note:
  your example config is missing `vertex_project`; LiteLLM's actual schema
  needs a project ID somewhere, either in config or inferred from the
  credentials file. Flag this as a config schema gap to resolve before
  implementation** — either add `vertex_project: <string>` to the schema or
  extract it from the service account JSON's `project_id` field at load
  time). Token refresh/caching for the GCP OAuth2 flow is a real chunk of
  work — budget for it separately from the Gemini translation logic, since
  it's shared infra (JWT signing, token cache with expiry) rather than
  request/response translation.

**Suggested build order for adapters:** OpenAI → Mistral → DeepSeek (these
three validate the server/router/config plumbing with near-zero translation
work) → Anthropic (exercises translation) → Gemini (exercises translation +
different auth) → Vertex (exercises translation + GCP OAuth2, hardest).

---

## 6. HTTP Surface

| Endpoint                  | Behavior                                                   |
|----------------------------|--------------------------------------------------------------|
| `POST /v1/chat/completions`| Main entry point. `model` field in body selects the group.  |
| `POST /v1/completions`     | Legacy completions, v0.2 if needed — skip for v0.1 unless you need it. |
| `POST /v1/embeddings`      | Same routing/failover logic, separate canonical shape.       |
| `GET /health`              | Liveness only — does NOT check upstream provider health (no circuit breaker state to report). |
| `GET /v1/models`           | Optional: list configured `model_name`s, for OpenAI-SDK compatibility with tools that call this. |

Auth: `Authorization: Bearer <master_key>` required on all `/v1/*` routes.
Reject with 401 before any routing/provider work happens.

---

## 7. Config → Your Pasted Example, Corrected

Your example as given actually works fine for the "two separate model
groups" case, EXCEPT the Vertex entries are missing `vertex_project` (see
§5). Once that's resolved, this is close to valid v0.1 config as-is:

```yaml
model_list:
  - model_name: mistral-fast
    litellm_params:
      model: mistral/mistral-small-latest
      api_key: os.environ/MISTRAL_API_KEY

  - model_name: codestrol-latest
    litellm_params:
      model: mistral/codestrol-latest
      api_key: os.environ/MISTRAL_API_KEY

  - model_name: vx-gemini-3.7-f
    litellm_params:
      model: vertex_ai/gemini-3.7-flash
      vertex_location: "global"
      vertex_project: os.environ/VERTEX_PROJECT_ID   # ADDED — see §5
      vertex_credentials: os.environ/VERTEX_CREDENTIALS_PATH

  - model_name: vx-gemini-3.6-f
    litellm_params:
      model: vertex_ai/gemini-3.6-flash
      vertex_location: "global"
      vertex_project: os.environ/VERTEX_PROJECT_ID
      vertex_credentials: os.environ/VERTEX_CREDENTIALS_PATH

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  max_retries: 2
  fallbacks:
    - vx-gemini-3.7-f: [vx-gemini-3.6-f, mistral-fast]
```

Each `model_name` here is its own group (no shared names in your example), so
there's no *implicit* failover between any of them without the explicit
`fallbacks:` block — added above as an illustration of the cross-group case.

---

## 8. Suggested Milestones

1. **Config loader + validation** — parse the schema above, resolve env vars,
   merge fallback groups, fail loudly on bad config. No server yet — a CLI
   subcommand that just loads and pretty-prints the resolved routing table is
   a good first deliverable and a genuinely useful `--validate` flag long-term.
2. **OpenAI adapter + non-streaming server** — `/v1/chat/completions`,
   single target, no failover yet. Prove the axum server, auth middleware,
   and canonical types work end to end against one real provider.
3. **Failover + retry loop** — wire the router's target-iteration logic using
   a mock/fault-injecting provider for tests (this is where you want good
   unit tests — simulate target 1 timing out, target 2 returning 429, target
   3 succeeding, and assert the client only ever sees the final success).
4. **Streaming** — SSE passthrough for OpenAI first, then the buffered-first-chunk
   failover logic from §4.
5. **Mistral + DeepSeek adapters** — should be quick, near-identical to OpenAI.
6. **Anthropic adapter** — first real translation layer.
7. **Gemini adapter** — second translation layer, static API key auth.
8. **Vertex adapter** — GCP OAuth2 token handling + same Gemini translation
   logic reused.
9. **Polish** — `/health`, `/v1/models`, SIGHUP config reload (stretch),
   structured logging, ARM cross-compile + Docker `FROM scratch` build for
   the Pi.

---

## 9. Open Questions / Decisions Needed Before Coding

- **`vertex_project`**: add to schema explicitly, or parse from the service
  account JSON at load time? (Recommend explicit config field — fewer
  surprises, matches how `vertex_location` is already explicit.)
- **`general_settings.fallbacks` syntax**: the sketch above
  (`- group_name: [target, target]`) is a list of single-key maps, which is
  a bit awkward in YAML/serde. Consider instead:
  ```yaml
  fallbacks:
    vx-gemini-3.7-f: [vx-gemini-3.6-f, mistral-fast]
  ```
  as a plain map, which is simpler to deserialize (`HashMap<String, Vec<String>>`)
  and still readable.
- **Per-target timeout defaults**: what's a sane default `request_timeout`
  before a target is considered failed and the router advances? Suggest
  30s default, overridable globally and per-target.
- **Logging**: even without a dashboard, you'll want structured request logs
  (which target served the request, how many targets were tried, latency) —
  worth deciding early whether that's just `tracing` to stdout (simplest,
  recommend this for v0.1) or something more.
