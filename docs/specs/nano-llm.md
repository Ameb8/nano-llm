# nano-llm — Rust LLM Gateway

**Spec v0.1**

A single-binary, config-driven LLM routing gateway in Rust. Its configuration is
LiteLLM-shaped for the documented subset that matters: model lists, provider
parameters, and fallback chains. This is migration familiarity, not a claim that
arbitrary LiteLLM configurations are compatible. No database, no built-in
dashboard, no enterprise gate. Config is YAML, loaded at startup. Hot reload via
SIGHUP is not part of v0.1; configuration is immutable for the process lifetime.

The primary user is a single operator or small trusted team that wants a
self-hosted reliability shim across a handful of LLM providers. nano-llm is not
an organization-wide AI platform: it favors one process, one config file, one
shared inbound key, static operator-defined routing, and troubleshooting-oriented
logs over multi-user administration, dynamic policy systems, metering, or
analytics. It should remain comfortable to run on a laptop, Raspberry Pi, small
VM, or minimal container.

### Release terminology

`v0.1` is the first public release described by this document. The milestones in
§8 are implementation slices, not separately conforming releases and not alternate
definitions of a "V1." A binary is v0.1-conformant only when it implements the
entire documented v0.1 surface, including every listed provider family, streaming,
tool calling, validation, operational endpoints, and release packaging. Earlier
milestones may be runnable and useful during development, but must identify
themselves as development builds rather than claiming v0.1 conformance.

---

## 1. Goals / Non-Goals

### Goals
- Familiar migration path: a `model_list` + `general_settings` YAML file using
  recognizable LiteLLM names, but governed by nano-llm's own strict, documented
  schema. Existing LiteLLM configs may require small edits and removal of
  unsupported fields before they validate.
- Real transparent failover: client sends one request, gateway tries target 1, on
  failure (timeout, 5xx, 429, connection error) transparently advances to target 2,
  3, etc. Client never sees the first failure unless *all* targets are exhausted.
- Streaming (SSE) support end-to-end, including failover *before* a streamed
  response has been committed to the client. Once a canonical SSE chunk has
  been sent, failover is no longer possible — see §4.
- One inbound generation surface: an OpenAI-compatible
  `/v1/chat/completions`. Clients use the same documented v0.1 wire format
  regardless of which upstream provider actually serves the request.
- Provider support: OpenAI, Anthropic, Gemini Generative Language API, Mistral,
  DeepSeek, and custom OpenAI-compatible endpoints.
- Minimal footprint: target <20MB idle RSS, single static binary, no runtime deps.
- Simple bearer-token auth on the gateway's own inbound endpoint (`master_key`).

### Non-goals (explicitly out of scope for v0.1)
- Circuit breaker / adaptive health tracking across requests (failover is
  per-request only; no cross-request cooldown state). Consequently, every
  request probes the primary even during an outage and may pay its configured
  timeout before falling back.
- Budgets, spend tracking, per-key rate limiting, usage dashboards.
- Semantic caching, prompt guardrails, PII redaction.
- Legacy `/v1/completions` and `/v1/embeddings` endpoints.
- Multimodal content, structured/JSON-schema output, log probabilities, and
  provider-specific request extensions. These must be rejected clearly in
  v0.1, including when the selected upstream could accept them, so every
  configured fallback target has the same interface.
- MCP gateway and agent tooling.
- Vertex AI and its GCP authentication stack. It is the first planned
  post-v0.1 provider and will reuse the v0.1 Gemini translation layer.
- Multi-tenant virtual keys / RBAC. One `master_key`, that's it.
- A web UI. Config is the UI.
- Inbound TLS termination, certificate management, and ACME. Remote deployments
  terminate TLS in a reverse proxy, tunnel, or load balancer.

---

## 2. Config Schema

### 2.1 Top-level shape

```yaml
model_list:
  - model_name: <string>          # the name clients request
    litellm_params:
      model: <provider>/<upstream-model-id>
      api_key: os.environ/<VAR>   # env reference required when key is present
      api_base: <string>          # optional override, e.g. self-hosted/proxy endpoints
      timeout: <int seconds>      # optional, per-target override

  - model_name: <string>          # SAME model_name reused = additional fallback target
    litellm_params:
      ...

general_settings:
  master_key: os.environ/<VAR>
  request_timeout: <int seconds>  # global default, per-target override wins
  overall_timeout: <int seconds>  # whole fallback chain; default 120 seconds
  max_in_flight: <int 1..65535>   # process-wide request cap; default 64
```

### 2.2 Key design decision: how fallback groups are formed

v0.1 has one routing mechanism: entries that repeat the same `model_name` form
one fixed-priority fallback route. Every request begins at the first entry;
later entries are attempted only after earlier failures. File order is attempt
order. A name with one entry has no fallback.

```yaml
model_list:
  - model_name: default
    litellm_params:
      model: mistral/mistral-small-latest
      api_key: os.environ/MISTRAL_API_KEY

  - model_name: default          # same model_name → 2nd target in same group
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

There are no cross-group references in v0.1. This makes every route locally
visible and avoids graph expansion, cycle detection, and deduplication rules.
Operators who want the same concrete target in multiple routes repeat its
configuration under each public `model_name`. Repeated entries do not
load-balance successful traffic: v0.1 has no round-robin, random, weighted,
latency-based, or cost-based target selection.

### 2.3 `os.environ/VAR` resolution

Environment references are recognized only in the secret fields `master_key`
and `api_key`. In those fields, a value of the exact form
`os.environ/VAR_NAME` is resolved from the process environment at config-load
time. Missing env var → fail fast at startup with a clear error naming the
config path and var name. No silent empty-string fallback. The same spelling in
`model_name`, `model`, `api_base`, or any other non-secret string is an ordinary
literal and is then validated by that field's normal rules.

Secret fields do not accept literal values in v0.1. `master_key` and every
present `api_key` must use the exact `os.environ/VAR_NAME` form; inline secrets
are config errors. The generic `openai_compatible/` adapter may omit `api_key`
entirely as described below.

At config-load time, every resolved `master_key` and `api_key` must be a
nonempty UTF-8 string containing no ASCII control characters. Secret values
are preserved exactly and are never trimmed. A value that cannot be encoded by
its configured HTTP header credential transport is a startup configuration
error rather than a runtime target failure. Gemini query credentials are not
subject to HTTP-header encoding, but are structurally percent-encoded as query
values as specified in §2.4. Each validation error names the configuration path
and environment-variable name needed to correct it, but never includes the
resolved value.

Secret values are treated specially. `master_key`, provider API keys,
authorization headers, and access tokens are never printed or logged. The
`--validate` output shows the resolved routing table with all secret values
replaced by `[REDACTED]`.

`api_base` means a versioned provider base URL, not a complete operation URL.
For example, `https://example.test/v1` is valid for an OpenAI-compatible
provider; the adapter appends `/chat/completions`. The Gemini adapter likewise
appends its documented model and operation paths. A trailing slash is accepted
and normalized away.

`api_base` is validated at startup. It must be an absolute `http` or `https`
URL with a host and must not contain userinfo, a query string, or a fragment.
Every branded provider requires `https`, including when `api_base` overrides
its preset. Any target with an `api_key` also requires `https`. Plaintext
`http` is accepted only for an `openai_compatible/` target that omits
`api_key`; this is the sole v0.1 exception for keyless local or trusted-network
endpoints.

### 2.4 Provider identification

The prefix before `/` in `litellm_params.model` selects the provider adapter:

| Prefix               | Provider/preset                    | Adapter family                |
|----------------------|------------------------------------|-------------------------------|
| `openai/`            | OpenAI                             | OpenAI-compatible             |
| `mistral/`           | Mistral                            | OpenAI-compatible             |
| `deepseek/`          | DeepSeek                           | OpenAI-compatible             |
| `openai_compatible/` | Custom compatible endpoint         | OpenAI-compatible             |
| `anthropic/`         | Anthropic                          | Anthropic translation         |
| `gemini/`            | Google Generative Language API     | Gemini translation            |

The portion of `litellm_params.model` after the first `/` is the upstream-model
suffix. Every suffix must be nonempty, no larger than 256 UTF-8 bytes, and free
of ASCII control characters (`U+0000`–`U+001F` and `U+007F`).

For OpenAI-compatible and Anthropic targets, the validated suffix is preserved
as the same JSON string in the provider's upstream `model` field. A Gemini
suffix must additionally match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` and is
inserted as one URL path segment in the operation paths below.

Provider-specific prefixes are thin presets, not independent adapter
implementations. `openai/`, `mistral/`, and `deepseek/` select the same
OpenAI-compatible adapter and differ only in defaults such as `api_base`.
`openai_compatible/` selects that adapter without branded defaults and therefore
requires an explicit `api_base`. Its `api_key` is optional: when present the
adapter sends it as a bearer token, and when absent it sends no authorization
header. This supports local and trusted-network endpoints without dummy keys.

The following branded presets are normative. Paths are appended to the
normalized `api_base` exactly as shown:

| Prefix | Default `api_base` | Non-stream operation | Stream operation | Credential transport | Mandatory headers and protocol version |
|--------|--------------------|----------------------|------------------|----------------------|----------------------------------------|
| `openai/` | `https://api.openai.com/v1` | `POST /chat/completions` | `POST /chat/completions` with canonical `stream: true` | `Authorization: Bearer <api_key>` | `Content-Type: application/json`; REST version `v1` is pinned by the base URL; no additional version header |
| `mistral/` | `https://api.mistral.ai/v1` | `POST /chat/completions` | `POST /chat/completions` with canonical `stream: true` | `Authorization: Bearer <api_key>` | `Content-Type: application/json`; API version `v1` is pinned by the base URL; no additional version header |
| `deepseek/` | `https://api.deepseek.com` | `POST /chat/completions` | `POST /chat/completions` with canonical `stream: true` | `Authorization: Bearer <api_key>` | `Content-Type: application/json`; the unversioned public API is the pinned v0.1 preset and no version header is sent |
| `anthropic/` | `https://api.anthropic.com/v1` | `POST /messages` | `POST /messages` with native `stream: true` | `x-api-key: <api_key>` | `Content-Type: application/json` and `anthropic-version: 2023-06-01`; API path version `v1` is pinned by the base URL |
| `gemini/` | `https://generativelanguage.googleapis.com/v1beta` | `POST /models/{model}:generateContent` | `POST /models/{model}:streamGenerateContent?alt=sse` | `key=<api_key>` query parameter | `Content-Type: application/json`; API version `v1beta` is pinned by the base URL; no version header |

Here `{model}` is the validated Gemini suffix. It is inserted as one URL path
segment and is never interpreted as a slash-delimited resource name, a query
component, or an already percent-encoded value.

Operation query parameters and credentials are assembled structurally rather
than by string concatenation. For Gemini streaming, `alt=sse` and `key` are
distinct query parameters; the API key must be percent-encoded as a query value.

All requests send only the mandatory headers in this table plus ordinary HTTP
transport headers generated by the client library. Provider-specific optional
organization, project, beta, or client-identification headers are unsupported
in v0.1. No provider SDK or HTTP-library default may change these preset values.

An explicit `api_base` on a branded target replaces the preset base URL only.
It does not change the preset's operation paths, request streaming flag,
credential transport, mandatory headers, or fixed version-header values. The
operator must include any desired path-based API version in the override;
nano-llm neither appends an implicit version segment nor derives behavior from
the override's hostname.

### 2.5 Validation at load time

- `model_list` must contain at least one entry. An empty list is a fatal startup
  validation error naming `model_list`.
- Every `model_list` entry must have `model_name` and `litellm_params.model`.
- `model_name` is case-sensitive, 1–128 characters, and must match
  `[A-Za-z0-9][A-Za-z0-9._:/-]*`.
- `litellm_params.model` must have a recognized provider prefix followed by a
  nonempty suffix no larger than 256 UTF-8 bytes and containing no ASCII
  control characters.
- A `gemini/` suffix must additionally match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`.
- Branded provider prefixes require `api_key`. `openai_compatible/` may omit it.
- Every explicit `api_base` must satisfy the URL and transport rules in §2.3.
- `request_timeout`, `overall_timeout`, and every per-target `timeout` must be
  YAML integers representing whole seconds from 1 through 86,400 inclusive.
  Floats (including integral-looking values such as `30.0`), strings, booleans,
  nulls, and out-of-range integers are config errors that name the field path.
  `request_timeout` defaults to 30 and `overall_timeout` defaults to 120.
- `max_in_flight` must be a YAML integer from 1 through 65,535 inclusive and
  defaults to 64. Floats (including integral-looking values such as `64.0`),
  strings, booleans, nulls, and out-of-range integers are config errors that
  name `general_settings.max_in_flight`.
- All `os.environ/*` references must resolve.
- `master_key` and every present `api_key` must be environment references, not
  inline literals.
- `general_settings.master_key` is required unless `--no-auth` flag is passed
  explicitly (for local dev only — should warn loudly on startup if unset).
- Config errors are fatal at startup, not runtime. Never start serving with a
  broken model group silently dropped.
- Unknown top-level, `general_settings`, and `litellm_params` keys are config
  errors in v0.1. This gateway implements a documented LiteLLM-shaped subset;
  it must not silently accept settings that it does not honor.
- Duplicate key names at any YAML mapping depth are fatal configuration errors
  that name the duplicate's configuration path. This rule applies during both
  normal startup and `--validate`.

### 2.6 Accepted YAML subset and CLI-dependent validation

The configuration file contains exactly one YAML document. Multiple documents,
custom tags, aliases, anchors, and merge keys (`<<`) are rejected, as are
non-string mapping keys. These restrictions keep duplicate detection, error
paths, and file-order routing deterministic across parser implementations.

`general_settings` is required during authenticated operation. Under
`--no-auth`, it may be omitted entirely, in which case all of its non-auth
members use their documented defaults. If `general_settings.master_key` is
present under `--no-auth`, it is still resolved and validated; the flag disables
authentication, not validation of configured data. `--validate --no-auth`
applies these same rules and still rejects a non-loopback `--bind`, even though
it exits before opening a listener. Route grouping and all other semantic
validation occur after secret resolution.

---

## 3. Runtime Architecture

```
                    ┌─────────────────────────────────┐
                    │        HTTP Server (axum)        │
                    │  /v1/chat/completions            │
                    │  /v1/models                      │
                    │  /health                          │
                    └────────────────┬──────────────────┘
                                     │  bearer auth check on /v1/*
                                     │  (/health bypasses auth)
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
                    │  structured TargetError; Router advances
                    │  to next target
                    ▼
              upstream provider API
```

### 3.1 Crate layout (suggested)

```
nano-llm/
  Cargo.toml
  src/
    main.rs              # CLI entry, config load, server bootstrap
    config/
      mod.rs              # schema structs (serde), validation
      env_resolve.rs       # os.environ/VAR resolution
      routes.rs            # merge repeated model_name entries in file order
    server/
      mod.rs               # axum router setup
      auth.rs              # bearer token middleware
      handlers.rs          # /v1/chat/completions etc — thin, delegates to router
    router/
      mod.rs               # core dispatch loop: try target[i], on fail advance
      error.rs             # structured target-attempt failures
    providers/
      mod.rs               # Provider trait
      openai_compatible.rs  # OpenAI wire family + branded endpoint presets
      anthropic.rs          # Anthropic wire family
      gemini.rs             # Gemini wire translation + API-key transport
    translate/
      mod.rs
      openai_wire.rs        # canonical request/response types (OpenAI shape)
      anthropic_wire.rs     # Anthropic ↔ OpenAI translation
      gemini_wire.rs        # Gemini ↔ OpenAI translation
    streaming/
      mod.rs                # SSE passthrough + re-framing helpers
```

### 3.2 Canonical chat interface and the provider seam

The gateway owns a strict canonical v0.1 chat shape. It parses and validates an
inbound request once, before routing, then passes the same immutable canonical
request to every target. This prevents adapters from interpreting the client
request differently and makes the provider interface the router's test surface.

Supported request fields are:

| Field | v0.1 behavior |
|-------|---------------|
| `model` | Required; selects a configured model group. |
| `messages` | Required; ordered canonical messages. Supports leading text `system` and `developer` instructions, text `user` and `assistant` messages, assistant tool calls, and text `tool` results linked by tool-call ID. |
| `stream` | Optional boolean; defaults to `false`. |
| `stream_options` | Optional only when streaming; the sole supported member is boolean `include_usage`. |
| `max_tokens` | Optional positive integer. Omission is preserved; the gateway does not invent a default. |
| `max_completion_tokens` | Optional alias for `max_tokens`. Supplying both fields is a 400 validation error. |
| `temperature` | Optional number from 0 through 1, the portable range across v0.1 providers. |
| `top_p` | Optional number from 0 through 1. |
| `stop` | Optional string or list of up to four nonempty strings, each at most 256 UTF-8 bytes. A single string is normalized to a one-element list. |
| `tools` | Optional OpenAI-shaped function-tool definitions. The portable v0.1 subset is translated by every adapter. |
| `tool_choice` | Optional; supports `auto` (default when tools are present), `none`, `required`, or one specifically named function in OpenAI's object form. |

`max_tokens` and `max_completion_tokens` accept JSON integers from 1 through
2,147,483,647. Booleans, floats, strings, nulls, zero, negative values, and
larger integers are validation errors. Canonical and emitted usage counters are
unsigned 64-bit integers; a provider value outside that range is invalid usage
and is handled under the usage-omission rule below.

The following nested request shapes are normative. Each object admits only the
members shown here, except for the opaque contents of `parameters`:

| Shape | Required members | Optional members |
|-------|------------------|------------------|
| `system`, `developer`, or `user` message | `role`, string `content` | none |
| `assistant` message | `role` | string-or-null `content`, nonempty `tool_calls` |
| `tool` message | `role`, string `tool_call_id`, string `content` | none |
| assistant tool call | string `id`, `type: "function"`, `function` | none |
| assistant tool-call `function` | string `name`, string `arguments` | none |
| tool definition | `type: "function"`, `function` | none |
| tool-definition `function` | string `name` | string `description`, object `parameters` |
| named `tool_choice` | `type: "function"`, `function` | none |
| named-choice `function` | string `name` | none |
| `stream_options` | boolean `include_usage` | none |

An assistant message must contain string content and/or at least one tool call;
`content: null` and omitted `content` are equivalent canonically. Empty strings
are valid content. `description` may be empty. An omitted `parameters` member is
preserved as omitted rather than replaced with an invented schema.

`messages` must be a nonempty JSON array. For `system`, `developer`, `user`,
and `tool` messages, `content` is required and must be a JSON string. For an
`assistant` message, `content` may be a JSON string; it may be `null` or
omitted only when the message contains at least one tool call. Every assistant
message must contain string content and/or at least one tool call. Content
arrays are unsupported in v0.1, including arrays containing only OpenAI
text-content parts, and return a gateway 400 before routing.

If the selected model group contains any Anthropic target, the request must
include exactly one of `max_tokens` or `max_completion_tokens`; omission is a
gateway 400 validation error before routing. This group-level rule applies to
Anthropic-only and mixed fallback routes so the same immutable canonical
request is valid for every target. The gateway does not invent an Anthropic
output-token default.

All other request fields are rejected with an OpenAI-shaped 400 error naming
the unsupported field. In particular, v0.1 does not silently discard
multimodal parts, response schemas, provider-specific extensions, or unknown
fields. This is deliberately narrower than the full OpenAI interface.

This strictness applies recursively to every gateway-defined request object,
including message objects, assistant tool calls, tool definitions, and
`tool_choice` objects. An unknown member at any of these levels is a gateway
400 `invalid_request` error rather than being ignored or forwarded to an
adapter. The sole opaque nested region is
`tools[].function.parameters`: after validating that it is a JSON object, the
gateway preserves all of its members as schema content. For a nested validation
error, `param` is the containing top-level field (`messages`, `tools`, or
`tool_choice`), and the safe error `message` identifies the precise nested path.

Duplicate JSON member names are rejected at every object depth before semantic
deserialization, including within `tools[].function.parameters`. This is a
syntax-level uniqueness rule and does not interpret or restrict JSON Schema
semantics. A duplicate returns 400 `invalid_request` before routing. For a
top-level duplicate, `param` is that duplicated field; for a nested duplicate,
`param` is its containing top-level field. The safe error `message` identifies
the precise duplicate path.

`system` and `developer` messages are accepted only as a leading instruction
prefix before the first `user`, `assistant`, or `tool` message. Every adapter
combines their content strings in request order using exactly two newline
characters (`\n\n`) between adjacent strings. It preserves every content string
verbatim, including empty strings, inserts no role labels or other text, and
sends the resulting single string through the provider's system-instruction
field. Either role appearing after the conversation begins is a 400 validation
error. `developer` is an inbound compatibility role, not a distinct provider
capability.

After that optional instruction prefix, the conversation must begin with a
`user` message and alternate user-side and assistant turns. A single `user`
message is one user-side turn. An immediately following contiguous group of
`tool` results that resolves the preceding assistant tool calls is also one
user-side turn. Consecutive `user` messages, consecutive assistant messages,
an initial assistant or tool message, and a `user` message immediately after a
tool-result group are validation errors on `messages`. The request must end on
a user-side turn (`user` or a complete tool-result group), because the endpoint
is asking the provider to generate the next assistant turn. This portable state
machine avoids relying on adapter-specific role-coalescing behavior.

Tool calling is part of the canonical interface rather than a provider-specific
extension. Tool definitions use the OpenAI function-tool shape; assistant tool
calls and tool-result messages are normalized into the same shape across
providers. A route is valid only if every adapter family used by it implements
this canonical tool contract. Responses may contain multiple tool calls, and
streaming adapters must preserve their indices while assembling argument
deltas. The `parallel_tool_calls` request field is rejected in v0.1: nano-llm
can represent multiple calls emitted by a provider but does not promise portable
control over whether providers generate them in parallel.

`tools`, when present, must be a nonempty array containing at least one valid
function definition. `tool_choice` is rejected when `tools` is absent. When
`tool_choice` is omitted, the canonical value is `auto` when tools are present
and `none` when tools are absent. A specifically named choice must match one of
the functions declared in `tools`. Violations are gateway 400 errors before
routing.

The gateway structurally validates each tool definition: `type` must be
`function`, the function name must match `[A-Za-z0-9_-]{1,64}` and be unique
within the request, and
`parameters` must be a JSON object when present. The contents of `parameters`
are otherwise treated as opaque JSON Schema and preserved while the surrounding
provider wire format is translated. nano-llm does not implement its own JSON
Schema validator, restrict schemas to a gateway-defined keyword subset, resolve
`$ref`, or rewrite schemas for provider compatibility in v0.1. Advanced-schema
portability across fallback targets is therefore best effort; a provider schema
rejection is a `TargetError` and advances to the next route entry.

Tool-call history is validated as a portable state machine. Every assistant
tool call must have a nonempty ID unique within the request and must name a tool
declared by the current request. A following `tool` message must reference an
unresolved call from that immediately preceding assistant tool-call turn. Each
call receives exactly one result, multiple results may arrive in any order, and
all calls must be resolved before another `user` or `assistant` turn begins.
Violations are gateway validation errors and return 400 before routing.

For inbound tool-call history, `function.arguments` is normally an opaque
string. If the selected route contains an Anthropic or Gemini target, every such
string must parse as exactly one JSON object with no trailing data; failure is a
gateway 400 on `messages` before routing. This is route-level canonical
validation, parallel to the Anthropic output-token rule, and ensures the same
request is representable by every fallback target. OpenAI-compatible adapters
preserve the original argument string unchanged. Translation adapters use the
parsed object without reinterpreting its schema.

For provider responses, `function.arguments` remains an opaque canonical
string. OpenAI-compatible adapters preserve it unchanged even when it is not
valid JSON, and nano-llm does not reject an otherwise well-formed successful
OpenAI-compatible response solely because a model emitted malformed JSON
arguments. Native object arguments from Anthropic or Gemini are serialized as
compact JSON without relying on object-member order.

The canonical non-streaming response contains `id`, `object`, `created`,
`model`, `choices`, and `usage` when the upstream reports usage. Each choice
contains `index`, an assistant message with text content and/or normalized tool
calls, and a normalized `finish_reason`.

The exact non-streaming choice shape is `index`, `message`, and
`finish_reason`. The exact assistant response-message shape is
`role: "assistant"`, `content`, and optional `tool_calls`: `content` is always
present and is either a string or `null`; `tool_calls` is present only when
nonempty and uses the assistant tool-call shape above. The response contains no
provider-specific members. `usage` is omitted when unavailable or invalid,
never emitted as `null`.

Because v0.1 does not accept a multi-choice request field, every successful
non-streaming canonical response contains exactly one choice with `index: 0`.
Every ordinary canonical streaming `ChatChunk` likewise contains exactly one
choice with `index: 0`; the final gateway-generated usage chunk described below
is the sole permitted `choices: []` exception. The terminal finish reason
belongs to choice 0.

An upstream response or ordinary stream chunk with zero choices, multiple
choices, or any nonzero or duplicate choice index is an `InvalidResponse`.
For a non-streaming response or an uncommitted stream this is a `TargetError`
and routing may advance to the next target. If it occurs after streaming
commitment, the gateway closes the stream without fallback under §4.

Canonical `usage` contains exactly three nonnegative integers:
`prompt_tokens`, `completion_tokens`, and `total_tokens`. Adapters map native
input or prompt counts to `prompt_tokens` and native output or candidate counts
to `completion_tokens`. `total_tokens` is their checked sum; overflow makes the
usage data invalid. Both component counts must be present and valid before the
gateway includes usage. If either is absent, non-integer, negative, or otherwise
invalid, the gateway omits the complete usage object and logs the condition
safely at debug level without failing an otherwise valid completion.
Provider-specific usage detail fields are not exposed in the canonical body.

Response metadata is gateway-owned. For each successful selected attempt, the
gateway generates one opaque unique ID beginning with `chatcmpl-` and one
`created` value equal to the current Unix timestamp in integer seconds. A
non-streaming response uses `object: "chat.completion"`. Every chunk in a
stream uses `object: "chat.completion.chunk"` and repeats the same gateway ID,
`created` timestamp, and client-requested `model`, including a final usage
chunk. Provider response IDs are not exposed in the canonical body. The
selected provider and upstream model remain available in structured logs.

Streaming produces OpenAI-shaped chat completion chunks with role, content,
and tool-call deltas, a final finish reason, and exactly one `data: [DONE]` on
a successful stream. A successful stream must yield at least one valid
canonical `ChatChunk` before its terminal signal; terminal-only output is not
a successful empty response. When `stream_options.include_usage` is true and
valid component counts are available, the gateway emits exactly one final
usage chunk immediately before `data: [DONE]`. This chunk repeats the stream's
gateway-owned `id`, `object`, `created`, and `model`, and contains exactly
`choices: []` plus the canonical `usage` object. When usage was not requested
or valid component counts are unavailable, the gateway emits no usage chunk
and completes normally; absent or malformed provider usage never terminates an
otherwise valid stream. Other `stream_options` members are rejected.

Every ordinary chunk contains exactly `id`, `object`, `created`, `model`, and
one choice with exactly `index`, `delta`, and `finish_reason`. `finish_reason`
is `null` on nonterminal chunks and one normalized value on the sole terminal
chunk. The first ordinary chunk must include `delta.role: "assistant"`; an
adapter synthesizes it when the native protocol does not. A delta may otherwise
contain string `content` and/or nonempty `tool_calls`. Empty native events that
would produce none of these members are ignored.

A streaming tool-call delta contains `index` and may contain `id`,
`type: "function"`, or a `function` object containing partial string `name`
and/or `arguments`. Indices start at zero, are introduced in increasing order,
and identify one call for the life of the stream. The adapter assembles each
call internally while forwarding deltas, rejects a changed ID or name, and at
the terminal chunk verifies that every call has a nonempty ID, a declared tool
name, and a complete native argument value. Exactly one terminal chunk is
required; an ordinary chunk after it, a second terminal chunk, or `[DONE]`
before it is an invalid stream.

Final finish reasons are normalized to four OpenAI-shaped values: normal end or
stop-sequence completion becomes `stop`, an output limit becomes `length`, tool
invocation becomes `tool_calls`, and a safety/policy block becomes
`content_filter`. An unknown terminal reason on an otherwise well-formed 2xx
response becomes `stop`; the safe upstream reason is logged at debug level and
does not trigger fallback.

The adapter for a configured target satisfies this interface:

```rust
#[async_trait]
trait Provider: Send + Sync {
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, TargetError>;

    async fn complete_stream(&self, req: &ChatRequest)
        -> Result<BoxStream<'static, Result<ChatChunk, TargetError>>, TargetError>;
}
```

Each adapter is constructed with one target's validated configuration, so the
router does not need to understand provider URLs, authentication, or wire
formats. Anthropic and Gemini adapters translate at this seam. Mistral,
DeepSeek, and OpenAI are near-passthrough adapters, but still validate and
normalize responses through the same interface.

### 3.3 Target errors and the public failure contract

Canonical request validation happens before routing. Once routing begins, every
provider failure has the same control-flow result: record the failure and
advance to the next route entry. Error kinds exist for diagnostics, not to
create separate routing policies.

```rust
struct TargetError {
    kind: TargetErrorKind,
    upstream_status: Option<u16>,
    safe_message: String,
}

enum TargetErrorKind {
    Timeout,
    ConnectionError,
    RateLimited,
    Authentication,
    PermissionDenied,
    RejectedRequest,
    InvalidResponse,
    Overloaded,
    UpstreamHttp,
}
```

Connection failures, timeouts, upstream 3xx/4xx/5xx responses, authentication
failures, provider rejections, overload responses, malformed bodies, and stream
setup failures all become `TargetError`. The router logs each kind safely and
advances. Adapters must not include secrets or raw upstream response bodies in
`safe_message`.

Any well-formed successful provider response ends routing. The gateway does not
fall back because of a safety refusal, content-filter finish reason,
length-limited output, empty text accompanied by valid tool calls, or a model's
natural-language refusal. nano-llm evaluates protocol success, not semantic
quality or policy outcomes.

The upstream HTTP client must not follow redirects. Any 3xx response becomes a
`TargetError` so bearer tokens and query-string API keys cannot be forwarded to
an unexpected host. Configured base URLs must point directly at the intended API
endpoint family.

The gateway owns the client-visible error contract. Unsupported or malformed
inbound media types return 415, malformed JSON returns 400, gateway validation
errors return 400, expiration of the request-body deadline returns 408, a
request body larger than 1 MiB returns 413, an unknown requested model returns
404, process-wide concurrency exhaustion returns 503, and expiration of
`overall_timeout` returns 504. If every route entry fails before that deadline,
return a 502 error with a stable gateway message.

Every error response on `/v1/*` uses `Content-Type: application/json` and this
exact envelope:

```json
{"error":{"message":"<safe string>","type":"<type>","param":null,"code":"<code>"}}
```

All four members are always present. `message` is a gateway-owned safe string;
`type` is one of `invalid_request_error`, `authentication_error`,
`not_found_error`, or `server_error`; `param` is the offending top-level
request field when applicable and `null` otherwise; and `code` is the stable
gateway-owned condition code from this mapping:

| Condition | HTTP status | `type` | `code` | `param` |
|-----------|-------------|--------|--------|---------|
| Request validation failure | 400 | `invalid_request_error` | `invalid_request` | offending top-level field, or `null` |
| Invalid UTF-8, malformed or non-object JSON, or trailing non-whitespace data | 400 | `invalid_request_error` | `invalid_json` | `null` |
| Authentication failure | 401 | `authentication_error` | `authentication_failed` | `null` |
| Unknown requested model | 404 | `not_found_error` | `model_not_found` | `model` |
| Unknown or unimplemented `/v1/*` route | 404 | `not_found_error` | `route_not_found` | `null` |
| Request body is not completely buffered within 30 seconds of generation-permit acquisition | 408 | `invalid_request_error` | `request_body_timeout` | `null` |
| Request body exceeds 1 MiB | 413 | `invalid_request_error` | `request_too_large` | `null` |
| Missing, repeated, comma-combined, malformed, or unsupported `Content-Type` | 415 | `invalid_request_error` | `unsupported_media_type` | `null` |
| All route entries fail | 502 | `server_error` | `upstream_exhausted` | `null` |
| Concurrency capacity exhausted | 503 | `server_error` | `capacity_exhausted` | `null` |
| `overall_timeout` expires | 504 | `server_error` | `overall_timeout` | `null` |

Upstream error objects are never copied into this envelope. Do not expose the
final provider's status merely because it happened to be last, and do not name
provider credentials or include raw upstream bodies in the response.

### 3.4 One attempt per route entry

The router attempts each route entry at most once and advances immediately on
any `TargetError`. Gateway request validation happens before this loop. There is
no implicit same-target retry, backoff, jitter, or `Retry-After` scheduling.
Operators who intentionally want another attempt against the same target repeat
that target as another entry in the route. This keeps attempt count and order
visible in configuration.

The maximum number of upstream attempts is therefore the number of entries in
the route, subject to the whole-request deadline.

`request_timeout` defaults to 30 seconds and may be overridden per target with
`litellm_params.timeout`. For non-streaming calls it limits one complete
upstream attempt. For streaming calls it first limits connection plus time to
the first canonical chunk; after commitment, it becomes an idle timeout that
resets after every canonical chunk. A committed stream that exceeds this idle
timeout is closed without fallback. `overall_timeout` defaults to 120 seconds
and starts only after the complete request body has been buffered and the
request has been parsed and successfully canonicalized. Inbound upload,
parsing, and validation time do not consume this budget. It covers the fallback
chain only until a response is committed. Each pre-commit attempt is clipped to
the time remaining in that overall deadline. There is no total generation-
duration timeout for a stream that continues making progress.

Only an emitted canonical chunk resets the committed-stream idle timer. Native
keepalives, comments, metadata, usage-only events, and other events consumed
internally by an adapter do not reset it. Time spent decoding and validating an
event is part of the same idle interval. Before commitment, the effective
attempt deadline is `min(request_timeout, remaining overall_timeout)`; if both
expire simultaneously, `overall_timeout` wins the public error classification.

The gateway cancels the active upstream request when the downstream client
disconnects and does not continue falling back. Fallback can cause more than one
provider to begin a billable generation when a connection fails after the
upstream accepted a request. v0.1 does not promise cross-provider
deduplication.

---

## 4. Streaming & Failover Interaction

This is the sharpest edge in the whole design, worth calling out explicitly.

### 4.1 Rule

**Failover is only possible before the first canonical SSE chunk and downstream
response headers have been sent to the client.** Once the response is
committed, a mid-stream provider failure terminates the stream; it never causes
a silent retry, because:
- The client may have already rendered/acted on partial tokens.
- Re-issuing the same prompt against a different provider mid-stream would
  either duplicate content or produce an incoherent transcript.

### 4.2 Implementation approach

The gateway obtains and validates the **first canonical chunk** from the
upstream stream before constructing the downstream streaming response. This
ensures axum has not sent a 200 status or SSE headers while failover is still
possible. It costs one chunk of latency but means:
- If target 1's stream fails/errors before yielding any chunk (including an
  immediate 4xx/5xx on the streaming request itself), the gateway can still
  transparently advance to target 2 — client never knows target 1 was tried.
- SSE comments, provider keepalives, and empty frames are ignored and do not
  count as the first canonical chunk. A valid role-only OpenAI chunk does count.
- A terminal signal received before any canonical chunk is an
  `InvalidResponse` `TargetError`. The response is still uncommitted, so the
  router advances to the next target.
- Once the first canonical chunk and response headers are sent, the gateway is
  "locked in" to that upstream for the rest of the request.

```
connect to target[i] (streaming)
  ├─ error before first canonical chunk? → classify, stop, or fall back
  └─ first canonical chunk validated?
       → forward it, then pipe remaining chunks directly (locked in)
       → upstream fails or emits malformed data? → close the stream. Do NOT
         emit a nonstandard error event or advance to target[i+1].
```

The streaming adapter normalizes provider termination into exactly one OpenAI
`data: [DONE]` event. If the provider ends without a valid terminal event, the
gateway closes the stream without inventing `[DONE]`. A downstream disconnect
cancels the upstream stream immediately. Each decoded upstream SSE event is
limited to 1 MiB. Exceeding the limit before commitment is a `TargetError` and
may fall back; after commitment it closes the stream. Total stream bytes are not
capped because the response is processed incrementally.

Native usage-only events are adapter metadata, not ordinary canonical chunks.
In particular, an OpenAI-compatible upstream `choices: []` usage event is
consumed by the adapter and retained for the optional gateway-generated usage
chunk; it is not subjected to the ordinary zero-choice rejection rule and is
never forwarded directly. Any other zero-choice native event is ignored only
when the adapter specification identifies it as metadata; otherwise it is an
`InvalidResponse`. On success, the downstream response uses status 200 and
headers `Content-Type: text/event-stream` and `Cache-Control: no-cache`; each
canonical chunk is encoded as one `data: <compact-json>\n\n` event and success
ends with exactly `data: [DONE]\n\n`.

### 4.3 Non-streaming requests

The gateway buffers and validates the full response before returning it. A
failure while reading or decoding the body is classified like any other
pre-commit provider failure, so the router may advance to the next target.
This can duplicate a generation, as described in §3.4. A buffered upstream body
is limited to 8 MiB; exceeding that fixed limit is a `TargetError`.

---

## 5. Provider Adapters — Notes Per Provider

### 5.1 Normative translation contract

Adapters preserve conversational order and content bytes; they may regroup
adjacent canonical messages into native content blocks only when the provider
wire format requires it. Regrouping inserts no separator text, role label, or
synthetic conversational content. A provider-specific inability to represent a
request that passed the route-level rules is an adapter defect, not permission
to silently omit a field.

| Canonical concept | OpenAI-compatible family | Anthropic | Gemini |
|-------------------|---------------------------|-----------|--------|
| Combined leading instructions | One leading `system` message | Top-level `system` string | `systemInstruction` text part |
| User text | `user` message | User text content block | `user` text part |
| Assistant text | `assistant` message | Assistant text content block | `model` text part |
| Assistant tool call | OpenAI `tool_calls` unchanged | Assistant `tool_use` block | Model `functionCall` part |
| Tool result | `tool` message keyed by `tool_call_id` | User `tool_result` block keyed by native call ID | User `functionResponse` part keyed by function name |
| `auto` / `none` / `required` / named choice | Native equivalent | Native equivalent | Native equivalent |
| `max_tokens` | Provider's supported OpenAI-compatible output-token field | `max_tokens` | Native maximum-output-token field |
| `stop` | `stop` scalar/list as supported by the shared wire family | Native stop-sequence list | Native stop-sequence list |
| `temperature`, `top_p` | Same-named fields | Same semantic native fields | Same semantic native fields |

For a Gemini tool result, the adapter resolves the canonical `tool_call_id` to
the function name recorded on the immediately preceding assistant call. If a
native provider response has a tool call but no stable call ID, the adapter
generates one opaque ID beginning with `call_`, uses the same ID for all deltas
of that call, and exposes it in the final canonical history. Native IDs are
preserved when present and valid. Generated IDs carry no provider meaning and
need only be unique within the response.

Adapters must have conformance tests for every row above in both request and
response directions, including multiple tool calls and multiple results in
non-call order. Provider request-field names and native event names belong to
the adapter implementation and its tests; they do not leak through the
canonical interface.

- **OpenAI-compatible family**: one near-passthrough adapter serves `openai/`,
  `mistral/`, `deepseek/`, and `openai_compatible/`. Canonical format is the
  OpenAI shape. Branded prefixes provide default base URLs; the generic prefix
  requires `api_base`. Authentication remains configurable per target, with the
  branded v0.1 presets using bearer tokens. Provider presets must not fork the
  wire implementation.
- **Anthropic**: real translation required. Messages API has a distinct
  request shape (`system` as top-level field not a message role,
  `max_tokens` required, content blocks, different streaming event names —
  `message_start`, `content_block_delta`, etc. instead of OpenAI's
  `choices[].delta`). This is the first adapter to write after OpenAI, since
  it exercises the translation layer fully.
- **Gemini (Generative Language API)**: `generateContent` /
  `streamGenerateContent` endpoints, API key as query param (`?key=`), request
  shape uses `contents[].parts[]`, roles are `user`/`model` not
  `user`/`assistant`. Translation is required for messages, tool calls, tool
  results, responses, and streaming events.

**Suggested build order for adapters:** OpenAI → Mistral → DeepSeek (these
three validate the server/router/config plumbing with near-zero translation
work) → Anthropic (exercises translation) → Gemini (exercises translation +
different auth).

---

## 6. HTTP Surface

### 6.1 Process CLI

```text
nano-llm --config <path> [--bind <address>] [--no-auth] [--validate]
```

- `--config` is required and names the YAML configuration file.
- `--bind` defaults to `127.0.0.1:4000`; external/container exposure must be
  requested explicitly, for example `--bind 0.0.0.0:4000`.
- `--validate` loads and validates configuration, prints the redacted resolved
  route table, and exits without binding a socket.
- `--no-auth` is for local development and is rejected unless the bind address
  is loopback.
- Routes, providers, keys, and timeout values cannot be overridden by CLI flags;
  YAML remains the single operational source of truth.
- On SIGINT or SIGTERM, the server stops accepting new connections and allows
  active requests and streams to finish. nano-llm has no internal shutdown
  deadline setting; the process supervisor or container runtime may impose a
  hard deadline externally.
- The listener serves plain HTTP. nano-llm has no inbound certificate or TLS
  configuration in v0.1; remote deployments must terminate TLS externally.
  Connections from nano-llm to public providers continue to use verified HTTPS
  with bundled trust roots.

### 6.2 Endpoints

For `POST /v1/chat/completions`, after successful inbound authentication and
before generation-permit acquisition or body buffering, the gateway requires
exactly one `Content-Type` header. Its media type is matched
ASCII-case-insensitively to `application/json`. The only permitted parameter is
a single `charset=utf-8`, whose name and value are also matched
ASCII-case-insensitively. An absent, repeated, comma-combined, malformed, or
unsupported value returns the canonical HTTP 415 `unsupported_media_type`
error with `type: invalid_request_error` and `param: null`, without route lookup,
permit acquisition, body buffering, or provider work.

Request bodies are limited to a fixed 1 MiB in v0.1. After inbound
authentication, `Content-Type` validation, and generation-permit acquisition,
the gateway races complete body buffering against a fixed 30-second deadline
and the 1 MiB limit. The first violation determines the response. If the
deadline expires first, the gateway stops reading the body, releases the
generation permit, and returns an HTTP 408 error with `type:
invalid_request_error`, `code: request_body_timeout`, and `param: null`.
Neither limit is configurable. After the complete body is buffered, it must
decode as UTF-8 and contain exactly one JSON object with no trailing
non-whitespace data. Invalid UTF-8, malformed JSON, a non-object top-level
value, or trailing non-whitespace data returns the canonical HTTP 400
`invalid_json` error with `type: invalid_request_error` and `param: null`,
releases the generation permit, and performs no route lookup or provider work.
Canonical validation then begins; `overall_timeout` starts only after parsing
and canonical validation both succeed.

The process also enforces `general_settings.max_in_flight`, default 64, with a
single semaphore constructed only from the load-time validated value. It is
applied only to `POST /v1/chat/completions` and acquired before buffering the
request body. One generation request holds one permit for its whole lifetime,
including fallback attempts and streaming. If no permit is immediately
available, return an OpenAI-shaped 503; v0.1 does not maintain an internal
waiting queue. `/health` and `/v1/models` bypass this semaphore so they remain
available during generation saturation. This is a resource guard, not a
per-key quota or rate limiter.

For every inbound request, the gateway selects the request ID at the start of
application handling, before authentication or any other application
processing, so early failures receive the same correlation behavior.

A client value is accepted only when exactly one raw `x-request-id` field is
present and its value is 1–128 bytes, all in the ASCII range `0x21`–`0x7E`
inclusive. The gateway preserves accepted bytes exactly, without trimming,
normalization, or comma splitting; a comma contained in the single header
field is therefore a literal permitted character, not list syntax. When the
ID is absent, repeated, or invalid, the gateway generates a lowercase
canonical UUIDv4 string and never echoes any rejected value.

The gateway attaches the chosen ID to every log record for the request and sets
exactly one `x-request-id` response header on every application response,
including health, authentication, validation, routing, and other error
responses. The gateway request ID is not forwarded upstream in v0.1.

| Endpoint                  | Behavior                                                   |
|----------------------------|--------------------------------------------------------------|
| `POST /v1/chat/completions`| Main entry point. `model` field in body selects the group.  |
| `GET /health`              | Liveness only — does NOT check upstream provider health (no circuit breaker state to report). |
| `GET /v1/models`           | Lists configured model-group names in the OpenAI models-list envelope. |

Only the method/path pairs in this table are registered. Any other method on a
listed path, and any other path, returns 404 rather than 405. Under `/v1/*` it
uses the canonical `route_not_found` envelope and is authenticated before route
resolution. Outside `/v1/*`, including a method mismatch on `/health`, it
returns that same envelope without authentication. All such responses still
carry the selected `x-request-id`.

Whenever the process can serve HTTP, `GET /health` returns HTTP 200 with
`Content-Type: application/json` and the exact body `{"status":"ok"}`. It
never inspects configured targets or upstream health.

`GET /v1/models` returns exactly one model object for each unique configured
`model_name`, using this exact envelope and field set for every entry:

```json
{"object":"list","data":[{"id":"<model_name>","object":"model","created":0,"owned_by":"nano-llm"}]}
```

Entries are ordered by the first appearance of each `model_name` in
`model_list`. Repeated fallback targets never create duplicate public entries.

`/v1/completions` and `/v1/embeddings` are not implemented in v0.1 and return
404. Adding either later requires its own canonical request/response types and
provider support matrix; it must not be squeezed through the chat interface.

When authentication is enabled, every `/v1/*` request must contain exactly one
`Authorization` header field. Its authentication scheme is matched
ASCII-case-insensitively to `Bearer` and must be followed by exactly one ASCII
space (`0x20`). Every byte after that delimiter is the credential and is
compared exactly, without trimming or normalization, to `master_key` using a
constant-time equality operation.

Absent, repeated, comma-combined, malformed, non-Bearer, and mismatched values
all return the same 401 `authentication_failed` envelope before route lookup,
generation-permit acquisition, body buffering, or provider work. Under
`--no-auth`, the gateway skips the authentication check entirely. `/health`
is always unauthenticated so a local supervisor or container runtime can
perform a liveness check. `--no-auth` remains restricted to loopback bind
addresses.

---

## 7. Example configuration

```yaml
model_list:
  - model_name: fast
    litellm_params:
      model: mistral/mistral-small-latest
      api_key: os.environ/MISTRAL_API_KEY

  - model_name: fast
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  overall_timeout: 120
```

The two `fast` entries form one fallback route in file order.

---

## 8. Suggested Milestones

1. **Config loader + validation** — parse the schema above, resolve env vars,
   merge repeated names into ordered routes, fail loudly on bad config. No server yet — a CLI
   subcommand that just loads and pretty-prints the resolved routing table is
   a good first deliverable and a genuinely useful `--validate` flag long-term.
2. **Canonical chat interface + OpenAI adapter** — validate the documented v0.1
   request subset, then serve non-streaming `/v1/chat/completions` against one
   target. Prove the axum server, auth middleware, and canonical types work end
   to end against one real provider.
3. **Failover loop** — wire the router's target-iteration logic using
   a mock/fault-injecting provider for tests (this is where you want good
   unit tests — simulate target 1 timing out, target 2 returning 429, target
   3 succeeding, and assert the client only ever sees the final success).
4. **Streaming** — SSE passthrough for OpenAI first, then the buffered-first-chunk
   failover logic from §4.
5. **Mistral + DeepSeek adapters** — should be quick, near-identical to OpenAI.
6. **Anthropic adapter** — first real translation layer.
7. **Gemini adapter** — second translation layer, static API key auth.
8. **Polish** — `/health`, `/v1/models`, structured logging, ARM cross-compile,
   and Docker `FROM scratch` build for the Pi.

---

## 9. Decisions for v0.1

- **Release boundary:** v0.1 means the complete surface in this document. The
  milestones are development slices, not smaller conforming releases; partial
  builds identify themselves as development builds.
- **Product boundary:** nano-llm serves a single operator or small trusted team
  needing a self-hosted reliability shim across a handful of providers. It is
  deliberately not an organization-wide AI platform. One process, one config
  file, one shared inbound key, static routing, and operational logs are in
  scope; tenants, user administration, policy engines, metering, and analytics
  are not.
- **LiteLLM relationship:** configuration is LiteLLM-shaped for migration
  familiarity, not LiteLLM-compatible or drop-in. nano-llm implements and
  validates its own documented subset; unsupported settings must be removed or
  translated rather than silently accepted.
- **Vertex timing:** Vertex AI is deferred until after v0.1. The first release
  establishes Gemini translation through the API-key Generative Language API
  without taking on ADC, OAuth token lifecycle, workload identity, and
  project/location endpoint construction.
- **Fallback syntax:** entries sharing a `model_name` form one ordered route in
  file order. v0.1 has no `general_settings.fallbacks` or other cross-group
  references; reuse across routes requires repeating the target configuration.
- **Model names:** public `model_name` aliases are case-sensitive, 1–128
  characters, and match `[A-Za-z0-9][A-Za-z0-9._:/-]*` so they remain safe in
  configuration output, API responses, and logs.
- **Routing policy:** routes use fixed priority. Every request begins at the
  first entry, and later entries receive traffic only after a failure. v0.1 has
  no load balancing, weights, or dynamic cost/latency selection.
- **Cross-request state:** there is no circuit breaker, health score, or passive
  cooldown. Each request starts from the first route entry independently. A
  failing primary may therefore add its timeout to every request until the
  operator lowers that timeout, reorders the route, or restores the target.
- **Config lifecycle:** configuration and referenced environment values are
  loaded and validated once at startup, then remain immutable for the process
  lifetime. v0.1 has no SIGHUP or API-driven reload; applying changes requires
  a process restart.
- **YAML subset:** accept exactly one document with string mapping keys and
  reject custom tags, aliases, anchors, merge keys, and duplicate keys.
  Environment-reference syntax is active only in `master_key` and `api_key`.
  Under `--no-auth`, `general_settings` may be omitted; configured values are
  still validated even when authentication is disabled.
- **CLI:** the process exposes only `--config`, `--bind`, `--validate`, and
  `--no-auth`. The config path is required, bind defaults to
  `127.0.0.1:4000`, validation exits before serving, and no routing or provider
  setting can be overridden from the command line.
- **Shutdown:** SIGINT and SIGTERM trigger graceful server shutdown: stop
  accepting new work and allow in-flight requests/streams to finish. There is
  no gateway shutdown-timeout knob; supervisors own any hard termination
  deadline.
- **TLS boundary:** the inbound listener is HTTP-only. TLS certificates,
  termination, and ACME belong to an external reverse proxy, tunnel, or load
  balancer. Upstream provider traffic still uses verified HTTPS with bundled
  trust roots.
- **Timeout:** `request_timeout` defaults to 30 seconds and is overridable
  globally and per target. `overall_timeout` defaults to 120 seconds for the
  entire pre-commit attempt/fallback chain. For a stream, `overall_timeout` ends
  at commitment; `request_timeout` then resets after each canonical chunk and
  closes a stream that remains idle for too long. Progressing streams have no
  total generation-duration limit.
- **Output-token default:** `max_tokens` is generally optional and has no
  gateway default. If the selected model group contains any Anthropic target,
  the request must instead include exactly one of `max_tokens` or
  `max_completion_tokens`; omission returns a gateway 400 before routing. For
  groups without Anthropic, omission lets each provider/model apply its native
  behavior. nano-llm does not impose an arbitrary truncation limit in pursuit
  of false cross-provider equivalence.
- **Output-token aliases:** clients may send `max_tokens` or
  `max_completion_tokens`, but not both. Either is normalized into one canonical
  output-token limit and mapped to the selected provider's preferred field.
- **Temperature range:** canonical `temperature` is limited to 0 through 1 so
  every v0.1 provider family can represent it. Values above 1 are rejected
  before routing even when a particular upstream would accept them.
- **Stop sequences:** accept one string or a list of at most four nonempty
  strings, each no larger than 256 UTF-8 bytes. Normalize the scalar form to a
  one-element list before routing.
- **Compatibility surface:** v0.1 implements text chat plus portable function
  tool calling through `/v1/chat/completions`, along with `/v1/models` and
  `/health`. Legacy completions, embeddings, multimodal input, structured
  output, and provider-specific extensions are deferred and rejected rather
  than silently degraded.
- **Message content:** `messages` is a nonempty array. `system`, `developer`,
  `user`, and `tool` messages require string content. Assistant content is a
  string, or may be null or omitted only when the message has tool calls.
  Every assistant message contains string content and/or tool calls. All
  content arrays, including text-only part arrays, return 400 before routing.
- **Streaming usage:** `stream_options.include_usage` is supported as a
  best-effort passthrough. If requested, a final usage chunk is emitted only
  when the selected provider supplies streaming usage; missing values are not
  estimated. Native usage-only events are consumed as metadata and only the
  gateway emits a zero-choice usage chunk. All other `stream_options` fields
  are rejected.
- **Tool-choice boundary:** `tools`, when present, is nonempty. `tool_choice`
  is rejected when `tools` is absent. Omission normalizes to `auto` with tools
  and `none` without tools. A named choice must match a declared function;
  violations return 400 before routing. Responses may contain multiple tool
  calls, but `parallel_tool_calls` is rejected because nano-llm does not promise
  equivalent parallel-generation behavior across providers.
- **Instruction roles:** leading `system` and `developer` messages are accepted,
  kept in order, and combined into each provider's system-instruction format.
  Either role is rejected if it appears after the conversational messages begin.
- **Tool-schema boundary:** validate the OpenAI function-tool wrapper, unique
  function names matching `[A-Za-z0-9_-]{1,64}`, and that `parameters` is a JSON
  object, but otherwise preserve the schema as opaque JSON. Apply the same name
  rule to named `tool_choice` values and assistant tool calls; adapters never
  rename tools. v0.1 does not validate a portable keyword subset, resolve
  references, or rewrite schemas. Advanced-schema fallback portability is best
  effort; an upstream rejection advances to the next target.
- **Tool-history validity:** assistant tool-call IDs must be nonempty and unique,
  and call names must exist in the request's tool definitions. Tool results may
  arrive in any order but must resolve each immediately preceding call exactly
  once before conversation continues. Invalid transcripts return 400 before
  routing.
- **Tool arguments:** model-generated canonical `function.arguments` is an
  opaque string and malformed JSON is not globally treated as an invalid
  successful OpenAI-compatible response. In inbound history, routes containing
  Anthropic or Gemini require every argument string to parse as exactly one JSON
  object during gateway validation; other routes preserve it opaquely. This
  route-level rule ensures every configured fallback can represent the request.
- **Attempt policy:** each configured route entry is attempted at most once.
  Every target error advances immediately; canonical validation errors stop
  before routing. Repeating a target in configuration is the explicit way to
  request another attempt. Client disconnection cancels all work.
- **Error contract:** every `/v1/*` error uses the exact JSON envelope from
  §3.3 with all four members present and `application/json`. Allowed `type`
  values are `invalid_request_error`, `authentication_error`,
  `not_found_error`, and `server_error`. Stable `code` values distinguish
  invalid requests, authentication failures, model or route not found,
  oversized requests, exhausted upstreams, exhausted capacity, and overall
  timeout. `param` names the offending top-level request field when applicable
  and is null otherwise. Upstream error objects, statuses, and raw bodies are
  never copied into the client response.
- **Semantic neutrality:** any well-formed successful provider response ends
  routing, including refusals, filtered or length-limited completions, and valid
  tool-call-only responses. nano-llm never grades content or triggers fallback
  based on semantic quality.
- **Finish reasons:** expose only `stop`, `length`, `tool_calls`, and
  `content_filter`. Unknown successful terminal reasons normalize to `stop` and
  are logged at debug level rather than causing fallback.
- **Redirect policy:** upstream redirects are never followed. Every 3xx is a
  target failure and advances the route, preventing credentials from being
  replayed to a redirect destination.
- **Request size:** inbound request bodies have a fixed 1 MiB limit applied
  before parsing and routing. Oversized requests return an OpenAI-shaped 413.
  v0.1 exposes no setting for this limit.
- **Upstream size limits:** buffered non-streaming response bodies are capped at
  8 MiB and individual decoded SSE events at 1 MiB. Total streaming bytes are
  uncapped and processed incrementally. A pre-commit violation may fall back;
  a post-commit SSE violation closes the stream. These limits are fixed in v0.1.
- **Concurrency guard:** `general_settings.max_in_flight` is a process-wide
  generation semaphore and defaults to 64. Capacity is acquired before body
  buffering and held for the request lifetime. When full, chat completions
  immediately return an OpenAI-shaped 503; there is no internal queue.
  `/health` and `/v1/models` bypass the guard.
- **Provider architecture:** implementation is organized by wire protocol, not
  provider brand. OpenAI, Mistral, DeepSeek, and custom compatible endpoints
  share one OpenAI-compatible adapter; branded prefixes are thin defaults.
  Post-v0.1 Vertex support will reuse the Gemini translation layer while adding
  its own GCP authentication and endpoint construction.
- **Provider preset determinism:** each branded prefix has the exact default
  base URL, streaming and non-streaming operation paths, credential transport,
  mandatory headers, and protocol/API version defined in §2.4. Implementations
  must not inherit changing provider-SDK or HTTP-library defaults. A branded
  `api_base` override replaces only the base URL and retains the preset's
  operations, authentication transport, and mandatory protocol headers.
- **Upstream model identifiers:** every provider suffix is nonempty, at most 256
  UTF-8 bytes, and contains no ASCII control characters. OpenAI-compatible and
  Anthropic adapters preserve the validated suffix as the upstream JSON model
  string. Gemini suffixes additionally match
  `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` and are inserted as one URL path segment;
  violations are fatal startup configuration errors.
- **API base security:** every explicit `api_base` is an absolute HTTP(S) URL
  with a host and no userinfo, query, or fragment. Branded providers and every
  target carrying an `api_key` require HTTPS. Plaintext HTTP is limited to
  keyless `openai_compatible/` targets, and violations are startup errors.
- **Custom endpoint authentication:** `openai_compatible/` requires `api_base`
  but not `api_key`. When a key is present it is sent as a bearer token; when
  absent, no authorization header is added. Branded cloud presets continue to
  require their API keys.
- **Secret sources:** `master_key` and every present provider `api_key` must use
  `os.environ/VAR_NAME`; inline secret literals are rejected. The generic
  OpenAI-compatible adapter may omit its key entirely.
- **Provider seam:** adapters receive one validated canonical `ChatRequest` and
  return a canonical `ChatResponse` or `ChatChunk` stream. Routing does not know
  provider authentication, URLs, or wire formats. The exact canonical nested
  shapes and translation table in §3.2 and §5.1 are part of this interface,
  not adapter discretion.
- **Streaming commitment:** no downstream status, headers, or body are sent
  until the first canonical chunk is valid. A successful stream requires at
  least one canonical chunk before its terminal signal. A terminal signal
  received first is an `InvalidResponse` `TargetError`; while the response is
  uncommitted, routing advances to the next target. After commitment, any
  upstream failure closes the stream and never triggers fallback. Only emitted
  canonical chunks reset the post-commit idle timer; native keepalives and
  metadata do not.
- **HTTP method handling:** only the three documented method/path pairs are
  registered. Every other path or method returns 404, not 405. `/v1/*` failures
  authenticate first and use `route_not_found`; non-v1 failures use the same
  envelope without authentication.
- **Logging:** use `tracing` and `tracing-subscriber` to write structured fields
  to stdout. Log one request-completion record per application request with its
  validated or generated request ID, outcome, status code, and elapsed
  milliseconds. Requested model, selected provider/model, and attempt count are
  nullable when processing did not reach those stages. Stream outcomes are
  `completed`, `client_disconnected`, or `upstream_failed_after_commit`; the
  latter two retain HTTP status 200 in the record when headers were already
  committed. Non-streaming outcome names use the stable gateway condition code
  or `completed`. Log failed attempts at debug level and the exhausted/fatal
  result at warn level. Never log request or
  response bodies, API keys, authorization headers, access tokens, or credential
  contents. Human-readable output is the only v0.1 formatter; JSON formatting,
  telemetry exporters, log storage, and analytics are out of scope.
- **Request IDs:** accept `x-request-id` only as 1–128 printable, non-whitespace
  ASCII characters; otherwise generate a UUID. Return the chosen ID in the same
  response header and include it in all request logs. Do not forward it upstream
  in v0.1.
- **Packaging:** release artifacts target statically linked Linux binaries for
  `x86_64` and `aarch64`, suitable for a `FROM scratch` container. TLS trust
  roots must be bundled, and the gateway must not shell out to provider CLIs at
  runtime. The <20MB idle-RSS goal is measured in a release build and treated
  as a target to verify, not a reason to compromise correctness.
