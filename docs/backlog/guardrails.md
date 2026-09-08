# Prompt and response guardrails

Add optional pre-call and post-call processing hooks for policies such as prompt-injection detection, PII masking, content filtering, and organization-specific validation. Guardrails could begin with a generic HTTP integration rather than embedding many vendor SDKs, with explicit timeout and failure-open or failure-closed behavior.

Central guardrails are useful when several clients need the same safety or compliance policy. They prevent each application from implementing inconsistent filtering and can remove sensitive data before it reaches an external model provider.

Guardrails sit directly in the request path and can alter content, latency, privacy, and availability. Their actions must be observable without leaking the protected data, streaming behavior needs careful treatment, and nano-llm should avoid claiming that a configurable filter provides universal model safety.
