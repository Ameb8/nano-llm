# Agent gateway and platform services

LiteLLM also exposes higher-level services such as MCP and A2A gateways, persistent memory, vector-store operations, prompt management, evaluations, and fine-tuning administration. Bringing any of these into nano-llm would require new protocols, authorization rules, persistence, and lifecycle APIs beyond model inference routing.

These services can give agent applications one governed entry point for models, tools, retrieval, and stored context. They are most useful in an organization-wide AI platform where centralized discovery and policy matter more than a minimal runtime footprint.

This should be treated as a collection of possible future products rather than one implementation item. Individual services should receive their own scoped backlog entry only after a concrete nano-llm use case appears; otherwise they would dilute the gateway's small reliability-focused design.
