# Observability and cost visibility

Extend the current troubleshooting logs with machine-readable JSON output and a small stable telemetry surface. Useful signals include request counts, latency, time to first token, active requests, per-target attempts and failures, fallback frequency, token usage, and estimated cost. Export could begin with Prometheus metrics or OpenTelemetry rather than building storage or dashboards into nano-llm.

This would make provider outages, capacity constraints, slow fallbacks, and unexpected spend visible before users report them. Per-provider latency and error data would also give operators evidence for adjusting route order and timeout values.

Telemetry must preserve nano-llm's existing secret and content-safety boundaries. Prompt and response bodies should remain excluded by default, dimensions should be bounded to avoid cardinality problems, and any cost estimate should clearly expose when pricing or usage data is unavailable.
