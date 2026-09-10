# Provider health and readiness

Keep `/health` as a cheap process-liveness check and add a separate readiness or diagnostic view of configured targets. Health could be derived from recent real attempts, optional active probes, or both, and should report safe status categories without revealing credentials, private URLs, or raw provider errors.

This would let orchestrators distinguish a running process from a gateway that currently has no usable upstreams. Operators would also gain a quick way to see which target is causing slow failovers or whether a provider has recovered from a cooldown.

Active checks should be opt-in because they may consume quota or incur cost, and model-generation probes are rarely side-effect free. Passive health based on actual traffic is a safer first step and naturally complements adaptive routing.
