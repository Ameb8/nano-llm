# Adaptive routing and outage memory

nano-llm currently starts every request at the first configured target and only moves forward after that attempt fails. Adaptive routing would retain a small amount of cross-request state so targets that are repeatedly timing out, overloaded, or rate limited can enter a temporary cooldown. It could also add bounded same-target retries, exponential backoff with jitter, and support for `Retry-After`, while keeping the existing overall request deadline authoritative.

This would prevent every request from paying for a known-bad primary during an outage and reduce avoidable traffic sent to a struggling provider. It is the most direct extension of nano-llm's reliability-shim role and can be useful without implementing LiteLLM's full distributed router.

A first version should remain deliberately simple: local in-process health state, fixed-priority selection among eligible targets, conservative retry defaults, and clear logs explaining why a target was attempted, skipped, or cooled down. More elaborate weighted, latency-based, cost-based, or distributed routing could be considered separately.
