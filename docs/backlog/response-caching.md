# Response caching

Introduce an optional cache for eligible deterministic or explicitly cacheable requests. A minimal implementation could use a bounded in-memory cache with configurable TTL and size, based on a stable hash of the canonical request and effective route. Distributed stores and semantic matching could remain separate future work.

Caching can reduce provider cost and latency for repeated prompts, especially in development, evaluation, and high-read workloads. It can also reduce pressure on rate-limited upstreams during traffic bursts.

Correctness and privacy need to be explicit. The cache key must include every response-affecting input, entries must not cross future tenant boundaries accidentally, streaming and tool results need defined behavior, and operators need controls to disable caching or bypass it per request. Semantic caching is substantially riskier than exact matching and should not be part of the first version.
