# Budgets, rate limits, and spend tracking

Track token usage and estimated provider cost, then optionally enforce limits such as requests per minute, tokens per minute, concurrent requests, or periodic monetary budgets. Limits could apply globally at first and later attach to virtual keys, users, teams, routes, or individual models if multi-tenancy is added.

These controls help prevent accidental bills, keep one workload from exhausting shared provider quotas, and attribute usage when several applications share the gateway. Even non-enforcing cost logs would make route choices and provider pricing easier to evaluate.

Reliable enforcement requires durable, concurrency-safe accounting and current model pricing. A single-process implementation can remain lightweight, but distributed enforcement would introduce a database or coordination service and should not be implied by an initial local-only version.
