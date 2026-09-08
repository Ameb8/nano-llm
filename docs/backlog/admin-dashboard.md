# Administrative dashboard

An administrative UI could expose configured routes, current target health, recent failures, request and cost metrics, virtual keys, budgets, and guarded configuration changes. It would also require authenticated management endpoints and a clear distinction between read-only operational data and mutations.

This is useful for teams that cannot or do not want to operate the gateway exclusively through YAML, logs, and command-line tools. It lowers the barrier to inspecting outages and delegating routine key or model administration.

A dashboard is not a small isolated feature: it creates a second API surface, browser assets, session security concerns, and usually persistent state. It should follow the underlying observability and administration capabilities rather than driving their design, and may remain inappropriate for nano-llm's minimal deployment goal.
