# Secret-manager integrations

Allow provider credentials and the inbound key to be resolved from external secret stores in addition to environment variables. Candidate integrations include cloud secret managers, Vault-compatible services, and mounted secret files, with credentials resolved at startup and eventually refreshed during configuration reload.

This reduces the need to place long-lived secrets directly in a process environment and fits organizations that already centralize rotation, access policy, and audit logging. It is especially valuable for cloud-specific providers whose credential lifecycle is more complex than a static API key.

Each integration adds authentication and failure modes of its own. A generic file-based secret source may cover many container and orchestration environments before native cloud SDK integrations are justified.
