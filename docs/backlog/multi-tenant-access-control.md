# Multi-tenant access control

Replace or supplement the single shared inbound key with multiple virtual keys that can expire and be restricted to particular public models. A larger version could introduce users, teams, service accounts, OIDC/JWT authentication, role-based administration, and audit records, which would likely require persistent storage.

This is useful when several applications or teams share one gateway. Separate identities make key rotation safer, limit the impact of credential leakage, and allow access to expensive or sensitive models to be granted selectively.

The feature changes nano-llm's trust model and operational footprint, so a small static-key map may be a more appropriate first step than reproducing LiteLLM's organization-management platform. Database-backed identity and RBAC should only follow if multi-team administration becomes a stated product goal.
