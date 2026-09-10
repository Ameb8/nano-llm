# Vertex AI

Add a native Vertex AI provider that reuses the existing Gemini request and response translation where possible. The provider-specific work would cover regional Vertex endpoints, project and location configuration, OAuth access-token acquisition, token refresh, and safe handling of a service-account key file.

Vertex AI is useful for operators who already run in Google Cloud, need regional or organizational controls, or want Gemini capacity that is independent from the public Generative Language API. It also creates a more meaningful Google-side fallback because authentication and service infrastructure differ from the existing Gemini API-key path.

The initial authentication scope should be service-account key files only. Application Default Credentials, workload identity, metadata-server credentials, and user credentials can remain out of scope until there is a clear deployment need for them.
