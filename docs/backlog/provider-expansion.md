# Broader provider support

nano-llm could add native adapters for important provider families that cannot be represented fully by the generic OpenAI-compatible adapter. Likely candidates include Azure OpenAI, AWS Bedrock, Vertex AI, and other services whose authentication, signing, endpoint discovery, API versioning, or wire format differs materially from the existing adapters.

Broader provider support increases the number of genuinely independent failure domains available in a fallback route. It also lets operators use cloud-managed credentials and regional deployments rather than embedding long-lived API keys or placing another compatibility proxy in front of nano-llm.

Providers should be added based on concrete demand rather than attempting LiteLLM-sized breadth. Each adapter carries a lasting compatibility and testing cost, so native support is most valuable where the generic adapter is insufficient or where the provider is especially useful as a cross-cloud fallback.
