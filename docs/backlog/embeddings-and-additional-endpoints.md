# Embeddings and additional model endpoints

Add API surfaces beyond Chat Completions, beginning with `/v1/embeddings`. Each endpoint should have its own canonical types and provider capability checks rather than being forced through the chat abstraction. Later demand could justify image generation, audio transcription or speech, reranking, moderation, batch, or realtime endpoints.

Embeddings are the most broadly useful first addition because they make nano-llm viable for retrieval, semantic search, and indexing pipelines. They are also naturally suited to routing across multiple deployments, although their batching and response-shape rules differ from generation.

The broader LiteLLM endpoint catalog should be treated as a menu, not a compatibility target. Every new surface increases the binary, adapter, and test matrix, so endpoints should be added independently when they serve a concrete nano-llm use case.
