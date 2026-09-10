# OpenAI Responses API

Add an OpenAI-compatible `/v1/responses` endpoint with its own canonical request, response, and streaming event types. The work would include translating Responses API inputs to providers that have equivalent native or chat-based APIs, normalizing tool and reasoning output, and defining which stateful features are intentionally unsupported.

The Responses API is becoming the primary interface for newer OpenAI models and reasoning-oriented workflows. Supporting it would let current OpenAI SDK applications use nano-llm directly instead of maintaining a client-side Chat Completions compatibility layer.

This should not be implemented as a thin path alias to `/v1/chat/completions`: the event model, input forms, tool behavior, and response objects differ enough to deserve a separate boundary. A useful first slice could support stateless text, streaming, and function tools while explicitly rejecting hosted tools and server-side conversation state.
