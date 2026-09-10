# Modern chat request features

The portable chat interface could be extended with commonly used capabilities that are currently rejected, especially multimodal image input, JSON and JSON-schema structured output, reasoning controls, log probabilities, and richer tool-call options. Each addition would require a canonical representation plus explicit translation and response behavior for every provider family allowed in the selected fallback route.

These features increasingly determine whether an application can use a gateway at all. Vision enables document and screenshot workflows, structured output makes responses safe to consume programmatically, and reasoning controls let callers balance latency, cost, and answer quality on newer model families.

The main design constraint is fallback portability. nano-llm should either prove that every target in a route can represent a requested feature or reject the request before contacting any provider. Feature-by-feature capability metadata would likely be more maintainable than continuing to apply one global request subset to every model.
