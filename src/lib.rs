pub mod cli;
pub mod config;
pub mod openai_compatible;
pub mod providers;
pub mod request;
pub mod response;
pub mod server;

pub use cli::{Cli, CliError, DEFAULT_BIND_ADDR, IS_DEV_BUILD};
pub use config::{
    build_runtime_config, parse_yaml_str, ConfigError, ConfigErrorKind, ConfigLocation,
    EnvLookupError, EnvProvider, ProviderKind, RawConfig, RuntimeConfig, RuntimeGeneralSettings,
    RuntimeRoute, RuntimeTarget, SecretString, SystemEnv,
};
pub use openai_compatible::{OpenAiCompatibleProvider, MAX_BUFFERED_RESPONSE_BYTES};
pub use providers::{
    build_provider, build_provider_with_transport, OutboundRequest, OutboundResponse,
    OutboundTransport, Provider, ProviderFuture, ProviderStream, SecureTransportPolicy,
    TargetError, TargetErrorKind, TargetProvider, TransportError, TransportErrorKind,
    TransportTlsVerification,
};
pub use request::{
    decode_chat_request, decode_chat_request_fields, decode_chat_request_for_route,
    decode_json_object, requested_model, CanonicalRequest, DecodeError, JsonValue, ToolChoice,
};
pub use response::{
    build_response, normalize_response, normalize_usage, safety_response, AssistantDelta,
    AssistantMessage, ChatChoice, ChatChunk, ChatResponse, ChunkChoice, FinishReason, FunctionCall,
    NativeChoice, NativeResponse, NativeTerminal, NativeToolCall, ResponseError, ResponseMetadata,
    StreamAssembler, ToolCall, ToolCallDelta, Usage,
};
pub use server::{
    app, app_with_provider_factory, app_with_provider_factory_and_clock, Application,
    AttemptOutcome, AttemptRecord, DownstreamCancellation, GatewayError, GatewayErrorKind,
    HttpRequest, HttpResponse, MonotonicClock, ProviderFactory, RouteDiagnostics, RouteDispatch,
    RouteExhausted,
};

/// Returns the package version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns true if this binary is a development build.
pub fn is_dev_build() -> bool {
    IS_DEV_BUILD && version().contains("-dev")
}
