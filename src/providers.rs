//! Provider-independent target attempt boundary.
//!
//! This module deliberately contains no provider wire format.  A router owns
//! one immutable canonical request and calls this trait; adapters own URL,
//! credentials, and their particular HTTP representation.

use crate::anthropic::AnthropicProvider;
use crate::config::{ProviderKind, RuntimeTarget};
use crate::openai_compatible::OpenAiCompatibleProvider;
use crate::request::CanonicalRequest;
use crate::response::{ChatChunk, ChatResponse};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// A future returned by an object-safe provider operation.
pub type ProviderFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A canonical stream returned after a provider has successfully set it up.
///
/// The streaming implementation is intentionally independent of an HTTP
/// library.  Later adapters can bridge their SSE decoder into this iterator
/// without changing the router-facing contract.
pub type ProviderStream = Box<dyn Iterator<Item = Result<ChatChunk, TargetError>> + Send>;

/// The only interface available to routing code.
///
/// Its arguments and results are canonical gateway values: provider URL,
/// authorization scheme, credentials, and wire payloads cannot leak into the
/// router's control flow.
pub trait Provider: Send + Sync {
    fn complete<'a>(
        &'a self,
        request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ChatResponse, TargetError>>;

    fn complete_stream<'a>(
        &'a self,
        request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>>;
}

/// Stable classifications for a failed target attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetErrorKind {
    Timeout,
    ConnectionError,
    RateLimited,
    Authentication,
    PermissionDenied,
    RejectedRequest,
    InvalidResponse,
    Overloaded,
    UpstreamHttp,
}

/// A safe diagnostic for a target attempt.
///
/// `safe_message` is deliberately not caller-supplied. This makes it
/// impossible for an adapter to accidentally put a credential, authorization
/// value, request body, or raw upstream response body into this error.
#[derive(Clone, PartialEq, Eq)]
pub struct TargetError {
    pub kind: TargetErrorKind,
    pub upstream_status: Option<u16>,
    /// Fixed, non-upstream diagnostic text. Static storage prevents raw
    /// response bodies and dynamically supplied credentials from entering it.
    pub safe_message: &'static str,
}

impl TargetError {
    pub const fn timeout() -> Self {
        Self::new(TargetErrorKind::Timeout, None)
    }

    pub const fn connection() -> Self {
        Self::new(TargetErrorKind::ConnectionError, None)
    }

    pub const fn invalid_response() -> Self {
        Self::new(TargetErrorKind::InvalidResponse, None)
    }

    /// A provider-native operational exhaustion reported in an otherwise
    /// successful HTTP response.
    pub const fn overloaded() -> Self {
        Self::new(TargetErrorKind::Overloaded, None)
    }

    /// Classify an upstream HTTP status without retaining its body.
    /// Redirects are intentionally ordinary target failures: callers must not
    /// inspect `Location` or issue a follow-up request.
    pub const fn from_upstream_status(status: u16) -> Self {
        let kind = match status {
            401 => TargetErrorKind::Authentication,
            403 => TargetErrorKind::PermissionDenied,
            408 | 504 => TargetErrorKind::Timeout,
            429 => TargetErrorKind::RateLimited,
            400..=499 => TargetErrorKind::RejectedRequest,
            503 | 529 => TargetErrorKind::Overloaded,
            _ => TargetErrorKind::UpstreamHttp,
        };
        Self::new(kind, Some(status))
    }

    pub const fn safe_message(&self) -> &'static str {
        self.safe_message
    }

    const fn new(kind: TargetErrorKind, upstream_status: Option<u16>) -> Self {
        let safe_message = match kind {
            TargetErrorKind::Timeout => "upstream request timed out",
            TargetErrorKind::ConnectionError => "could not connect to upstream",
            TargetErrorKind::RateLimited => "upstream rate limited the request",
            TargetErrorKind::Authentication => "upstream authentication failed",
            TargetErrorKind::PermissionDenied => "upstream permission denied",
            TargetErrorKind::RejectedRequest => "upstream rejected the request",
            TargetErrorKind::InvalidResponse => "upstream returned an invalid response",
            TargetErrorKind::Overloaded => "upstream is overloaded",
            TargetErrorKind::UpstreamHttp => "upstream returned an HTTP error",
        };
        Self {
            kind,
            upstream_status,
            safe_message,
        }
    }
}

impl fmt::Debug for TargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TargetError")
            .field("kind", &self.kind)
            .field("upstream_status", &self.upstream_status)
            .field("safe_message", &self.safe_message)
            .finish()
    }
}

impl fmt::Display for TargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.safe_message)
    }
}

impl std::error::Error for TargetError {}

/// TLS verification required for provider traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportTlsVerification {
    /// Verify every HTTPS peer using trust roots packaged with the HTTP client.
    BundledRoots,
}

/// Non-negotiable outbound HTTP security settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SecureTransportPolicy {
    pub tls_verification: TransportTlsVerification,
    /// Always false.  A redirect response is returned as a failed attempt.
    pub follow_redirects: bool,
}

impl Default for SecureTransportPolicy {
    fn default() -> Self {
        Self {
            tls_verification: TransportTlsVerification::BundledRoots,
            follow_redirects: false,
        }
    }
}

/// Minimal outbound request representation for adapter-owned HTTP transports.
/// It is not visible to routing code.
#[derive(Clone, PartialEq, Eq)]
pub struct OutboundRequest {
    pub method: &'static str,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl fmt::Debug for OutboundRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OutboundRequest")
            .field("method", &self.method)
            .field("url", &self.url)
            .field("headers", &"[REDACTED]")
            .field(
                "body",
                &format_args!("[{} bytes redacted]", self.body.len()),
            )
            .finish()
    }
}

/// A successful non-redirect HTTP response.  Adapters parse the body locally.
#[derive(Clone, PartialEq, Eq)]
pub struct OutboundResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// A response body delivered as transport-sized byte fragments.  Fragments do
/// not have any SSE framing meaning and may split anywhere.
pub type OutboundByteStream = Box<dyn Iterator<Item = Result<Vec<u8>, TransportError>> + Send>;

/// Successful streaming HTTP response.  Unlike [`OutboundResponse`], its body
/// is deliberately not retained by the transport.
pub struct OutboundStreamResponse {
    pub status: u16,
    pub body: OutboundByteStream,
}

impl fmt::Debug for OutboundResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OutboundResponse")
            .field("status", &self.status)
            .field(
                "body",
                &format_args!("[{} bytes redacted]", self.body.len()),
            )
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportErrorKind {
    Timeout,
    Connection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransportError {
    pub kind: TransportErrorKind,
}

/// Adapter-owned transport implementation. A production implementation must
/// honor [`SecureTransportPolicy`]; it must use bundled roots and return any
/// 3xx response without issuing another request.
pub trait OutboundTransport: Send + Sync {
    fn execute(
        &self,
        policy: SecureTransportPolicy,
        request: OutboundRequest,
    ) -> Result<OutboundResponse, TransportError>;

    /// Start a response whose body is consumed incrementally.  The default
    /// keeps existing buffered-only transports explicit: they cannot be used
    /// to accidentally buffer a native SSE response.
    fn execute_stream(
        &self,
        _policy: SecureTransportPolicy,
        _request: OutboundRequest,
    ) -> Result<OutboundStreamResponse, TransportError> {
        Err(TransportError {
            kind: TransportErrorKind::Connection,
        })
    }
}

/// A target-bound provider construction. The target is copied only after
/// configuration validation; credentials remain private and redacted by their
/// `SecretString` implementation.
#[derive(Debug, Clone)]
pub struct TargetProvider {
    target: RuntimeTarget,
    transport_policy: SecureTransportPolicy,
}

impl TargetProvider {
    pub fn from_validated(target: RuntimeTarget) -> Self {
        Self {
            target,
            transport_policy: SecureTransportPolicy::default(),
        }
    }

    pub fn target(&self) -> &RuntimeTarget {
        &self.target
    }

    pub const fn transport_policy(&self) -> SecureTransportPolicy {
        self.transport_policy
    }
}

/// Construct the adapter for exactly one validated target.  Provider-family
/// request and response translation is intentionally deferred; the family is
/// selected here, outside the router-facing [`Provider`] trait.
pub fn build_provider(target: RuntimeTarget) -> Box<dyn Provider> {
    match target.provider {
        ProviderKind::OpenAi
        | ProviderKind::Mistral
        | ProviderKind::DeepSeek
        | ProviderKind::OpenAiCompatible
        | ProviderKind::Anthropic
        | ProviderKind::Gemini => Box::new(TargetProvider::from_validated(target)),
    }
}

/// Construct a provider using the adapter-owned outbound transport.  Tests and
/// the eventual HTTP runtime use this seam to capture the exact provider wire
/// request without exposing it to router code.
pub fn build_provider_with_transport(
    target: RuntimeTarget,
    transport: Arc<dyn OutboundTransport>,
) -> Box<dyn Provider> {
    match target.provider {
        ProviderKind::OpenAi
        | ProviderKind::Mistral
        | ProviderKind::DeepSeek
        | ProviderKind::OpenAiCompatible => {
            Box::new(OpenAiCompatibleProvider::new(target, transport))
        }
        ProviderKind::Anthropic => Box::new(AnthropicProvider::new(target, transport)),
        ProviderKind::Gemini => Box::new(TargetProvider::from_validated(target)),
    }
}

impl Provider for TargetProvider {
    fn complete<'a>(
        &'a self,
        _request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ChatResponse, TargetError>> {
        Box::pin(async { Err(TargetError::invalid_response()) })
    }

    fn complete_stream<'a>(
        &'a self,
        _request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>> {
        Box::pin(async { Err(TargetError::invalid_response()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SecretString;
    use crate::request::decode_chat_request;
    use crate::response::{ChatChoice, FinishReason};

    fn target(secret: &str) -> RuntimeTarget {
        RuntimeTarget {
            model: "openai/gpt-test".into(),
            provider: ProviderKind::OpenAi,
            model_suffix: "gpt-test".into(),
            api_key: Some(SecretString::new(secret.into())),
            api_base: "https://example.test/v1".into(),
            timeout: 30,
            explicit_timeout: None,
        }
    }

    #[test]
    fn target_errors_cover_the_contract_without_raw_diagnostics() {
        let secret = "provider-secret-value";
        for status in [
            300, 301, 302, 303, 307, 308, 400, 401, 403, 429, 503, 529, 500,
        ] {
            let error = TargetError::from_upstream_status(status);
            assert_eq!(error.upstream_status, Some(status));
            assert!(!format!("{error:?}").contains(secret));
            assert!(!error.safe_message().contains(secret));
        }
        assert_eq!(
            TargetError::from_upstream_status(302).kind,
            TargetErrorKind::UpstreamHttp
        );
        assert_eq!(
            TargetError::from_upstream_status(429).kind,
            TargetErrorKind::RateLimited
        );
        assert_eq!(
            TargetError::from_upstream_status(401).kind,
            TargetErrorKind::Authentication
        );
        assert_eq!(
            TargetError::from_upstream_status(403).kind,
            TargetErrorKind::PermissionDenied
        );
        assert_eq!(
            TargetError::from_upstream_status(400).kind,
            TargetErrorKind::RejectedRequest
        );
        assert_eq!(
            TargetError::from_upstream_status(503).kind,
            TargetErrorKind::Overloaded
        );
    }

    #[test]
    fn configured_target_uses_bundled_verification_and_never_follows_redirects() {
        let provider = TargetProvider::from_validated(target("provider-secret-value"));
        assert_eq!(
            provider.transport_policy().tls_verification,
            TransportTlsVerification::BundledRoots
        );
        assert!(!provider.transport_policy().follow_redirects);
        assert!(!format!("{provider:?}").contains("provider-secret-value"));
    }

    #[test]
    fn outbound_debug_representations_redact_credentials_and_bodies() {
        let request = OutboundRequest {
            method: "POST",
            url: "https://example.test/v1/chat/completions".into(),
            headers: vec![(
                "Authorization".into(),
                "Bearer provider-secret-value".into(),
            )],
            body: b"raw upstream request body".to_vec(),
        };
        let response = OutboundResponse {
            status: 500,
            body: b"raw upstream response body".to_vec(),
        };
        let debug = format!("{request:?} {response:?}");
        assert!(!debug.contains("provider-secret-value"));
        assert!(!debug.contains("raw upstream"));
    }

    struct FakeProvider;

    impl Provider for FakeProvider {
        fn complete<'a>(
            &'a self,
            request: &'a CanonicalRequest,
        ) -> ProviderFuture<'a, Result<ChatResponse, TargetError>> {
            let model = request.model.clone();
            Box::pin(async move {
                Ok(ChatResponse {
                    id: "test".into(),
                    object: "chat.completion",
                    created: 0,
                    model,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: crate::response::AssistantMessage {
                            role: "assistant",
                            content: Some("ok".into()),
                            tool_calls: None,
                        },
                        finish_reason: FinishReason::Stop,
                    }],
                    usage: None,
                })
            })
        }
        fn complete_stream<'a>(
            &'a self,
            _request: &'a CanonicalRequest,
        ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>> {
            Box::pin(async {
                Ok(
                    Box::new(std::iter::empty::<Result<ChatChunk, TargetError>>())
                        as ProviderStream,
                )
            })
        }
    }

    #[test]
    fn fake_provider_is_usable_at_the_router_contract() {
        let request = decode_chat_request(
            br#"{"model":"public","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        let provider: Box<dyn Provider> = Box::new(FakeProvider);
        let response = crate::providers::tests::block_on(provider.complete(&request)).unwrap();
        assert_eq!(response.model, "public");
    }

    fn block_on<T>(mut future: Pin<Box<dyn Future<Output = T> + Send + '_>>) -> T {
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        fn no_op(_: *const ()) {}
        fn clone(_: *const ()) -> RawWaker {
            RawWaker::new(std::ptr::null(), &VTABLE)
        }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
        let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
        let mut context = Context::from_waker(&waker);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => value,
            Poll::Pending => panic!("test future unexpectedly pending"),
        }
    }
}
