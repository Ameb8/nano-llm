//! Dependency-free inbound HTTP contract and router-level test seam.

use crate::config::RuntimeConfig;
use crate::providers::{build_provider, Provider};
use crate::request::{
    decode_chat_request_fields, decode_json_object, requested_model, DecodeError,
};
use crate::response::{ChatResponse, ToolCall};
use std::fmt::Write;
use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const REQUEST_ID_HEADER: &str = "x-request-id";
const MAX_REQUEST_BODY_BYTES: usize = 1024 * 1024;
const REQUEST_BODY_TIMEOUT: Duration = Duration::from_secs(30);

/// An inbound request whose headers retain every raw field occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, Vec<u8>)>,
    /// Fully buffered inbound body at this dependency-free router seam.
    pub body: Vec<u8>,
    body_chunks: Option<Vec<(Duration, Vec<u8>)>>,
    cancellation: DownstreamCancellation,
}

/// A transport-owned signal that the downstream peer has disconnected.
/// Dropping the in-flight provider future when this becomes cancelled gives
/// transports a single, explicit cancellation boundary.
#[derive(Debug, Clone, Default)]
pub struct DownstreamCancellation(Arc<std::sync::atomic::AtomicBool>);

impl PartialEq for DownstreamCancellation {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for DownstreamCancellation {}

impl DownstreamCancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl HttpRequest {
    pub fn new(method: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
            headers: Vec::new(),
            body: Vec::new(),
            body_chunks: None,
            cancellation: DownstreamCancellation::default(),
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Attach an inbound body. Network adapters must enforce the same fixed
    /// 30-second complete-buffer deadline before constructing this request.
    pub fn with_body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = body.into();
        self
    }

    /// Attach a body as chunks whose offsets are measured from permit
    /// acquisition. This lets a transport adapter preserve the body deadline
    /// without exposing its reader implementation to routing code.
    pub fn with_body_chunks<I>(mut self, chunks: I) -> Self
    where
        I: IntoIterator<Item = (Duration, Vec<u8>)>,
    {
        self.body_chunks = Some(chunks.into_iter().collect());
        self
    }

    /// Attach the downstream disconnect signal supplied by a transport.
    pub fn with_downstream_cancellation(mut self, cancellation: DownstreamCancellation) -> Self {
        self.cancellation = cancellation;
        self
    }

    fn headers_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a [u8]> + 'a {
        self.headers
            .iter()
            .filter(move |(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_slice())
    }
}

/// A materialized application response suitable for exact snapshot tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn header_values<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        self.headers
            .iter()
            .filter(move |(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Stable public error conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayErrorKind {
    InvalidRequest,
    InvalidJson,
    AuthenticationFailed,
    ModelNotFound,
    RouteNotFound,
    RequestBodyTimeout,
    RequestTooLarge,
    UnsupportedMediaType,
    UpstreamExhausted,
    CapacityExhausted,
    OverallTimeout,
}

/// A gateway-owned error; it never carries an upstream body or credential.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayError {
    pub kind: GatewayErrorKind,
    pub message: String,
    pub param: Option<String>,
}

impl GatewayError {
    pub fn new(kind: GatewayErrorKind, message: impl Into<String>, param: Option<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            param,
        }
    }
    fn contract(&self) -> (u16, &'static str, &'static str) {
        match self.kind {
            GatewayErrorKind::InvalidRequest => (400, "invalid_request_error", "invalid_request"),
            GatewayErrorKind::InvalidJson => (400, "invalid_request_error", "invalid_json"),
            GatewayErrorKind::AuthenticationFailed => {
                (401, "authentication_error", "authentication_failed")
            }
            GatewayErrorKind::ModelNotFound => (404, "not_found_error", "model_not_found"),
            GatewayErrorKind::RouteNotFound => (404, "not_found_error", "route_not_found"),
            GatewayErrorKind::RequestBodyTimeout => {
                (408, "invalid_request_error", "request_body_timeout")
            }
            GatewayErrorKind::RequestTooLarge => {
                (413, "invalid_request_error", "request_too_large")
            }
            GatewayErrorKind::UnsupportedMediaType => {
                (415, "invalid_request_error", "unsupported_media_type")
            }
            GatewayErrorKind::UpstreamExhausted => (502, "server_error", "upstream_exhausted"),
            GatewayErrorKind::CapacityExhausted => (503, "server_error", "capacity_exhausted"),
            GatewayErrorKind::OverallTimeout => (504, "server_error", "overall_timeout"),
        }
    }

    /// Render the exact canonical JSON envelope. Application dispatch adds its
    /// request-ID header afterwards.
    pub fn response(&self) -> HttpResponse {
        error_response(self.clone())
    }
}

/// Application state and exact route dispatch behavior.
#[derive(Clone)]
pub struct Application {
    config: RuntimeConfig,
    no_auth: bool,
    generation_capacity: Arc<GenerationCapacity>,
    provider_factory: Arc<ProviderFactory>,
}

/// Constructs one target-bound provider after routing has selected its target.
/// The factory is an application seam: routing sees only canonical requests and
/// providers never see route selection or inbound HTTP details.
pub type ProviderFactory = dyn Fn(crate::config::RuntimeTarget) -> Box<dyn Provider> + Send + Sync;

/// Construct an application from immutable, validated configuration.
pub fn app(config: RuntimeConfig, no_auth: bool) -> Application {
    app_with_provider_factory(config, no_auth, Arc::new(build_provider))
}

/// Construct an application with an adapter factory. This is primarily useful
/// for a transport-backed server runtime and deterministic integration tests.
pub fn app_with_provider_factory(
    config: RuntimeConfig,
    no_auth: bool,
    provider_factory: Arc<ProviderFactory>,
) -> Application {
    let max_in_flight = config.general_settings.max_in_flight;
    Application {
        config,
        no_auth,
        generation_capacity: Arc::new(GenerationCapacity::new(max_in_flight)),
        provider_factory,
    }
}

impl Application {
    /// Select an ID before any route or authentication processing, then dispatch.
    pub fn handle(&self, request: &HttpRequest) -> HttpResponse {
        let request_id = select_request_id(request);
        let response = if request.path == "/health" && request.method == "GET" {
            json_response(200, "{\"status\":\"ok\"}".to_owned())
        } else if request.path.starts_with("/v1/") {
            if !self.is_authenticated(request) {
                error_response(GatewayError::new(
                    GatewayErrorKind::AuthenticationFailed,
                    "Authentication failed",
                    None,
                ))
            } else if request.path == "/v1/models" && request.method == "GET" {
                self.models_response()
            } else if request.path == "/v1/chat/completions" && request.method == "POST" {
                self.chat_response(request)
            } else {
                route_not_found()
            }
        } else {
            route_not_found()
        };
        with_request_id(response, request_id)
    }

    fn is_authenticated(&self, request: &HttpRequest) -> bool {
        if self.no_auth {
            return true;
        }
        let Some(master_key) = self.config.general_settings.master_key.as_ref() else {
            return false;
        };
        let values: Vec<_> = request.headers_named("authorization").collect();
        values.len() == 1 && exact_bearer_matches(values[0], master_key.expose_secret().as_bytes())
    }

    fn chat_response(&self, request: &HttpRequest) -> HttpResponse {
        if !valid_json_content_type(request) {
            return error_response(GatewayError::new(
                GatewayErrorKind::UnsupportedMediaType,
                "Unsupported media type",
                None,
            ));
        }

        let Some(_permit) = self.generation_capacity.try_acquire() else {
            return error_response(GatewayError::new(
                GatewayErrorKind::CapacityExhausted,
                "Generation capacity exhausted",
                None,
            ));
        };

        let body = match buffer_body(request) {
            Ok(body) => body,
            Err(kind) => {
                return error_response(GatewayError::new(
                    kind,
                    match kind {
                        GatewayErrorKind::RequestBodyTimeout => "Request body timed out",
                        GatewayErrorKind::RequestTooLarge => "Request body is too large",
                        _ => unreachable!("body buffering returns only body-limit errors"),
                    },
                    None,
                ))
            }
        };
        let fields = match decode_json_object(&body) {
            Ok(fields) => fields,
            Err(error) => return decode_error_response(error),
        };
        let model = match requested_model(&fields) {
            Ok(model) => model,
            Err(error) => return decode_error_response(error),
        };
        let Some(route) = self.config.get_route(&model) else {
            return error_response(GatewayError::new(
                GatewayErrorKind::ModelNotFound,
                "Model not found",
                Some("model".into()),
            ));
        };
        let canonical = match decode_chat_request_fields(fields) {
            Ok(request) => request,
            Err(error) => return decode_error_response(error),
        };
        if let Err(error) = canonical.validate_for_route(route) {
            return decode_error_response(error);
        }
        if canonical.stream {
            return error_response(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "Streaming is not supported",
                Some("stream".into()),
            ));
        }

        // This slice deliberately invokes only the first configured target.
        // The response is materialized by the adapter before this function
        // creates any successful HTTP response, preserving non-streaming
        // commitment semantics.
        let target = route.targets[0].clone();
        let provider = (self.provider_factory)(target);
        match block_on(provider.complete(&canonical), &request.cancellation) {
            Ok(Ok(response)) => json_response(200, serialize_chat_response(&response)),
            Ok(Err(_)) | Err(()) => error_response(GatewayError::new(
                GatewayErrorKind::UpstreamExhausted,
                "All configured upstream targets failed",
                None,
            )),
        }
    }

    fn models_response(&self) -> HttpResponse {
        let mut body = String::from("{\"object\":\"list\",\"data\":[");
        for (index, route) in self.config.routes.iter().enumerate() {
            if index != 0 {
                body.push(',');
            }
            // Runtime validation limits model names to JSON-safe ASCII.
            write!(
                body,
                "{{\"id\":\"{}\",\"object\":\"model\",\"created\":0,\"owned_by\":\"nano-llm\"}}",
                route.model_name
            )
            .expect("writing to String cannot fail");
        }
        body.push_str("]}");
        json_response(200, body)
    }
}

fn buffer_body(request: &HttpRequest) -> Result<Vec<u8>, GatewayErrorKind> {
    let Some(chunks) = request.body_chunks.as_deref() else {
        if request.body.len() > MAX_REQUEST_BODY_BYTES {
            return Err(GatewayErrorKind::RequestTooLarge);
        }
        return Ok(request.body.clone());
    };

    let mut body = Vec::new();
    for (arrival, chunk) in chunks {
        // A chunk that arrives after the deadline cannot make its size breach
        // win; the timeout happened first. At exactly 30 seconds the deadline
        // wins deterministically as well.
        if *arrival >= REQUEST_BODY_TIMEOUT {
            return Err(GatewayErrorKind::RequestBodyTimeout);
        }
        if body.len().saturating_add(chunk.len()) > MAX_REQUEST_BODY_BYTES {
            return Err(GatewayErrorKind::RequestTooLarge);
        }
        body.extend_from_slice(chunk);
    }
    Ok(body)
}

fn decode_error_response(error: DecodeError) -> HttpResponse {
    let (kind, message, param) = match error {
        DecodeError::InvalidJson { message } => (GatewayErrorKind::InvalidJson, message, None),
        DecodeError::Validation { message, param } => {
            (GatewayErrorKind::InvalidRequest, message, param)
        }
    };
    error_response(GatewayError::new(kind, message, param))
}

fn exact_bearer_matches(value: &[u8], expected: &[u8]) -> bool {
    let Some(space) = value.iter().position(|byte| *byte == b' ') else {
        return false;
    };
    if !value[..space].eq_ignore_ascii_case(b"Bearer") {
        return false;
    }
    constant_time_eq(&value[space + 1..], expected)
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let max_length = left.len().max(right.len());
    let mut difference = left.len() ^ right.len();
    for index in 0..max_length {
        difference |= usize::from(*left.get(index).unwrap_or(&0) ^ *right.get(index).unwrap_or(&0));
    }
    difference == 0
}

fn valid_json_content_type(request: &HttpRequest) -> bool {
    let values: Vec<_> = request.headers_named("content-type").collect();
    values.len() == 1 && valid_json_media_type(values[0])
}

fn valid_json_media_type(value: &[u8]) -> bool {
    let Ok(value) = std::str::from_utf8(value) else {
        return false;
    };
    let mut parts = value.split(';');
    let Some(media_type) = parts.next() else {
        return false;
    };
    if !media_type
        .trim_matches([' ', '\t'])
        .eq_ignore_ascii_case("application/json")
    {
        return false;
    }
    match (parts.next(), parts.next()) {
        (None, None) => true,
        (Some(parameter), None) => {
            let mut pair = parameter.trim_matches([' ', '\t']).split('=');
            matches!(
                (pair.next(), pair.next(), pair.next()),
                (Some(name), Some(value), None)
                    if name.trim_matches([' ', '\t']).eq_ignore_ascii_case("charset")
                        && value.trim_matches([' ', '\t']).eq_ignore_ascii_case("utf-8")
            )
        }
        _ => false,
    }
}

struct GenerationCapacity {
    available: AtomicUsize,
}

impl GenerationCapacity {
    fn new(limit: usize) -> Self {
        Self {
            available: AtomicUsize::new(limit),
        }
    }

    fn try_acquire(self: &Arc<Self>) -> Option<GenerationPermit> {
        let mut available = self.available.load(Ordering::Acquire);
        loop {
            if available == 0 {
                return None;
            }
            match self.available.compare_exchange_weak(
                available,
                available - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(GenerationPermit(self.clone())),
                Err(current) => available = current,
            }
        }
    }
}

struct GenerationPermit(Arc<GenerationCapacity>);

impl Drop for GenerationPermit {
    fn drop(&mut self) {
        self.0.available.fetch_add(1, Ordering::Release);
    }
}

fn select_request_id(request: &HttpRequest) -> String {
    let values: Vec<_> = request.headers_named(REQUEST_ID_HEADER).collect();
    if values.len() == 1
        && (1..=128).contains(&values[0].len())
        && values[0].iter().all(|byte| (0x21..=0x7e).contains(byte))
    {
        return String::from_utf8(values[0].to_vec()).expect("printable ASCII is UTF-8");
    }
    generated_uuid_v4()
}

fn generated_uuid_v4() -> String {
    let mut bytes = [0_u8; 16];
    File::open("/dev/urandom")
        .and_then(|mut source| source.read_exact(&mut bytes))
        .expect("operating-system randomness is required for request IDs");
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!("{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}", bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15])
}

fn route_not_found() -> HttpResponse {
    error_response(GatewayError::new(
        GatewayErrorKind::RouteNotFound,
        "Route not found",
        None,
    ))
}

fn error_response(error: GatewayError) -> HttpResponse {
    let (status, error_type, code) = error.contract();
    let param = match error.kind {
        GatewayErrorKind::InvalidRequest => error.param,
        GatewayErrorKind::ModelNotFound => Some("model".to_owned()),
        _ => None,
    }
    .map(|value| format!("\"{}\"", json_escape(&value)))
    .unwrap_or_else(|| "null".to_owned());
    json_response(status, format!("{{\"error\":{{\"message\":\"{}\",\"type\":\"{error_type}\",\"param\":{param},\"code\":\"{code}\"}}}}", json_escape(&error.message)))
}

/// Encode the already validated canonical completion. This happens only after
/// a provider has returned its complete response, so no partial success can be
/// committed if provider response validation fails.
fn serialize_chat_response(response: &ChatResponse) -> String {
    let mut body = format!(
        "{{\"id\":\"{}\",\"object\":\"{}\",\"created\":{},\"model\":\"{}\",\"choices\":[",
        json_escape(&response.id),
        response.object,
        response.created,
        json_escape(&response.model),
    );
    for (index, choice) in response.choices.iter().enumerate() {
        if index != 0 {
            body.push(',');
        }
        body.push_str(&format!(
            "{{\"index\":{},\"message\":{{\"role\":\"assistant\",\"content\":{}",
            choice.index,
            choice
                .message
                .content
                .as_deref()
                .map(|value| format!("\"{}\"", json_escape(value)))
                .unwrap_or_else(|| "null".into())
        ));
        if let Some(calls) = &choice.message.tool_calls {
            body.push_str(",\"tool_calls\":[");
            for (call_index, call) in calls.iter().enumerate() {
                if call_index != 0 {
                    body.push(',');
                }
                serialize_tool_call(&mut body, call);
            }
            body.push(']');
        }
        body.push_str(&format!(
            "}},\"finish_reason\":\"{}\"}}",
            choice.finish_reason.as_str()
        ));
    }
    body.push(']');
    if let Some(usage) = &response.usage {
        body.push_str(&format!(
            ",\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        ));
    }
    body.push('}');
    body
}

fn serialize_tool_call(body: &mut String, call: &ToolCall) {
    body.push_str(&format!(
        "{{\"id\":\"{}\",\"type\":\"function\",\"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}",
        json_escape(&call.id),
        json_escape(&call.function.name),
        json_escape(&call.function.arguments),
    ));
}

fn block_on<T>(
    mut future: crate::providers::ProviderFuture<'_, T>,
    cancellation: &DownstreamCancellation,
) -> Result<T, ()> {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn no_op(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut context = Context::from_waker(&waker);
    loop {
        if cancellation.is_cancelled() {
            return Err(());
        }
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(value) => return Ok(value),
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn json_response(status: u16, body: String) -> HttpResponse {
    HttpResponse {
        status,
        headers: vec![("content-type".into(), "application/json".into())],
        body: body.into_bytes(),
    }
}

fn with_request_id(mut response: HttpResponse, request_id: String) -> HttpResponse {
    response
        .headers
        .retain(|(name, _)| !name.eq_ignore_ascii_case(REQUEST_ID_HEADER));
    response
        .headers
        .push((REQUEST_ID_HEADER.into(), request_id));
    response
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => write!(escaped, "\\u{:04x}", character as u32)
                .expect("writing to String cannot fail"),
            character => escaped.push(character),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RuntimeGeneralSettings, RuntimeRoute};
    use crate::SecretString;

    fn application(no_auth: bool, max_in_flight: usize) -> Application {
        app(
            RuntimeConfig {
                general_settings: RuntimeGeneralSettings {
                    master_key: Some(SecretString::new("master-key".into())),
                    max_in_flight,
                    ..RuntimeGeneralSettings::default()
                },
                routes: Vec::<RuntimeRoute>::new(),
            },
            no_auth,
        )
    }

    fn chat() -> HttpRequest {
        HttpRequest::new("POST", "/v1/chat/completions")
            .with_header("authorization", "Bearer master-key")
            .with_header("content-type", "application/json")
            .with_body(br#"{}"#.to_vec())
    }

    #[test]
    fn authentication_is_exact_and_precedes_every_chat_gate() {
        let application = application(false, 1);
        for authorization in [
            None,
            Some("Bearer"),
            Some("Bearer\tmaster-key"),
            Some("Basic master-key"),
            Some("Bearer wrong"),
        ] {
            let mut request = HttpRequest::new("POST", "/v1/chat/completions")
                .with_header("content-type", "text/plain")
                .with_body(vec![b'x'; MAX_REQUEST_BODY_BYTES + 1]);
            if let Some(value) = authorization {
                request = request.with_header("authorization", value);
            }
            assert_eq!(application.handle(&request).status, 401);
        }
        let repeated = HttpRequest::new("POST", "/v1/chat/completions")
            .with_header("authorization", "Bearer master-key")
            .with_header("authorization", "Bearer master-key");
        assert_eq!(application.handle(&repeated).status, 401);
        assert_eq!(application.handle(&chat()).status, 400);
    }

    #[test]
    fn content_type_is_strict_and_precedes_capacity_and_body() {
        let application = application(false, 1);
        for content_type in [
            "application/json, text/plain",
            "application/json; charset=utf-8; boundary=x",
            "application/json; charset=latin1",
            "text/plain",
        ] {
            let request = HttpRequest::new("POST", "/v1/chat/completions")
                .with_header("authorization", "Bearer master-key")
                .with_header("content-type", content_type)
                .with_body(vec![b'x'; MAX_REQUEST_BODY_BYTES + 1]);
            assert_eq!(application.handle(&request).status, 415, "{content_type}");
        }
        let repeated = chat().with_header("content-type", "application/json");
        assert_eq!(application.handle(&repeated).status, 415);
        let charset = HttpRequest::new("POST", "/v1/chat/completions")
            .with_header("authorization", "bEaReR master-key")
            .with_header("content-type", "APPLICATION/JSON; CHARSET=UTF-8")
            .with_body(br#"{}"#.to_vec());
        assert_eq!(application.handle(&charset).status, 400);
    }

    #[test]
    fn buffering_limits_and_all_early_paths_release_the_permit() {
        let application = application(false, 1);
        let cases = [
            chat().with_body(vec![b'x'; MAX_REQUEST_BODY_BYTES + 1]),
            chat().with_body(b"not json".to_vec()),
            chat().with_body_chunks([(Duration::from_secs(31), br#"{}"#.to_vec())]),
            chat().with_body_chunks([(
                Duration::from_secs(29),
                vec![b'x'; MAX_REQUEST_BODY_BYTES + 1],
            )]),
            chat().with_body_chunks([(
                Duration::from_secs(31),
                vec![b'x'; MAX_REQUEST_BODY_BYTES + 1],
            )]),
        ];
        let expected = [413, 400, 408, 413, 408];
        for (request, status) in cases.into_iter().zip(expected) {
            assert_eq!(application.handle(&request).status, status);
            let permit = application
                .generation_capacity
                .try_acquire()
                .expect("terminal response must release the permit");
            drop(permit);
        }
    }

    #[test]
    fn exhausted_capacity_never_enters_chat_but_public_routes_bypass_it() {
        let application = application(false, 1);
        let permit = application.generation_capacity.try_acquire().unwrap();
        assert_eq!(application.handle(&chat()).status, 503);
        assert_eq!(
            application
                .handle(&HttpRequest::new("GET", "/health"))
                .status,
            200
        );
        let models =
            HttpRequest::new("GET", "/v1/models").with_header("authorization", "Bearer master-key");
        assert_eq!(application.handle(&models).status, 200);
        drop(permit);
    }
}
