//! Dependency-free inbound HTTP contract and router-level test seam.

use crate::config::RuntimeConfig;
use std::fmt::Write;
use std::fs::File;
use std::io::Read;

const REQUEST_ID_HEADER: &str = "x-request-id";

/// An inbound request whose headers retain every raw field occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, Vec<u8>)>,
}

impl HttpRequest {
    pub fn new(method: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            path: path.into(),
            headers: Vec::new(),
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        self.headers.push((name.into(), value.into()));
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
}

/// Construct an application from immutable, validated configuration.
pub fn app(config: RuntimeConfig, no_auth: bool) -> Application {
    Application { config, no_auth }
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
                // Provider dispatch is outside this slice. Do not invent a success path.
                error_response(GatewayError::new(
                    GatewayErrorKind::UpstreamExhausted,
                    "All configured upstream targets failed",
                    None,
                ))
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
        values.len() == 1
            && values[0] == format!("Bearer {}", master_key.expose_secret()).as_bytes()
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
