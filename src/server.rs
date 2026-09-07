//! Dependency-free inbound HTTP contract and router-level test seam.

use crate::config::{ProviderKind, RuntimeConfig};
use crate::providers::{build_provider, Provider, TargetError, TargetErrorKind};
use crate::request::{
    decode_chat_request_fields, decode_json_object, requested_model, DecodeError,
};
use crate::response::{ChatChunk, ChatResponse, ToolCall};
use std::fmt::Write;
use std::fmt::{self, Display};
use std::fs::File;
use std::io::{self, Read, Write as IoWrite};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const REQUEST_ID_HEADER: &str = "x-request-id";
const MAX_REQUEST_BODY_BYTES: usize = 1024 * 1024;
const REQUEST_BODY_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_HTTP_HEADER_BYTES: usize = 32 * 1024;
// These are transport safeguards, not public configuration: a peer that does
// not complete its headers or consume a response may occupy one bounded worker
// only for this long.
const SOCKET_HEADER_TIMEOUT: Duration = Duration::from_secs(30);
const SOCKET_WRITE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_CONNECTION_WORKERS: usize = 128;
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(10);
type RawHeaders = Vec<(String, Vec<u8>)>;
type ParsedHttpHead = (String, String, RawHeaders, usize);

// Signal handlers are intentionally limited to setting this lock-free flag.
// In particular, they do not close sockets or join worker threads; those are
// ordinary runtime operations performed by the listener loop.
static TERMINATION_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// A programmatic shutdown handle for embedding and integration tests.
///
/// The process runtime also observes SIGINT and SIGTERM.  Requesting shutdown
/// only stops the accept loop; already accepted connections are retained until
/// their application work has completed.
#[derive(Clone, Default)]
pub struct Shutdown(Arc<std::sync::atomic::AtomicBool>);

impl Shutdown {
    pub fn request(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn requested(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Run the plain-HTTP listener until SIGINT or SIGTERM requests graceful
/// shutdown. There is deliberately no shutdown deadline: the process owner
/// decides whether and when to force termination.
pub fn serve(application: Application, bind: SocketAddr) -> io::Result<()> {
    install_termination_handlers()?;
    TERMINATION_REQUESTED.store(false, Ordering::Release);
    run_listener(application, bind, Shutdown::default(), true)
}

/// Run a listener using an explicit shutdown handle. This is the same runtime
/// used by [`serve`], exposed so transports can be tested without process
/// signals. It does not install or observe process signal handlers.
pub fn serve_until(
    application: Application,
    bind: SocketAddr,
    shutdown: Shutdown,
) -> io::Result<()> {
    run_listener(application, bind, shutdown, false)
}

fn run_listener(
    application: Application,
    bind: SocketAddr,
    shutdown: Shutdown,
    observe_signals: bool,
) -> io::Result<()> {
    let listener = TcpListener::bind(bind)?;
    listener.set_nonblocking(true)?;
    let mut workers = Vec::new();
    let active_connections = Arc::new(AtomicUsize::new(0));

    while !(shutdown.requested()
        || (observe_signals && TERMINATION_REQUESTED.load(Ordering::Acquire)))
    {
        reap_workers(&mut workers);
        match listener.accept() {
            Ok((stream, _peer)) => {
                // A signal may have arrived while accept was waiting. Do not
                // start new application work in that case.
                if shutdown.requested()
                    || (observe_signals && TERMINATION_REQUESTED.load(Ordering::Acquire))
                {
                    drop(stream);
                    break;
                }
                if !reserve_connection_slot(&active_connections) {
                    // Keep the listener bounded under connection churn.  This
                    // is deliberately a close rather than an unbounded queue.
                    drop(stream);
                    continue;
                }
                let application = application.clone();
                let active_connections = active_connections.clone();
                workers.push(thread::spawn(move || {
                    let _slot = ConnectionSlot(active_connections);
                    serve_connection(stream, application)
                }));
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_POLL_INTERVAL);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }

    // No timeout belongs here. A worker owns an already accepted request or
    // stream and is allowed to make progress until its normal completion.
    for worker in workers {
        let _ = worker.join();
    }
    Ok(())
}

fn reap_workers(workers: &mut Vec<thread::JoinHandle<()>>) {
    let mut index = 0;
    while index < workers.len() {
        if workers[index].is_finished() {
            let worker = workers.swap_remove(index);
            let _ = worker.join();
        } else {
            index += 1;
        }
    }
}

fn reserve_connection_slot(active: &AtomicUsize) -> bool {
    let mut current = active.load(Ordering::Acquire);
    loop {
        if current >= MAX_CONNECTION_WORKERS {
            return false;
        }
        match active.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return true,
            Err(observed) => current = observed,
        }
    }
}

struct ConnectionSlot(Arc<AtomicUsize>);

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

fn serve_connection(mut stream: TcpStream, application: Application) {
    let Ok((mut request, content_length, initial_body)) = read_http_head(&mut stream) else {
        return;
    };
    match application.socket_admit(&request) {
        SocketAdmission::Respond(response) => {
            let _ = write_http_response(&mut stream, &response);
        }
        SocketAdmission::Chat {
            permit,
            request_id,
            started,
        } => {
            let cancellation = DownstreamCancellation::default();
            let _watcher = DisconnectWatcher::start(&stream, cancellation.clone());
            request = request.with_downstream_cancellation(cancellation);
            match read_http_body(&mut stream, content_length, initial_body) {
                Ok(body) => {
                    request.body = body;
                    match application
                        .handle_admitted_socket_chat(&request, permit, request_id, started)
                    {
                        SocketChatResult::Stream(delivery) => delivery.write_to(&mut stream),
                        SocketChatResult::Response(response) => {
                            let _ = write_http_response(&mut stream, &response);
                        }
                    }
                }
                Err(kind) => {
                    drop(permit);
                    let response = with_request_id(
                        error_response(GatewayError::new(
                            kind,
                            if kind == GatewayErrorKind::RequestBodyTimeout {
                                "Request body timed out"
                            } else {
                                "Request body is too large"
                            },
                            None,
                        )),
                        request_id,
                    );
                    emit_completion(
                        &response,
                        gateway_outcome(kind),
                        started,
                        &CompletionContext::default(),
                    );
                    let _ = write_http_response(&mut stream, &response);
                }
            }
        }
    }
}

/// Read only the HTTP head.  Header admission intentionally precedes body
/// ingestion so rejected requests never wait for an attacker-controlled body.
fn read_http_head(stream: &mut TcpStream) -> io::Result<(HttpRequest, usize, Vec<u8>)> {
    stream.set_read_timeout(Some(SOCKET_HEADER_TIMEOUT))?;
    let mut bytes = Vec::new();
    let header_end = loop {
        if bytes.len() > MAX_HTTP_HEADER_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request headers are too large",
            ));
        }
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete request headers",
            ));
        }
        bytes.extend_from_slice(&chunk[..read]);
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break end + 4;
        }
    };

    let (method, path, headers, content_length) = parse_http_head(&bytes[..header_end])?;
    Ok((
        HttpRequest::new(method, path).with_headers(headers),
        content_length,
        bytes[header_end..].to_vec(),
    ))
}

fn read_http_body(
    stream: &mut TcpStream,
    content_length: usize,
    mut body: Vec<u8>,
) -> Result<Vec<u8>, GatewayErrorKind> {
    if content_length > MAX_REQUEST_BODY_BYTES {
        return Err(GatewayErrorKind::RequestTooLarge);
    }
    body.truncate(content_length);
    let deadline = Instant::now() + REQUEST_BODY_TIMEOUT;
    while body.len() < content_length {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(GatewayErrorKind::RequestBodyTimeout);
        }
        stream
            .set_read_timeout(Some(remaining))
            .map_err(|_| GatewayErrorKind::RequestBodyTimeout)?;
        let mut chunk = [0_u8; 4096];
        let read = match stream.read(&mut chunk) {
            Ok(read) => read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                ) =>
            {
                return Err(GatewayErrorKind::RequestBodyTimeout)
            }
            Err(_) => return Err(GatewayErrorKind::RequestBodyTimeout),
        };
        if read == 0 {
            return Err(GatewayErrorKind::RequestBodyTimeout);
        }
        let needed = content_length - body.len();
        body.extend_from_slice(&chunk[..read.min(needed)]);
    }
    Ok(body)
}

fn parse_http_head(bytes: &[u8]) -> io::Result<ParsedHttpHead> {
    let Some(request_line_end) = bytes.windows(2).position(|window| window == b"\r\n") else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing request line",
        ));
    };
    let request_line = std::str::from_utf8(&bytes[..request_line_end])
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 request line"))?;
    let mut request_parts = request_line.split_ascii_whitespace();
    let (Some(method), Some(target), Some(version), None) = (
        request_parts.next(),
        request_parts.next(),
        request_parts.next(),
        request_parts.next(),
    ) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid request line",
        ));
    };
    if !version.starts_with("HTTP/") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid HTTP version",
        ));
    }
    let method = method.to_owned();
    if !target.starts_with('/') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported request target",
        ));
    }
    let path = target
        .split_once('?')
        .map_or(target, |(path, _)| path)
        .to_owned();

    let mut headers = Vec::new();
    let mut content_length = None;
    let mut remaining = &bytes[request_line_end + 2..];
    while !remaining.is_empty() {
        let Some(line_end) = remaining.windows(2).position(|window| window == b"\r\n") else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid header"));
        };
        let line = &remaining[..line_end];
        remaining = &remaining[line_end + 2..];
        if line.is_empty() {
            break;
        }
        let Some(colon) = line.iter().position(|byte| *byte == b':') else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid header"));
        };
        let name = std::str::from_utf8(&line[..colon])
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-UTF-8 header name"))?;
        if name.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "empty header name",
            ));
        }
        let value = line[colon + 1..]
            .iter()
            .copied()
            .skip_while(|byte| matches!(byte, b' ' | b'\t'))
            .collect::<Vec<_>>();
        if name.eq_ignore_ascii_case("content-length") {
            let parsed = std::str::from_utf8(&value)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid content length")
                })?;
            if content_length.replace(parsed).is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "repeated content length",
                ));
            }
        }
        headers.push((name.to_owned(), value));
    }
    Ok((method, path, headers, content_length.unwrap_or(0)))
}

fn write_http_response(stream: &mut TcpStream, response: &HttpResponse) -> io::Result<()> {
    stream.set_write_timeout(Some(SOCKET_WRITE_TIMEOUT))?;
    let reason = match response.status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    };
    write!(stream, "HTTP/1.1 {} {reason}\r\n", response.status)?;
    for (name, value) in &response.headers {
        write!(stream, "{name}: {value}\r\n")?;
    }
    write!(
        stream,
        "content-length: {}\r\nconnection: close\r\n\r\n",
        response.body.len()
    )?;
    stream.write_all(&response.body)
}

/// Write the commitment point for an SSE response.  Unlike ordinary responses
/// this deliberately has no Content-Length: every following write is observed
/// by the peer as the selected canonical stream makes progress.
fn write_sse_head(stream: &mut TcpStream, response: &HttpResponse) -> io::Result<()> {
    stream.set_write_timeout(Some(SOCKET_WRITE_TIMEOUT))?;
    write!(stream, "HTTP/1.1 {} OK\r\n", response.status)?;
    for (name, value) in &response.headers {
        write!(stream, "{name}: {value}\r\n")?;
    }
    write!(stream, "connection: close\r\n\r\n")
}

#[cfg(unix)]
fn install_termination_handlers() -> io::Result<()> {
    type SignalHandler = extern "C" fn(std::os::raw::c_int);
    unsafe extern "C" {
        fn signal(signal: std::os::raw::c_int, handler: SignalHandler) -> SignalHandler;
    }
    extern "C" fn request_shutdown(_: std::os::raw::c_int) {
        TERMINATION_REQUESTED.store(true, Ordering::Release);
    }
    // `signal` is used only to install an async-signal-safe handler which
    // performs a lock-free atomic store. Failure is reported rather than
    // silently serving without graceful termination behavior.
    unsafe {
        if signal(2, request_shutdown) as usize == usize::MAX
            || signal(15, request_shutdown) as usize == usize::MAX
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn install_termination_handlers() -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "graceful signal shutdown requires a Unix platform",
    ))
}

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

/// A socket-level disconnect observer.  Router cancellation is useful only
/// when it reflects the peer, so the listener owns this short-lived watcher
/// rather than relying on tests to toggle a token.  It is joined before the
/// connection worker exits and never becomes detached background work.
struct DisconnectWatcher {
    stop: Arc<std::sync::atomic::AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl DisconnectWatcher {
    fn start(stream: &TcpStream, cancellation: DownstreamCancellation) -> Self {
        let Ok(probe) = stream.try_clone() else {
            return Self {
                stop: Arc::new(std::sync::atomic::AtomicBool::new(true)),
                worker: None,
            };
        };
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watcher_stop = stop.clone();
        let worker = thread::spawn(move || {
            let _ = probe.set_nonblocking(true);
            let mut byte = [0_u8; 1];
            while !watcher_stop.load(Ordering::Acquire) {
                match probe.peek(&mut byte) {
                    Ok(0) => {
                        cancellation.cancel();
                        return;
                    }
                    Ok(_) => {}
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(_) => {
                        cancellation.cancel();
                        return;
                    }
                }
                thread::sleep(Duration::from_millis(10));
            }
        });
        Self {
            stop,
            worker: Some(worker),
        }
    }
}

impl Drop for DisconnectWatcher {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
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

    fn with_headers(mut self, headers: Vec<(String, Vec<u8>)>) -> Self {
        self.headers = headers;
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

/// Safe routing information retained for an attempt.
///
/// This is deliberately limited to configured target identity and the stable
/// target-error classification. It contains neither request data nor provider
/// credentials, URLs, or upstream response bodies, and is suitable for a
/// later structured-logging boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptRecord {
    /// Zero-based position in the selected route.
    pub route_index: usize,
    pub provider: ProviderKind,
    pub target_model: String,
    pub outcome: AttemptOutcome,
}

/// The protocol-level outcome of one configured route entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    Succeeded,
    /// The downstream peer disconnected while this target operation was
    /// active. This is terminal for the route, not a target failure.
    Cancelled,
    Failed {
        kind: TargetErrorKind,
        upstream_status: Option<u16>,
    },
}

/// The complete safe context retained by routing for a later diagnostics or
/// logging boundary. It intentionally has no URL, credential, request body,
/// upstream error object, response body, or provider-supplied message field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDiagnostics {
    pub requested_model: String,
    pub selected_provider: Option<ProviderKind>,
    pub selected_model: Option<String>,
    pub attempt_count: usize,
    pub error_kind: Option<TargetErrorKind>,
    pub upstream_status: Option<u16>,
}

/// The result of trying a complete route in its configured order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDispatch {
    pub response: ChatResponse,
    pub attempts: Vec<AttemptRecord>,
    pub diagnostics: RouteDiagnostics,
}

/// A stream selected while the downstream response is still uncommitted.
///
/// `first_chunk` is deliberately separate from `stream`: routing consumed and
/// validated it before reporting success, so the HTTP boundary can commit only
/// after it has something canonical to send.
pub struct StreamRouteDispatch {
    pub first_chunk: ChatChunk,
    pub stream: crate::providers::ProviderStream,
    pub attempts: Vec<AttemptRecord>,
    pub diagnostics: RouteDiagnostics,
    // Commitment ends the route-wide timeout.  These values are retained for
    // the locked-in stream only: no post-commit outcome can resume routing.
    idle_timeout: Duration,
    cancellation: DownstreamCancellation,
    clock: Arc<dyn MonotonicClock>,
    include_usage: bool,
}

/// Safe attempt history when no target produced a response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteExhausted {
    pub attempts: Vec<AttemptRecord>,
    pub diagnostics: RouteDiagnostics,
    /// True when the route-wide deadline elapsed. In that case no later
    /// target was started, even if entries remain in the configured route.
    pub overall_timeout: bool,
    /// The downstream peer disconnected; no later route entry was started.
    pub cancelled: bool,
}

/// Source of monotonic time used by pre-commit routing deadlines. Keeping this
/// as a narrow application seam permits deterministic deadline tests without
/// coupling the router to an async runtime.
pub trait MonotonicClock: Send + Sync {
    fn now(&self) -> Duration;
}

struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl MonotonicClock for SystemClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
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
    clock: Arc<dyn MonotonicClock>,
}

/// The only values allowed to cross from request processing into logs.
/// Bodies, headers, credentials, URLs, and provider messages are deliberately
/// absent from this type.
#[derive(Default)]
struct CompletionContext {
    requested_model: Option<String>,
    selected_provider: Option<ProviderKind>,
    selected_model: Option<String>,
    attempt_count: Option<usize>,
}

impl CompletionContext {
    fn from_diagnostics(diagnostics: &RouteDiagnostics) -> Self {
        Self {
            requested_model: Some(diagnostics.requested_model.clone()),
            selected_provider: diagnostics.selected_provider,
            selected_model: diagnostics.selected_model.clone(),
            attempt_count: Some(diagnostics.attempt_count),
        }
    }
}

struct ChatHandling {
    response: HttpResponse,
    outcome: &'static str,
    context: CompletionContext,
    attempts: Vec<AttemptRecord>,
}

enum SocketAdmission {
    Respond(HttpResponse),
    Chat {
        permit: GenerationPermit,
        request_id: String,
        started: Instant,
    },
}

enum SocketChatResult {
    Response(HttpResponse),
    Stream(Box<SocketStreamDelivery>),
}

impl ChatHandling {
    fn early(kind: GatewayErrorKind, message: &'static str) -> Self {
        Self::with_context(
            error_response(GatewayError::new(kind, message, None)),
            gateway_outcome(kind),
            CompletionContext::default(),
            Vec::new(),
        )
    }

    fn decode_error(error: DecodeError) -> Self {
        let kind = decode_error_kind(&error);
        Self::with_context(
            decode_error_response(error),
            gateway_outcome(kind),
            CompletionContext::default(),
            Vec::new(),
        )
    }

    fn decode_error_with_context(error: DecodeError, context: CompletionContext) -> Self {
        let kind = decode_error_kind(&error);
        Self::with_context(
            decode_error_response(error),
            gateway_outcome(kind),
            context,
            Vec::new(),
        )
    }

    fn with_diagnostics(
        response: HttpResponse,
        outcome: &'static str,
        diagnostics: RouteDiagnostics,
        attempts: Vec<AttemptRecord>,
    ) -> Self {
        Self::with_context(
            response,
            outcome,
            CompletionContext::from_diagnostics(&diagnostics),
            attempts,
        )
    }

    fn with_context(
        response: HttpResponse,
        outcome: &'static str,
        context: CompletionContext,
        attempts: Vec<AttemptRecord>,
    ) -> Self {
        Self {
            response,
            outcome,
            context,
            attempts,
        }
    }
}

fn decode_error_kind(error: &DecodeError) -> GatewayErrorKind {
    match error {
        DecodeError::InvalidJson { .. } => GatewayErrorKind::InvalidJson,
        DecodeError::Validation { .. } => GatewayErrorKind::InvalidRequest,
    }
}

fn gateway_outcome(kind: GatewayErrorKind) -> &'static str {
    match kind {
        GatewayErrorKind::InvalidRequest => "invalid_request",
        GatewayErrorKind::InvalidJson => "invalid_json",
        GatewayErrorKind::AuthenticationFailed => "authentication_failed",
        GatewayErrorKind::ModelNotFound => "model_not_found",
        GatewayErrorKind::RouteNotFound => "route_not_found",
        GatewayErrorKind::RequestBodyTimeout => "request_body_timeout",
        GatewayErrorKind::RequestTooLarge => "request_too_large",
        GatewayErrorKind::UnsupportedMediaType => "unsupported_media_type",
        GatewayErrorKind::UpstreamExhausted => "upstream_exhausted",
        GatewayErrorKind::CapacityExhausted => "capacity_exhausted",
        GatewayErrorKind::OverallTimeout => "overall_timeout",
    }
}

struct Nullable<'a, T>(&'a Option<T>);

impl<T: Display> Display for Nullable<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(value) => value.fmt(formatter),
            None => formatter.write_str("null"),
        }
    }
}

fn emit_completion(
    response: &HttpResponse,
    outcome: &str,
    started: Instant,
    context: &CompletionContext,
) {
    tracing::info!(
        event = "request_completed",
        outcome = %outcome,
        status = response.status,
        elapsed_ms = started.elapsed().as_millis() as u64,
        requested_model = %Nullable(&context.requested_model),
        selected_provider = %Nullable(&context.selected_provider),
        selected_model = %Nullable(&context.selected_model),
        attempt_count = %Nullable(&context.attempt_count),
    );
}

fn socket_early_response(
    request_id: &str,
    started: Instant,
    response: HttpResponse,
) -> HttpResponse {
    let response = with_request_id(response, request_id.to_owned());
    let outcome = match response.status {
        401 => "authentication_failed",
        415 => "unsupported_media_type",
        503 => "capacity_exhausted",
        _ => "invalid_request",
    };
    let span = tracing::info_span!("request", request_id = %request_id);
    let _guard = span.enter();
    emit_completion(&response, outcome, started, &CompletionContext::default());
    response
}

fn emit_attempt_failures(attempts: &[AttemptRecord]) {
    for attempt in attempts {
        if let AttemptOutcome::Failed {
            kind,
            upstream_status,
        } = attempt.outcome
        {
            tracing::debug!(
                event = "attempt_failed",
                route_index = attempt.route_index,
                provider = %attempt.provider,
                target_model = %attempt.target_model,
                error_kind = ?kind,
                upstream_status = %Nullable(&upstream_status),
            );
        }
    }
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
    app_with_provider_factory_and_clock(
        config,
        no_auth,
        provider_factory,
        Arc::new(SystemClock::new()),
    )
}

/// Construct an application with an explicit monotonic clock. Production uses
/// [`app_with_provider_factory`]; this constructor keeps timing behavior
/// deterministically testable at the route boundary.
pub fn app_with_provider_factory_and_clock(
    config: RuntimeConfig,
    no_auth: bool,
    provider_factory: Arc<ProviderFactory>,
    clock: Arc<dyn MonotonicClock>,
) -> Application {
    let max_in_flight = config.general_settings.max_in_flight;
    Application {
        config,
        no_auth,
        generation_capacity: Arc::new(GenerationCapacity::new(max_in_flight)),
        provider_factory,
        clock,
    }
}

impl Application {
    /// Apply every header-only rule before allowing the transport to ingest an
    /// application body.  In particular, capacity is deliberately acquired
    /// after authentication and media-type validation and before buffering.
    fn socket_admit(&self, request: &HttpRequest) -> SocketAdmission {
        if request.path != "/v1/chat/completions" || request.method != "POST" {
            return SocketAdmission::Respond(self.handle(request));
        }
        let request_id = select_request_id(request);
        let started = Instant::now();
        if !self.is_authenticated(request) {
            return SocketAdmission::Respond(socket_early_response(
                &request_id,
                started,
                error_response(GatewayError::new(
                    GatewayErrorKind::AuthenticationFailed,
                    "Authentication failed",
                    None,
                )),
            ));
        }
        if !valid_json_content_type(request) {
            return SocketAdmission::Respond(socket_early_response(
                &request_id,
                started,
                error_response(GatewayError::new(
                    GatewayErrorKind::UnsupportedMediaType,
                    "Unsupported media type",
                    None,
                )),
            ));
        }
        let Some(permit) = self.generation_capacity.try_acquire() else {
            return SocketAdmission::Respond(socket_early_response(
                &request_id,
                started,
                error_response(GatewayError::new(
                    GatewayErrorKind::CapacityExhausted,
                    "Generation capacity exhausted",
                    None,
                )),
            ));
        };
        SocketAdmission::Chat {
            permit,
            request_id,
            started,
        }
    }

    fn handle_admitted_socket_chat(
        &self,
        request: &HttpRequest,
        permit: GenerationPermit,
        request_id: String,
        started: Instant,
    ) -> SocketChatResult {
        if self.is_socket_stream_candidate(request) {
            let result = self
                .begin_socket_stream_with_permit(request, permit, request_id.clone(), started)
                .expect("stream candidate was validated before delivery");
            return match result {
                Ok(delivery) => SocketChatResult::Stream(Box::new(delivery)),
                Err(response) => SocketChatResult::Response(response),
            };
        }
        let handled = self.chat_response_with_permit(request, permit);
        emit_attempt_failures(&handled.attempts);
        if handled.response.status >= 500 {
            tracing::warn!(event = "request_failed", outcome = %handled.outcome, status = handled.response.status);
        }
        let response = with_request_id(handled.response, request_id);
        emit_completion(&response, handled.outcome, started, &handled.context);
        SocketChatResult::Response(response)
    }

    fn is_socket_stream_candidate(&self, request: &HttpRequest) -> bool {
        let Ok(body) = buffer_body(request) else {
            return false;
        };
        let Ok(fields) = decode_json_object(&body) else {
            return false;
        };
        let Ok(model) = requested_model(&fields) else {
            return false;
        };
        let Some(route) = self.config.get_route(&model) else {
            return false;
        };
        let Ok(canonical) = decode_chat_request_fields(fields) else {
            return false;
        };
        canonical.stream && canonical.validate_for_route(route).is_ok()
    }

    /// Select an ID before any route or authentication processing, then dispatch.
    pub fn handle(&self, request: &HttpRequest) -> HttpResponse {
        let request_id = select_request_id(request);
        let request_span = tracing::info_span!("request", request_id = %request_id);
        let _request_guard = request_span.enter();
        let started = Instant::now();
        let response = if request.path == "/health" && request.method == "GET" {
            let response = json_response(200, "{\"status\":\"ok\"}".to_owned());
            emit_completion(
                &response,
                "completed",
                started,
                &CompletionContext::default(),
            );
            response
        } else if request.path.starts_with("/v1/") {
            if !self.is_authenticated(request) {
                let response = error_response(GatewayError::new(
                    GatewayErrorKind::AuthenticationFailed,
                    "Authentication failed",
                    None,
                ));
                emit_completion(
                    &response,
                    "authentication_failed",
                    started,
                    &CompletionContext::default(),
                );
                response
            } else if request.path == "/v1/models" && request.method == "GET" {
                let response = self.models_response();
                emit_completion(
                    &response,
                    "completed",
                    started,
                    &CompletionContext::default(),
                );
                response
            } else if request.path == "/v1/chat/completions" && request.method == "POST" {
                let handled = self.chat_response(request);
                emit_attempt_failures(&handled.attempts);
                if handled.response.status >= 500
                    || handled.outcome == "upstream_failed_after_commit"
                {
                    tracing::warn!(
                        event = "request_failed",
                        outcome = %handled.outcome,
                        status = handled.response.status
                    );
                }
                emit_completion(
                    &handled.response,
                    handled.outcome,
                    started,
                    &handled.context,
                );
                handled.response
            } else {
                let response = route_not_found();
                emit_completion(
                    &response,
                    "route_not_found",
                    started,
                    &CompletionContext::default(),
                );
                response
            }
        } else {
            let response = route_not_found();
            emit_completion(
                &response,
                "route_not_found",
                started,
                &CompletionContext::default(),
            );
            response
        };
        with_request_id(response, request_id)
    }

    /// Start only a request that is known to be a valid streaming chat
    /// request.  Returning `None` leaves ordinary handling (including all
    /// validation errors) on the long-standing materialized response seam.
    /// A selected stream is intentionally returned, not consumed here, so no
    /// downstream bytes exist until the listener has its first chunk.
    fn begin_socket_stream_with_permit(
        &self,
        request: &HttpRequest,
        permit: GenerationPermit,
        request_id: String,
        started: Instant,
    ) -> Option<Result<SocketStreamDelivery, HttpResponse>> {
        if request.path != "/v1/chat/completions" || request.method != "POST" {
            return None;
        }
        // Decode before taking the special delivery path.  Invalid requests
        // retain the existing error/permit behavior in `handle`.
        let body = buffer_body(request).ok()?;
        let fields = decode_json_object(&body).ok()?;
        let model = requested_model(&fields).ok()?;
        let route = self.config.get_route(&model)?;
        let canonical = decode_chat_request_fields(fields).ok()?;
        if !canonical.stream || canonical.validate_for_route(route).is_err() {
            return None;
        }

        let request_span = tracing::info_span!("request", request_id = %request_id);
        let _request_guard = request_span.enter();
        match self.dispatch_stream_route(route, &canonical, &request.cancellation) {
            Ok(dispatch) => {
                let context = CompletionContext::from_diagnostics(&dispatch.diagnostics);
                emit_attempt_failures(&dispatch.attempts);
                Some(Ok(SocketStreamDelivery {
                    dispatch,
                    _permit: permit,
                    response: with_request_id(empty_sse_response(), request_id),
                    started,
                    context,
                }))
            }
            Err(exhausted) => {
                emit_attempt_failures(&exhausted.attempts);
                let (error, outcome) = if exhausted.overall_timeout {
                    (
                        GatewayError::new(
                            GatewayErrorKind::OverallTimeout,
                            "Overall request timed out",
                            None,
                        ),
                        "overall_timeout",
                    )
                } else if exhausted.cancelled {
                    (
                        GatewayError::new(
                            GatewayErrorKind::UpstreamExhausted,
                            "All configured upstream targets failed",
                            None,
                        ),
                        "client_disconnected",
                    )
                } else {
                    (
                        GatewayError::new(
                            GatewayErrorKind::UpstreamExhausted,
                            "All configured upstream targets failed",
                            None,
                        ),
                        "upstream_exhausted",
                    )
                };
                let response = with_request_id(error_response(error), request_id);
                emit_completion(
                    &response,
                    outcome,
                    started,
                    &CompletionContext::from_diagnostics(&exhausted.diagnostics),
                );
                Some(Err(response))
            }
        }
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

    fn chat_response(&self, request: &HttpRequest) -> ChatHandling {
        if !valid_json_content_type(request) {
            return ChatHandling::early(
                GatewayErrorKind::UnsupportedMediaType,
                "Unsupported media type",
            );
        }

        let Some(_permit) = self.generation_capacity.try_acquire() else {
            return ChatHandling::early(
                GatewayErrorKind::CapacityExhausted,
                "Generation capacity exhausted",
            );
        };

        self.chat_response_with_permit(request, _permit)
    }

    fn chat_response_with_permit(
        &self,
        request: &HttpRequest,
        _permit: GenerationPermit,
    ) -> ChatHandling {
        let body = match buffer_body(request) {
            Ok(body) => body,
            Err(kind) => {
                return ChatHandling::early(
                    kind,
                    match kind {
                        GatewayErrorKind::RequestBodyTimeout => "Request body timed out",
                        GatewayErrorKind::RequestTooLarge => "Request body is too large",
                        _ => unreachable!("body buffering returns only body-limit errors"),
                    },
                )
            }
        };
        let fields = match decode_json_object(&body) {
            Ok(fields) => fields,
            Err(error) => return ChatHandling::decode_error(error),
        };
        let model = match requested_model(&fields) {
            Ok(model) => model,
            Err(error) => return ChatHandling::decode_error(error),
        };
        let requested_context = CompletionContext {
            requested_model: Some(model.clone()),
            ..CompletionContext::default()
        };
        let Some(route) = self.config.get_route(&model) else {
            return ChatHandling::with_context(
                error_response(GatewayError::new(
                    GatewayErrorKind::ModelNotFound,
                    "Model not found",
                    Some("model".into()),
                )),
                "model_not_found",
                requested_context,
                Vec::new(),
            );
        };
        let canonical = match decode_chat_request_fields(fields) {
            Ok(request) => request,
            Err(error) => return ChatHandling::decode_error_with_context(error, requested_context),
        };
        if let Err(error) = canonical.validate_for_route(route) {
            return ChatHandling::decode_error_with_context(error, requested_context);
        }
        if canonical.stream {
            return match self.dispatch_stream_route(route, &canonical, &request.cancellation) {
                Ok(dispatch) => {
                    let context = CompletionContext::from_diagnostics(&dispatch.diagnostics);
                    let attempts = dispatch.attempts.clone();
                    let (response, outcome) = sse_response(dispatch);
                    ChatHandling::with_context(response, outcome.as_str(), context, attempts)
                }
                Err(exhausted) if exhausted.overall_timeout => ChatHandling::with_diagnostics(
                    error_response(GatewayError::new(
                        GatewayErrorKind::OverallTimeout,
                        "Overall request timed out",
                        None,
                    )),
                    "overall_timeout",
                    exhausted.diagnostics,
                    exhausted.attempts,
                ),
                Err(exhausted) => {
                    let outcome = if exhausted.cancelled {
                        "client_disconnected"
                    } else {
                        "upstream_exhausted"
                    };
                    ChatHandling::with_diagnostics(
                        error_response(GatewayError::new(
                            GatewayErrorKind::UpstreamExhausted,
                            "All configured upstream targets failed",
                            None,
                        )),
                        outcome,
                        exhausted.diagnostics,
                        exhausted.attempts,
                    )
                }
            };
        }

        // The response is materialized by the adapter before this function
        // creates any successful HTTP response, preserving non-streaming
        // commitment semantics.
        // This is deliberately after complete buffering, JSON parsing, and
        // canonical route validation: none of that inbound work consumes the
        // fallback-chain budget.
        match self.dispatch_route(route, &canonical, &request.cancellation) {
            Ok(dispatch) => ChatHandling::with_diagnostics(
                json_response(200, serialize_chat_response(&dispatch.response)),
                "completed",
                dispatch.diagnostics,
                dispatch.attempts,
            ),
            Err(exhausted) if exhausted.overall_timeout => ChatHandling::with_diagnostics(
                error_response(GatewayError::new(
                    GatewayErrorKind::OverallTimeout,
                    "Overall request timed out",
                    None,
                )),
                "overall_timeout",
                exhausted.diagnostics,
                exhausted.attempts,
            ),
            Err(exhausted) => ChatHandling::with_diagnostics(
                error_response(GatewayError::new(
                    GatewayErrorKind::UpstreamExhausted,
                    "All configured upstream targets failed",
                    None,
                )),
                "upstream_exhausted",
                exhausted.diagnostics,
                exhausted.attempts,
            ),
        }
    }

    /// Try every configured entry once, in route order, until protocol success.
    /// All [`TargetError`] kinds deliberately share the exact same advance
    /// behavior. Each advance is a distinct provider operation and may be
    /// billable; v0.1 deliberately has no cross-provider deduplication. A
    /// downstream cancellation stops this loop immediately.
    pub fn dispatch_route(
        &self,
        route: &crate::config::RuntimeRoute,
        request: &crate::request::CanonicalRequest,
        cancellation: &DownstreamCancellation,
    ) -> Result<RouteDispatch, RouteExhausted> {
        let overall_deadline = self.clock.now().saturating_add(Duration::from_secs(
            self.config.general_settings.overall_timeout,
        ));
        let mut attempts = Vec::with_capacity(route.targets.len());
        for (route_index, target) in route.targets.iter().cloned().enumerate() {
            // Check before constructing or invoking the next provider. Besides
            // avoiding needless allocation, this prevents an adapter with
            // eager setup from beginning work after the client is gone.
            if cancellation.is_cancelled() {
                return Err(route_exhausted(request, attempts, false, true));
            }
            let now = self.clock.now();
            // Equality belongs to the route-wide deadline. Never construct a
            // provider after it has expired, avoiding a billable invocation.
            if now >= overall_deadline {
                return Err(route_exhausted(request, attempts, true, false));
            }
            let provider_kind = target.provider;
            let target_model = target.model.clone();
            let target_timeout = target.timeout;
            let provider = (self.provider_factory)(target);
            // Provider construction is deliberately distinct from invoking its
            // operation. Re-read time here so adapter setup cannot enlarge the
            // operation's share of the route-wide budget.
            let attempt_start = self.clock.now();
            if attempt_start >= overall_deadline {
                return Err(route_exhausted(request, attempts, true, false));
            }
            if cancellation.is_cancelled() {
                return Err(route_exhausted(request, attempts, false, true));
            }
            let attempt_deadline = attempt_start
                .saturating_add(Duration::from_secs(target_timeout))
                .min(overall_deadline);
            match block_on_until(
                provider.complete(request),
                cancellation,
                &*self.clock,
                attempt_deadline,
                overall_deadline,
            ) {
                Ok(Ok(response)) => {
                    attempts.push(AttemptRecord {
                        route_index,
                        provider: provider_kind,
                        target_model,
                        outcome: AttemptOutcome::Succeeded,
                    });
                    return Ok(RouteDispatch {
                        response,
                        diagnostics: route_diagnostics(request, &attempts),
                        attempts,
                    });
                }
                Ok(Err(error)) => attempts.push(failed_attempt(
                    route_index,
                    provider_kind,
                    target_model,
                    error,
                )),
                Err(DeadlineOutcome::AttemptTimeout) => attempts.push(failed_attempt(
                    route_index,
                    provider_kind,
                    target_model,
                    TargetError::timeout(),
                )),
                Err(DeadlineOutcome::OverallTimeout) => {
                    return Err(route_exhausted(request, attempts, true, false))
                }
                // Do not start another potentially billable upstream request
                // after the downstream client has disconnected.
                Err(DeadlineOutcome::Cancelled) => {
                    attempts.push(AttemptRecord {
                        route_index,
                        provider: provider_kind,
                        target_model,
                        outcome: AttemptOutcome::Cancelled,
                    });
                    return Err(route_exhausted(request, attempts, false, true));
                }
            }
        }
        Err(route_exhausted(request, attempts, false, false))
    }

    /// Select a streaming target without committing a downstream response.
    /// Every setup error, terminal-before-chunk, decoder error, or deadline
    /// expiry is still a normal target failure and advances exactly once.
    pub fn dispatch_stream_route(
        &self,
        route: &crate::config::RuntimeRoute,
        request: &crate::request::CanonicalRequest,
        cancellation: &DownstreamCancellation,
    ) -> Result<StreamRouteDispatch, RouteExhausted> {
        let overall_deadline = self.clock.now().saturating_add(Duration::from_secs(
            self.config.general_settings.overall_timeout,
        ));
        let mut attempts = Vec::with_capacity(route.targets.len());
        for (route_index, target) in route.targets.iter().cloned().enumerate() {
            if cancellation.is_cancelled() {
                return Err(route_exhausted(request, attempts, false, true));
            }
            if self.clock.now() >= overall_deadline {
                return Err(route_exhausted(request, attempts, true, false));
            }
            let provider_kind = target.provider;
            let target_model = target.model.clone();
            let target_timeout = target.timeout;
            let provider = (self.provider_factory)(target);
            let attempt_start = self.clock.now();
            if attempt_start >= overall_deadline {
                return Err(route_exhausted(request, attempts, true, false));
            }
            let attempt_deadline = attempt_start
                .saturating_add(Duration::from_secs(target_timeout))
                .min(overall_deadline);
            let stream = match block_on_until(
                provider.complete_stream(request),
                cancellation,
                &*self.clock,
                attempt_deadline,
                overall_deadline,
            ) {
                Ok(Ok(stream)) => stream,
                Ok(Err(error)) => {
                    attempts.push(failed_attempt(
                        route_index,
                        provider_kind,
                        target_model,
                        error,
                    ));
                    continue;
                }
                Err(DeadlineOutcome::AttemptTimeout) => {
                    attempts.push(failed_attempt(
                        route_index,
                        provider_kind,
                        target_model,
                        TargetError::timeout(),
                    ));
                    continue;
                }
                Err(DeadlineOutcome::OverallTimeout) => {
                    return Err(route_exhausted(request, attempts, true, false))
                }
                Err(DeadlineOutcome::Cancelled) => {
                    attempts.push(AttemptRecord {
                        route_index,
                        provider: provider_kind,
                        target_model,
                        outcome: AttemptOutcome::Cancelled,
                    });
                    return Err(route_exhausted(request, attempts, false, true));
                }
            };
            let mut stream = stream;
            // Iterators are adapter-owned synchronous decoders. Check both
            // sides of `next` so decoder work counts toward the same first
            // canonical-chunk budget as connection setup.
            let first = if cancellation.is_cancelled() {
                attempts.push(AttemptRecord {
                    route_index,
                    provider: provider_kind,
                    target_model,
                    outcome: AttemptOutcome::Cancelled,
                });
                return Err(route_exhausted(request, attempts, false, true));
            } else if self.clock.now() >= overall_deadline {
                return Err(route_exhausted(request, attempts, true, false));
            } else if self.clock.now() >= attempt_deadline {
                attempts.push(failed_attempt(
                    route_index,
                    provider_kind,
                    target_model,
                    TargetError::timeout(),
                ));
                continue;
            } else {
                stream.next()
            };
            if self.clock.now() >= overall_deadline {
                return Err(route_exhausted(request, attempts, true, false));
            }
            if cancellation.is_cancelled() {
                attempts.push(AttemptRecord {
                    route_index,
                    provider: provider_kind,
                    target_model,
                    outcome: AttemptOutcome::Cancelled,
                });
                return Err(route_exhausted(request, attempts, false, true));
            }
            if self.clock.now() >= attempt_deadline {
                attempts.push(failed_attempt(
                    route_index,
                    provider_kind,
                    target_model,
                    TargetError::timeout(),
                ));
                continue;
            }
            match first {
                Some(Ok(first_chunk))
                    if StreamSuccessState::first(&first_chunk, request.include_usage).is_some()
                        && first_chunk.model == request.model =>
                {
                    attempts.push(AttemptRecord {
                        route_index,
                        provider: provider_kind,
                        target_model,
                        outcome: AttemptOutcome::Succeeded,
                    });
                    return Ok(StreamRouteDispatch {
                        first_chunk,
                        stream,
                        diagnostics: route_diagnostics(request, &attempts),
                        attempts,
                        idle_timeout: Duration::from_secs(target_timeout),
                        cancellation: cancellation.clone(),
                        clock: self.clock.clone(),
                        include_usage: request.include_usage,
                    });
                }
                Some(Ok(_)) => attempts.push(failed_attempt(
                    route_index,
                    provider_kind,
                    target_model,
                    TargetError::invalid_response(),
                )),
                Some(Err(error)) => attempts.push(failed_attempt(
                    route_index,
                    provider_kind,
                    target_model,
                    error,
                )),
                // A terminal signal before an ordinary canonical chunk is an
                // invalid response, not an empty successful stream.
                None => attempts.push(failed_attempt(
                    route_index,
                    provider_kind,
                    target_model,
                    TargetError::invalid_response(),
                )),
            }
        }
        Err(route_exhausted(request, attempts, false, false))
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

fn route_exhausted(
    request: &crate::request::CanonicalRequest,
    attempts: Vec<AttemptRecord>,
    overall_timeout: bool,
    cancelled: bool,
) -> RouteExhausted {
    RouteExhausted {
        diagnostics: route_diagnostics(request, &attempts),
        attempts,
        overall_timeout,
        cancelled,
    }
}

fn route_diagnostics(
    request: &crate::request::CanonicalRequest,
    attempts: &[AttemptRecord],
) -> RouteDiagnostics {
    let selected = attempts.last();
    let (error_kind, upstream_status) = match selected.map(|attempt| attempt.outcome) {
        Some(AttemptOutcome::Failed {
            kind,
            upstream_status,
        }) => (Some(kind), upstream_status),
        Some(AttemptOutcome::Succeeded | AttemptOutcome::Cancelled) | None => (None, None),
    };
    RouteDiagnostics {
        requested_model: request.model.clone(),
        selected_provider: selected.map(|attempt| attempt.provider),
        selected_model: selected.map(|attempt| attempt.target_model.clone()),
        attempt_count: attempts.len(),
        error_kind,
        upstream_status,
    }
}

fn failed_attempt(
    route_index: usize,
    provider: ProviderKind,
    target_model: String,
    error: TargetError,
) -> AttemptRecord {
    AttemptRecord {
        route_index,
        provider,
        target_model,
        outcome: AttemptOutcome::Failed {
            kind: error.kind,
            upstream_status: error.upstream_status,
        },
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

/// A selected stream plus every resource whose lifetime must extend through
/// downstream delivery.  In particular, retaining the permit here prevents a
/// slow client from admitting another generation before this one is finished.
struct SocketStreamDelivery {
    dispatch: StreamRouteDispatch,
    _permit: GenerationPermit,
    response: HttpResponse,
    started: Instant,
    context: CompletionContext,
}

impl SocketStreamDelivery {
    fn write_to(mut self, stream: &mut TcpStream) {
        let outcome = if write_sse_head(stream, &self.response).is_ok() {
            write_sse_stream(stream, &mut self.dispatch)
        } else {
            self.dispatch.cancellation.cancel();
            StreamCompletionOutcome::ClientDisconnected
        };
        if matches!(outcome, StreamCompletionOutcome::UpstreamFailedAfterCommit) {
            tracing::warn!(
                event = "request_failed",
                outcome = %outcome.as_str(),
                status = 200
            );
        }
        emit_completion(
            &self.response,
            outcome.as_str(),
            self.started,
            &self.context,
        );
        // `dispatch` (and therefore its upstream iterator) drops before the
        // permit. This is the common cleanup path for EOF, decode failure,
        // write failure, timeout, and downstream cancellation.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeadlineOutcome {
    AttemptTimeout,
    OverallTimeout,
    Cancelled,
}

/// Poll an upstream operation until it completes or reaches the earlier of its
/// per-attempt and route-wide deadlines. The caller owns the future, so every
/// timeout return drops it and therefore cancels the active upstream work.
fn block_on_until<T>(
    mut future: crate::providers::ProviderFuture<'_, T>,
    cancellation: &DownstreamCancellation,
    clock: &dyn MonotonicClock,
    attempt_deadline: Duration,
    overall_deadline: Duration,
) -> Result<T, DeadlineOutcome> {
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
            return Err(DeadlineOutcome::Cancelled);
        }
        // The overall deadline is checked first both before and after polling,
        // making an exact tie deterministic even if the future changes time
        // during its poll implementation.
        if clock.now() >= overall_deadline {
            return Err(DeadlineOutcome::OverallTimeout);
        }
        if clock.now() >= attempt_deadline {
            return Err(DeadlineOutcome::AttemptTimeout);
        }
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(value) => {
                if clock.now() >= overall_deadline {
                    return Err(DeadlineOutcome::OverallTimeout);
                }
                if clock.now() >= attempt_deadline {
                    return Err(DeadlineOutcome::AttemptTimeout);
                }
                return Ok(value);
            }
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

/// Materialize the selected stream at this dependency-free HTTP seam. The
/// selection routine has already buffered the first canonical chunk, so these
/// SSE headers cannot be observed for a target that later proves invalid
/// before its first chunk. A real network adapter may write the same events
/// incrementally after this commitment point.
#[derive(Clone, Copy)]
enum StreamCompletionOutcome {
    Completed,
    ClientDisconnected,
    UpstreamFailedAfterCommit,
}

impl StreamCompletionOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::ClientDisconnected => "client_disconnected",
            Self::UpstreamFailedAfterCommit => "upstream_failed_after_commit",
        }
    }
}

/// Deliver a committed stream directly to the socket.  At most one encoded
/// event is allocated at a time; `write_all` supplies natural TCP backpressure
/// instead of building an unbounded application queue.
fn write_sse_stream(
    stream: &mut TcpStream,
    dispatch: &mut StreamRouteDispatch,
) -> StreamCompletionOutcome {
    let expected = StreamMetadata::from(&dispatch.first_chunk);
    let mut state = match StreamSuccessState::first(&dispatch.first_chunk, dispatch.include_usage) {
        Some(state) => state,
        None => return StreamCompletionOutcome::UpstreamFailedAfterCommit,
    };
    let mut event = String::new();
    serialize_sse_chunk(&mut event, &dispatch.first_chunk);
    if stream.write_all(event.as_bytes()).is_err() {
        dispatch.cancellation.cancel();
        return StreamCompletionOutcome::ClientDisconnected;
    }
    let mut idle_deadline = dispatch.clock.now().saturating_add(dispatch.idle_timeout);
    loop {
        if dispatch.cancellation.is_cancelled() {
            return StreamCompletionOutcome::ClientDisconnected;
        }
        if dispatch.clock.now() >= idle_deadline {
            return StreamCompletionOutcome::UpstreamFailedAfterCommit;
        }
        let item = dispatch.stream.next();
        if dispatch.cancellation.is_cancelled() {
            return StreamCompletionOutcome::ClientDisconnected;
        }
        if dispatch.clock.now() >= idle_deadline {
            return StreamCompletionOutcome::UpstreamFailedAfterCommit;
        }
        let Some(item) = item else {
            if state.terminal && stream.write_all(b"data: [DONE]\n\n").is_ok() {
                return StreamCompletionOutcome::Completed;
            }
            if state.terminal {
                dispatch.cancellation.cancel();
                return StreamCompletionOutcome::ClientDisconnected;
            }
            return StreamCompletionOutcome::UpstreamFailedAfterCommit;
        };
        match item {
            Ok(chunk) if state.accept(&chunk, &expected) => {
                event.clear();
                serialize_sse_chunk(&mut event, &chunk);
                if stream.write_all(event.as_bytes()).is_err() {
                    dispatch.cancellation.cancel();
                    return StreamCompletionOutcome::ClientDisconnected;
                }
                if !chunk.choices.is_empty() {
                    idle_deadline = dispatch.clock.now().saturating_add(dispatch.idle_timeout);
                }
            }
            Ok(_) | Err(_) => return StreamCompletionOutcome::UpstreamFailedAfterCommit,
        }
    }
}

fn sse_response(mut dispatch: StreamRouteDispatch) -> (HttpResponse, StreamCompletionOutcome) {
    let mut body = String::new();
    let expected = StreamMetadata::from(&dispatch.first_chunk);
    let mut state = match StreamSuccessState::first(&dispatch.first_chunk, dispatch.include_usage) {
        Some(state) => state,
        // This should be unreachable for adapters, but the HTTP commitment
        // boundary must remain defensive if an implementation violates the
        // provider contract.
        None => {
            return (
                empty_sse_response(),
                StreamCompletionOutcome::UpstreamFailedAfterCommit,
            )
        }
    };
    serialize_sse_chunk(&mut body, &dispatch.first_chunk);
    // The first chunk is the only event buffered before commitment. Emitting
    // it starts the post-commit idle interval; metadata and keepalives never
    // reach this boundary and therefore cannot reset it.
    let mut idle_deadline = dispatch.clock.now().saturating_add(dispatch.idle_timeout);
    let mut outcome = StreamCompletionOutcome::UpstreamFailedAfterCommit;
    loop {
        if dispatch.cancellation.is_cancelled() {
            outcome = StreamCompletionOutcome::ClientDisconnected;
            break;
        }
        if dispatch.clock.now() >= idle_deadline {
            break;
        }
        let item = dispatch.stream.next();
        // Decoding can itself consume the idle budget. A chunk obtained after
        // expiry is not emitted and cannot revive the stream.
        if dispatch.cancellation.is_cancelled() {
            outcome = StreamCompletionOutcome::ClientDisconnected;
            break;
        }
        if dispatch.clock.now() >= idle_deadline {
            break;
        }
        let Some(item) = item else {
            if state.terminal {
                outcome = StreamCompletionOutcome::Completed;
            }
            break;
        };
        match item {
            Ok(chunk) if state.accept(&chunk, &expected) => {
                let ordinary = !chunk.choices.is_empty();
                serialize_sse_chunk(&mut body, &chunk);
                // A usage-only chunk is adapter metadata made observable only
                // at successful completion. It deliberately does not count as
                // progress for the idle deadline.
                if ordinary {
                    idle_deadline = dispatch.clock.now().saturating_add(dispatch.idle_timeout);
                }
            }
            Ok(_) => break,
            // Failures after the first chunk are deliberately not retried and
            // do not receive a gateway-invented terminal event.
            Err(_) => break,
        }
    }
    if matches!(outcome, StreamCompletionOutcome::Completed) {
        body.push_str("data: [DONE]\n\n");
    }
    (empty_sse_response_with_body(body), outcome)
}

fn empty_sse_response() -> HttpResponse {
    empty_sse_response_with_body(String::new())
}

fn empty_sse_response_with_body(body: String) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![
            ("content-type".into(), "text/event-stream".into()),
            ("cache-control".into(), "no-cache".into()),
        ],
        body: body.into_bytes(),
    }
}

#[derive(Clone)]
struct StreamMetadata {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
}

impl From<&ChatChunk> for StreamMetadata {
    fn from(chunk: &ChatChunk) -> Self {
        Self {
            id: chunk.id.clone(),
            object: chunk.object,
            created: chunk.created,
            model: chunk.model.clone(),
        }
    }
}

/// Final validation immediately before bytes become public. Adapters normally
/// establish these invariants, but retaining this small gate makes a custom
/// provider unable to leak inconsistent canonical chunks after commitment.
struct StreamSuccessState {
    terminal: bool,
    usage_seen: bool,
    include_usage: bool,
}

impl StreamSuccessState {
    fn first(chunk: &ChatChunk, include_usage: bool) -> Option<Self> {
        let mut state = Self {
            terminal: false,
            usage_seen: false,
            include_usage,
        };
        let has_initial_role = matches!(
            chunk.choices.as_slice(),
            [choice] if choice.delta.role == Some("assistant")
        );
        (has_initial_role && state.accept(chunk, &StreamMetadata::from(chunk))).then_some(state)
    }

    fn accept(&mut self, chunk: &ChatChunk, metadata: &StreamMetadata) -> bool {
        if chunk.id != metadata.id
            || chunk.object != metadata.object
            || chunk.created != metadata.created
            || chunk.model != metadata.model
        {
            return false;
        }
        if chunk.choices.is_empty() {
            let Some(usage) = &chunk.usage else {
                return false;
            };
            if !self.include_usage
                || self.usage_seen
                || !self.terminal
                || usage.prompt_tokens.checked_add(usage.completion_tokens)
                    != Some(usage.total_tokens)
            {
                return false;
            }
            self.usage_seen = true;
            return true;
        }
        if self.terminal || chunk.usage.is_some() || chunk.choices.len() != 1 {
            return false;
        }
        let choice = &chunk.choices[0];
        if choice.index != 0 {
            return false;
        }
        let has_delta = choice.delta.role.is_some()
            || choice.delta.content.is_some()
            || !choice.delta.tool_calls.is_empty();
        if (!has_delta && choice.finish_reason.is_none())
            || matches!(choice.delta.role, Some(role) if role != "assistant")
        {
            return false;
        }
        if choice.finish_reason.is_some() {
            self.terminal = true;
        }
        true
    }
}

fn serialize_sse_chunk(body: &mut String, chunk: &ChatChunk) {
    body.push_str("data: {");
    body.push_str(&format!(
        "\"id\":\"{}\",\"object\":\"{}\",\"created\":{},\"model\":\"{}\",\"choices\":[",
        json_escape(&chunk.id),
        chunk.object,
        chunk.created,
        json_escape(&chunk.model)
    ));
    for (choice_index, choice) in chunk.choices.iter().enumerate() {
        if choice_index != 0 {
            body.push(',');
        }
        body.push_str(&format!("{{\"index\":{},\"delta\":{{", choice.index));
        let mut needs_comma = false;
        if let Some(role) = choice.delta.role {
            body.push_str(&format!("\"role\":\"{}\"", role));
            needs_comma = true;
        }
        if let Some(content) = &choice.delta.content {
            if needs_comma {
                body.push(',');
            }
            body.push_str(&format!("\"content\":\"{}\"", json_escape(content)));
            needs_comma = true;
        }
        if !choice.delta.tool_calls.is_empty() {
            if needs_comma {
                body.push(',');
            }
            body.push_str("\"tool_calls\":[");
            for (call_index, call) in choice.delta.tool_calls.iter().enumerate() {
                if call_index != 0 {
                    body.push(',');
                }
                body.push_str(&format!("{{\"index\":{}", call.index));
                if let Some(id) = &call.id {
                    body.push_str(&format!(",\"id\":\"{}\"", json_escape(id)));
                }
                if let Some(kind) = call.r#type {
                    body.push_str(&format!(",\"type\":\"{}\"", kind));
                }
                if call.name.is_some() || call.arguments.is_some() {
                    body.push_str(",\"function\":{");
                    if let Some(name) = &call.name {
                        body.push_str(&format!("\"name\":\"{}\"", json_escape(name)));
                    }
                    if let Some(arguments) = &call.arguments {
                        if call.name.is_some() {
                            body.push(',');
                        }
                        body.push_str(&format!("\"arguments\":\"{}\"", json_escape(arguments)));
                    }
                    body.push('}');
                }
                body.push('}');
            }
            body.push(']');
        }
        body.push_str("},\"finish_reason\":");
        match choice.finish_reason {
            Some(reason) => body.push_str(&format!("\"{}\"", reason.as_str())),
            None => body.push_str("null"),
        }
        body.push('}');
    }
    body.push(']');
    if let Some(usage) = &chunk.usage {
        body.push_str(&format!(
            ",\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        ));
    }
    body.push_str("}\n\n");
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
