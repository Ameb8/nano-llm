use nano_llm::{
    app_with_provider_factory, build_runtime_config, parse_yaml_str, serve_until, AssistantDelta,
    ChatChunk, ChatResponse, ChunkChoice, FinishReason, Provider, ProviderFactory, ProviderFuture,
    ProviderStream, Shutdown, TargetError,
};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn unused_loopback_addr() -> SocketAddr {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    listener.local_addr().unwrap()
}

fn connect(address: SocketAddr) -> TcpStream {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if let Ok(stream) = TcpStream::connect(address) {
            return stream;
        }
        assert!(Instant::now() < deadline, "server did not start listening");
        thread::sleep(Duration::from_millis(5));
    }
}

fn request(address: SocketAddr, request: &[u8]) -> String {
    let mut stream = connect(address);
    stream.write_all(request).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
}

struct BlockingProvider {
    started: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
}

struct GatedStreamProvider {
    release: Arc<AtomicBool>,
}

impl Provider for GatedStreamProvider {
    fn complete<'a>(
        &'a self,
        _request: &'a nano_llm::CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ChatResponse, TargetError>> {
        Box::pin(async { Err(TargetError::connection()) })
    }

    fn complete_stream<'a>(
        &'a self,
        _request: &'a nano_llm::CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>> {
        let release = self.release.clone();
        Box::pin(async move { Ok(Box::new(GatedStream { release, stage: 0 }) as ProviderStream) })
    }
}

struct GatedStream {
    release: Arc<AtomicBool>,
    stage: u8,
}

impl Iterator for GatedStream {
    type Item = Result<ChatChunk, TargetError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stage {
            0 => {
                self.stage = 1;
                Some(Ok(stream_chunk(Some("first"), None)))
            }
            1 => {
                while !self.release.load(Ordering::Acquire) {
                    thread::yield_now();
                }
                self.stage = 2;
                Some(Ok(stream_chunk(None, Some(FinishReason::Stop))))
            }
            _ => None,
        }
    }
}

fn stream_chunk(content: Option<&str>, finish_reason: Option<FinishReason>) -> ChatChunk {
    ChatChunk {
        id: "stream-id".into(),
        object: "chat.completion.chunk",
        created: 1,
        model: "alpha".into(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: AssistantDelta {
                role: Some("assistant"),
                content: content.map(str::to_owned),
                tool_calls: Vec::new(),
            },
            finish_reason,
        }],
        usage: None,
    }
}

impl Provider for BlockingProvider {
    fn complete<'a>(
        &'a self,
        _request: &'a nano_llm::CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ChatResponse, TargetError>> {
        self.started.store(true, Ordering::Release);
        Box::pin(async move {
            while !self.release.load(Ordering::Acquire) {
                thread::yield_now();
            }
            Err(TargetError::connection())
        })
    }

    fn complete_stream<'a>(
        &'a self,
        _request: &'a nano_llm::CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>> {
        Box::pin(async { Err(TargetError::connection()) })
    }
}

#[test]
fn operational_endpoints_bypass_saturation_and_shutdown_waits_for_active_work() {
    let yaml = r#"
model_list:
  - model_name: alpha
    litellm_params:
      model: openai_compatible/test
      api_base: http://localhost:8000/v1
general_settings:
  max_in_flight: 1
"#;
    let config = build_runtime_config(
        parse_yaml_str(yaml).unwrap(),
        true,
        &HashMap::<String, String>::new(),
    )
    .unwrap();
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let factory: Arc<ProviderFactory> = {
        let started = started.clone();
        let release = release.clone();
        Arc::new(move |_| {
            Box::new(BlockingProvider {
                started: started.clone(),
                release: release.clone(),
            })
        })
    };
    let application = app_with_provider_factory(config, true, factory);
    let address = unused_loopback_addr();
    let shutdown = Shutdown::default();
    let (finished_tx, finished_rx) = mpsc::channel();
    let server_shutdown = shutdown.clone();
    let server = thread::spawn(move || {
        serve_until(application, address, server_shutdown).unwrap();
        finished_tx.send(()).unwrap();
    });

    let mut generation = connect(address);
    generation
        .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 64\r\n\r\n{\"model\":\"alpha\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}")
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    while !started.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "generation did not reach provider"
        );
        thread::sleep(Duration::from_millis(5));
    }

    let health = request(
        address,
        b"GET /health HTTP/1.1\r\nHost: localhost\r\nX-Request-Id: health-id\r\n\r\n",
    );
    assert!(health.starts_with("HTTP/1.1 200"));
    assert!(health.contains("x-request-id: health-id\r\n"));
    assert!(health.ends_with("{\"status\":\"ok\"}"));
    let models = request(
        address,
        b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n",
    );
    assert!(models.starts_with("HTTP/1.1 200"));
    assert!(models.contains("\"id\":\"alpha\""));
    let mismatch = request(
        address,
        b"POST /v1/models HTTP/1.1\r\nHost: localhost\r\nX-Request-Id: mismatch-id\r\n\r\n",
    );
    assert!(mismatch.starts_with("HTTP/1.1 404"));
    assert!(mismatch.contains("x-request-id: mismatch-id\r\n"));
    assert!(mismatch.contains("\"code\":\"route_not_found\""));
    let saturated = request(address, b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 64\r\n\r\n{\"model\":\"alpha\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}");
    assert!(saturated.starts_with("HTTP/1.1 503"));
    assert!(saturated.contains("\"code\":\"capacity_exhausted\""));

    shutdown.request();
    assert!(finished_rx
        .recv_timeout(Duration::from_millis(100))
        .is_err());
    release.store(true, Ordering::Release);
    let mut completed = String::new();
    generation.read_to_string(&mut completed).unwrap();
    assert!(completed.starts_with("HTTP/1.1 502"));
    finished_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    server.join().unwrap();
}

#[test]
fn listener_flushes_first_sse_event_before_upstream_stream_completes() {
    let yaml = r#"
model_list:
  - model_name: alpha
    litellm_params:
      model: openai_compatible/test
      api_base: http://localhost:8000/v1
general_settings:
  max_in_flight: 1
"#;
    let config =
        build_runtime_config(parse_yaml_str(yaml).unwrap(), true, &HashMap::new()).unwrap();
    let release = Arc::new(AtomicBool::new(false));
    let factory: Arc<ProviderFactory> = {
        let release = release.clone();
        Arc::new(move |_| {
            Box::new(GatedStreamProvider {
                release: release.clone(),
            })
        })
    };
    let address = unused_loopback_addr();
    let shutdown = Shutdown::default();
    let server_shutdown = shutdown.clone();
    let server = thread::spawn(move || {
        serve_until(
            app_with_provider_factory(config, true, factory),
            address,
            server_shutdown,
        )
        .unwrap()
    });

    let mut client = connect(address);
    client
        .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 78\r\n\r\n{\"model\":\"alpha\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}")
        .unwrap();
    client
        .set_read_timeout(Some(Duration::from_millis(300)))
        .unwrap();
    let mut response = String::new();
    let deadline = Instant::now() + Duration::from_millis(300);
    while !response.contains("data: {") {
        let mut bytes = [0_u8; 4096];
        match client.read(&mut bytes) {
            Ok(0) => panic!("stream closed before its first event"),
            Ok(read) => response.push_str(std::str::from_utf8(&bytes[..read]).unwrap()),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) && Instant::now() < deadline =>
            {
                continue
            }
            Err(error) => panic!("first SSE event must not wait for upstream EOF: {error}"),
        }
    }
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
    assert!(response.contains("content-type: text/event-stream\r\n"));
    assert!(response.contains("cache-control: no-cache\r\n"));
    assert!(response.contains("data: {"));
    assert!(response.contains("\"role\":\"assistant\""));
    assert!(response.contains("\"content\":\"first\""));
    assert!(!response.contains("[DONE]"));

    let saturated = request(
        address,
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 64\r\n\r\n{\"model\":\"alpha\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}",
    );
    assert!(saturated.starts_with("HTTP/1.1 503"));

    release.store(true, Ordering::Release);
    let mut tail = String::new();
    client.read_to_string(&mut tail).unwrap();
    assert!(tail.contains("data: [DONE]\n\n"));
    shutdown.request();
    server.join().unwrap();
}
