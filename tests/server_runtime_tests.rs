use nano_llm::{
    app_with_provider_factory, build_runtime_config, parse_yaml_str, serve_until, ChatResponse,
    Provider, ProviderFactory, ProviderFuture, ProviderStream, Shutdown, TargetError,
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
