//! Process-level provider conformance evidence.
//!
//! This deliberately starts `nano-llm` through its normal CLI and talks to the
//! default provider factory.  The TLS peer is a loopback fixture whose CA is
//! added only by the `executable-test-tls` Cargo feature; verification and
//! hostname checks remain enabled in the production transport.

#![cfg(feature = "executable-test-tls")]

use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::{ServerConfig, ServerConnection, StreamOwned};
use std::fs;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const CERT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/executable-tls/cert.pem"
);
const CA_CERT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/executable-tls/ca-cert.pem"
);
const KEY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/executable-tls/key.pem"
);

#[derive(Clone, Copy)]
enum Preset {
    OpenAi,
    Mistral,
    DeepSeek,
    Compatible,
    Anthropic,
    Gemini,
}

impl Preset {
    const ALL: [Self; 6] = [
        Self::OpenAi,
        Self::Mistral,
        Self::DeepSeek,
        Self::Compatible,
        Self::Anthropic,
        Self::Gemini,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Mistral => "mistral",
            Self::DeepSeek => "deepseek",
            Self::Compatible => "openai_compatible",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        }
    }
    fn model(self) -> &'static str {
        match self {
            Self::Anthropic => "claude-test",
            Self::Gemini => "gemini-test",
            _ => "model-test",
        }
    }
    fn credential(self) -> bool {
        !matches!(self, Self::Compatible)
    }
    fn expected_path(self, stream: bool) -> String {
        match self {
            Self::Anthropic => "/messages".into(),
            Self::Gemini if stream => format!(
                "/models/{}:streamGenerateContent?alt=sse&key=provider-canary",
                self.model()
            ),
            Self::Gemini => format!(
                "/models/{}:generateContent?key=provider-canary",
                self.model()
            ),
            _ => "/chat/completions".into(),
        }
    }
    fn response(self, kind: Kind) -> String {
        match (self, kind) {
            (Self::Anthropic, Kind::Text) => r#"{"type":"message","role":"assistant","content":[{"type":"text","text":"fixture text"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":3}}"#.into(),
            (Self::Anthropic, Kind::Tool) => r#"{"type":"message","role":"assistant","content":[{"type":"tool_use","id":"fixture-call","name":"weather","input":{"city":"Paris"}}],"stop_reason":"tool_use","usage":{"input_tokens":2,"output_tokens":3}}"#.into(),
            (Self::Anthropic, Kind::Safety) => r#"{"type":"message","role":"assistant","content":[],"stop_reason":"refusal","usage":{"input_tokens":2,"output_tokens":0}}"#.into(),
            (Self::Gemini, Kind::Text) => r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"fixture text"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3}}"#.into(),
            (Self::Gemini, Kind::Tool) => r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"fixture-call","name":"weather","args":{"city":"Paris"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":3}}"#.into(),
            (Self::Gemini, Kind::Safety) => r#"{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":0}}"#.into(),
            (_, Kind::Text) => r#"{"id":"native","object":"chat.completion","created":1,"model":"private","choices":[{"index":0,"message":{"role":"assistant","content":"fixture text"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#.into(),
            (_, Kind::Tool) => r#"{"id":"native","object":"chat.completion","created":1,"model":"private","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"fixture-call","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#.into(),
            (_, Kind::Safety) => r#"{"id":"native","object":"chat.completion","created":1,"model":"private","choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":2,"completion_tokens":0}}"#.into(),
        }
    }
}
#[derive(Clone, Copy)]
enum Kind {
    Text,
    Tool,
    Safety,
}

struct Fixture {
    address: SocketAddr,
    seen: Arc<Mutex<Vec<String>>>,
    worker: thread::JoinHandle<()>,
}
impl Fixture {
    fn start(responses: Vec<(u16, String)>) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let captured = seen.clone();
        let config = tls_config();
        let worker = thread::spawn(move || {
            for (status, body) in responses {
                let (socket, _) = listener.accept().unwrap();
                let mut stream =
                    StreamOwned::new(ServerConnection::new(config.clone()).unwrap(), socket);
                let request = read_request(&mut stream);
                captured.lock().unwrap().push(request);
                let reason = if status == 200 {
                    "OK"
                } else {
                    "Service Unavailable"
                };
                write!(stream, "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).unwrap();
                stream.flush().unwrap();
            }
        });
        Self {
            address,
            seen,
            worker,
        }
    }
    fn finish(self) -> Vec<String> {
        self.worker.join().unwrap();
        Arc::try_unwrap(self.seen).unwrap().into_inner().unwrap()
    }
}

fn tls_config() -> Arc<ServerConfig> {
    let mut cert_reader = std::io::BufReader::new(fs::File::open(CERT).unwrap());
    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut cert_reader)
        .collect::<Result<_, _>>()
        .unwrap();
    let mut key_reader = std::io::BufReader::new(fs::File::open(KEY).unwrap());
    let key: PrivateKeyDer<'static> = rustls_pemfile::private_key(&mut key_reader)
        .unwrap()
        .unwrap();
    Arc::new(
        ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)
            .unwrap(),
    )
}
fn read_request(stream: &mut impl Read) -> String {
    let mut bytes = Vec::new();
    let mut chunk = [0; 4096];
    loop {
        let n = stream.read(&mut chunk).unwrap();
        bytes.extend_from_slice(&chunk[..n]);
        if let Some(head) = bytes.windows(4).position(|x| x == b"\r\n\r\n") {
            let text = String::from_utf8_lossy(&bytes[..head + 4]);
            let len = text
                .lines()
                .find_map(|l| {
                    l.strip_prefix("content-length: ")
                        .or_else(|| l.strip_prefix("Content-Length: "))
                })
                .and_then(|x| x.parse::<usize>().ok())
                .unwrap_or(0);
            if bytes.len() >= head + 4 + len {
                return String::from_utf8(bytes).unwrap();
            }
        }
    }
}

struct Gateway {
    child: Child,
    address: SocketAddr,
    config: std::path::PathBuf,
}
impl Gateway {
    fn start(preset: Preset, upstream: SocketAddr) -> Self {
        let address = unused_addr();
        let config = std::env::temp_dir().join(format!(
            "nano-llm-matrix-{}-{}.yaml",
            preset.name(),
            std::process::id()
        ));
        let key = if preset.credential() {
            "      api_key: os.environ/NANO_LLM_MATRIX_PROVIDER_KEY\n"
        } else {
            ""
        };
        fs::write(&config, format!("model_list:\n  - model_name: public\n    litellm_params:\n      model: {}/{}\n{}      api_base: https://localhost:{}\n      timeout: 3\ngeneral_settings:\n  master_key: os.environ/NANO_LLM_MATRIX_MASTER_KEY\n  overall_timeout: 5\n", preset.name(), preset.model(), key, upstream.port())).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_nano-llm"))
            .args([
                "--config",
                config.to_str().unwrap(),
                "--bind",
                &address.to_string(),
            ])
            .env("NANO_LLM_MATRIX_MASTER_KEY", "master-canary")
            .env("NANO_LLM_MATRIX_PROVIDER_KEY", "provider-canary")
            .env("NANO_LLM_EXECUTABLE_TEST_CA_PEM", CA_CERT)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        wait_listener(address);
        Self {
            child,
            address,
            config,
        }
    }
    fn request(&self, body: &str) -> String {
        let mut s = TcpStream::connect(self.address).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
        write!(s, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer master-canary\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }
    fn stop(self) {
        unsafe {
            extern "C" {
                fn kill(pid: i32, signal: i32) -> i32;
            }
            kill(self.child.id() as i32, 15);
        }
        let out = self.child.wait_with_output().unwrap();
        fs::remove_file(self.config).ok();
        let all = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!all.contains("provider-canary"));
        assert!(!all.contains("master-canary"));
    }
}
fn unused_addr() -> SocketAddr {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .unwrap()
        .local_addr()
        .unwrap()
}
fn wait_listener(address: SocketAddr) {
    let until = Instant::now() + Duration::from_secs(3);
    while TcpStream::connect(address).is_err() {
        assert!(Instant::now() < until, "gateway failed to listen");
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn production_binary_conforms_for_each_provider_preset() {
    for preset in Preset::ALL {
        let fixture = Fixture::start(vec![
            (200, preset.response(Kind::Text)),
            (200, preset.response(Kind::Tool)),
            (200, preset.response(Kind::Safety)),
            (503, "upstream-canary-body".into()),
            (200, preset.response(Kind::Text)),
        ]);
        let gateway = Gateway::start(preset, fixture.address);
        let text = gateway.request(
            r#"{"model":"public","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}"#,
        );
        assert!(text.contains("HTTP/1.1 200"));
        assert!(text.contains("\"model\":\"public\""));
        assert!(text.contains("fixture text"));
        assert!(text.contains("\"total_tokens\":5"));
        let tool = gateway.request(r#"{"model":"public","max_tokens":8,"messages":[{"role":"user","content":"weather"}],"tools":[{"type":"function","function":{"name":"weather"}}]}"#);
        assert!(tool.contains("fixture-call"));
        assert!(tool.contains("tool_calls"));
        let safety = gateway.request(
            r#"{"model":"public","max_tokens":8,"messages":[{"role":"user","content":"refuse"}]}"#,
        );
        assert!(safety.contains("content_filter"));
        let failed = gateway.request(
            r#"{"model":"public","max_tokens":8,"messages":[{"role":"user","content":"error"}]}"#,
        );
        assert!(failed.contains("HTTP/1.1 502"));
        assert!(failed.contains("upstream_exhausted"));
        assert!(!failed.contains("upstream-canary-body"));
        let restarted = gateway.request(
            r#"{"model":"public","max_tokens":8,"messages":[{"role":"user","content":"again"}]}"#,
        );
        assert!(restarted.contains("fixture text"));
        gateway.stop();
        let seen = fixture.finish();
        assert_eq!(seen.len(), 5);
        assert!(
            seen.iter()
                .all(|r| r.starts_with(&format!("POST {} HTTP/1.1", preset.expected_path(false)))),
            "unexpected path for {}: {:?}",
            preset.name(),
            seen
        );
        if preset.credential() && !matches!(preset, Preset::Gemini) {
            assert!(
                seen[0]
                    .to_ascii_lowercase()
                    .contains("authorization: bearer provider-canary")
                    || seen[0]
                        .to_ascii_lowercase()
                        .contains("x-api-key: provider-canary")
            );
        }
        assert!(seen.iter().all(|r| !r.contains("\"model\":\"public\"")));
    }
}

#[test]
fn binary_fallback_is_file_ordered_and_does_not_follow_redirects() {
    let primary = Fixture::start(vec![(302, String::new()), (302, String::new())]);
    let secondary = Fixture::start(vec![
        (200, Preset::Compatible.response(Kind::Text)),
        (200, Preset::Compatible.response(Kind::Text)),
    ]);
    let address = unused_addr();
    let config =
        std::env::temp_dir().join(format!("nano-llm-fallback-{}.yaml", std::process::id()));
    fs::write(&config, format!("model_list:\n  - model_name: public\n    litellm_params:\n      model: openai_compatible/first\n      api_base: https://localhost:{}\n  - model_name: public\n    litellm_params:\n      model: openai_compatible/second\n      api_base: https://localhost:{}\ngeneral_settings:\n  master_key: os.environ/NANO_LLM_MATRIX_MASTER_KEY\n", primary.address.port(), secondary.address.port())).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_nano-llm"))
        .args([
            "--config",
            config.to_str().unwrap(),
            "--bind",
            &address.to_string(),
        ])
        .env("NANO_LLM_MATRIX_MASTER_KEY", "master-canary")
        .env("NANO_LLM_EXECUTABLE_TEST_CA_PEM", CA_CERT)
        .spawn()
        .unwrap();
    wait_listener(address);
    for _ in 0..2 {
        let mut s = TcpStream::connect(address).unwrap();
        let body = r#"{"model":"public","messages":[{"role":"user","content":"x"}]}"#;
        write!(s, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer master-canary\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        assert!(out.contains("fixture text"));
    }
    unsafe {
        extern "C" {
            fn kill(pid: i32, signal: i32) -> i32;
        }
        kill(child.id() as i32, 15);
    }
    assert!(child.wait().unwrap().success());
    fs::remove_file(config).ok();
    assert_eq!(primary.finish().len(), 2);
    assert_eq!(secondary.finish().len(), 2);
}
