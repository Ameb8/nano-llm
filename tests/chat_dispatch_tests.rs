use nano_llm::{
    app_with_provider_factory, CanonicalRequest, ChatChoice, ChatResponse, FinishReason,
    HttpRequest, Provider, ProviderFuture, ProviderStream, RuntimeConfig, RuntimeGeneralSettings,
    RuntimeRoute, RuntimeTarget, TargetError,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

fn target(suffix: &str) -> RuntimeTarget {
    RuntimeTarget {
        model: format!("openai/{suffix}"),
        provider: nano_llm::ProviderKind::OpenAi,
        model_suffix: suffix.into(),
        api_key: None,
        api_base: "https://example.test/v1".into(),
        timeout: 30,
        explicit_timeout: None,
    }
}

fn application(
    targets: Vec<RuntimeTarget>,
    seen_targets: Arc<Mutex<Vec<String>>>,
    seen_requests: Arc<Mutex<Vec<CanonicalRequest>>>,
    fail: bool,
) -> nano_llm::Application {
    let config = RuntimeConfig {
        general_settings: RuntimeGeneralSettings {
            max_in_flight: 1,
            ..RuntimeGeneralSettings::default()
        },
        routes: vec![RuntimeRoute {
            model_name: "public".into(),
            targets,
        }],
    };
    let factory = Arc::new(move |target: RuntimeTarget| -> Box<dyn Provider> {
        seen_targets.lock().unwrap().push(target.model_suffix);
        Box::new(RecordingProvider {
            seen_requests: seen_requests.clone(),
            fail,
        })
    });
    app_with_provider_factory(config, true, factory)
}

struct RecordingProvider {
    seen_requests: Arc<Mutex<Vec<CanonicalRequest>>>,
    fail: bool,
}

impl Provider for RecordingProvider {
    fn complete<'a>(
        &'a self,
        request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ChatResponse, TargetError>> {
        self.seen_requests.lock().unwrap().push(request.clone());
        let result = if self.fail {
            Err(TargetError::connection())
        } else {
            Ok(ChatResponse {
                id: "complete-id".into(),
                object: "chat.completion",
                created: 42,
                model: request.model.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: nano_llm::AssistantMessage {
                        role: "assistant",
                        content: Some("complete reply".into()),
                        tool_calls: None,
                    },
                    finish_reason: FinishReason::Stop,
                }],
                usage: None,
            })
        };
        Box::pin(async move { result })
    }

    fn complete_stream<'a>(
        &'a self,
        _request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>> {
        Box::pin(async { Err(TargetError::invalid_response()) })
    }
}

fn chat(body: &'static [u8]) -> HttpRequest {
    HttpRequest::new("POST", "/v1/chat/completions")
        .with_header("content-type", "application/json")
        .with_body(body)
}

#[test]
fn unknown_model_and_canonical_failures_do_not_construct_a_provider() {
    let targets = Arc::new(Mutex::new(Vec::new()));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let application = application(
        vec![target("first")],
        targets.clone(),
        requests.clone(),
        false,
    );

    let unknown = application.handle(&chat(
        br#"{"model":"missing","messages":[{"role":"user","content":"hello"}]}"#,
    ));
    assert_eq!(unknown.status, 404);
    assert_eq!(unknown.body, br#"{"error":{"message":"Model not found","type":"not_found_error","param":"model","code":"model_not_found"}}"#);

    let invalid = application.handle(&chat(
        br#"{"model":"public","messages":[{"role":"assistant","content":"bad"}]}"#,
    ));
    assert_eq!(invalid.status, 400);
    assert!(std::str::from_utf8(&invalid.body)
        .unwrap()
        .contains("invalid_request"));
    assert!(targets.lock().unwrap().is_empty());
    assert!(requests.lock().unwrap().is_empty());
}

#[test]
fn first_target_receives_one_immutable_canonical_request_and_full_success_is_serialized() {
    let targets = Arc::new(Mutex::new(Vec::new()));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let application = application(
        vec![target("first"), target("must-not-run")],
        targets.clone(),
        requests.clone(),
        false,
    );
    let response = application.handle(&chat(
        br#"{"model":"public","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":7}"#,
    ));

    assert_eq!(response.status, 200);
    assert_eq!(response.body, br#"{"id":"complete-id","object":"chat.completion","created":42,"model":"public","choices":[{"index":0,"message":{"role":"assistant","content":"complete reply"},"finish_reason":"stop"}]}"#);
    assert_eq!(*targets.lock().unwrap(), vec!["first"]);
    let recorded = requests.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].model, "public");
    assert_eq!(recorded[0].max_tokens, Some(7));
    assert_eq!(recorded[0].fields[0].0, "model");
}

#[test]
fn first_target_failure_is_a_safe_502_and_never_tries_a_fallback() {
    let targets = Arc::new(Mutex::new(Vec::new()));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let application = application(
        vec![target("first"), target("fallback")],
        targets.clone(),
        requests,
        true,
    );
    let response = application.handle(&chat(
        br#"{"model":"public","messages":[{"role":"user","content":"hello"}]}"#,
    ));

    assert_eq!(response.status, 502);
    assert_eq!(response.body, br#"{"error":{"message":"All configured upstream targets failed","type":"server_error","param":null,"code":"upstream_exhausted"}}"#);
    assert_eq!(*targets.lock().unwrap(), vec!["first"]);
}

#[test]
fn route_aware_validation_and_stream_rejection_precede_provider_work_and_release_capacity() {
    let targets = Arc::new(Mutex::new(Vec::new()));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut anthropic = target("claude");
    anthropic.provider = nano_llm::ProviderKind::Anthropic;
    let application = application(vec![anthropic], targets.clone(), requests.clone(), false);

    let missing_limit = application.handle(&chat(
        br#"{"model":"public","messages":[{"role":"user","content":"hello"}]}"#,
    ));
    assert_eq!(missing_limit.status, 400);
    assert!(targets.lock().unwrap().is_empty());

    let stream = application.handle(&chat(
        br#"{"model":"public","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"stream":true}"#,
    ));
    assert_eq!(stream.status, 400);
    assert!(targets.lock().unwrap().is_empty());

    let valid =
        br#"{"model":"public","messages":[{"role":"user","content":"hello"}],"max_tokens":1}"#;
    assert_eq!(application.handle(&chat(valid)).status, 200);
    // The second request would receive 503 if the successful request's permit
    // were not released exactly once at the end of provider dispatch.
    assert_eq!(application.handle(&chat(valid)).status, 200);
    assert_eq!(*targets.lock().unwrap(), vec!["claude", "claude"]);
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[test]
fn downstream_cancellation_drops_active_provider_work() {
    struct PendingUntilDropped {
        cancellation: nano_llm::DownstreamCancellation,
        dropped: Arc<AtomicBool>,
    }
    impl std::future::Future for PendingUntilDropped {
        type Output = Result<ChatResponse, TargetError>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _context: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            self.cancellation.cancel();
            std::task::Poll::Pending
        }
    }
    impl Drop for PendingUntilDropped {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }
    struct PendingProvider {
        cancellation: nano_llm::DownstreamCancellation,
        dropped: Arc<AtomicBool>,
    }
    impl Provider for PendingProvider {
        fn complete<'a>(
            &'a self,
            _request: &'a CanonicalRequest,
        ) -> ProviderFuture<'a, Result<ChatResponse, TargetError>> {
            Box::pin(PendingUntilDropped {
                cancellation: self.cancellation.clone(),
                dropped: self.dropped.clone(),
            })
        }

        fn complete_stream<'a>(
            &'a self,
            _request: &'a CanonicalRequest,
        ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>> {
            Box::pin(async { Err(TargetError::invalid_response()) })
        }
    }

    let cancellation = nano_llm::DownstreamCancellation::default();
    let dropped = Arc::new(AtomicBool::new(false));
    let cancellation_for_factory = cancellation.clone();
    let dropped_for_factory = dropped.clone();
    let config = RuntimeConfig {
        general_settings: RuntimeGeneralSettings::default(),
        routes: vec![RuntimeRoute {
            model_name: "public".into(),
            targets: vec![target("first")],
        }],
    };
    let application = app_with_provider_factory(
        config,
        true,
        Arc::new(move |_| {
            Box::new(PendingProvider {
                cancellation: cancellation_for_factory.clone(),
                dropped: dropped_for_factory.clone(),
            })
        }),
    );

    let response = application.handle(
        &chat(br#"{"model":"public","messages":[{"role":"user","content":"hello"}]}"#)
            .with_downstream_cancellation(cancellation),
    );
    assert_eq!(response.status, 502);
    assert!(dropped.load(Ordering::Acquire));
}
