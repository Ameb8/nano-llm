use nano_llm::{
    app, build_runtime_config, parse_yaml_str, GatewayError, GatewayErrorKind, HttpRequest,
};
use std::collections::HashMap;

fn application(no_auth: bool) -> nano_llm::Application {
    let yaml = r#"
model_list:
  - model_name: alpha
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OAI_KEY
  - model_name: beta
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OAI_KEY
  - model_name: alpha
    litellm_params:
      model: mistral/mistral-small
      api_key: os.environ/MIS_KEY
general_settings:
  master_key: os.environ/MASTER_KEY
"#;
    let env = HashMap::from([
        ("OAI_KEY".to_owned(), "openai-key".to_owned()),
        ("MIS_KEY".to_owned(), "mistral-key".to_owned()),
        ("MASTER_KEY".to_owned(), "master-key".to_owned()),
    ]);
    app(
        build_runtime_config(parse_yaml_str(yaml).unwrap(), no_auth, &env).unwrap(),
        no_auth,
    )
}

fn id(response: &nano_llm::HttpResponse) -> String {
    let values: Vec<_> = response.header_values("x-request-id").collect();
    assert_eq!(values.len(), 1);
    values[0].to_owned()
}

fn is_uuid_v4(value: &str) -> bool {
    value.len() == 36
        && value.as_bytes()[8] == b'-'
        && value.as_bytes()[13] == b'-'
        && value.as_bytes()[18] == b'-'
        && value.as_bytes()[23] == b'-'
        && value.as_bytes()[14] == b'4'
        && matches!(value.as_bytes()[19], b'8' | b'9' | b'a' | b'b')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

#[test]
fn request_id_preserves_only_a_single_valid_raw_header() {
    let application = application(true);
    let valid = application
        .handle(&HttpRequest::new("GET", "/health").with_header("X-Request-Id", "abc,def"));
    assert_eq!(id(&valid), "abc,def");

    for request in [
        HttpRequest::new("GET", "/health"),
        HttpRequest::new("GET", "/health")
            .with_header("x-request-id", "one")
            .with_header("x-request-id", "two"),
        HttpRequest::new("GET", "/health").with_header("x-request-id", ""),
        HttpRequest::new("GET", "/health").with_header("x-request-id", b"bad\nvalue".to_vec()),
        HttpRequest::new("GET", "/health").with_header("x-request-id", " "),
        HttpRequest::new("GET", "/health").with_header("x-request-id", "x".repeat(129)),
    ] {
        assert!(is_uuid_v4(&id(&application.handle(&request))));
    }
}

#[test]
fn health_and_models_have_exact_bodies_and_one_request_id() {
    let application = application(true);
    let health = application
        .handle(&HttpRequest::new("GET", "/health").with_header("x-request-id", "health-id"));
    assert_eq!(health.status, 200);
    assert_eq!(
        health.header_values("content-type").collect::<Vec<_>>(),
        ["application/json"]
    );
    assert_eq!(health.body, br#"{"status":"ok"}"#);
    assert_eq!(id(&health), "health-id");

    let models = application
        .handle(&HttpRequest::new("GET", "/v1/models").with_header("x-request-id", "models-id"));
    assert_eq!(models.status, 200);
    assert_eq!(models.body, br#"{"object":"list","data":[{"id":"alpha","object":"model","created":0,"owned_by":"nano-llm"},{"id":"beta","object":"model","created":0,"owned_by":"nano-llm"}]}"#);
    assert_eq!(id(&models), "models-id");
}

#[test]
fn route_mismatches_are_404_and_v1_mismatches_authenticate_first() {
    let application = application(false);
    let unauthenticated = application
        .handle(&HttpRequest::new("POST", "/v1/models").with_header("x-request-id", "unauth"));
    assert_eq!(unauthenticated.status, 401);
    assert_eq!(String::from_utf8(unauthenticated.body).unwrap(), "{\"error\":{\"message\":\"Authentication failed\",\"type\":\"authentication_error\",\"param\":null,\"code\":\"authentication_failed\"}}");

    let authorized = application.handle(
        &HttpRequest::new("POST", "/v1/models")
            .with_header("authorization", "Bearer master-key")
            .with_header("x-request-id", "authorized"),
    );
    assert_eq!(authorized.status, 404);
    assert_eq!(std::str::from_utf8(&authorized.body).unwrap(), "{\"error\":{\"message\":\"Route not found\",\"type\":\"not_found_error\",\"param\":null,\"code\":\"route_not_found\"}}");
    assert_eq!(id(&authorized), "authorized");

    let public = application.handle(&HttpRequest::new("POST", "/health"));
    assert_eq!(public.status, 404);
    assert_eq!(public.body, br#"{"error":{"message":"Route not found","type":"not_found_error","param":null,"code":"route_not_found"}}"#);
}

#[test]
fn every_stable_error_condition_has_the_canonical_envelope() {
    let cases = [
        (
            GatewayErrorKind::InvalidRequest,
            400,
            "invalid_request_error",
            "invalid_request",
        ),
        (
            GatewayErrorKind::InvalidJson,
            400,
            "invalid_request_error",
            "invalid_json",
        ),
        (
            GatewayErrorKind::AuthenticationFailed,
            401,
            "authentication_error",
            "authentication_failed",
        ),
        (
            GatewayErrorKind::ModelNotFound,
            404,
            "not_found_error",
            "model_not_found",
        ),
        (
            GatewayErrorKind::RouteNotFound,
            404,
            "not_found_error",
            "route_not_found",
        ),
        (
            GatewayErrorKind::RequestBodyTimeout,
            408,
            "invalid_request_error",
            "request_body_timeout",
        ),
        (
            GatewayErrorKind::RequestTooLarge,
            413,
            "invalid_request_error",
            "request_too_large",
        ),
        (
            GatewayErrorKind::UnsupportedMediaType,
            415,
            "invalid_request_error",
            "unsupported_media_type",
        ),
        (
            GatewayErrorKind::UpstreamExhausted,
            502,
            "server_error",
            "upstream_exhausted",
        ),
        (
            GatewayErrorKind::CapacityExhausted,
            503,
            "server_error",
            "capacity_exhausted",
        ),
        (
            GatewayErrorKind::OverallTimeout,
            504,
            "server_error",
            "overall_timeout",
        ),
    ];
    for (kind, status, error_type, code) in cases {
        let response = GatewayError::new(kind, "safe", Some("model".to_owned())).response();
        assert_eq!(response.status, status);
        assert_eq!(
            response.header_values("content-type").collect::<Vec<_>>(),
            ["application/json"]
        );
        let param = match kind {
            GatewayErrorKind::InvalidRequest | GatewayErrorKind::ModelNotFound => "\"model\"",
            _ => "null",
        };
        assert_eq!(String::from_utf8(response.body).unwrap(), format!("{{\"error\":{{\"message\":\"safe\",\"type\":\"{error_type}\",\"param\":{param},\"code\":\"{code}\"}}}}"));
    }
}
