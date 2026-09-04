//! OpenAI-compatible non-streaming provider family.
//!
//! OpenAI, Mistral, DeepSeek, and generic compatible targets intentionally
//! share this one implementation.  Their differences have already been
//! validated into [`RuntimeTarget`]; this module only applies the common wire
//! protocol.

use crate::config::{ProviderKind, RuntimeTarget};
use crate::providers::{
    OutboundRequest, OutboundTransport, Provider, ProviderFuture, ProviderStream,
    SecureTransportPolicy, TargetError, TransportErrorKind,
};
use crate::request::{CanonicalRequest, JsonValue};
use crate::response::{
    normalize_response, normalize_usage, NativeChoice, NativeResponse, NativeTerminal,
    NativeToolCall, ResponseError, ResponseMetadata,
};
use std::sync::Arc;

/// Fixed maximum body retained for a buffered provider completion.
pub const MAX_BUFFERED_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// One target of the OpenAI-compatible provider family.
pub struct OpenAiCompatibleProvider {
    target: RuntimeTarget,
    transport: Arc<dyn OutboundTransport>,
    transport_policy: SecureTransportPolicy,
}

impl std::fmt::Debug for OpenAiCompatibleProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiCompatibleProvider")
            .field("target", &self.target)
            .field("transport_policy", &self.transport_policy)
            .finish_non_exhaustive()
    }
}

impl OpenAiCompatibleProvider {
    /// Construct the shared adapter for a validated compatible-family target.
    pub fn new(target: RuntimeTarget, transport: Arc<dyn OutboundTransport>) -> Self {
        debug_assert!(matches!(
            target.provider,
            ProviderKind::OpenAi
                | ProviderKind::Mistral
                | ProviderKind::DeepSeek
                | ProviderKind::OpenAiCompatible
        ));
        Self {
            target,
            transport,
            transport_policy: SecureTransportPolicy::default(),
        }
    }

    fn outbound_request(&self, request: &CanonicalRequest) -> OutboundRequest {
        let mut fields = request.fields.clone();
        replace_field(
            &mut fields,
            "model",
            JsonValue::String(self.target.model_suffix.clone()),
        );
        replace_field(&mut fields, "messages", compatible_messages(request));

        let mut headers = vec![("Content-Type".into(), "application/json".into())];
        if let Some(key) = &self.target.api_key {
            headers.push((
                "Authorization".into(),
                format!("Bearer {}", key.expose_secret()),
            ));
        }
        OutboundRequest {
            method: "POST",
            url: format!("{}/chat/completions", self.target.api_base),
            headers,
            body: encode_object(&fields).into_bytes(),
        }
    }
}

impl Provider for OpenAiCompatibleProvider {
    fn complete<'a>(
        &'a self,
        request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<crate::response::ChatResponse, TargetError>> {
        Box::pin(async move {
            let response = self
                .transport
                .execute(self.transport_policy, self.outbound_request(request))
                .map_err(|error| match error.kind {
                    TransportErrorKind::Timeout => TargetError::timeout(),
                    TransportErrorKind::Connection => TargetError::connection(),
                })?;
            if !(200..300).contains(&response.status) {
                return Err(TargetError::from_upstream_status(response.status));
            }
            if response.body.len() > MAX_BUFFERED_RESPONSE_BYTES {
                return Err(TargetError::invalid_response());
            }
            let native = parse_response(&response.body, self.target.provider)?;
            normalize_response(request, ResponseMetadata::for_model(&request.model), native)
                .map_err(response_error)
        })
    }

    fn complete_stream<'a>(
        &'a self,
        _request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>> {
        // Streaming parsing is deliberately a separate delivery slice.  Do
        // not accidentally send a buffered request with stream semantics.
        Box::pin(async { Err(TargetError::invalid_response()) })
    }
}

fn response_error(error: ResponseError) -> TargetError {
    match error {
        ResponseError::Overloaded => TargetError::overloaded(),
        ResponseError::InvalidResponse(_) => TargetError::invalid_response(),
    }
}

fn parse_response(bytes: &[u8], provider: ProviderKind) -> Result<NativeResponse, TargetError> {
    let root =
        crate::request::decode_json_object(bytes).map_err(|_| TargetError::invalid_response())?;
    let choices = array(required(&root, "choices")?)?;
    let native_choices = choices
        .iter()
        .map(|choice| parse_choice(choice, provider))
        .collect::<Result<Vec<_>, _>>()?;
    let usage = match get(&root, "usage") {
        None | Some(JsonValue::Null) => None,
        Some(value) => parse_usage(value),
    };
    Ok(NativeResponse {
        choices: native_choices,
        usage,
    })
}

fn parse_choice(value: &JsonValue, provider: ProviderKind) -> Result<NativeChoice, TargetError> {
    let choice = object(value)?;
    let index = unsigned(required(choice, "index")?)?;
    let message = object(required(choice, "message")?)?;
    if !matches!(get(message, "role"), Some(JsonValue::String(role)) if role == "assistant") {
        return Err(TargetError::invalid_response());
    }
    let content = match get(message, "content") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::String(value)) => Some(value.clone()),
        Some(_) => return Err(TargetError::invalid_response()),
    };
    let calls = match get(message, "tool_calls") {
        None => Vec::new(),
        Some(value) => {
            let values = array(value)?;
            if values.is_empty() {
                return Err(TargetError::invalid_response());
            }
            values
                .iter()
                .map(parse_tool_call)
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let terminal = match required(choice, "finish_reason")? {
        JsonValue::String(value) => terminal(value, provider),
        // A non-terminal choice in a non-streaming response is malformed.
        JsonValue::Null => NativeTerminal::Invalid,
        _ => return Err(TargetError::invalid_response()),
    };
    Ok(NativeChoice {
        index,
        content,
        calls,
        terminal,
    })
}

fn parse_tool_call(value: &JsonValue) -> Result<NativeToolCall, TargetError> {
    let call = object(value)?;
    let id = match get(call, "id") {
        None => None,
        Some(JsonValue::String(value)) => Some(value.clone()),
        Some(_) => return Err(TargetError::invalid_response()),
    };
    if !matches!(get(call, "type"), Some(JsonValue::String(kind)) if kind == "function") {
        return Err(TargetError::invalid_response());
    }
    let function = object(required(call, "function")?)?;
    let name = string(required(function, "name")?)?.to_owned();
    let arguments = string(required(function, "arguments")?)?.to_owned();
    Ok(NativeToolCall {
        id,
        name,
        arguments,
    })
}

fn parse_usage(value: &JsonValue) -> Option<crate::response::Usage> {
    let JsonValue::Object(usage) = value else {
        return None;
    };
    // Invalid usage is intentionally omitted, rather than turning an otherwise
    // valid completion into a failed attempt.
    let prompt = get(usage, "prompt_tokens").and_then(unsigned_optional);
    let completion = get(usage, "completion_tokens").and_then(unsigned_optional);
    normalize_usage(prompt, completion)
}

fn unsigned_optional(value: &JsonValue) -> Option<u128> {
    unsigned(value).ok().map(u128::from)
}

fn terminal(value: &str, provider: ProviderKind) -> NativeTerminal {
    match value {
        "stop" => NativeTerminal::Stop,
        "length" => NativeTerminal::Length,
        "tool_calls" => NativeTerminal::ToolCalls,
        "content_filter" => NativeTerminal::ContentFilter,
        "insufficient_system_resource" if provider == ProviderKind::DeepSeek => {
            NativeTerminal::Overloaded
        }
        _ => NativeTerminal::Unknown,
    }
}

fn required<'a>(
    fields: &'a [(String, JsonValue)],
    name: &str,
) -> Result<&'a JsonValue, TargetError> {
    get(fields, name).ok_or_else(TargetError::invalid_response)
}

fn get<'a>(fields: &'a [(String, JsonValue)], name: &str) -> Option<&'a JsonValue> {
    fields
        .iter()
        .find_map(|(key, value)| (key == name).then_some(value))
}

fn object(value: &JsonValue) -> Result<&[(String, JsonValue)], TargetError> {
    match value {
        JsonValue::Object(value) => Ok(value),
        _ => Err(TargetError::invalid_response()),
    }
}

fn array(value: &JsonValue) -> Result<&[JsonValue], TargetError> {
    match value {
        JsonValue::Array(value) => Ok(value),
        _ => Err(TargetError::invalid_response()),
    }
}

fn string(value: &JsonValue) -> Result<&str, TargetError> {
    match value {
        JsonValue::String(value) => Ok(value),
        _ => Err(TargetError::invalid_response()),
    }
}

fn unsigned(value: &JsonValue) -> Result<u64, TargetError> {
    let JsonValue::Number(value) = value else {
        return Err(TargetError::invalid_response());
    };
    if value.contains(['.', 'e', 'E']) {
        return Err(TargetError::invalid_response());
    }
    value.parse().map_err(|_| TargetError::invalid_response())
}

fn compatible_messages(request: &CanonicalRequest) -> JsonValue {
    let Some(JsonValue::Array(messages)) = get(&request.fields, "messages") else {
        // The canonical decoder guarantees this. Keep the adapter defensive.
        return JsonValue::Array(Vec::new());
    };
    let leading = messages
        .iter()
        .take_while(|message| matches!(role(message), Some("system" | "developer")))
        .count();
    if leading == 0 {
        return JsonValue::Array(messages.clone());
    }
    let mut result = Vec::with_capacity(messages.len() - leading + 1);
    result.push(JsonValue::Object(vec![
        ("role".into(), JsonValue::String("system".into())),
        (
            "content".into(),
            JsonValue::String(request.instruction.clone().unwrap_or_default()),
        ),
    ]));
    result.extend_from_slice(&messages[leading..]);
    JsonValue::Array(result)
}

fn role(value: &JsonValue) -> Option<&str> {
    let JsonValue::Object(fields) = value else {
        return None;
    };
    match get(fields, "role") {
        Some(JsonValue::String(role)) => Some(role),
        _ => None,
    }
}

fn replace_field(fields: &mut Vec<(String, JsonValue)>, name: &str, value: JsonValue) {
    if let Some((_, existing)) = fields.iter_mut().find(|(key, _)| key == name) {
        *existing = value;
    } else {
        fields.push((name.into(), value));
    }
}

fn encode_object(fields: &[(String, JsonValue)]) -> String {
    encode(&JsonValue::Object(fields.to_vec()))
}

fn encode(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".into(),
        JsonValue::Bool(value) => value.to_string(),
        JsonValue::Number(value) => value.clone(),
        JsonValue::String(value) => quote(value),
        JsonValue::Array(values) => format!(
            "[{}]",
            values.iter().map(encode).collect::<Vec<_>>().join(",")
        ),
        JsonValue::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!("{}:{}", quote(key), encode(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn quote(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0C}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1F}' => {
                use std::fmt::Write as _;
                let _ = write!(output, "\\u{:04x}", character as u32);
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SecretString;
    use crate::providers::{OutboundResponse, TransportError};
    use crate::request::decode_chat_request;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    struct CaptureTransport {
        requests: Mutex<Vec<OutboundRequest>>,
        response: Mutex<Result<OutboundResponse, TransportError>>,
    }

    impl CaptureTransport {
        fn returning(status: u16, body: impl Into<Vec<u8>>) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: Mutex::new(Ok(OutboundResponse {
                    status,
                    body: body.into(),
                })),
            }
        }
        fn request(&self) -> OutboundRequest {
            self.requests.lock().unwrap()[0].clone()
        }
    }

    impl OutboundTransport for CaptureTransport {
        fn execute(
            &self,
            policy: SecureTransportPolicy,
            request: OutboundRequest,
        ) -> Result<OutboundResponse, TransportError> {
            assert_eq!(policy, SecureTransportPolicy::default());
            self.requests.lock().unwrap().push(request);
            self.response.lock().unwrap().clone()
        }
    }

    fn target(provider: ProviderKind, key: Option<&str>) -> RuntimeTarget {
        RuntimeTarget {
            model: format!("{}/upstream-model", provider.name()),
            provider,
            model_suffix: "upstream-model".into(),
            api_key: key.map(|key| SecretString::new(key.into())),
            api_base: "https://capture.test/preset".into(),
            timeout: 30,
            explicit_timeout: None,
        }
    }

    fn request() -> CanonicalRequest {
        decode_chat_request(
            br#"{"model":"alias","messages":[{"role":"system","content":"s"},{"role":"developer","content":"d"},{"role":"user","content":"hello"}],"max_tokens":12,"temperature":0.2,"top_p":0.8,"stop":["END"],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","x-any":"kept"}}}],"tool_choice":"auto"}"#,
        )
        .unwrap()
    }

    const TEXT_RESPONSE: &str = r#"{"id":"upstream","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":99}}"#;

    #[test]
    fn all_presets_share_the_exact_compatible_wire_contract() {
        for provider in [
            ProviderKind::OpenAi,
            ProviderKind::Mistral,
            ProviderKind::DeepSeek,
            ProviderKind::OpenAiCompatible,
        ] {
            let transport = Arc::new(CaptureTransport::returning(200, TEXT_RESPONSE));
            let adapter =
                OpenAiCompatibleProvider::new(target(provider, Some("key")), transport.clone());
            let response = block_on(adapter.complete(&request())).unwrap();
            assert_eq!(response.model, "alias");
            assert_eq!(response.usage.unwrap().total_tokens, 5);
            let captured = transport.request();
            assert_eq!(captured.method, "POST");
            assert_eq!(captured.url, "https://capture.test/preset/chat/completions");
            assert_eq!(
                captured.headers,
                vec![
                    ("Content-Type".into(), "application/json".into()),
                    ("Authorization".into(), "Bearer key".into()),
                ]
            );
            let fields = crate::request::decode_json_object(&captured.body).unwrap();
            assert_eq!(
                get(&fields, "model"),
                Some(&JsonValue::String("upstream-model".into()))
            );
            assert_eq!(
                get(&fields, "max_tokens"),
                Some(&JsonValue::Number("12".into()))
            );
            assert!(get(&fields, "tools").is_some());
            assert!(get(&fields, "tool_choice").is_some());
            let JsonValue::Array(messages) = get(&fields, "messages").unwrap() else {
                panic!()
            };
            assert_eq!(messages.len(), 2);
            let JsonValue::Object(system) = &messages[0] else {
                panic!()
            };
            assert_eq!(
                get(system, "role"),
                Some(&JsonValue::String("system".into()))
            );
            assert_eq!(
                get(system, "content"),
                Some(&JsonValue::String("s\n\nd".into()))
            );
        }
    }

    #[test]
    fn keyless_compatible_target_omits_authorization_and_override_is_only_a_base() {
        let transport = Arc::new(CaptureTransport::returning(200, TEXT_RESPONSE));
        let mut branded = target(ProviderKind::DeepSeek, Some("key"));
        branded.api_base = "https://override.test/custom-version".into();
        let adapter = OpenAiCompatibleProvider::new(branded, transport.clone());
        block_on(adapter.complete(&request())).unwrap();
        assert_eq!(
            transport.request().url,
            "https://override.test/custom-version/chat/completions"
        );

        let keyless = Arc::new(CaptureTransport::returning(200, TEXT_RESPONSE));
        let adapter = OpenAiCompatibleProvider::new(
            target(ProviderKind::OpenAiCompatible, None),
            keyless.clone(),
        );
        block_on(adapter.complete(&request())).unwrap();
        assert_eq!(
            keyless.request().headers,
            vec![("Content-Type".into(), "application/json".into())]
        );
    }

    #[test]
    fn invalid_successes_and_over_limit_body_are_target_errors() {
        for body in [
            b"not json".to_vec(),
            br#"{"choices":[]}"#.to_vec(),
            br#"{"choices":[{"index":0,"message":{"content":"x"},"finish_reason":"tool_calls"}]}"#
                .to_vec(),
        ] {
            let transport = Arc::new(CaptureTransport::returning(200, body));
            let adapter =
                OpenAiCompatibleProvider::new(target(ProviderKind::OpenAi, Some("key")), transport);
            assert_eq!(
                block_on(adapter.complete(&request())).unwrap_err().kind,
                crate::providers::TargetErrorKind::InvalidResponse
            );
        }
        let transport = Arc::new(CaptureTransport::returning(
            200,
            vec![b'x'; MAX_BUFFERED_RESPONSE_BYTES + 1],
        ));
        let adapter =
            OpenAiCompatibleProvider::new(target(ProviderKind::OpenAi, Some("key")), transport);
        assert_eq!(
            block_on(adapter.complete(&request())).unwrap_err().kind,
            crate::providers::TargetErrorKind::InvalidResponse
        );

        let mut at_limit = TEXT_RESPONSE.as_bytes().to_vec();
        at_limit.resize(MAX_BUFFERED_RESPONSE_BYTES, b' ');
        let transport = Arc::new(CaptureTransport::returning(200, at_limit));
        let adapter =
            OpenAiCompatibleProvider::new(target(ProviderKind::OpenAi, Some("key")), transport);
        assert!(block_on(adapter.complete(&request())).is_ok());
    }

    #[test]
    fn deepseek_operational_exhaustion_is_overloaded_and_tool_calls_normalize() {
        let exhausted = Arc::new(CaptureTransport::returning(
            200,
            br#"{"choices":[{"index":0,"message":{"role":"assistant","content":"x"},"finish_reason":"insufficient_system_resource"}]}"#,
        ));
        let adapter =
            OpenAiCompatibleProvider::new(target(ProviderKind::DeepSeek, Some("key")), exhausted);
        assert_eq!(
            block_on(adapter.complete(&request())).unwrap_err().kind,
            crate::providers::TargetErrorKind::Overloaded
        );

        let tool_response = Arc::new(CaptureTransport::returning(
            200,
            br#"{"choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"native","type":"function","function":{"name":"weather","arguments":"not-json"}}]},"finish_reason":"stop"}]}"#,
        ));
        let adapter =
            OpenAiCompatibleProvider::new(target(ProviderKind::OpenAi, Some("key")), tool_response);
        let response = block_on(adapter.complete(&request())).unwrap();
        assert_eq!(response.choices[0].finish_reason.as_str(), "tool_calls");
        assert_eq!(
            response.choices[0].message.tool_calls.as_ref().unwrap()[0]
                .function
                .arguments,
            "not-json"
        );
        assert!(response.id.starts_with("chatcmpl-"));
    }

    #[test]
    fn status_and_transport_failures_are_safely_classified() {
        for (status, kind) in [
            (302, crate::providers::TargetErrorKind::UpstreamHttp),
            (401, crate::providers::TargetErrorKind::Authentication),
            (403, crate::providers::TargetErrorKind::PermissionDenied),
            (429, crate::providers::TargetErrorKind::RateLimited),
            (503, crate::providers::TargetErrorKind::Overloaded),
            (500, crate::providers::TargetErrorKind::UpstreamHttp),
        ] {
            let transport = Arc::new(CaptureTransport::returning(status, b"ignored".to_vec()));
            let adapter =
                OpenAiCompatibleProvider::new(target(ProviderKind::OpenAi, Some("key")), transport);
            assert_eq!(
                block_on(adapter.complete(&request())).unwrap_err().kind,
                kind
            );
        }
        let transport = Arc::new(CaptureTransport {
            requests: Mutex::new(Vec::new()),
            response: Mutex::new(Err(TransportError {
                kind: TransportErrorKind::Timeout,
            })),
        });
        let adapter =
            OpenAiCompatibleProvider::new(target(ProviderKind::OpenAi, Some("key")), transport);
        assert_eq!(
            block_on(adapter.complete(&request())).unwrap_err().kind,
            crate::providers::TargetErrorKind::Timeout
        );
    }

    fn block_on<T>(mut future: Pin<Box<dyn Future<Output = T> + Send + '_>>) -> T {
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
