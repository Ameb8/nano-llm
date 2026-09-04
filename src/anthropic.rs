//! Anthropic Messages translation.
//!
//! The buffered response parser is deliberately strict: a 2xx status only
//! succeeds after its Messages payload has been converted through the shared
//! canonical response conformance boundary.  SSE remains a separate slice.

use crate::config::{ProviderKind, RuntimeTarget};
use crate::providers::{
    OutboundRequest, OutboundTransport, Provider, ProviderFuture, ProviderStream,
    SecureTransportPolicy, TargetError, TransportErrorKind,
};
use crate::request::{decode_json_value, CanonicalRequest, JsonValue, ToolChoice};
use crate::response::{
    build_response, normalize_usage, ChatResponse, NativeTerminal, NativeToolCall, ResponseError,
    ResponseMetadata,
};
use std::sync::Arc;

/// Anthropic Messages adapter for one validated target.
pub struct AnthropicProvider {
    target: RuntimeTarget,
    transport: Arc<dyn OutboundTransport>,
    transport_policy: SecureTransportPolicy,
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicProvider")
            .field("target", &self.target)
            .field("transport_policy", &self.transport_policy)
            .finish_non_exhaustive()
    }
}

impl AnthropicProvider {
    pub fn new(target: RuntimeTarget, transport: Arc<dyn OutboundTransport>) -> Self {
        debug_assert_eq!(target.provider, ProviderKind::Anthropic);
        Self {
            target,
            transport,
            transport_policy: SecureTransportPolicy::default(),
        }
    }

    fn outbound_request(&self, request: &CanonicalRequest, stream: bool) -> OutboundRequest {
        let mut body = vec![
            (
                "model".into(),
                JsonValue::String(self.target.model_suffix.clone()),
            ),
            (
                "max_tokens".into(),
                JsonValue::Number(
                    request
                        .max_tokens
                        .expect("route validation requires max_tokens")
                        .to_string(),
                ),
            ),
            ("messages".into(), JsonValue::Array(messages(request))),
        ];
        if let Some(system) = &request.instruction {
            body.push(("system".into(), JsonValue::String(system.clone())));
        }
        if let Some(temperature) = request.temperature {
            body.push((
                "temperature".into(),
                JsonValue::Number(temperature.to_string()),
            ));
        }
        if let Some(top_p) = request.top_p {
            body.push(("top_p".into(), JsonValue::Number(top_p.to_string())));
        }
        if let Some(stop) = &request.stop {
            body.push((
                "stop_sequences".into(),
                JsonValue::Array(stop.iter().cloned().map(JsonValue::String).collect()),
            ));
        }
        if let Some(tools) = field(&request.fields, "tools") {
            body.push(("tools".into(), anthropic_tools(tools)));
            body.push(("tool_choice".into(), tool_choice(&request.tool_choice)));
        }
        if stream {
            body.push(("stream".into(), JsonValue::Bool(true)));
        }

        OutboundRequest {
            method: "POST",
            url: format!("{}/messages", self.target.api_base),
            headers: vec![
                ("Content-Type".into(), "application/json".into()),
                (
                    "x-api-key".into(),
                    self.target
                        .api_key
                        .as_ref()
                        .expect("validated Anthropic key")
                        .expose_secret()
                        .into(),
                ),
                ("anthropic-version".into(), "2023-06-01".into()),
            ],
            body: encode(&JsonValue::Object(body)).into_bytes(),
        }
    }
}

impl Provider for AnthropicProvider {
    fn complete<'a>(
        &'a self,
        request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ChatResponse, TargetError>> {
        Box::pin(async move {
            if request.max_tokens.is_none() {
                return Err(TargetError::invalid_response());
            }
            let response = self
                .transport
                .execute(self.transport_policy, self.outbound_request(request, false))
                .map_err(transport_error)?;
            if !(200..300).contains(&response.status) {
                return Err(TargetError::from_upstream_status(response.status));
            }
            if response.body.len() > MAX_BUFFERED_RESPONSE_BYTES {
                return Err(TargetError::invalid_response());
            }
            let native = parse_response(&response.body)?;
            build_response(
                request,
                ResponseMetadata::for_model(&request.model),
                native.content,
                native.calls,
                native.terminal,
                native.usage,
            )
            .map_err(response_error)
        })
    }

    fn complete_stream<'a>(
        &'a self,
        request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>> {
        Box::pin(async move {
            if request.max_tokens.is_none() {
                return Err(TargetError::invalid_response());
            }
            let response = self
                .transport
                .execute_stream(self.transport_policy, self.outbound_request(request, true))
                .map_err(transport_error)?;
            if !(200..300).contains(&response.status) {
                return Err(TargetError::from_upstream_status(response.status));
            }
            Err(TargetError::invalid_response())
        })
    }
}

/// Fixed bound for a buffered native completion.  This matches the shared
/// compatible-family bound while keeping this adapter independently safe.
const MAX_BUFFERED_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

struct NativeMessage {
    content: Option<String>,
    calls: Vec<NativeToolCall>,
    terminal: NativeTerminal,
    usage: Option<crate::response::Usage>,
}

fn response_error(error: ResponseError) -> TargetError {
    match error {
        ResponseError::Overloaded => TargetError::overloaded(),
        ResponseError::InvalidResponse(_) => TargetError::invalid_response(),
    }
}

/// Decode exactly the Messages response fields that define canonical output.
/// Provider response IDs and all other provider-only metadata are ignored.
fn parse_response(bytes: &[u8]) -> Result<NativeMessage, TargetError> {
    let root =
        crate::request::decode_json_object(bytes).map_err(|_| TargetError::invalid_response())?;
    if !matches!(get(&root, "type"), Some(JsonValue::String(kind)) if kind == "message")
        || !matches!(get(&root, "role"), Some(JsonValue::String(role)) if role == "assistant")
    {
        return Err(TargetError::invalid_response());
    }
    let blocks = array(required(&root, "content")?)?;
    let (content, calls) = parse_blocks(blocks)?;
    let terminal = match required(&root, "stop_reason")? {
        JsonValue::String(reason) => terminal(reason),
        // A non-streaming response must be terminal.
        _ => return Err(TargetError::invalid_response()),
    };
    let usage = match get(&root, "usage") {
        None | Some(JsonValue::Null) => None,
        Some(value) => parse_usage(value),
    };
    Ok(NativeMessage {
        content,
        calls,
        terminal,
        usage,
    })
}

/// Anthropic content blocks may interleave text and tool uses.  The canonical
/// representation has one text field plus an ordered call list, so text is
/// joined with no synthetic separator and tool calls retain their native order.
fn parse_blocks(
    blocks: &[JsonValue],
) -> Result<(Option<String>, Vec<NativeToolCall>), TargetError> {
    let mut text = String::new();
    let mut saw_text = false;
    let mut calls = Vec::new();
    for block in blocks {
        let block = object(block)?;
        match string(required(block, "type")?)? {
            "text" => {
                text.push_str(string(required(block, "text")?)?);
                saw_text = true;
            }
            "tool_use" => {
                let id = match get(block, "id") {
                    None => None,
                    Some(JsonValue::String(id)) => Some(id.clone()),
                    Some(_) => return Err(TargetError::invalid_response()),
                };
                let name = string(required(block, "name")?)?.to_owned();
                let input = object(required(block, "input")?)?;
                calls.push(NativeToolCall {
                    id,
                    name,
                    arguments: encode(&JsonValue::Object(input.to_vec())),
                });
            }
            // v0.1 only has canonical counterparts for text and completed
            // function calls.  Silently dropping a native block would make a
            // malformed/unsupported 2xx response look successful.
            _ => return Err(TargetError::invalid_response()),
        }
    }
    Ok((saw_text.then_some(text), calls))
}

fn parse_usage(value: &JsonValue) -> Option<crate::response::Usage> {
    let JsonValue::Object(usage) = value else {
        return None;
    };
    normalize_usage(
        get(usage, "input_tokens").and_then(unsigned_optional),
        get(usage, "output_tokens").and_then(unsigned_optional),
    )
}

fn unsigned_optional(value: &JsonValue) -> Option<u128> {
    unsigned(value).ok().map(u128::from)
}

fn terminal(reason: &str) -> NativeTerminal {
    match reason {
        "end_turn" | "stop_sequence" => NativeTerminal::Stop,
        "max_tokens" | "model_context_window_exceeded" => NativeTerminal::Length,
        "tool_use" => NativeTerminal::ToolCalls,
        "refusal" => NativeTerminal::ContentFilter,
        // Server-side tool continuation has no portable v0.1 representation.
        "pause_turn" => NativeTerminal::Invalid,
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
        JsonValue::Object(fields) => Ok(fields),
        _ => Err(TargetError::invalid_response()),
    }
}

fn array(value: &JsonValue) -> Result<&[JsonValue], TargetError> {
    match value {
        JsonValue::Array(values) => Ok(values),
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

fn transport_error(error: crate::providers::TransportError) -> TargetError {
    match error.kind {
        TransportErrorKind::Timeout => TargetError::timeout(),
        TransportErrorKind::Connection => TargetError::connection(),
    }
}

fn messages(request: &CanonicalRequest) -> Vec<JsonValue> {
    let JsonValue::Array(source) = field(&request.fields, "messages").expect("validated messages")
    else {
        unreachable!()
    };
    let mut output = Vec::new();
    let mut index = 0;
    while index < source.len() {
        let JsonValue::Object(message) = &source[index] else {
            unreachable!()
        };
        let JsonValue::String(role) = field(message, "role").expect("validated role") else {
            unreachable!()
        };
        match role.as_str() {
            "system" | "developer" => index += 1,
            "user" => {
                output.push(native_message(
                    "user",
                    vec![text_block(string_field(message, "content").clone())],
                ));
                index += 1;
            }
            "assistant" => {
                let mut blocks = Vec::new();
                if let Some(JsonValue::String(content)) = field(message, "content") {
                    blocks.push(text_block(content.clone()));
                }
                if let Some(JsonValue::Array(calls)) = field(message, "tool_calls") {
                    for call in calls {
                        let JsonValue::Object(call) = call else {
                            unreachable!()
                        };
                        let JsonValue::Object(function) =
                            field(call, "function").expect("validated function")
                        else {
                            unreachable!()
                        };
                        let JsonValue::Object(input) =
                            decode_json_value(string_field(function, "arguments").as_bytes())
                                .expect("route-validated arguments")
                        else {
                            unreachable!()
                        };
                        blocks.push(JsonValue::Object(vec![
                            ("type".into(), JsonValue::String("tool_use".into())),
                            (
                                "id".into(),
                                JsonValue::String(string_field(call, "id").clone()),
                            ),
                            (
                                "name".into(),
                                JsonValue::String(string_field(function, "name").clone()),
                            ),
                            ("input".into(), JsonValue::Object(input)),
                        ]));
                    }
                }
                output.push(native_message("assistant", blocks));
                index += 1;
            }
            "tool" => {
                let mut blocks = Vec::new();
                while index < source.len() {
                    let JsonValue::Object(result) = &source[index] else {
                        unreachable!()
                    };
                    if !matches!(field(result, "role"), Some(JsonValue::String(value)) if value == "tool")
                    {
                        break;
                    }
                    blocks.push(JsonValue::Object(vec![
                        ("type".into(), JsonValue::String("tool_result".into())),
                        (
                            "tool_use_id".into(),
                            JsonValue::String(string_field(result, "tool_call_id").clone()),
                        ),
                        (
                            "content".into(),
                            JsonValue::String(string_field(result, "content").clone()),
                        ),
                    ]));
                    index += 1;
                }
                output.push(native_message("user", blocks));
            }
            _ => unreachable!(),
        }
    }
    output
}

fn native_message(role: &str, content: Vec<JsonValue>) -> JsonValue {
    JsonValue::Object(vec![
        ("role".into(), JsonValue::String(role.into())),
        ("content".into(), JsonValue::Array(content)),
    ])
}

fn text_block(text: String) -> JsonValue {
    JsonValue::Object(vec![
        ("type".into(), JsonValue::String("text".into())),
        ("text".into(), JsonValue::String(text)),
    ])
}

fn anthropic_tools(value: &JsonValue) -> JsonValue {
    let JsonValue::Array(tools) = value else {
        unreachable!()
    };
    JsonValue::Array(
        tools
            .iter()
            .map(|tool| {
                let JsonValue::Object(tool) = tool else {
                    unreachable!()
                };
                let JsonValue::Object(function) =
                    field(tool, "function").expect("validated function")
                else {
                    unreachable!()
                };
                let mut native = vec![(
                    "name".into(),
                    JsonValue::String(string_field(function, "name").clone()),
                )];
                if let Some(JsonValue::String(description)) = field(function, "description") {
                    native.push(("description".into(), JsonValue::String(description.clone())));
                }
                if let Some(parameters) = field(function, "parameters") {
                    native.push(("input_schema".into(), parameters.clone()));
                }
                JsonValue::Object(native)
            })
            .collect(),
    )
}

fn tool_choice(choice: &ToolChoice) -> JsonValue {
    let fields = match choice {
        ToolChoice::None => vec![("type".into(), JsonValue::String("none".into()))],
        ToolChoice::Auto => vec![("type".into(), JsonValue::String("auto".into()))],
        ToolChoice::Required => vec![("type".into(), JsonValue::String("any".into()))],
        ToolChoice::Named(name) => vec![
            ("type".into(), JsonValue::String("tool".into())),
            ("name".into(), JsonValue::String(name.clone())),
        ],
    };
    JsonValue::Object(fields)
}

fn field<'a>(fields: &'a [(String, JsonValue)], name: &str) -> Option<&'a JsonValue> {
    fields
        .iter()
        .find_map(|(key, value)| (key == name).then_some(value))
}

fn string_field<'a>(fields: &'a [(String, JsonValue)], name: &str) -> &'a String {
    let Some(JsonValue::String(value)) = field(fields, name) else {
        unreachable!()
    };
    value
}

fn encode(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "null".into(),
        JsonValue::Bool(value) => value.to_string(),
        JsonValue::Number(value) => value.clone(),
        JsonValue::String(value) => format!("\"{}\"", escape(value)),
        JsonValue::Array(values) => format!(
            "[{}]",
            values.iter().map(encode).collect::<Vec<_>>().join(",")
        ),
        JsonValue::Object(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| format!("\"{}\":{}", escape(key), encode(value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn escape(value: &str) -> String {
    let mut output = String::new();
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{1f}' => {
                output.push_str(&format!("\\u{:04x}", character as u32))
            }
            character => output.push(character),
        }
    }
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
    use std::task::{Context, Poll, Wake, Waker};

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

        fn execute_stream(
            &self,
            policy: SecureTransportPolicy,
            request: OutboundRequest,
        ) -> Result<crate::providers::OutboundStreamResponse, TransportError> {
            assert_eq!(policy, SecureTransportPolicy::default());
            self.requests.lock().unwrap().push(request);
            Ok(crate::providers::OutboundStreamResponse {
                status: 500,
                body: Box::new(std::iter::empty()),
            })
        }
    }

    struct Noop;
    impl Wake for Noop {
        fn wake(self: Arc<Self>) {}
    }

    fn block_on<T>(mut future: Pin<Box<dyn Future<Output = T> + Send + '_>>) -> T {
        let waker = Waker::from(Arc::new(Noop));
        let mut context = Context::from_waker(&waker);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    fn target() -> RuntimeTarget {
        RuntimeTarget {
            model: "anthropic/claude-test".into(),
            provider: ProviderKind::Anthropic,
            model_suffix: "claude-test".into(),
            api_key: Some(SecretString::new("secret".into())),
            api_base: "https://capture.test/v1".into(),
            timeout: 30,
            explicit_timeout: None,
        }
    }

    fn captured(body: &str) -> Vec<(String, JsonValue)> {
        let request = decode_chat_request(body.as_bytes()).unwrap();
        let transport = Arc::new(CaptureTransport::returning(500, Vec::new()));
        let provider = AnthropicProvider::new(target(), transport.clone());
        assert_eq!(
            block_on(provider.complete(&request)).unwrap_err().kind,
            crate::providers::TargetErrorKind::UpstreamHttp
        );
        let outbound = transport.request();
        assert_eq!(outbound.method, "POST");
        assert_eq!(outbound.url, "https://capture.test/v1/messages");
        assert_eq!(
            outbound.headers,
            vec![
                ("Content-Type".into(), "application/json".into()),
                ("x-api-key".into(), "secret".into()),
                ("anthropic-version".into(), "2023-06-01".into()),
            ]
        );
        crate::request::decode_json_object(&outbound.body).unwrap()
    }

    #[test]
    fn captured_text_and_instructions_use_the_pinned_messages_contract() {
        let body = captured(
            r#"{"model":"public","messages":[{"role":"system","content":"one"},{"role":"developer","content":"two"},{"role":"user","content":"hi\nthere"}],"max_completion_tokens":12,"temperature":0.2,"top_p":0.8,"stop":"END"}"#,
        );
        assert_eq!(
            field(&body, "model"),
            Some(&JsonValue::String("claude-test".into()))
        );
        assert_eq!(
            field(&body, "max_tokens"),
            Some(&JsonValue::Number("12".into()))
        );
        assert_eq!(
            field(&body, "system"),
            Some(&JsonValue::String("one\n\ntwo".into()))
        );
        assert_eq!(
            field(&body, "stop_sequences"),
            Some(&JsonValue::Array(vec![JsonValue::String("END".into())]))
        );
        let Some(JsonValue::Array(messages)) = field(&body, "messages") else {
            panic!()
        };
        assert_eq!(
            messages,
            &vec![native_message("user", vec![text_block("hi\nthere".into())])]
        );
    }

    #[test]
    fn captured_history_preserves_block_order_argument_objects_and_result_ids() {
        let body = captured(
            r#"{"model":"public","messages":[{"role":"user","content":"start"},{"role":"assistant","content":"thinking","tool_calls":[{"id":"call-b","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Montréal\"}"}},{"id":"call-a","type":"function","function":{"name":"weather","arguments":"{\"city\":\"東京\"}"}}]},{"role":"tool","tool_call_id":"call-a","content":"A"},{"role":"tool","tool_call_id":"call-b","content":"B"},{"role":"assistant","content":"done"},{"role":"user","content":"next"}],"max_tokens":8,"tools":[{"type":"function","function":{"name":"weather","description":"forecast","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"tool_choice":"auto"}"#,
        );
        let Some(JsonValue::Array(messages)) = field(&body, "messages") else {
            panic!()
        };
        assert_eq!(messages.len(), 5);
        let JsonValue::Object(assistant) = &messages[1] else {
            panic!()
        };
        let Some(JsonValue::Array(blocks)) = field(assistant, "content") else {
            panic!()
        };
        assert_eq!(blocks[0], text_block("thinking".into()));
        let JsonValue::Object(first_call) = &blocks[1] else {
            panic!()
        };
        assert_eq!(
            field(first_call, "id"),
            Some(&JsonValue::String("call-b".into()))
        );
        assert_eq!(
            field(first_call, "input"),
            Some(&JsonValue::Object(vec![(
                "city".into(),
                JsonValue::String("Montréal".into())
            )]))
        );
        let JsonValue::Object(results) = &messages[2] else {
            panic!()
        };
        let Some(JsonValue::Array(result_blocks)) = field(results, "content") else {
            panic!()
        };
        assert_eq!(
            field(
                match &result_blocks[0] {
                    JsonValue::Object(v) => v,
                    _ => panic!(),
                },
                "tool_use_id"
            ),
            Some(&JsonValue::String("call-a".into()))
        );
        assert_eq!(
            field(
                match &result_blocks[1] {
                    JsonValue::Object(v) => v,
                    _ => panic!(),
                },
                "tool_use_id"
            ),
            Some(&JsonValue::String("call-b".into()))
        );
        let Some(JsonValue::Array(tools)) = field(&body, "tools") else {
            panic!()
        };
        let JsonValue::Object(tool) = &tools[0] else {
            panic!()
        };
        assert!(field(tool, "input_schema").is_some());
    }

    #[test]
    fn every_tool_choice_has_its_native_equivalent() {
        for (choice, expected) in [
            (r#""none""#, "none"),
            (r#""auto""#, "auto"),
            (r#""required""#, "any"),
            (
                r#"{"type":"function","function":{"name":"weather"}}"#,
                "tool",
            ),
        ] {
            let body = captured(&format!(
                r#"{{"model":"public","messages":[{{"role":"user","content":"hi"}}],"max_tokens":1,"tools":[{{"type":"function","function":{{"name":"weather"}}}}],"tool_choice":{choice}}}"#
            ));
            let Some(JsonValue::Object(native)) = field(&body, "tool_choice") else {
                panic!()
            };
            assert_eq!(
                field(native, "type"),
                Some(&JsonValue::String(expected.into()))
            );
        }
    }

    #[test]
    fn streaming_uses_the_same_messages_operation_with_native_stream_flag() {
        let request = decode_chat_request(
            br#"{"model":"public","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":true}"#,
        )
        .unwrap();
        let transport = Arc::new(CaptureTransport::returning(500, Vec::new()));
        let provider = AnthropicProvider::new(target(), transport.clone());
        let result = block_on(provider.complete_stream(&request));
        assert!(matches!(
            result,
            Err(error) if error.kind == crate::providers::TargetErrorKind::UpstreamHttp
        ));
        let body = crate::request::decode_json_object(&transport.request().body).unwrap();
        assert_eq!(field(&body, "stream"), Some(&JsonValue::Bool(true)));
    }

    fn completion_request(with_tools: bool) -> CanonicalRequest {
        let tools = if with_tools {
            r#", "tools":[{"type":"function","function":{"name":"weather"}}],"tool_choice":"auto""#
        } else {
            ""
        };
        decode_chat_request(
            format!(
                r#"{{"model":"public","messages":[{{"role":"user","content":"hi"}}],"max_tokens":8{tools}}}"#
            )
            .as_bytes(),
        )
        .unwrap()
    }

    fn complete(body: impl Into<Vec<u8>>, with_tools: bool) -> Result<ChatResponse, TargetError> {
        let transport = Arc::new(CaptureTransport::returning(200, body));
        block_on(
            AnthropicProvider::new(target(), transport).complete(&completion_request(with_tools)),
        )
    }

    #[test]
    fn buffered_messages_normalize_text_tools_usage_and_gateway_metadata() {
        let response = complete(
            r#"{"id":"msg_provider_private","type":"message","role":"assistant","content":[{"type":"text","text":"one"},{"type":"tool_use","id":"native","name":"weather","input":{"city":"Montréal","days":2}},{"type":"text","text":" two"},{"type":"tool_use","id":"native","name":"weather","input":{}}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":5}}"#,
            true,
        )
        .unwrap();
        assert!(response.id.starts_with("chatcmpl-"));
        assert_eq!(response.model, "public");
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("one two")
        );
        assert_eq!(response.choices[0].finish_reason.as_str(), "tool_calls");
        let calls = response.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "native");
        assert!(calls[1].id.starts_with("call_"));
        assert_ne!(calls[0].id, calls[1].id);
        assert_eq!(
            calls[0].function.arguments,
            r#"{"city":"Montréal","days":2}"#
        );
        assert_eq!(calls[1].function.arguments, "{}");
        assert_eq!(response.usage.unwrap().total_tokens, 8);
    }

    #[test]
    fn documented_stop_reasons_and_explicit_refusal_are_normalized() {
        for (reason, expected) in [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("model_context_window_exceeded", "length"),
        ] {
            let response = complete(
                format!(
                    r#"{{"type":"message","role":"assistant","content":[{{"type":"text","text":"ok"}}],"stop_reason":"{reason}"}}"#
                ),
                false,
            )
            .unwrap();
            assert_eq!(response.choices[0].finish_reason.as_str(), expected);
        }
        let tool = complete(
            r#"{"type":"message","role":"assistant","content":[{"type":"tool_use","id":"x","name":"weather","input":{}}],"stop_reason":"tool_use"}"#,
            true,
        )
        .unwrap();
        assert_eq!(tool.choices[0].finish_reason.as_str(), "tool_calls");

        // Anthropic's explicit refusal is a successful policy outcome even
        // when it has no candidate blocks to expose.
        let refusal = complete(
            r#"{"type":"message","role":"assistant","content":[],"stop_reason":"refusal"}"#,
            false,
        )
        .unwrap();
        assert_eq!(refusal.choices[0].finish_reason.as_str(), "content_filter");
        assert_eq!(refusal.choices[0].message.content, None);
    }

    #[test]
    fn malformed_2xx_unknown_empty_and_pause_turn_are_target_errors() {
        for body in [
            br#"not json"#.as_slice(),
            br#"{"type":"message","role":"assistant","content":[{"type":"tool_use","name":"weather","input":[]}],"stop_reason":"tool_use"}"#
                .as_slice(),
            br#"{"type":"message","role":"assistant","content":[],"stop_reason":"pause_turn"}"#
                .as_slice(),
            br#"{"type":"message","role":"assistant","content":[],"stop_reason":"future_reason"}"#
                .as_slice(),
        ] {
            assert_eq!(
                complete(body.to_vec(), true).unwrap_err().kind,
                crate::providers::TargetErrorKind::InvalidResponse
            );
        }
        let unknown_with_text = complete(
            br#"{"type":"message","role":"assistant","content":[{"type":"text","text":"still valid"}],"stop_reason":"future_reason","usage":{"input_tokens":-1,"output_tokens":2}}"#,
            false,
        )
        .unwrap();
        assert_eq!(unknown_with_text.choices[0].finish_reason.as_str(), "stop");
        assert!(unknown_with_text.usage.is_none());
    }

    #[test]
    fn native_http_and_transport_failures_stay_structured() {
        let request = completion_request(false);
        for (status, expected) in [
            (401, crate::providers::TargetErrorKind::Authentication),
            (429, crate::providers::TargetErrorKind::RateLimited),
            (503, crate::providers::TargetErrorKind::Overloaded),
            (500, crate::providers::TargetErrorKind::UpstreamHttp),
        ] {
            let transport = Arc::new(CaptureTransport::returning(status, b"ignored".to_vec()));
            assert_eq!(
                block_on(AnthropicProvider::new(target(), transport).complete(&request))
                    .unwrap_err()
                    .kind,
                expected
            );
        }
        let transport = Arc::new(CaptureTransport {
            requests: Mutex::new(Vec::new()),
            response: Mutex::new(Err(TransportError {
                kind: TransportErrorKind::Connection,
            })),
        });
        assert_eq!(
            block_on(AnthropicProvider::new(target(), transport).complete(&request))
                .unwrap_err()
                .kind,
            crate::providers::TargetErrorKind::ConnectionError
        );
        assert_eq!(
            complete(vec![b'x'; MAX_BUFFERED_RESPONSE_BYTES + 1], false)
                .unwrap_err()
                .kind,
            crate::providers::TargetErrorKind::InvalidResponse
        );
    }
}
