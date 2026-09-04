//! Anthropic Messages request translation.
//!
//! This module intentionally owns only the request half of the Messages API.
//! Response and SSE event translation are separate delivery slices, so a 2xx
//! response is conservatively treated as an invalid response until that work
//! exists rather than being mistaken for a successful canonical completion.

use crate::config::{ProviderKind, RuntimeTarget};
use crate::providers::{
    OutboundRequest, OutboundTransport, Provider, ProviderFuture, ProviderStream,
    SecureTransportPolicy, TargetError, TransportErrorKind,
};
use crate::request::{decode_json_value, CanonicalRequest, JsonValue, ToolChoice};
use crate::response::ChatResponse;
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
            Err(TargetError::invalid_response())
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

    struct CaptureTransport(Mutex<Vec<OutboundRequest>>);

    impl CaptureTransport {
        fn request(&self) -> OutboundRequest {
            self.0.lock().unwrap()[0].clone()
        }
    }

    impl OutboundTransport for CaptureTransport {
        fn execute(
            &self,
            policy: SecureTransportPolicy,
            request: OutboundRequest,
        ) -> Result<OutboundResponse, TransportError> {
            assert_eq!(policy, SecureTransportPolicy::default());
            self.0.lock().unwrap().push(request);
            Ok(OutboundResponse {
                status: 500,
                body: vec![],
            })
        }

        fn execute_stream(
            &self,
            policy: SecureTransportPolicy,
            request: OutboundRequest,
        ) -> Result<crate::providers::OutboundStreamResponse, TransportError> {
            assert_eq!(policy, SecureTransportPolicy::default());
            self.0.lock().unwrap().push(request);
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
        let transport = Arc::new(CaptureTransport(Mutex::new(Vec::new())));
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
        let transport = Arc::new(CaptureTransport(Mutex::new(Vec::new())));
        let provider = AnthropicProvider::new(target(), transport.clone());
        let result = block_on(provider.complete_stream(&request));
        assert!(matches!(
            result,
            Err(error) if error.kind == crate::providers::TargetErrorKind::UpstreamHttp
        ));
        let body = crate::request::decode_json_object(&transport.request().body).unwrap();
        assert_eq!(field(&body, "stream"), Some(&JsonValue::Bool(true)));
    }
}
