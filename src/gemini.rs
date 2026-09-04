//! Gemini Generative Language request translation.
//!
//! This module owns the pinned v1beta operation shape and the conversion from
//! the gateway's canonical conversation into Gemini `contents`. Response and
//! SSE decoding intentionally belong to a later adapter slice.

use crate::config::{ProviderKind, RuntimeTarget};
use crate::providers::{
    OutboundRequest, OutboundTransport, Provider, ProviderFuture, ProviderStream,
    SecureTransportPolicy, TargetError, TransportErrorKind,
};
use crate::request::{decode_json_value, CanonicalRequest, JsonValue, ToolChoice};
use crate::response::{
    normalize_response, normalize_usage, safety_response, ChatResponse, NativeChoice,
    NativeResponse, NativeTerminal, NativeToolCall, ResponseError, ResponseMetadata, Usage,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Gemini Generative Language API adapter for a single validated target.
pub struct GeminiProvider {
    target: RuntimeTarget,
    transport: Arc<dyn OutboundTransport>,
    transport_policy: SecureTransportPolicy,
}

impl std::fmt::Debug for GeminiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiProvider")
            .field("target", &self.target)
            .field("transport_policy", &self.transport_policy)
            .finish_non_exhaustive()
    }
}

impl GeminiProvider {
    pub fn new(target: RuntimeTarget, transport: Arc<dyn OutboundTransport>) -> Self {
        debug_assert_eq!(target.provider, ProviderKind::Gemini);
        Self {
            target,
            transport,
            transport_policy: SecureTransportPolicy::default(),
        }
    }

    fn outbound_request(&self, request: &CanonicalRequest, stream: bool) -> OutboundRequest {
        let endpoint = GeminiEndpoint::for_target(&self.target, stream);
        let mut body = vec![("contents".into(), JsonValue::Array(contents(request)))];
        if let Some(instruction) = &request.instruction {
            body.push((
                "systemInstruction".into(),
                JsonValue::Object(vec![(
                    "parts".into(),
                    JsonValue::Array(vec![text_part(instruction.clone())]),
                )]),
            ));
        }
        let mut generation = Vec::new();
        if let Some(value) = request.max_tokens {
            generation.push((
                "maxOutputTokens".into(),
                JsonValue::Number(value.to_string()),
            ));
        }
        if let Some(value) = request.temperature {
            generation.push(("temperature".into(), JsonValue::Number(value.to_string())));
        }
        if let Some(value) = request.top_p {
            generation.push(("topP".into(), JsonValue::Number(value.to_string())));
        }
        if let Some(value) = &request.stop {
            generation.push((
                "stopSequences".into(),
                JsonValue::Array(value.iter().cloned().map(JsonValue::String).collect()),
            ));
        }
        if !generation.is_empty() {
            body.push(("generationConfig".into(), JsonValue::Object(generation)));
        }
        if let Some(tools) = field(&request.fields, "tools") {
            body.push((
                "tools".into(),
                JsonValue::Array(vec![JsonValue::Object(vec![(
                    "functionDeclarations".into(),
                    gemini_tools(tools),
                )])]),
            ));
            body.push((
                "toolConfig".into(),
                JsonValue::Object(vec![(
                    "functionCallingConfig".into(),
                    gemini_tool_choice(&request.tool_choice),
                )]),
            ));
        }
        OutboundRequest {
            method: "POST",
            url: endpoint.into_url(),
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: encode(&JsonValue::Object(body)).into_bytes(),
        }
    }
}

impl Provider for GeminiProvider {
    fn complete<'a>(
        &'a self,
        request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ChatResponse, TargetError>> {
        Box::pin(async move {
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
            let metadata = ResponseMetadata::for_model(&request.model);
            match parse_response(&response.body)? {
                ParsedResponse::Candidate(native) => normalize_response(request, metadata, native),
                ParsedResponse::SafetyBlock { usage } => safety_response(request, metadata, usage),
            }
            .map_err(response_error)
        })
    }

    fn complete_stream<'a>(
        &'a self,
        request: &'a CanonicalRequest,
    ) -> ProviderFuture<'a, Result<ProviderStream, TargetError>> {
        Box::pin(async move {
            let response = self
                .transport
                .execute_stream(self.transport_policy, self.outbound_request(request, true))
                .map_err(transport_error)?;
            if !(200..300).contains(&response.status) {
                return Err(TargetError::from_upstream_status(response.status));
            }
            // Gemini SSE parsing is explicitly outside this slice.
            Err(TargetError::invalid_response())
        })
    }
}

/// Keep a buffered provider response bounded before decoding it.  Gemini SSE
/// uses a separate decoder and is deliberately not covered by this limit.
const MAX_BUFFERED_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

enum ParsedResponse {
    Candidate(NativeResponse),
    /// Gemini can block a prompt before constructing any candidate.  This is
    /// the only no-candidate response which is a canonical success.
    SafetyBlock {
        usage: Option<Usage>,
    },
}

fn response_error(error: ResponseError) -> TargetError {
    match error {
        ResponseError::Overloaded => TargetError::overloaded(),
        ResponseError::InvalidResponse(_) => TargetError::invalid_response(),
    }
}

/// Decode Gemini's buffered `generateContent` response into gateway-native
/// response inputs.  Fields outside the canonical contract are deliberately
/// ignored, but every field used to establish output semantics is strict.
fn parse_response(bytes: &[u8]) -> Result<ParsedResponse, TargetError> {
    let root =
        crate::request::decode_json_object(bytes).map_err(|_| TargetError::invalid_response())?;
    let usage = match field(&root, "usageMetadata") {
        None | Some(JsonValue::Null) => None,
        Some(value) => parse_usage(value),
    };

    match field(&root, "candidates") {
        Some(JsonValue::Array(candidates)) if candidates.len() == 1 => {
            Ok(ParsedResponse::Candidate(NativeResponse {
                choices: vec![parse_candidate(&candidates[0])?],
                usage,
            }))
        }
        // Gemini provides this explicit signal when a prompt is blocked before
        // it has a candidate.  An empty or missing array on its own is never a
        // successful completion.
        None | Some(JsonValue::Array(_)) if explicit_policy_block(&root) => {
            Ok(ParsedResponse::SafetyBlock { usage })
        }
        _ => Err(TargetError::invalid_response()),
    }
}

fn parse_candidate(value: &JsonValue) -> Result<NativeChoice, TargetError> {
    let candidate = object(value)?;
    if let Some(index) = field(candidate, "index") {
        if unsigned(index)? != 0 {
            return Err(TargetError::invalid_response());
        }
    }
    let content = object(required(candidate, "content")?)?;
    match field(content, "role") {
        None => {}
        Some(JsonValue::String(role)) if role == "model" => {}
        _ => return Err(TargetError::invalid_response()),
    }
    let parts = array(required(content, "parts")?)?;
    if parts.is_empty() {
        return Err(TargetError::invalid_response());
    }
    let (content, calls) = parse_parts(parts)?;
    let terminal = match required(candidate, "finishReason")? {
        JsonValue::String(reason) => terminal(reason),
        _ => return Err(TargetError::invalid_response()),
    };
    Ok(NativeChoice {
        index: 0,
        content,
        calls,
        terminal,
    })
}

/// Canonical output has one text field and an ordered call list.  Text parts
/// therefore concatenate in native order without introducing content.
fn parse_parts(parts: &[JsonValue]) -> Result<(Option<String>, Vec<NativeToolCall>), TargetError> {
    let mut text = String::new();
    let mut saw_text = false;
    let mut calls = Vec::new();
    for part in parts {
        let part = object(part)?;
        match (field(part, "text"), field(part, "functionCall")) {
            (Some(JsonValue::String(value)), None) => {
                text.push_str(value);
                saw_text = true;
            }
            (None, Some(value)) => {
                let call = object(value)?;
                let id = match field(call, "id") {
                    None => None,
                    Some(JsonValue::String(id)) => Some(id.clone()),
                    Some(_) => return Err(TargetError::invalid_response()),
                };
                let name = string(required(call, "name")?)?.to_owned();
                let JsonValue::Object(arguments) = required(call, "args")? else {
                    return Err(TargetError::invalid_response());
                };
                calls.push(NativeToolCall {
                    id,
                    name,
                    arguments: encode(&JsonValue::Object(arguments.clone())),
                });
            }
            // A native part cannot represent two canonical outputs at once,
            // and v0.1 has no counterpart for any other Gemini part type.
            _ => return Err(TargetError::invalid_response()),
        }
    }
    Ok((saw_text.then_some(text), calls))
}

fn parse_usage(value: &JsonValue) -> Option<Usage> {
    let JsonValue::Object(usage) = value else {
        return None;
    };
    normalize_usage(
        field(usage, "promptTokenCount").and_then(unsigned_optional),
        field(usage, "candidatesTokenCount").and_then(unsigned_optional),
    )
}

fn unsigned_optional(value: &JsonValue) -> Option<u128> {
    let JsonValue::Number(value) = value else {
        return None;
    };
    if value.contains(['.', 'e', 'E']) {
        return None;
    }
    value.parse().ok()
}

fn explicit_policy_block(root: &[(String, JsonValue)]) -> bool {
    let Some(JsonValue::Object(feedback)) = field(root, "promptFeedback") else {
        return false;
    };
    matches!(field(feedback, "blockReason"),
        Some(JsonValue::String(reason)) if matches!(reason.as_str(),
            "SAFETY" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "IMAGE_SAFETY" | "JAILBREAK" | "MODEL_ARMOR" | "OTHER"))
}

fn terminal(reason: &str) -> NativeTerminal {
    match reason {
        "STOP" => NativeTerminal::Stop,
        "MAX_TOKENS" => NativeTerminal::Length,
        "SAFETY" | "RECITATION" | "LANGUAGE" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" => {
            NativeTerminal::ContentFilter
        }
        "MALFORMED_FUNCTION_CALL"
        | "UNEXPECTED_TOOL_CALL"
        | "TOO_MANY_TOOL_CALLS"
        | "MISSING_THOUGHT_SIGNATURE"
        | "MALFORMED_RESPONSE"
        | "FINISH_REASON_UNSPECIFIED" => NativeTerminal::Invalid,
        _ => NativeTerminal::Unknown,
    }
}

fn required<'a>(
    fields: &'a [(String, JsonValue)],
    name: &str,
) -> Result<&'a JsonValue, TargetError> {
    field(fields, name).ok_or_else(TargetError::invalid_response)
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

/// URL components are kept distinct until final serialization. The configured
/// base has already been URL-validated, while the model is a separately
/// validated Gemini path segment and the key is encoded solely as a query value.
struct GeminiEndpoint<'a> {
    base: &'a str,
    model: &'a str,
    key: &'a str,
    stream: bool,
}
impl<'a> GeminiEndpoint<'a> {
    fn for_target(target: &'a RuntimeTarget, stream: bool) -> Self {
        Self {
            base: &target.api_base,
            model: &target.model_suffix,
            key: target
                .api_key
                .as_ref()
                .expect("validated Gemini key")
                .expose_secret(),
            stream,
        }
    }
    fn into_url(self) -> String {
        let operation = if self.stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let mut url =
            String::with_capacity(self.base.len() + self.model.len() + self.key.len() + 64);
        url.push_str(self.base);
        url.push_str("/models/");
        // The config validator admits only the Gemini segment grammar, so this
        // component cannot contribute slash, query, fragment, or escape syntax.
        url.push_str(self.model);
        url.push(':');
        url.push_str(operation);
        url.push('?');
        if self.stream {
            url.push_str("alt=sse&");
        }
        url.push_str("key=");
        url.push_str(&percent_encode_query_value(self.key));
        url
    }
}

fn percent_encode_query_value(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push('%');
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 15) as usize] as char);
        }
    }
    encoded
}

fn contents(request: &CanonicalRequest) -> Vec<JsonValue> {
    let JsonValue::Array(source) = field(&request.fields, "messages").expect("validated messages")
    else {
        unreachable!()
    };
    let mut output = Vec::new();
    let mut calls = HashMap::<String, String>::new();
    let mut index = 0;
    while index < source.len() {
        let JsonValue::Object(message) = &source[index] else {
            unreachable!()
        };
        match string_field(message, "role").as_str() {
            "system" | "developer" => index += 1,
            "user" => {
                output.push(content(
                    "user",
                    vec![text_part(string_field(message, "content").clone())],
                ));
                index += 1;
            }
            "assistant" => {
                let mut parts = Vec::new();
                if let Some(JsonValue::String(text)) = field(message, "content") {
                    parts.push(text_part(text.clone()));
                }
                if let Some(JsonValue::Array(tool_calls)) = field(message, "tool_calls") {
                    for call in tool_calls {
                        let JsonValue::Object(call) = call else {
                            unreachable!()
                        };
                        let JsonValue::Object(function) =
                            field(call, "function").expect("validated function")
                        else {
                            unreachable!()
                        };
                        let id = string_field(call, "id").clone();
                        let name = string_field(function, "name").clone();
                        let JsonValue::Object(args) =
                            decode_json_value(string_field(function, "arguments").as_bytes())
                                .expect("route validated arguments")
                        else {
                            unreachable!()
                        };
                        calls.insert(id.clone(), name.clone());
                        parts.push(JsonValue::Object(vec![(
                            "functionCall".into(),
                            JsonValue::Object(vec![
                                ("id".into(), JsonValue::String(id)),
                                ("name".into(), JsonValue::String(name)),
                                ("args".into(), JsonValue::Object(args)),
                            ]),
                        )]));
                    }
                }
                output.push(content("model", parts));
                index += 1;
            }
            "tool" => {
                let mut parts = Vec::new();
                while index < source.len() {
                    let JsonValue::Object(result) = &source[index] else {
                        unreachable!()
                    };
                    if string_field(result, "role") != "tool" {
                        break;
                    }
                    let id = string_field(result, "tool_call_id").clone();
                    let name = calls
                        .get(&id)
                        .expect("canonical result resolves a preceding call")
                        .clone();
                    parts.push(JsonValue::Object(vec![(
                        "functionResponse".into(),
                        JsonValue::Object(vec![
                            ("id".into(), JsonValue::String(id)),
                            ("name".into(), JsonValue::String(name)),
                            (
                                "response".into(),
                                JsonValue::Object(vec![(
                                    "result".into(),
                                    JsonValue::String(string_field(result, "content").clone()),
                                )]),
                            ),
                        ]),
                    )]));
                    index += 1;
                }
                output.push(content("user", parts));
            }
            _ => unreachable!(),
        }
    }
    output
}

fn content(role: &str, parts: Vec<JsonValue>) -> JsonValue {
    JsonValue::Object(vec![
        ("role".into(), JsonValue::String(role.into())),
        ("parts".into(), JsonValue::Array(parts)),
    ])
}
fn text_part(text: String) -> JsonValue {
    JsonValue::Object(vec![("text".into(), JsonValue::String(text))])
}
fn gemini_tools(value: &JsonValue) -> JsonValue {
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
                let mut declaration = vec![(
                    "name".into(),
                    JsonValue::String(string_field(function, "name").clone()),
                )];
                if let Some(JsonValue::String(description)) = field(function, "description") {
                    declaration
                        .push(("description".into(), JsonValue::String(description.clone())));
                }
                if let Some(parameters) = field(function, "parameters") {
                    declaration.push(("parameters".into(), parameters.clone()));
                }
                JsonValue::Object(declaration)
            })
            .collect(),
    )
}
fn gemini_tool_choice(choice: &ToolChoice) -> JsonValue {
    let mut fields = vec![(
        "mode".into(),
        JsonValue::String(
            match choice {
                ToolChoice::None => "NONE",
                ToolChoice::Auto => "AUTO",
                ToolChoice::Required | ToolChoice::Named(_) => "ANY",
            }
            .into(),
        ),
    )];
    if let ToolChoice::Named(name) = choice {
        fields.push((
            "allowedFunctionNames".into(),
            JsonValue::Array(vec![JsonValue::String(name.clone())]),
        ));
    }
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
fn transport_error(error: crate::providers::TransportError) -> TargetError {
    match error.kind {
        TransportErrorKind::Timeout => TargetError::timeout(),
        TransportErrorKind::Connection => TargetError::connection(),
    }
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
    use crate::providers::{OutboundResponse, OutboundStreamResponse, TransportError};
    use crate::request::decode_chat_request;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::task::{Context, Poll, Wake, Waker};

    struct Capture {
        request: Mutex<Option<OutboundRequest>>,
    }
    impl Capture {
        fn take(&self) -> OutboundRequest {
            self.request.lock().unwrap().take().unwrap()
        }
    }
    impl OutboundTransport for Capture {
        fn execute(
            &self,
            _: SecureTransportPolicy,
            request: OutboundRequest,
        ) -> Result<OutboundResponse, TransportError> {
            *self.request.lock().unwrap() = Some(request);
            Ok(OutboundResponse {
                status: 500,
                body: vec![],
            })
        }
        fn execute_stream(
            &self,
            _: SecureTransportPolicy,
            request: OutboundRequest,
        ) -> Result<OutboundStreamResponse, TransportError> {
            *self.request.lock().unwrap() = Some(request);
            Ok(OutboundStreamResponse {
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
            model: "gemini/gemini-2.5-flash".into(),
            provider: ProviderKind::Gemini,
            model_suffix: "gemini-2.5-flash".into(),
            api_key: Some(SecretString::new("a& b/🔑".into())),
            api_base: "https://capture.test/v1beta".into(),
            timeout: 30,
            explicit_timeout: None,
        }
    }
    fn captured(source: &str, stream: bool) -> (OutboundRequest, Vec<(String, JsonValue)>) {
        let request = decode_chat_request(source.as_bytes()).unwrap();
        let capture = Arc::new(Capture {
            request: Mutex::new(None),
        });
        let provider = GeminiProvider::new(target(), capture.clone());
        let result = if stream {
            block_on(provider.complete_stream(&request)).map(|_| ())
        } else {
            block_on(provider.complete(&request)).map(|_| ())
        };
        assert!(
            matches!(result, Err(error) if error.kind == crate::providers::TargetErrorKind::UpstreamHttp)
        );
        let outbound = capture.take();
        let body = crate::request::decode_json_object(&outbound.body).unwrap();
        (outbound, body)
    }

    #[test]
    fn captured_urls_are_pinned_and_query_components_are_encoded() {
        let (normal, _) = captured(
            r#"{"model":"public","messages":[{"role":"user","content":"hi"}]}"#,
            false,
        );
        assert_eq!(normal.url, "https://capture.test/v1beta/models/gemini-2.5-flash:generateContent?key=a%26%20b%2F%F0%9F%94%91");
        assert_eq!(
            normal.headers,
            vec![("Content-Type".into(), "application/json".into())]
        );
        let (stream, _) = captured(
            r#"{"model":"public","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
            true,
        );
        assert_eq!(stream.url, "https://capture.test/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=a%26%20b%2F%F0%9F%94%91");
    }

    #[test]
    fn translates_instructions_options_tools_and_reordered_same_name_results() {
        let (outbound, body) = captured(
            r#"{"model":"public","messages":[{"role":"system","content":"one"},{"role":"developer","content":"two"},{"role":"user","content":"start"},{"role":"assistant","content":"thinking","tool_calls":[{"id":"call-b","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Montréal\"}"}},{"id":"call-a","type":"function","function":{"name":"weather","arguments":"{\"city\":\"東京\"}"}}]},{"role":"tool","tool_call_id":"call-a","content":"{\"raw\":true}"},{"role":"tool","tool_call_id":"call-b","content":"exact string"}],"max_tokens":9,"temperature":0.2,"top_p":0.8,"stop":["END"],"tools":[{"type":"function","function":{"name":"weather","description":"forecast","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"weather"}}}"#,
            false,
        );
        assert_eq!(outbound.method, "POST");
        let Some(JsonValue::Object(system)) = field(&body, "systemInstruction") else {
            panic!()
        };
        assert_eq!(
            field(system, "parts"),
            Some(&JsonValue::Array(vec![text_part("one\n\ntwo".into())]))
        );
        let Some(JsonValue::Object(config)) = field(&body, "generationConfig") else {
            panic!()
        };
        assert_eq!(
            field(config, "maxOutputTokens"),
            Some(&JsonValue::Number("9".into()))
        );
        assert_eq!(
            field(config, "stopSequences"),
            Some(&JsonValue::Array(vec![JsonValue::String("END".into())]))
        );
        let Some(JsonValue::Array(contents)) = field(&body, "contents") else {
            panic!()
        };
        let JsonValue::Object(model) = &contents[1] else {
            panic!()
        };
        let Some(JsonValue::Array(parts)) = field(model, "parts") else {
            panic!()
        };
        let JsonValue::Object(first) = &parts[1] else {
            panic!()
        };
        let Some(JsonValue::Object(first_call)) = field(first, "functionCall") else {
            panic!()
        };
        assert_eq!(
            field(first_call, "id"),
            Some(&JsonValue::String("call-b".into()))
        );
        let JsonValue::Object(results) = &contents[2] else {
            panic!()
        };
        let Some(JsonValue::Array(result_parts)) = field(results, "parts") else {
            panic!()
        };
        let JsonValue::Object(first_result) = &result_parts[0] else {
            panic!()
        };
        let Some(JsonValue::Object(response)) = field(first_result, "functionResponse") else {
            panic!()
        };
        assert_eq!(
            field(response, "id"),
            Some(&JsonValue::String("call-a".into()))
        );
        assert_eq!(
            field(response, "name"),
            Some(&JsonValue::String("weather".into()))
        );
        assert_eq!(
            field(response, "response"),
            Some(&JsonValue::Object(vec![(
                "result".into(),
                JsonValue::String("{\"raw\":true}".into())
            )]))
        );
        let Some(JsonValue::Object(tool_config)) = field(&body, "toolConfig") else {
            panic!()
        };
        let Some(JsonValue::Object(choice)) = field(tool_config, "functionCallingConfig") else {
            panic!()
        };
        assert_eq!(
            field(choice, "mode"),
            Some(&JsonValue::String("ANY".into()))
        );
        assert_eq!(
            field(choice, "allowedFunctionNames"),
            Some(&JsonValue::Array(vec![JsonValue::String("weather".into())]))
        );
    }

    #[test]
    fn every_canonical_tool_choice_is_emitted() {
        for (choice, mode) in [
            (r#""none""#, "NONE"),
            (r#""auto""#, "AUTO"),
            (r#""required""#, "ANY"),
            (
                r#"{"type":"function","function":{"name":"weather"}}"#,
                "ANY",
            ),
        ] {
            let (_, body) = captured(
                &format!(
                    r#"{{"model":"public","messages":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"function","function":{{"name":"weather"}}}}],"tool_choice":{choice}}}"#
                ),
                false,
            );
            let Some(JsonValue::Object(config)) = field(&body, "toolConfig") else {
                panic!()
            };
            let Some(JsonValue::Object(functions)) = field(config, "functionCallingConfig") else {
                panic!()
            };
            assert_eq!(
                field(functions, "mode"),
                Some(&JsonValue::String(mode.into()))
            );
        }
    }

    fn normalized(source: &str, payload: &str) -> Result<ChatResponse, TargetError> {
        let request = decode_chat_request(source.as_bytes()).unwrap();
        let metadata = ResponseMetadata::for_model(&request.model);
        match parse_response(payload.as_bytes())? {
            ParsedResponse::Candidate(native) => normalize_response(&request, metadata, native),
            ParsedResponse::SafetyBlock { usage } => safety_response(&request, metadata, usage),
        }
        .map_err(response_error)
    }

    #[test]
    fn normalizes_one_candidate_text_calls_ids_and_usage() {
        let response = normalized(
            r#"{"model":"public","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"weather"}}]}"#,
            r#"{"candidates":[{"index":0,"content":{"parts":[{"text":"The "},{"text":"forecast:"},{"functionCall":{"id":"native","name":"weather","args":{"city":"Paris","days":2}}},{"functionCall":{"id":"native","name":"weather","args":{}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":5,"totalTokenCount":999}}"#,
        )
        .unwrap();
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_deref(), Some("The forecast:"));
        assert_eq!(choice.finish_reason.as_str(), "tool_calls");
        let calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].id, "native");
        assert!(calls[1].id.starts_with("call_"));
        assert_ne!(calls[0].id, calls[1].id);
        assert_eq!(calls[0].function.arguments, r#"{"city":"Paris","days":2}"#);
        assert_eq!(
            response.usage,
            Some(Usage {
                prompt_tokens: 3,
                completion_tokens: 5,
                total_tokens: 8,
            })
        );
    }

    #[test]
    fn maps_every_gemini_finish_reason_and_rejects_protocol_reasons() {
        let request = r#"{"model":"public","messages":[{"role":"user","content":"hi"}]}"#;
        for (native, canonical) in [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "content_filter"),
            ("RECITATION", "content_filter"),
            ("LANGUAGE", "content_filter"),
            ("BLOCKLIST", "content_filter"),
            ("PROHIBITED_CONTENT", "content_filter"),
            ("SPII", "content_filter"),
            ("unrecognized_future_reason", "stop"),
        ] {
            let payload = format!(
                r#"{{"candidates":[{{"content":{{"parts":[{{"text":"x"}}]}},"finishReason":"{native}"}}]}}"#
            );
            assert_eq!(
                normalized(request, &payload).unwrap().choices[0]
                    .finish_reason
                    .as_str(),
                canonical
            );
        }
        for native in [
            "MALFORMED_FUNCTION_CALL",
            "UNEXPECTED_TOOL_CALL",
            "TOO_MANY_TOOL_CALLS",
            "MISSING_THOUGHT_SIGNATURE",
            "MALFORMED_RESPONSE",
            "FINISH_REASON_UNSPECIFIED",
        ] {
            let payload = format!(
                r#"{{"candidates":[{{"content":{{"parts":[{{"text":"x"}}]}},"finishReason":"{native}"}}]}}"#
            );
            assert!(matches!(normalized(request, &payload), Err(error)
                if error.kind == crate::providers::TargetErrorKind::InvalidResponse));
        }
    }

    #[test]
    fn only_explicit_prompt_policy_blocks_can_synthesize_a_choice() {
        let request = r#"{"model":"public","messages":[{"role":"user","content":"hi"}]}"#;
        let blocked = normalized(
            request,
            r#"{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":0}}"#,
        )
        .unwrap();
        assert_eq!(blocked.choices[0].finish_reason.as_str(), "content_filter");
        assert_eq!(blocked.choices[0].message.content, None);
        assert_eq!(blocked.choices[0].message.tool_calls, None);
        assert_eq!(blocked.usage.as_ref().unwrap().total_tokens, 2);

        for payload in [
            r#"{}"#,
            r#"{"candidates":[]}"#,
            r#"{"promptFeedback":{"blockReason":"BLOCK_REASON_UNSPECIFIED"}}"#,
            r#"{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}"#,
            r#"{"candidates":[{"content":{"role":"user","parts":[{"text":"x"}]},"finishReason":"STOP"}]}"#,
            r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"missing_args"}}]},"finishReason":"STOP"}]}"#,
        ] {
            assert!(matches!(normalized(request, payload), Err(error)
                if error.kind == crate::providers::TargetErrorKind::InvalidResponse));
        }
    }

    #[test]
    fn omits_invalid_usage_and_rejects_noncanonical_candidate_sets() {
        let request = r#"{"model":"public","messages":[{"role":"user","content":"hi"}]}"#;
        let omitted = normalized(
            request,
            r#"{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1}}"#,
        )
        .unwrap();
        assert_eq!(omitted.usage, None);
        for payload in [
            r#"{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"},{"content":{"parts":[{"text":"y"}]},"finishReason":"STOP"}]}"#,
            r#"{"candidates":[{"index":1,"content":{"parts":[{"text":"x"}]},"finishReason":"STOP"}]}"#,
            r#"{"candidates":[{"content":{"parts":[{"text":"x"}]}}]}"#,
        ] {
            assert!(matches!(normalized(request, payload), Err(error)
                if error.kind == crate::providers::TargetErrorKind::InvalidResponse));
        }
    }
}
