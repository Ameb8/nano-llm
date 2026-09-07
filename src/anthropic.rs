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
    build_response, normalize_usage, AssistantDelta, ChatChunk, ChatResponse, NativeTerminal,
    NativeToolCall, ResponseError, ResponseMetadata, StreamAssembler, ToolCallDelta,
};
use std::collections::{HashMap, VecDeque};
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
            Ok(Box::new(AnthropicSseDecoder::new(
                request.clone(),
                ResponseMetadata::for_model(&request.model),
                response.body,
            )) as ProviderStream)
        })
    }
}

/// Maximum decoded payload of one native SSE event.  Transport fragments have
/// no framing significance and may split a field or JSON string anywhere.
const MAX_SSE_EVENT_BYTES: usize = 1024 * 1024;

#[derive(Default)]
struct BlockState {
    kind: BlockKind,
    call_index: Option<usize>,
    arguments: String,
}

#[derive(Default, PartialEq, Eq)]
enum BlockKind {
    #[default]
    Text,
    Tool,
}

/// Strict incremental decoder for the Anthropic Messages SSE lifecycle.
pub struct AnthropicSseDecoder {
    source: crate::providers::OutboundByteStream,
    assembler: StreamAssembler,
    input: Vec<u8>,
    data: Vec<u8>,
    event: Option<String>,
    pending: VecDeque<Result<ChatChunk, TargetError>>,
    blocks: HashMap<usize, BlockState>,
    next_block: usize,
    next_call: usize,
    started: bool,
    saw_message_delta: bool,
    saw_message_stop: bool,
    done: bool,
    exhausted: bool,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

impl AnthropicSseDecoder {
    fn new(
        request: CanonicalRequest,
        metadata: ResponseMetadata,
        source: crate::providers::OutboundByteStream,
    ) -> Self {
        Self {
            source,
            assembler: StreamAssembler::new(&request, metadata),
            input: Vec::new(),
            data: Vec::new(),
            event: None,
            pending: VecDeque::new(),
            blocks: HashMap::new(),
            next_block: 0,
            next_call: 0,
            started: false,
            saw_message_delta: false,
            saw_message_stop: false,
            done: false,
            exhausted: false,
            input_tokens: None,
            output_tokens: None,
        }
    }

    fn invalid() -> TargetError {
        TargetError::invalid_response()
    }

    fn line(&mut self, mut line: Vec<u8>) -> Result<(), TargetError> {
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        if line.is_empty() {
            return self.dispatch();
        }
        if line[0] == b':' {
            return Ok(());
        }
        if let Some(value) = line.strip_prefix(b"event:") {
            if self.event.is_some() {
                return Err(Self::invalid());
            }
            let value = value.strip_prefix(b" ").unwrap_or(value);
            self.event = Some(
                std::str::from_utf8(value)
                    .map_err(|_| Self::invalid())?
                    .to_owned(),
            );
        } else if let Some(value) = line.strip_prefix(b"data:") {
            let value = value.strip_prefix(b" ").unwrap_or(value);
            let extra = value.len() + usize::from(!self.data.is_empty());
            if self.data.len().saturating_add(extra) > MAX_SSE_EVENT_BYTES {
                return Err(Self::invalid());
            }
            if !self.data.is_empty() {
                self.data.push(b'\n');
            }
            self.data.extend_from_slice(value);
        }
        Ok(())
    }

    fn dispatch(&mut self) -> Result<(), TargetError> {
        if self.data.is_empty() {
            self.event = None;
            return Ok(());
        }
        let payload = std::mem::take(&mut self.data);
        let event = self.event.take().ok_or_else(Self::invalid)?;
        if self.done {
            return Err(Self::invalid());
        }
        let value = crate::request::decode_json_object(&payload).map_err(|_| Self::invalid())?;
        if !matches!(get(&value, "type"), Some(JsonValue::String(kind)) if kind == &event) {
            return Err(Self::invalid());
        }
        match event.as_str() {
            "message_start" => self.message_start(&value),
            "content_block_start" => self.block_start(&value),
            "content_block_delta" => self.block_delta(&value),
            "content_block_stop" => self.block_stop(&value),
            "message_delta" => self.message_delta(&value),
            "message_stop" => self.message_stop(&value),
            _ => Err(Self::invalid()),
        }
    }

    fn message_start(&mut self, value: &[(String, JsonValue)]) -> Result<(), TargetError> {
        if self.started || self.saw_message_delta || self.saw_message_stop {
            return Err(Self::invalid());
        }
        let message = object(required(value, "message")?)?;
        if !matches!(get(message, "type"), Some(JsonValue::String(kind)) if kind == "message")
            || !matches!(get(message, "role"), Some(JsonValue::String(role)) if role == "assistant")
        {
            return Err(Self::invalid());
        }
        if let Some(usage) = get(message, "usage") {
            self.input_tokens = usage_count(usage, "input_tokens");
        }
        self.started = true;
        // The shared assembler synthesizes the first role on the first
        // observable delta.  Deferring it lets a terminal refusal become the
        // single required role-bearing terminal chunk.
        Ok(())
    }

    fn block_start(&mut self, value: &[(String, JsonValue)]) -> Result<(), TargetError> {
        if !self.started || self.saw_message_delta {
            return Err(Self::invalid());
        }
        let index =
            usize::try_from(unsigned(required(value, "index")?)?).map_err(|_| Self::invalid())?;
        if index != self.next_block {
            return Err(Self::invalid());
        }
        let block = object(required(value, "content_block")?)?;
        let kind = string(required(block, "type")?)?;
        let mut state = BlockState::default();
        let mut delta = AssistantDelta::default();
        match kind {
            "text" => {
                state.kind = BlockKind::Text;
                if let Some(text) = get(block, "text") {
                    delta.content = Some(string(text)?.to_owned());
                }
            }
            "tool_use" => {
                state.kind = BlockKind::Tool;
                let id = match get(block, "id") {
                    None => None,
                    Some(JsonValue::String(id)) => Some(id.clone()),
                    _ => return Err(Self::invalid()),
                };
                let name = string(required(block, "name")?)?.to_owned();
                // Anthropic starts streamed tools with an empty object and sends
                // the actual object incrementally as input_json_delta fragments.
                if !matches!(get(block, "input"), Some(JsonValue::Object(input)) if input.is_empty())
                {
                    return Err(Self::invalid());
                }
                state.call_index = Some(self.next_call);
                self.next_call += 1;
                delta.tool_calls.push(ToolCallDelta {
                    index: state.call_index.unwrap(),
                    id,
                    r#type: Some("function"),
                    name: Some(name),
                    arguments: None,
                });
            }
            _ => return Err(Self::invalid()),
        }
        self.blocks.insert(index, state);
        self.next_block += 1;
        self.push(delta, None)
    }

    fn block_delta(&mut self, value: &[(String, JsonValue)]) -> Result<(), TargetError> {
        if !self.started || self.saw_message_delta {
            return Err(Self::invalid());
        }
        let index =
            usize::try_from(unsigned(required(value, "index")?)?).map_err(|_| Self::invalid())?;
        let state = self.blocks.get_mut(&index).ok_or_else(Self::invalid)?;
        let delta = object(required(value, "delta")?)?;
        let (content, arguments) = match state.kind {
            BlockKind::Text => match string(required(delta, "type")?)? {
                "text_delta" => (Some(string(required(delta, "text")?)?.to_owned()), None),
                _ => return Err(Self::invalid()),
            },
            BlockKind::Tool => match string(required(delta, "type")?)? {
                "input_json_delta" => (
                    None,
                    Some(string(required(delta, "partial_json")?)?.to_owned()),
                ),
                _ => return Err(Self::invalid()),
            },
        };
        if let Some(arguments) = &arguments {
            state.arguments.push_str(arguments);
        }
        let mut output = AssistantDelta {
            content,
            ..Default::default()
        };
        if let Some(arguments) = arguments {
            output.tool_calls.push(ToolCallDelta {
                index: state.call_index.unwrap(),
                id: None,
                r#type: None,
                name: None,
                arguments: Some(arguments),
            });
        }
        self.push(output, None)
    }

    fn block_stop(&mut self, value: &[(String, JsonValue)]) -> Result<(), TargetError> {
        if !self.started || self.saw_message_delta {
            return Err(Self::invalid());
        }
        let index =
            usize::try_from(unsigned(required(value, "index")?)?).map_err(|_| Self::invalid())?;
        let state = self.blocks.remove(&index).ok_or_else(Self::invalid)?;
        if state.kind == BlockKind::Tool {
            let parsed =
                decode_json_value(state.arguments.as_bytes()).map_err(|_| Self::invalid())?;
            if !matches!(parsed, JsonValue::Object(_)) {
                return Err(Self::invalid());
            }
        }
        Ok(())
    }

    fn message_delta(&mut self, value: &[(String, JsonValue)]) -> Result<(), TargetError> {
        if !self.started || self.saw_message_delta || !self.blocks.is_empty() {
            return Err(Self::invalid());
        }
        let delta = object(required(value, "delta")?)?;
        let reason = string(required(delta, "stop_reason")?)?;
        if let Some(usage) = get(value, "usage") {
            self.output_tokens = usage_count(usage, "output_tokens");
        }
        self.saw_message_delta = true;
        self.push(AssistantDelta::default(), Some(terminal(reason)))
    }

    fn message_stop(&mut self, _value: &[(String, JsonValue)]) -> Result<(), TargetError> {
        if !self.saw_message_delta || self.saw_message_stop {
            return Err(Self::invalid());
        }
        self.saw_message_stop = true;
        self.assembler.finish().map_err(response_error)?;
        self.done = true;
        if let Some(usage) = normalize_usage(
            self.input_tokens.map(u128::from),
            self.output_tokens.map(u128::from),
        ) {
            if let Some(chunk) = self.assembler.usage_chunk(usage).map_err(response_error)? {
                self.pending.push_back(Ok(chunk));
            }
        }
        Ok(())
    }

    fn push(
        &mut self,
        delta: AssistantDelta,
        terminal: Option<NativeTerminal>,
    ) -> Result<(), TargetError> {
        if let Some(chunk) = self
            .assembler
            .push(0, delta, terminal)
            .map_err(response_error)?
        {
            self.pending.push_back(Ok(chunk));
        }
        Ok(())
    }

    fn next_event(&mut self) -> Result<bool, TargetError> {
        loop {
            if let Some(newline) = self.input.iter().position(|byte| *byte == b'\n') {
                let line: Vec<_> = self.input.drain(..=newline).collect();
                self.line(line[..line.len() - 1].to_vec())?;
                return Ok(true);
            }
            match self.source.next() {
                Some(Ok(bytes)) => {
                    if self.input.len().saturating_add(bytes.len())
                        > MAX_SSE_EVENT_BYTES + 64 * 1024
                    {
                        return Err(Self::invalid());
                    }
                    self.input.extend_from_slice(&bytes);
                }
                Some(Err(_)) => return Err(TargetError::connection()),
                None => {
                    self.exhausted = true;
                    if !self.input.is_empty() || !self.data.is_empty() || !self.done {
                        return Err(Self::invalid());
                    }
                    return Ok(false);
                }
            }
        }
    }
}

impl Iterator for AnthropicSseDecoder {
    type Item = Result<ChatChunk, TargetError>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.pending.pop_front() {
            return Some(item);
        }
        if self.exhausted {
            return None;
        }
        loop {
            match self.next_event() {
                Ok(false) => return self.pending.pop_front(),
                Ok(true) => {
                    if let Some(item) = self.pending.pop_front() {
                        return Some(item);
                    }
                }
                Err(error) => {
                    self.exhausted = true;
                    return Some(Err(error));
                }
            }
        }
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

/// A streaming usage update carries input and output counts in separate
/// lifecycle events, so retain each valid component until `message_stop`.
fn usage_count(value: &JsonValue, name: &str) -> Option<u64> {
    let JsonValue::Object(fields) = value else {
        return None;
    };
    get(fields, name).and_then(|value| unsigned(value).ok())
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

    fn stream_request(include_usage: bool, tools: bool) -> CanonicalRequest {
        let body = match (include_usage, tools) {
            (true, true) => {
                r#"{"model":"public","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true,"stream_options":{"include_usage":true},"tools":[{"type":"function","function":{"name":"weather"}}],"tool_choice":"auto"}"#
            }
            (false, false) => {
                r#"{"model":"public","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true}"#
            }
            _ => unreachable!(),
        };
        decode_chat_request(body.as_bytes()).unwrap()
    }

    fn decode_native(
        request: CanonicalRequest,
        frames: &[&str],
    ) -> Vec<Result<ChatChunk, TargetError>> {
        let source = frames
            .iter()
            .map(|frame| Ok(frame.as_bytes().to_vec()))
            .collect::<Vec<_>>()
            .into_iter();
        AnthropicSseDecoder::new(
            request,
            ResponseMetadata::for_model("public"),
            Box::new(source),
        )
        .collect()
    }

    #[test]
    fn streaming_translates_fragmented_text_tools_and_usage() {
        let frames = [
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"type\":\"message\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":2}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"hel\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"native\",\"name\":\"weather\",\"input\":{}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":3}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ];
        // Deliberately split inside both a JSON object and an SSE field.
        let joined = frames.concat();
        let split = [&joined[..97], &joined[97..401], &joined[401..]];
        let output = decode_native(stream_request(true, true), &split);
        assert!(output.iter().all(Result::is_ok));
        let output: Vec<_> = output.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            output.last().unwrap().usage.as_ref().unwrap().total_tokens,
            5
        );
        let terminal = &output[5];
        assert_eq!(
            terminal.choices[0].finish_reason.unwrap().as_str(),
            "tool_calls"
        );
        let tool = &output[2].choices[0].delta.tool_calls[0];
        assert_eq!(tool.index, 0);
        assert_eq!(tool.id.as_deref(), Some("native"));
        assert_eq!(tool.name.as_deref(), Some("weather"));
        assert_eq!(
            output[4].choices[0].delta.tool_calls[0]
                .arguments
                .as_deref(),
            Some("}")
        );
    }

    #[test]
    fn streaming_refusal_is_a_single_role_bearing_content_filter_terminal() {
        let output = decode_native(stream_request(false, false), &[
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"type\":\"message\",\"role\":\"assistant\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\"}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]);
        assert_eq!(output.len(), 1);
        let chunk = output[0].as_ref().unwrap();
        assert_eq!(chunk.choices[0].delta.role, Some("assistant"));
        assert_eq!(
            chunk.choices[0].finish_reason.unwrap().as_str(),
            "content_filter"
        );
    }

    #[test]
    fn streaming_rejects_missing_repeated_inconsistent_and_post_terminal_state() {
        let start = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"type\":\"message\",\"role\":\"assistant\"}}\n\n";
        let terminal = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n";
        let stop = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let text_start = "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"x\"}}\n\n";
        for frames in [
            vec![start],
            vec![start, terminal, stop, stop],
            vec![start, text_start, terminal],
            vec![start, terminal, stop, text_start],
            vec![start, "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"pause_turn\"}}\n\n"],
        ] {
            let output = decode_native(stream_request(false, false), &frames);
            assert!(matches!(output.last(), Some(Err(error)) if error.kind == crate::providers::TargetErrorKind::InvalidResponse));
        }
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
