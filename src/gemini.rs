//! Gemini Generative Language request translation.
//!
//! This module owns the pinned v1beta operation shape and the conversion from
//! the gateway's canonical conversation into Gemini `contents`, plus strict
//! normalization of Gemini responses and incremental SSE events.

use crate::config::{ProviderKind, RuntimeTarget};
use crate::providers::{
    OutboundRequest, OutboundTransport, Provider, ProviderFuture, ProviderStream,
    SecureTransportPolicy, TargetError, TransportErrorKind,
};
use crate::request::{decode_json_value, CanonicalRequest, JsonValue, ToolChoice};
use crate::response::{
    normalize_response, normalize_usage, safety_response, AssistantDelta, ChatChunk, ChatResponse,
    NativeChoice, NativeResponse, NativeTerminal, NativeToolCall, ResponseError, ResponseMetadata,
    StreamAssembler, ToolCallDelta, Usage,
};
use std::collections::{HashMap, HashSet, VecDeque};
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
            Ok(Box::new(GeminiSseDecoder::new(
                request.clone(),
                ResponseMetadata::for_model(&request.model),
                response.body,
            )) as ProviderStream)
        })
    }
}

/// Maximum decoded payload of one Gemini SSE event. Transport fragments may
/// split an SSE field or JSON value at arbitrary byte boundaries.
const MAX_SSE_EVENT_BYTES: usize = 1024 * 1024;

/// Incremental decoder for Gemini's `streamGenerateContent` response. Gemini
/// does not send an OpenAI-style `[DONE]` marker: a valid native terminal
/// candidate followed by EOF is the successful lifecycle.
pub struct GeminiSseDecoder {
    source: crate::providers::OutboundByteStream,
    assembler: StreamAssembler,
    input: Vec<u8>,
    data: Vec<u8>,
    pending: VecDeque<Result<ChatChunk, TargetError>>,
    native_calls: HashMap<String, GeminiCall>,
    /// An ID repeated for separate calls in one native candidate cannot safely
    /// identify a later delta.  Do not silently attach that delta to either
    /// call.
    ambiguous_native_ids: HashSet<String>,
    declared_tools: HashSet<String>,
    tool_choice: ToolChoice,
    next_call: usize,
    done: bool,
    exhausted: bool,
    prompt_tokens: Option<u128>,
    completion_tokens: Option<u128>,
    invalid_usage: bool,
}

#[derive(Clone)]
struct GeminiCall {
    index: usize,
    name: String,
}

impl GeminiSseDecoder {
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
            pending: VecDeque::new(),
            native_calls: HashMap::new(),
            ambiguous_native_ids: HashSet::new(),
            declared_tools: declared_tool_names(&request),
            tool_choice: request.tool_choice.clone(),
            next_call: 0,
            done: false,
            exhausted: false,
            prompt_tokens: None,
            completion_tokens: None,
            invalid_usage: false,
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
        if let Some(value) = line.strip_prefix(b"data:") {
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
            return Ok(());
        }
        let payload = std::mem::take(&mut self.data);
        let root = crate::request::decode_json_object(&payload).map_err(|_| Self::invalid())?;
        self.capture_usage(&root);
        // Gemini may report usage in a separate event after the terminal
        // candidate.  It remains adapter metadata, not a second completion.
        if self.done {
            return field(&root, "candidates")
                .is_none()
                .then_some(())
                .filter(|_| field(&root, "usageMetadata").is_some())
                .ok_or_else(Self::invalid);
        }
        match field(&root, "candidates") {
            Some(JsonValue::Array(candidates)) if candidates.len() == 1 => {
                self.candidate(&candidates[0])
            }
            // A usage-only event is metadata, never an observable zero-choice
            // chunk. Missing candidates without this native metadata remains
            // malformed rather than becoming an empty completion.
            None if field(&root, "usageMetadata").is_some() => Ok(()),
            Some(JsonValue::Array(candidates))
                if candidates.is_empty() && explicit_policy_block(&root) =>
            {
                self.policy_terminal()
            }
            None if explicit_policy_block(&root) => self.policy_terminal(),
            _ => Err(Self::invalid()),
        }
    }

    fn capture_usage(&mut self, root: &[(String, JsonValue)]) {
        let Some(value) = field(root, "usageMetadata") else {
            return;
        };
        let JsonValue::Object(usage) = value else {
            self.invalid_usage = true;
            return;
        };
        match usage_component(field(usage, "promptTokenCount")) {
            Ok(Some(value)) => self.prompt_tokens = Some(value),
            Ok(None) => {}
            Err(()) => self.invalid_usage = true,
        }
        match usage_component(field(usage, "candidatesTokenCount")) {
            Ok(Some(value)) => self.completion_tokens = Some(value),
            Ok(None) => {}
            Err(()) => self.invalid_usage = true,
        }
    }

    fn candidate(&mut self, value: &JsonValue) -> Result<(), TargetError> {
        let candidate = object(value)?;
        if let Some(index) = field(candidate, "index") {
            if unsigned(index)? != 0 {
                return Err(Self::invalid());
            }
        }
        let terminal = match field(candidate, "finishReason") {
            None => None,
            Some(JsonValue::String(reason)) => Some(terminal(reason)),
            Some(_) => return Err(Self::invalid()),
        };
        let delta = match field(candidate, "content") {
            Some(content) => self.content(content)?,
            None if matches!(terminal, Some(NativeTerminal::ContentFilter)) => {
                AssistantDelta::default()
            }
            None => return Err(Self::invalid()),
        };
        self.push(delta, terminal)
    }

    fn content(&mut self, value: &JsonValue) -> Result<AssistantDelta, TargetError> {
        let content = object(value)?;
        match field(content, "role") {
            None => {}
            Some(JsonValue::String(role)) if role == "model" => {}
            _ => return Err(Self::invalid()),
        }
        let parts = array(required(content, "parts")?)?;
        let mut delta = AssistantDelta::default();
        let mut event_call_ids = HashSet::new();
        for part in parts {
            let part = object(part)?;
            match (field(part, "text"), field(part, "functionCall")) {
                (Some(JsonValue::String(text)), None) => {
                    let output = delta.content.get_or_insert_with(String::new);
                    output.push_str(text);
                }
                (None, Some(call)) => delta.tool_calls.push(self.call(call, &mut event_call_ids)?),
                _ => return Err(Self::invalid()),
            }
        }
        Ok(delta)
    }

    fn call(
        &mut self,
        value: &JsonValue,
        event_call_ids: &mut HashSet<String>,
    ) -> Result<ToolCallDelta, TargetError> {
        let call = object(value)?;
        let native_id = match field(call, "id") {
            None => None,
            Some(JsonValue::String(id)) if !id.is_empty() => Some(id.clone()),
            Some(JsonValue::String(_)) => None,
            Some(_) => return Err(Self::invalid()),
        };
        let name = string(required(call, "name")?)?.to_owned();
        if name.is_empty()
            || !self.declared_tools.contains(&name)
            || matches!(&self.tool_choice, ToolChoice::None)
            || matches!(&self.tool_choice, ToolChoice::Named(expected) if expected != &name)
        {
            return Err(Self::invalid());
        }
        let JsonValue::Object(args) = required(call, "args")? else {
            return Err(Self::invalid());
        };
        let arguments = encode(&JsonValue::Object(args.to_vec()));
        if let Some(native_id) = native_id {
            if self.ambiguous_native_ids.contains(&native_id) {
                return Err(Self::invalid());
            }
            if !event_call_ids.insert(native_id.clone()) {
                // Native duplicate IDs identify distinct calls only in this
                // event.  Give the later call a gateway ID and reject any
                // subsequent ambiguous update rather than coalescing calls.
                self.ambiguous_native_ids.insert(native_id);
                return Ok(self.new_call(name, arguments));
            }
            if let Some(existing) = self.native_calls.get(&native_id) {
                if existing.name != name {
                    return Err(Self::invalid());
                }
                return Ok(ToolCallDelta {
                    index: existing.index,
                    id: Some(native_id),
                    r#type: None,
                    name: None,
                    arguments: Some(arguments),
                });
            }
            let index = self.next_call;
            self.next_call += 1;
            self.native_calls.insert(
                native_id.clone(),
                GeminiCall {
                    index,
                    name: name.clone(),
                },
            );
            Ok(ToolCallDelta {
                index,
                id: Some(native_id),
                r#type: Some("function"),
                name: Some(name),
                arguments: Some(arguments),
            })
        } else {
            // Without a native correlation ID, Gemini provides no safe way to
            // merge same-name calls. Treat each part as a new call so repeated
            // names remain distinct rather than silently coalescing them.
            Ok(self.new_call(name, arguments))
        }
    }

    fn new_call(&mut self, name: String, arguments: String) -> ToolCallDelta {
        let index = self.next_call;
        self.next_call += 1;
        ToolCallDelta {
            index,
            id: None,
            r#type: Some("function"),
            name: Some(name),
            arguments: Some(arguments),
        }
    }

    fn policy_terminal(&mut self) -> Result<(), TargetError> {
        self.push(
            AssistantDelta::default(),
            Some(NativeTerminal::ContentFilter),
        )
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
        if terminal.is_some() {
            self.done = true;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), TargetError> {
        self.assembler.finish().map_err(response_error)?;
        if !self.invalid_usage {
            if let Some(usage) = normalize_usage(self.prompt_tokens, self.completion_tokens) {
                if let Some(chunk) = self.assembler.usage_chunk(usage).map_err(response_error)? {
                    self.pending.push_back(Ok(chunk));
                }
            }
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
                    self.finish()?;
                    return Ok(false);
                }
            }
        }
    }
}

impl Iterator for GeminiSseDecoder {
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

/// A streaming usage event may report its two components at different times.
/// Preserve valid components independently, but poison the optional usage
/// result if a component it does report is malformed.
fn usage_component(value: Option<&JsonValue>) -> Result<Option<u128>, ()> {
    let Some(value) = value else {
        return Ok(None);
    };
    let JsonValue::Number(value) = value else {
        return Err(());
    };
    if value.contains(['.', 'e', 'E']) {
        return Err(());
    }
    value.parse().map(Some).map_err(|_| ())
}

fn declared_tool_names(request: &CanonicalRequest) -> HashSet<String> {
    let Some(JsonValue::Array(tools)) = field(&request.fields, "tools") else {
        return HashSet::new();
    };
    tools
        .iter()
        .filter_map(|tool| {
            let JsonValue::Object(tool) = tool else {
                return None;
            };
            let JsonValue::Object(function) = field(tool, "function")? else {
                return None;
            };
            let JsonValue::String(name) = field(function, "name")? else {
                return None;
            };
            Some(name.clone())
        })
        .collect()
}

fn explicit_policy_block(root: &[(String, JsonValue)]) -> bool {
    let Some(JsonValue::Object(feedback)) = field(root, "promptFeedback") else {
        return false;
    };
    matches!(field(feedback, "blockReason"),
        Some(JsonValue::String(reason)) if matches!(reason.as_str(),
            "SAFETY" | "RECITATION" | "LANGUAGE" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII"
            | "IMAGE_SAFETY" | "JAILBREAK" | "MODEL_ARMOR" | "OTHER"))
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

    fn streamed(request: &str, fragments: &[&str]) -> Vec<Result<ChatChunk, TargetError>> {
        let request = decode_chat_request(request.as_bytes()).unwrap();
        let fragments: Vec<_> = fragments
            .iter()
            .map(|part| Ok(part.as_bytes().to_vec()))
            .collect();
        GeminiSseDecoder::new(
            request.clone(),
            ResponseMetadata::for_model(&request.model),
            Box::new(fragments.into_iter()),
        )
        .collect()
    }

    #[test]
    fn streams_fragmented_text_distinct_same_name_calls_and_usage() {
        let events = concat!(
            "data: {\"candidates\":[{\"index\":0,\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"one\",\"name\":\"weather\",\"args\":{\"city\":\"Paris\"}}},{\"functionCall\":{\"id\":\"two\",\"name\":\"weather\",\"args\":{\"city\":\"Rome\"}}}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":3}}\n\n"
        );
        // Deliberately split inside both SSE and JSON framing.
        let output = streamed(
            r#"{"model":"public","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"weather"}}]}"#,
            &[&events[..41], &events[41..187], &events[187..]],
        );
        assert_eq!(output.len(), 4);
        let first = output[0].as_ref().unwrap();
        assert_eq!(first.choices[0].delta.role, Some("assistant"));
        assert_eq!(first.choices[0].delta.content.as_deref(), Some("Hel"));
        let calls = &output[1].as_ref().unwrap().choices[0].delta.tool_calls;
        assert_eq!(calls.len(), 2);
        assert_eq!((calls[0].index, calls[0].id.as_deref()), (0, Some("one")));
        assert_eq!((calls[1].index, calls[1].id.as_deref()), (1, Some("two")));
        let terminal = output[2].as_ref().unwrap();
        assert_eq!(terminal.choices[0].delta.content.as_deref(), Some("lo"));
        assert_eq!(
            terminal.choices[0].finish_reason.unwrap().as_str(),
            "tool_calls"
        );
        let usage = output[3].as_ref().unwrap();
        assert!(usage.choices.is_empty());
        assert_eq!(usage.usage.as_ref().unwrap().total_tokens, 5);
    }

    #[test]
    fn streams_candidate_less_policy_as_role_bearing_terminal() {
        let output = streamed(
            r#"{"model":"public","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
            &["data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\"}}\n\n"],
        );
        assert_eq!(output.len(), 1);
        let chunk = output[0].as_ref().unwrap();
        assert_eq!(chunk.choices[0].delta.role, Some("assistant"));
        assert_eq!(chunk.choices[0].delta.content, None);
        assert_eq!(
            chunk.choices[0].finish_reason.unwrap().as_str(),
            "content_filter"
        );
    }

    #[test]
    fn retains_usage_sent_after_the_terminal_candidate() {
        let output = streamed(
            r#"{"model":"public","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}"#,
            &[
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n",
                "data: {\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":3}}\n\n",
            ],
        );
        assert_eq!(output.len(), 2);
        assert_eq!(
            output[1]
                .as_ref()
                .unwrap()
                .usage
                .as_ref()
                .unwrap()
                .total_tokens,
            5
        );
    }

    #[test]
    fn rejects_an_undeclared_call_before_emitting_a_chunk() {
        let output = streamed(
            r#"{"model":"public","stream":true,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"weather"}}]}"#,
            &["data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"other\",\"args\":{}}}]}}]}\n\n"],
        );
        assert!(
            matches!(output.as_slice(), [Err(error)] if error.kind == crate::providers::TargetErrorKind::InvalidResponse)
        );
    }

    #[test]
    fn rejects_missing_repeated_and_malformed_stream_state() {
        let request =
            r#"{"model":"public","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
        for events in [
            "data: {\"usageMetadata\":{\"promptTokenCount\":1}}\n\n",
            concat!(
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"x\"}]},\"finishReason\":\"STOP\"}]}\n\n",
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"y\"}]}}]}\n\n"
            ),
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"x\",\"args\":[]}}]}}]}\n\n",
        ] {
            let output = streamed(request, &[events]);
            assert!(output.last().is_some_and(|item| matches!(item, Err(error) if error.kind == crate::providers::TargetErrorKind::InvalidResponse)));
        }
    }
}
