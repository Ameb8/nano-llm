//! Gateway-owned canonical response and streaming conformance helpers.
//!
//! Provider adapters translate their wire protocol into the small input types in
//! this module.  They must not construct public response bodies themselves.

use crate::request::{CanonicalRequest, JsonValue, ToolChoice};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseError {
    InvalidResponse(String),
    Overloaded,
}

impl std::fmt::Display for ResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidResponse(message) => f.write_str(message),
            Self::Overloaded => f.write_str("upstream is overloaded"),
        }
    }
}

impl std::error::Error for ResponseError {}

/// The only finish reasons exposed by the canonical API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

impl FinishReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
        }
    }
}

/// A provider terminal condition after provider-specific parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeTerminal {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    /// A structurally valid but undocumented terminal reason.
    Unknown,
    /// DeepSeek's `insufficient_system_resource` and equivalent exhaustion.
    Overloaded,
    /// A native protocol failure presented in a successful HTTP response.
    Invalid,
}

/// Complete, validated canonical usage.  There are no provider detail fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Normalize provider counts.  Missing, negative, non-integral, out-of-range,
/// or overflowing counts deliberately become no usage rather than an error.
pub fn normalize_usage(
    prompt_tokens: Option<u128>,
    completion_tokens: Option<u128>,
) -> Option<Usage> {
    let prompt_tokens = u64::try_from(prompt_tokens?).ok()?;
    let completion_tokens = u64::try_from(completion_tokens?).ok()?;
    let total_tokens = prompt_tokens.checked_add(completion_tokens)?;
    Some(Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    })
}

fn valid_usage(usage: Usage) -> Option<Usage> {
    (usage.prompt_tokens.checked_add(usage.completion_tokens)? == usage.total_tokens)
        .then_some(usage)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseMetadata {
    id: String,
    created: u64,
}

impl ResponseMetadata {
    /// Generate metadata once per selected attempt.  Reuse it for every chunk.
    pub fn for_model(_model: impl Into<String>) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let sequence = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            id: format!("chatcmpl-{created:x}-{sequence:x}"),
            created,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub const fn created(&self) -> u64 {
        self.created
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    /// Opaque model output; intentionally not parsed as JSON.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub r#type: &'static str,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatChoice {
    pub index: u8,
    pub message: AssistantMessage,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
}

/// One parsed upstream candidate.  Adapters must pass the whole candidate
/// array to [`normalize_response`] so this boundary can reject multi-choice
/// and non-zero-index responses rather than silently selecting one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeChoice {
    pub index: u64,
    pub content: Option<String>,
    pub calls: Vec<NativeToolCall>,
    pub terminal: NativeTerminal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeResponse {
    pub choices: Vec<NativeChoice>,
    pub usage: Option<Usage>,
}

/// Validate and normalize a parsed non-streaming provider response.
pub fn normalize_response(
    request: &CanonicalRequest,
    metadata: ResponseMetadata,
    native: NativeResponse,
) -> Result<ChatResponse, ResponseError> {
    let [choice] = native.choices.as_slice() else {
        return Err(ResponseError::InvalidResponse(
            "provider response must contain exactly one choice".into(),
        ));
    };
    if choice.index != 0 {
        return Err(ResponseError::InvalidResponse(
            "provider response choice index must be zero".into(),
        ));
    }
    build_response(
        request,
        metadata,
        choice.content.clone(),
        choice.calls.clone(),
        choice.terminal,
        native.usage,
    )
}

/// Build a canonical buffered response from one provider candidate.
pub fn build_response(
    request: &CanonicalRequest,
    metadata: ResponseMetadata,
    content: Option<String>,
    calls: Vec<NativeToolCall>,
    terminal: NativeTerminal,
    usage: Option<Usage>,
) -> Result<ChatResponse, ResponseError> {
    let calls = normalize_calls(request, calls, &mut HashSet::new())?;
    let finish_reason = normalize_terminal(terminal, content.as_deref(), !calls.is_empty())?;
    validate_outcome(request, content.as_deref(), &calls, finish_reason)?;
    Ok(ChatResponse {
        id: metadata.id,
        object: "chat.completion",
        created: metadata.created,
        // Never trust an upstream model identifier: this is always the alias
        // that the client requested.
        model: request.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content,
                tool_calls: (!calls.is_empty()).then_some(calls),
            },
            finish_reason,
        }],
        usage: usage.and_then(valid_usage),
    })
}

/// Synthesize only an explicit provider-level safety/policy block with no candidate.
pub fn safety_response(
    request: &CanonicalRequest,
    metadata: ResponseMetadata,
    usage: Option<Usage>,
) -> Result<ChatResponse, ResponseError> {
    build_response(
        request,
        metadata,
        None,
        Vec::new(),
        NativeTerminal::ContentFilter,
        usage,
    )
}

fn declared_tools(request: &CanonicalRequest) -> HashSet<String> {
    let Some(JsonValue::Array(tools)) = request
        .fields
        .iter()
        .find_map(|(k, v)| (k == "tools").then_some(v))
    else {
        return HashSet::new();
    };
    tools
        .iter()
        .filter_map(|tool| {
            let JsonValue::Object(tool) = tool else {
                return None;
            };
            let JsonValue::Object(function) = tool
                .iter()
                .find_map(|(k, v)| (k == "function").then_some(v))?
            else {
                return None;
            };
            let JsonValue::String(name) = function
                .iter()
                .find_map(|(k, v)| (k == "name").then_some(v))?
            else {
                return None;
            };
            Some(name.clone())
        })
        .collect()
}

fn generated_call_id(used: &mut HashSet<String>) -> String {
    loop {
        let id = format!("call_{:x}", NEXT_ID.fetch_add(1, Ordering::Relaxed));
        if used.insert(id.clone()) {
            return id;
        }
    }
}

fn normalize_calls(
    request: &CanonicalRequest,
    native: Vec<NativeToolCall>,
    used: &mut HashSet<String>,
) -> Result<Vec<ToolCall>, ResponseError> {
    let declared = declared_tools(request);
    native
        .into_iter()
        .map(|call| {
            if !declared.contains(&call.name) {
                return Err(ResponseError::InvalidResponse(
                    "tool call names an undeclared function".into(),
                ));
            }
            let id = match call.id.filter(|id| !id.is_empty()) {
                Some(id) if used.insert(id.clone()) => id,
                _ => generated_call_id(used),
            };
            Ok(ToolCall {
                id,
                r#type: "function",
                function: FunctionCall {
                    name: call.name,
                    arguments: call.arguments,
                },
            })
        })
        .collect()
}

fn normalize_terminal(
    terminal: NativeTerminal,
    content: Option<&str>,
    has_calls: bool,
) -> Result<FinishReason, ResponseError> {
    match terminal {
        NativeTerminal::Overloaded => Err(ResponseError::Overloaded),
        NativeTerminal::Invalid => Err(ResponseError::InvalidResponse(
            "invalid provider terminal condition".into(),
        )),
        NativeTerminal::ContentFilter => Ok(FinishReason::ContentFilter),
        NativeTerminal::ToolCalls if !has_calls => Err(ResponseError::InvalidResponse(
            "tool_calls finish reason has no completed call".into(),
        )),
        _ if has_calls => Ok(FinishReason::ToolCalls),
        NativeTerminal::ToolCalls => unreachable!(),
        NativeTerminal::Length => Ok(FinishReason::Length),
        NativeTerminal::Stop => Ok(FinishReason::Stop),
        NativeTerminal::Unknown if content.is_some() => Ok(FinishReason::Stop),
        NativeTerminal::Unknown => Err(ResponseError::InvalidResponse(
            "unknown terminal reason has no candidate content".into(),
        )),
    }
}

fn validate_outcome(
    request: &CanonicalRequest,
    content: Option<&str>,
    calls: &[ToolCall],
    finish: FinishReason,
) -> Result<(), ResponseError> {
    if finish != FinishReason::ContentFilter && content.is_none() && calls.is_empty() {
        return Err(ResponseError::InvalidResponse(
            "assistant response has neither content nor tool calls".into(),
        ));
    }
    let calls_present = !calls.is_empty();
    let tool_calls_finish = finish == FinishReason::ToolCalls;
    if calls_present ^ tool_calls_finish {
        return Err(ResponseError::InvalidResponse(
            "completed calls and finish reason disagree".into(),
        ));
    }
    if finish != FinishReason::ContentFilter {
        match &request.tool_choice {
            ToolChoice::None if !calls.is_empty() => {
                return Err(ResponseError::InvalidResponse(
                    "tool_choice none forbids tool calls".into(),
                ))
            }
            ToolChoice::Required if calls.is_empty() => {
                return Err(ResponseError::InvalidResponse(
                    "tool_choice required needs a tool call".into(),
                ))
            }
            ToolChoice::Named(name)
                if calls.is_empty() || calls.iter().any(|call| call.function.name != *name) =>
            {
                return Err(ResponseError::InvalidResponse(
                    "tool calls violate named tool_choice".into(),
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AssistantDelta {
    pub role: Option<&'static str>,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub r#type: Option<&'static str>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkChoice {
    pub index: u8,
    pub delta: AssistantDelta,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Default)]
struct PartialCall {
    id: String,
    /// A generated ID deliberately ignores later native IDs.  A preserved
    /// native ID instead has to remain stable for the life of its call.
    preserved_native_id: bool,
    name: String,
    arguments: String,
}

/// Stateful conformance boundary for one native stream.  `push` returns only
/// ordinary chunks; `usage_chunk` returns the one permitted zero-choice chunk.
pub struct StreamAssembler<'a> {
    request: &'a CanonicalRequest,
    metadata: ResponseMetadata,
    calls: HashMap<usize, PartialCall>,
    used_ids: HashSet<String>,
    next_index: usize,
    started: bool,
    saw_content: bool,
    terminal: bool,
    emitted_usage: bool,
}

impl<'a> StreamAssembler<'a> {
    pub fn new(request: &'a CanonicalRequest, metadata: ResponseMetadata) -> Self {
        Self {
            request,
            metadata,
            calls: HashMap::new(),
            used_ids: HashSet::new(),
            next_index: 0,
            started: false,
            saw_content: false,
            terminal: false,
            emitted_usage: false,
        }
    }

    /// Normalize one parsed native choice.  Empty native events return `None`;
    /// callers must not turn them into observable canonical chunks.
    pub fn push(
        &mut self,
        choice_index: u64,
        mut delta: AssistantDelta,
        terminal: Option<NativeTerminal>,
    ) -> Result<Option<ChatChunk>, ResponseError> {
        if choice_index != 0 {
            return Err(ResponseError::InvalidResponse(
                "ordinary stream choice index must be zero".into(),
            ));
        }
        if self.terminal {
            return Err(ResponseError::InvalidResponse(
                "ordinary chunk after terminal chunk".into(),
            ));
        }
        if matches!(delta.role, Some(role) if role != "assistant") {
            return Err(ResponseError::InvalidResponse(
                "stream delta role must be assistant".into(),
            ));
        }
        for part in &mut delta.tool_calls {
            self.accept_call_delta(part)?;
        }
        if delta.role.is_none()
            && delta.content.is_none()
            && delta.tool_calls.is_empty()
            && terminal.is_none()
        {
            return Ok(None);
        }
        if !self.started {
            delta.role = Some("assistant");
            self.started = true;
        }
        self.saw_content |= delta.content.is_some();
        let finish_reason = if let Some(native) = terminal {
            let calls = self.completed_calls()?;
            let content = self.saw_content.then_some("");
            let finish = normalize_terminal(native, content, !calls.is_empty())?;
            validate_outcome(self.request, content, &calls, finish)?;
            self.terminal = true;
            Some(finish)
        } else {
            None
        };
        Ok(Some(self.chunk(delta, finish_reason)))
    }

    pub fn finish(&self) -> Result<(), ResponseError> {
        if !self.started {
            return Err(ResponseError::InvalidResponse(
                "stream ended before an ordinary chunk".into(),
            ));
        }
        if !self.terminal {
            return Err(ResponseError::InvalidResponse(
                "stream ended without a terminal chunk".into(),
            ));
        }
        Ok(())
    }

    pub fn usage_chunk(&mut self, usage: Usage) -> Result<Option<ChatChunk>, ResponseError> {
        self.finish()?;
        if !self.request.include_usage || self.emitted_usage {
            return Ok(None);
        }
        let Some(usage) = valid_usage(usage) else {
            return Ok(None);
        };
        self.emitted_usage = true;
        Ok(Some(ChatChunk {
            id: self.metadata.id.clone(),
            object: "chat.completion.chunk",
            created: self.metadata.created,
            model: self.request.model.clone(),
            choices: Vec::new(),
            usage: Some(usage),
        }))
    }

    fn chunk(&self, delta: AssistantDelta, finish_reason: Option<FinishReason>) -> ChatChunk {
        ChatChunk {
            id: self.metadata.id.clone(),
            object: "chat.completion.chunk",
            created: self.metadata.created,
            model: self.request.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage: None,
        }
    }

    fn accept_call_delta(&mut self, part: &mut ToolCallDelta) -> Result<(), ResponseError> {
        if part.index > self.next_index {
            return Err(ResponseError::InvalidResponse(
                "tool-call indices must be introduced in increasing order".into(),
            ));
        }
        if part.index == self.next_index {
            let (id, preserved_native_id) = match part.id.as_ref().filter(|id| !id.is_empty()) {
                Some(id) if self.used_ids.insert(id.clone()) => (id.clone(), true),
                _ => (generated_call_id(&mut self.used_ids), false),
            };
            part.id = Some(id.clone());
            self.calls.insert(
                part.index,
                PartialCall {
                    id,
                    preserved_native_id,
                    ..Default::default()
                },
            );
            self.next_index += 1;
        }
        let call = self.calls.get_mut(&part.index).ok_or_else(|| {
            ResponseError::InvalidResponse("tool-call index was not introduced".into())
        })?;
        if let Some(id) = &part.id {
            if call.preserved_native_id && id != &call.id {
                return Err(ResponseError::InvalidResponse(
                    "tool-call ID changed during stream".into(),
                ));
            }
        }
        if !call.preserved_native_id {
            part.id = Some(call.id.clone());
        }
        if matches!(part.r#type, Some(kind) if kind != "function") {
            return Err(ResponseError::InvalidResponse(
                "stream tool-call type must be function".into(),
            ));
        }
        if let Some(name) = &part.name {
            call.name.push_str(name);
        }
        if let Some(arguments) = &part.arguments {
            call.arguments.push_str(arguments);
        }
        Ok(())
    }

    fn completed_calls(&self) -> Result<Vec<ToolCall>, ResponseError> {
        let declared = declared_tools(self.request);
        let mut calls: Vec<_> = self.calls.iter().collect();
        calls.sort_by_key(|(index, _)| **index);
        calls
            .into_iter()
            .map(|(_, call)| {
                if call.name.is_empty() || !declared.contains(&call.name) {
                    return Err(ResponseError::InvalidResponse(
                        "streamed tool call is incomplete or undeclared".into(),
                    ));
                }
                Ok(ToolCall {
                    id: call.id.clone(),
                    r#type: "function",
                    function: FunctionCall {
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                })
            })
            .collect()
    }
}
