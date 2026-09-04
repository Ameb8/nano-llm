//! Strict decoding for the gateway-owned chat request wire format.
//!
//! This deliberately does not use a map-backed JSON representation: maps erase
//! duplicate members before validation can report them.

use crate::config::{ProviderKind, RuntimeRoute};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalRequest {
    /// The validated request. Object members retain their input order.
    pub fields: Vec<(String, JsonValue)>,
    /// Gateway-owned, provider-neutral normalized values.
    pub model: String,
    pub stream: bool,
    pub include_usage: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stop: Option<Vec<String>>,
    /// Leading `system` and `developer` content, joined with exactly `\n\n`.
    pub instruction: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Bytes are not exactly one UTF-8 JSON object.
    InvalidJson { message: String },
    /// JSON is syntactically sound but violates the gateway request shape.
    Validation {
        param: Option<String>,
        message: String,
    },
}

impl DecodeError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidJson { .. } => "invalid_json",
            Self::Validation { .. } => "invalid_request",
        }
    }

    pub fn param(&self) -> Option<&str> {
        match self {
            Self::InvalidJson { .. } => None,
            Self::Validation { param, .. } => param.as_deref(),
        }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson { message } | Self::Validation { message, .. } => {
                f.write_str(message)
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Decode one complete request body and close every gateway-defined object.
///
/// Values in `tools[].function.parameters` are retained as opaque JSON Schema,
/// except that JSON syntax and duplicate-member checks still apply there.
pub fn decode_chat_request(bytes: &[u8]) -> Result<CanonicalRequest, DecodeError> {
    let text = std::str::from_utf8(bytes).map_err(|_| DecodeError::InvalidJson {
        message: "request body is not valid UTF-8".into(),
    })?;
    let mut parser = Parser::new(text);
    parser.ws();
    let value = parser.value("", None)?;
    parser.ws();
    if !parser.eof() {
        return Err(DecodeError::InvalidJson {
            message: "request body contains trailing data".into(),
        });
    }
    let JsonValue::Object(fields) = value else {
        return Err(DecodeError::InvalidJson {
            message: "request JSON must be an object".into(),
        });
    };
    validate_top(&fields)?;
    canonicalize(fields)
}

/// Decode and validate a request against the selected immutable fallback route.
///
/// This is deliberately separate from JSON decoding because only the selected
/// route can establish whether an output-token limit is required.
pub fn decode_chat_request_for_route(
    bytes: &[u8],
    route: &RuntimeRoute,
) -> Result<CanonicalRequest, DecodeError> {
    let request = decode_chat_request(bytes)?;
    request.validate_for_route(route)?;
    Ok(request)
}

impl CanonicalRequest {
    /// Validate route-wide representability requirements before routing begins.
    pub fn validate_for_route(&self, route: &RuntimeRoute) -> Result<(), DecodeError> {
        if route
            .targets
            .iter()
            .any(|target| target.provider == ProviderKind::Anthropic)
            && self.max_tokens.is_none()
        {
            return Err(validation(
                Some("max_tokens"),
                "'max_tokens' or 'max_completion_tokens' is required for routes containing Anthropic targets",
            ));
        }
        Ok(())
    }
}

fn canonicalize(fields: Vec<(String, JsonValue)>) -> Result<CanonicalRequest, DecodeError> {
    let model = match field(&fields, "model").unwrap() {
        JsonValue::String(value) => value.clone(),
        _ => unreachable!("validate_top validates model"),
    };
    let stream = matches!(field(&fields, "stream"), Some(JsonValue::Bool(true)));
    let include_usage = match field(&fields, "stream_options") {
        Some(JsonValue::Object(options)) => {
            matches!(field(options, "include_usage"), Some(JsonValue::Bool(true)))
        }
        None => false,
        _ => unreachable!("validate_top validates stream_options"),
    };
    let max_tokens = match (
        field(&fields, "max_tokens"),
        field(&fields, "max_completion_tokens"),
    ) {
        (Some(_), Some(_)) => {
            return Err(validation(
                Some("max_tokens"),
                "'max_tokens' and 'max_completion_tokens' cannot both be supplied",
            ))
        }
        (Some(value), None) | (None, Some(value)) => Some(positive_i32(value, "max_tokens")?),
        (None, None) => None,
    };
    let temperature = optional_probability(field(&fields, "temperature"), "temperature")?;
    let top_p = optional_probability(field(&fields, "top_p"), "top_p")?;
    let stop = match field(&fields, "stop") {
        Some(JsonValue::String(value)) => Some(vec![value.clone()]),
        Some(JsonValue::Array(values)) => Some(
            values
                .iter()
                .map(|value| match value {
                    JsonValue::String(value) => value.clone(),
                    _ => unreachable!("validate_top validates stop"),
                })
                .collect(),
        ),
        None => None,
        _ => unreachable!("validate_top validates stop"),
    };
    let instruction = combined_instruction(field(&fields, "messages").unwrap());
    Ok(CanonicalRequest {
        fields,
        model,
        stream,
        include_usage,
        max_tokens,
        temperature,
        top_p,
        stop,
        instruction,
    })
}

fn positive_i32(value: &JsonValue, param: &str) -> Result<u32, DecodeError> {
    let JsonValue::Number(text) = value else {
        unreachable!("caller validates number")
    };
    if text.contains(['.', 'e', 'E']) {
        return Err(validation(
            Some(param),
            format!("'{param}' must be an integer from 1 through 2147483647"),
        ));
    }
    match text.parse::<u32>() {
        Ok(value @ 1..=2_147_483_647) => Ok(value),
        _ => Err(validation(
            Some(param),
            format!("'{param}' must be an integer from 1 through 2147483647"),
        )),
    }
}

fn optional_probability(
    value: Option<&JsonValue>,
    param: &str,
) -> Result<Option<f64>, DecodeError> {
    let Some(JsonValue::Number(text)) = value else {
        return Ok(None);
    };
    match text.parse::<f64>() {
        Ok(value) if (0.0..=1.0).contains(&value) => Ok(Some(value)),
        _ => Err(validation(
            Some(param),
            format!("'{param}' must be a number from 0 through 1"),
        )),
    }
}

fn combined_instruction(messages: &JsonValue) -> Option<String> {
    let JsonValue::Array(messages) = messages else {
        unreachable!("validate_top validates messages")
    };
    let parts: Vec<&str> = messages.iter().take_while(|message| match message {
        JsonValue::Object(fields) => matches!(field(fields, "role"), Some(JsonValue::String(role)) if role == "system" || role == "developer"),
        _ => false,
    }).map(|message| match message {
        JsonValue::Object(fields) => match field(fields, "content") { Some(JsonValue::String(content)) => content.as_str(), _ => unreachable!("message validation requires instruction content") },
        _ => unreachable!(),
    }).collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

fn validation(param: Option<&str>, message: impl Into<String>) -> DecodeError {
    DecodeError::Validation {
        param: param.map(str::to_owned),
        message: message.into(),
    }
}

fn path_member(path: &str, member: &str) -> String {
    if path.is_empty() {
        member.to_owned()
    } else {
        format!("{path}.{member}")
    }
}

fn path_index(path: &str, index: usize) -> String {
    format!("{path}[{index}]")
}

fn field<'a>(fields: &'a [(String, JsonValue)], name: &str) -> Option<&'a JsonValue> {
    fields
        .iter()
        .find_map(|(key, value)| (key == name).then_some(value))
}

fn object<'a>(
    value: &'a JsonValue,
    param: &str,
    path: &str,
) -> Result<&'a [(String, JsonValue)], DecodeError> {
    match value {
        JsonValue::Object(fields) => Ok(fields),
        _ => Err(validation(
            Some(param),
            format!("'{path}' must be an object"),
        )),
    }
}

fn array<'a>(
    value: &'a JsonValue,
    param: &str,
    path: &str,
) -> Result<&'a [JsonValue], DecodeError> {
    match value {
        JsonValue::Array(items) => Ok(items),
        _ => Err(validation(
            Some(param),
            format!("'{path}' must be an array"),
        )),
    }
}

fn string(value: &JsonValue, param: &str, path: &str) -> Result<(), DecodeError> {
    matches!(value, JsonValue::String(_))
        .then_some(())
        .ok_or_else(|| validation(Some(param), format!("'{path}' must be a string")))
}

fn boolean(value: &JsonValue, param: &str, path: &str) -> Result<(), DecodeError> {
    matches!(value, JsonValue::Bool(_))
        .then_some(())
        .ok_or_else(|| validation(Some(param), format!("'{path}' must be a boolean")))
}

fn number(value: &JsonValue, param: &str, path: &str) -> Result<(), DecodeError> {
    matches!(value, JsonValue::Number(_))
        .then_some(())
        .ok_or_else(|| validation(Some(param), format!("'{path}' must be a number")))
}

fn closed(
    fields: &[(String, JsonValue)],
    allowed: &[&str],
    required: &[&str],
    param: &str,
    path: &str,
) -> Result<(), DecodeError> {
    for (key, _) in fields {
        if !allowed.contains(&key.as_str()) {
            return Err(validation(
                Some(param),
                format!("unknown member '{}' at '{path}'", key),
            ));
        }
    }
    for key in required {
        if field(fields, key).is_none() {
            return Err(validation(
                Some(param),
                format!("missing required member '{key}' at '{path}'"),
            ));
        }
    }
    Ok(())
}

fn validate_top(fields: &[(String, JsonValue)]) -> Result<(), DecodeError> {
    const TOP: &[&str] = &[
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
    ];
    for (key, _) in fields {
        if !TOP.contains(&key.as_str()) {
            return Err(validation(
                Some(key),
                format!("unknown top-level member '{key}'"),
            ));
        }
    }
    for key in ["model", "messages"] {
        if field(fields, key).is_none() {
            return Err(validation(
                Some(key),
                format!("missing required top-level member '{key}'"),
            ));
        }
    }
    string(field(fields, "model").unwrap(), "model", "model")?;
    validate_messages(field(fields, "messages").unwrap())?;
    let stream = match field(fields, "stream") {
        Some(value) => {
            boolean(value, "stream", "stream")?;
            matches!(value, JsonValue::Bool(true))
        }
        None => false,
    };
    if let Some(value) = field(fields, "stream_options") {
        if !stream {
            return Err(validation(
                Some("stream_options"),
                "'stream_options' is allowed only when 'stream' is true",
            ));
        }
        validate_stream_options(value)?;
    }
    if let Some(value) = field(fields, "tools") {
        validate_tools(value)?;
    }
    if let Some(value) = field(fields, "tool_choice") {
        validate_tool_choice(value)?;
    }
    for key in [
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
    ] {
        if let Some(value) = field(fields, key) {
            number(value, key, key)?;
        }
    }
    if field(fields, "max_tokens").is_some() && field(fields, "max_completion_tokens").is_some() {
        return Err(validation(
            Some("max_tokens"),
            "'max_tokens' and 'max_completion_tokens' cannot both be supplied",
        ));
    }
    if let Some(value) = field(fields, "stop") {
        validate_stop(value)?;
    }
    Ok(())
}

fn validate_stop(value: &JsonValue) -> Result<(), DecodeError> {
    match value {
        JsonValue::String(value) => validate_stop_string(value, "stop"),
        JsonValue::Array(values) => {
            if values.len() > 4 {
                return Err(validation(
                    Some("stop"),
                    "'stop' may contain at most four strings",
                ));
            }
            for (index, value) in values.iter().enumerate() {
                string(value, "stop", &path_index("stop", index))?;
                let JsonValue::String(value) = value else {
                    unreachable!()
                };
                validate_stop_string(value, &path_index("stop", index))?;
            }
            Ok(())
        }
        _ => Err(validation(
            Some("stop"),
            "'stop' must be a string or an array of strings",
        )),
    }
}

fn validate_stop_string(value: &str, path: &str) -> Result<(), DecodeError> {
    if value.is_empty() || value.len() > 256 {
        return Err(validation(
            Some("stop"),
            format!("'{path}' must be a nonempty string no larger than 256 UTF-8 bytes"),
        ));
    }
    Ok(())
}

fn validate_messages(value: &JsonValue) -> Result<(), DecodeError> {
    let messages = array(value, "messages", "messages")?;
    if messages.is_empty() {
        return Err(validation(
            Some("messages"),
            "'messages' must be a nonempty array",
        ));
    }
    let mut state = TurnState::Instruction;
    for (index, value) in messages.iter().enumerate() {
        let path = path_index("messages", index);
        let fields = object(value, "messages", &path)?;
        let role = field(fields, "role").ok_or_else(|| {
            validation(
                Some("messages"),
                format!("missing required member 'role' at '{path}'"),
            )
        })?;
        let JsonValue::String(role) = role else {
            return Err(validation(
                Some("messages"),
                format!("'{path}.role' must be a string"),
            ));
        };
        match role.as_str() {
            "system" | "developer" => {
                closed(
                    fields,
                    &["role", "content"],
                    &["role", "content"],
                    "messages",
                    &path,
                )?;
                string(
                    field(fields, "content").unwrap(),
                    "messages",
                    &path_member(&path, "content"),
                )?;
                if state != TurnState::Instruction {
                    return Err(validation(
                        Some("messages"),
                        format!("'{path}.role' instructions must precede conversation turns"),
                    ));
                }
            }
            "user" => {
                closed(
                    fields,
                    &["role", "content"],
                    &["role", "content"],
                    "messages",
                    &path,
                )?;
                string(
                    field(fields, "content").unwrap(),
                    "messages",
                    &path_member(&path, "content"),
                )?;
                if !matches!(state, TurnState::Instruction | TurnState::AfterAssistant) {
                    return Err(validation(
                        Some("messages"),
                        format!("unexpected user message at '{path}'"),
                    ));
                }
                state = TurnState::AfterUser;
            }
            "assistant" => {
                closed(
                    fields,
                    &["role", "content", "tool_calls"],
                    &["role"],
                    "messages",
                    &path,
                )?;
                if let Some(content) = field(fields, "content") {
                    if !matches!(content, JsonValue::String(_) | JsonValue::Null) {
                        return Err(validation(
                            Some("messages"),
                            format!("'{path}.content' must be a string or null"),
                        ));
                    }
                }
                if let Some(calls) = field(fields, "tool_calls") {
                    validate_tool_calls(calls, &path)?;
                }
                let has_content = matches!(field(fields, "content"), Some(JsonValue::String(_)));
                let has_calls = field(fields, "tool_calls").is_some();
                if !has_content && !has_calls {
                    return Err(validation(
                        Some("messages"),
                        format!("'{path}' must contain string content and/or nonempty tool_calls"),
                    ));
                }
                if state != TurnState::AfterUser {
                    return Err(validation(
                        Some("messages"),
                        format!("unexpected assistant message at '{path}'"),
                    ));
                }
                state = if has_calls {
                    TurnState::ToolResults
                } else {
                    TurnState::AfterAssistant
                };
            }
            "tool" => {
                closed(
                    fields,
                    &["role", "tool_call_id", "content"],
                    &["role", "tool_call_id", "content"],
                    "messages",
                    &path,
                )?;
                string(
                    field(fields, "tool_call_id").unwrap(),
                    "messages",
                    &path_member(&path, "tool_call_id"),
                )?;
                if !matches!(state, TurnState::ToolResults | TurnState::AfterToolResult) {
                    return Err(validation(
                        Some("messages"),
                        format!("unexpected tool result at '{path}'"),
                    ));
                }
                state = TurnState::AfterToolResult;
                string(
                    field(fields, "content").unwrap(),
                    "messages",
                    &path_member(&path, "content"),
                )?;
            }
            _ => {
                return Err(validation(
                    Some("messages"),
                    format!("unsupported role at '{path}.role'"),
                ))
            }
        }
    }
    if !matches!(state, TurnState::AfterUser | TurnState::AfterToolResult) {
        return Err(validation(
            Some("messages"),
            "'messages' must end on a user-side turn",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TurnState {
    Instruction,
    AfterUser,
    AfterAssistant,
    ToolResults,
    AfterToolResult,
}

fn validate_tool_calls(value: &JsonValue, parent: &str) -> Result<(), DecodeError> {
    let calls = array(value, "messages", &path_member(parent, "tool_calls"))?;
    if calls.is_empty() {
        return Err(validation(
            Some("messages"),
            format!("'{parent}.tool_calls' must be nonempty"),
        ));
    }
    for (index, value) in calls.iter().enumerate() {
        let path = path_index(&path_member(parent, "tool_calls"), index);
        let fields = object(value, "messages", &path)?;
        closed(
            fields,
            &["id", "type", "function"],
            &["id", "type", "function"],
            "messages",
            &path,
        )?;
        string(
            field(fields, "id").unwrap(),
            "messages",
            &path_member(&path, "id"),
        )?;
        string(
            field(fields, "type").unwrap(),
            "messages",
            &path_member(&path, "type"),
        )?;
        validate_call_function(
            field(fields, "function").unwrap(),
            "messages",
            &path_member(&path, "function"),
        )?;
    }
    Ok(())
}

fn validate_call_function(value: &JsonValue, param: &str, path: &str) -> Result<(), DecodeError> {
    let fields = object(value, param, path)?;
    closed(
        fields,
        &["name", "arguments"],
        &["name", "arguments"],
        param,
        path,
    )?;
    string(
        field(fields, "name").unwrap(),
        param,
        &path_member(path, "name"),
    )?;
    string(
        field(fields, "arguments").unwrap(),
        param,
        &path_member(path, "arguments"),
    )
}

fn validate_tools(value: &JsonValue) -> Result<(), DecodeError> {
    let tools = array(value, "tools", "tools")?;
    for (index, value) in tools.iter().enumerate() {
        let path = path_index("tools", index);
        let fields = object(value, "tools", &path)?;
        closed(
            fields,
            &["type", "function"],
            &["type", "function"],
            "tools",
            &path,
        )?;
        string(
            field(fields, "type").unwrap(),
            "tools",
            &path_member(&path, "type"),
        )?;
        let function_path = path_member(&path, "function");
        let function = object(field(fields, "function").unwrap(), "tools", &function_path)?;
        closed(
            function,
            &["name", "description", "parameters"],
            &["name"],
            "tools",
            &function_path,
        )?;
        string(
            field(function, "name").unwrap(),
            "tools",
            &path_member(&function_path, "name"),
        )?;
        if let Some(description) = field(function, "description") {
            string(
                description,
                "tools",
                &path_member(&function_path, "description"),
            )?;
        }
        if let Some(parameters) = field(function, "parameters") {
            object(
                parameters,
                "tools",
                &path_member(&function_path, "parameters"),
            )?;
        }
    }
    Ok(())
}

fn validate_tool_choice(value: &JsonValue) -> Result<(), DecodeError> {
    if matches!(value, JsonValue::String(_)) {
        return Ok(());
    }
    let fields = object(value, "tool_choice", "tool_choice")?;
    closed(
        fields,
        &["type", "function"],
        &["type", "function"],
        "tool_choice",
        "tool_choice",
    )?;
    string(
        field(fields, "type").unwrap(),
        "tool_choice",
        "tool_choice.type",
    )?;
    let function = object(
        field(fields, "function").unwrap(),
        "tool_choice",
        "tool_choice.function",
    )?;
    closed(
        function,
        &["name"],
        &["name"],
        "tool_choice",
        "tool_choice.function",
    )?;
    string(
        field(function, "name").unwrap(),
        "tool_choice",
        "tool_choice.function.name",
    )
}

fn validate_stream_options(value: &JsonValue) -> Result<(), DecodeError> {
    let fields = object(value, "stream_options", "stream_options")?;
    closed(
        fields,
        &["include_usage"],
        &["include_usage"],
        "stream_options",
        "stream_options",
    )?;
    boolean(
        field(fields, "include_usage").unwrap(),
        "stream_options",
        "stream_options.include_usage",
    )
}

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }
    fn eof(&self) -> bool {
        self.pos == self.input.len()
    }
    fn ws(&mut self) {
        while matches!(self.byte(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }
    fn byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }
    fn invalid<T>(&self, message: impl Into<String>) -> Result<T, DecodeError> {
        Err(DecodeError::InvalidJson {
            message: message.into(),
        })
    }
    fn take(&mut self, expected: u8) -> Result<(), DecodeError> {
        if self.byte() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            self.invalid("malformed JSON")
        }
    }
    fn value(&mut self, path: &str, owner: Option<&str>) -> Result<JsonValue, DecodeError> {
        self.ws();
        match self.byte() {
            Some(b'{') => self.object(path, owner),
            Some(b'[') => self.array(path, owner),
            Some(b'"') => self.string().map(JsonValue::String),
            Some(b't') => {
                self.literal("true")?;
                Ok(JsonValue::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(JsonValue::Bool(false))
            }
            Some(b'n') => {
                self.literal("null")?;
                Ok(JsonValue::Null)
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => self.invalid("malformed JSON"),
        }
    }
    fn literal(&mut self, literal: &str) -> Result<(), DecodeError> {
        if self.input[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            Ok(())
        } else {
            self.invalid("malformed JSON")
        }
    }
    fn object(&mut self, path: &str, owner: Option<&str>) -> Result<JsonValue, DecodeError> {
        self.take(b'{')?;
        self.ws();
        let mut fields = Vec::new();
        if self.byte() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(fields));
        }
        loop {
            self.ws();
            if self.byte() != Some(b'"') {
                return self.invalid("malformed JSON object");
            }
            let key = self.string()?;
            self.ws();
            self.take(b':')?;
            let member_path = path_member(path, &key);
            let member_owner = if path.is_empty() {
                Some(key.as_str())
            } else {
                owner
            };
            let value = self.value(&member_path, member_owner)?;
            if fields.iter().any(|(seen, _)| seen == &key) {
                let param = if path.is_empty() {
                    key.clone()
                } else {
                    owner.unwrap_or(&key).to_owned()
                };
                return Err(validation(
                    Some(&param),
                    format!("duplicate member at '{member_path}'"),
                ));
            }
            fields.push((key, value));
            self.ws();
            match self.byte() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return self.invalid("malformed JSON object"),
            }
        }
        Ok(JsonValue::Object(fields))
    }
    fn array(&mut self, path: &str, owner: Option<&str>) -> Result<JsonValue, DecodeError> {
        self.take(b'[')?;
        self.ws();
        let mut values = Vec::new();
        if self.byte() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(values));
        }
        loop {
            let index_path = path_index(path, values.len());
            values.push(self.value(&index_path, owner)?);
            self.ws();
            match self.byte() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return self.invalid("malformed JSON array"),
            }
        }
        Ok(JsonValue::Array(values))
    }
    fn number(&mut self) -> Result<JsonValue, DecodeError> {
        let start = self.pos;
        if self.byte() == Some(b'-') {
            self.pos += 1;
        }
        match self.byte() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.byte(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return self.invalid("malformed JSON number"),
        }
        if self.byte() == Some(b'.') {
            self.pos += 1;
            let decimal = self.pos;
            while matches!(self.byte(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if decimal == self.pos {
                return self.invalid("malformed JSON number");
            }
        }
        if matches!(self.byte(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.byte(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let exponent = self.pos;
            while matches!(self.byte(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
            if exponent == self.pos {
                return self.invalid("malformed JSON number");
            }
        }
        Ok(JsonValue::Number(self.input[start..self.pos].to_owned()))
    }
    fn string(&mut self) -> Result<String, DecodeError> {
        self.take(b'"')?;
        let mut out = String::new();
        loop {
            let Some(byte) = self.byte() else {
                return self.invalid("unterminated JSON string");
            };
            self.pos += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let Some(escape) = self.byte() else {
                        return self.invalid("unterminated JSON escape");
                    };
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            if (0xD800..=0xDBFF).contains(&first) {
                                if self.byte() != Some(b'\\')
                                    || self.input.as_bytes().get(self.pos + 1) != Some(&b'u')
                                {
                                    return self.invalid("invalid JSON unicode escape");
                                }
                                self.pos += 2;
                                let second = self.hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&second) {
                                    return self.invalid("invalid JSON unicode escape");
                                }
                                let scalar = 0x10000
                                    + (((first - 0xD800) as u32) << 10)
                                    + (second - 0xDC00) as u32;
                                out.push(char::from_u32(scalar).unwrap());
                            } else if (0xDC00..=0xDFFF).contains(&first) {
                                return self.invalid("invalid JSON unicode escape");
                            } else {
                                out.push(char::from_u32(first as u32).unwrap());
                            }
                        }
                        _ => return self.invalid("invalid JSON escape"),
                    }
                }
                0..=0x1F => return self.invalid("control character in JSON string"),
                _ => {
                    let start = self.pos - 1;
                    let ch = self.input[start..].chars().next().unwrap();
                    out.push(ch);
                    self.pos = start + ch.len_utf8();
                }
            }
        }
    }
    fn hex4(&mut self) -> Result<u16, DecodeError> {
        let end = self
            .pos
            .checked_add(4)
            .filter(|end| *end <= self.input.len())
            .ok_or_else(|| DecodeError::InvalidJson {
                message: "invalid JSON unicode escape".into(),
            })?;
        let digits = &self.input[self.pos..end];
        if !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return self.invalid("invalid JSON unicode escape");
        }
        self.pos = end;
        u16::from_str_radix(digits, 16).map_err(|_| DecodeError::InvalidJson {
            message: "invalid JSON unicode escape".into(),
        })
    }
}
