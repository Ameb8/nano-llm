//! Strict decoding for the gateway-owned chat request wire format.
//!
//! This deliberately does not use a map-backed JSON representation: maps erase
//! duplicate members before validation can report them.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalRequest {
    /// The validated request. Object members retain their input order.
    pub fields: Vec<(String, JsonValue)>,
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
    Ok(CanonicalRequest { fields })
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
    if let Some(value) = field(fields, "stream") {
        boolean(value, "stream", "stream")?;
    }
    if let Some(value) = field(fields, "stream_options") {
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
    if let Some(value) = field(fields, "stop") {
        validate_stop(value)?;
    }
    Ok(())
}

fn validate_stop(value: &JsonValue) -> Result<(), DecodeError> {
    match value {
        JsonValue::String(_) => Ok(()),
        JsonValue::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                string(value, "stop", &path_index("stop", index))?;
            }
            Ok(())
        }
        _ => Err(validation(
            Some("stop"),
            "'stop' must be a string or an array of strings",
        )),
    }
}

fn validate_messages(value: &JsonValue) -> Result<(), DecodeError> {
    let messages = array(value, "messages", "messages")?;
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
            "system" | "developer" | "user" => {
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
    Ok(())
}

fn validate_tool_calls(value: &JsonValue, parent: &str) -> Result<(), DecodeError> {
    let calls = array(value, "messages", &path_member(parent, "tool_calls"))?;
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
