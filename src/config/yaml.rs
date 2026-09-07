use crate::config::error::{ConfigError, ConfigErrorKind, ConfigLocation};
use crate::config::raw::{RawConfig, RawGeneralSettings, RawLiteLlmParams, RawModelEntry};
use saphyr_parser::{Event, Marker, Parser, ScalarStyle, Span};
use std::collections::HashSet;

/// Intermediate AST node retaining source location and scalar styles.
#[derive(Debug, Clone, PartialEq)]
pub enum SpannedNode {
    Scalar {
        value: String,
        style: ScalarStyle,
        location: ConfigLocation,
    },
    Sequence {
        items: Vec<SpannedNode>,
        location: ConfigLocation,
    },
    Mapping {
        entries: Vec<(String, SpannedNode, ConfigLocation)>,
        location: ConfigLocation,
    },
}

impl SpannedNode {
    pub fn type_name(&self) -> &'static str {
        match self {
            SpannedNode::Scalar { value, style, .. } => {
                if *style == ScalarStyle::Plain {
                    if let Some(t) = is_plain_non_string(value) {
                        return t;
                    }
                }
                "string"
            }
            SpannedNode::Sequence { .. } => "sequence",
            SpannedNode::Mapping { .. } => "mapping",
        }
    }

    pub fn location(&self) -> ConfigLocation {
        match self {
            SpannedNode::Scalar { location, .. }
            | SpannedNode::Sequence { location, .. }
            | SpannedNode::Mapping { location, .. } => *location,
        }
    }
}

fn marker_to_location(marker: &Marker) -> ConfigLocation {
    ConfigLocation {
        line: marker.line(),
        column: marker.col() + 1,
    }
}

/// Identifies whether an unquoted scalar represents a non-string YAML type.
fn is_plain_non_string(s: &str) -> Option<&'static str> {
    if s.is_empty() || s == "~" || s.eq_ignore_ascii_case("null") {
        return Some("null");
    }
    match s {
        "true" | "True" | "TRUE" | "false" | "False" | "FALSE" | "yes" | "Yes" | "YES" | "no"
        | "No" | "NO" | "on" | "On" | "ON" | "off" | "Off" | "OFF" => {
            return Some("boolean");
        }
        _ => {}
    }
    // Hex, octal, binary integers
    if (s.starts_with("0x") || s.starts_with("0X"))
        && s.len() > 2
        && s[2..].chars().all(|c| c.is_ascii_hexdigit())
    {
        return Some("integer");
    }
    if (s.starts_with("0o") || s.starts_with("0O"))
        && s.len() > 2
        && s[2..].chars().all(|c| ('0'..='7').contains(&c))
    {
        return Some("integer");
    }
    if (s.starts_with("0b") || s.starts_with("0B"))
        && s.len() > 2
        && s[2..].chars().all(|c| c == '0' || c == '1')
    {
        return Some("integer");
    }
    if is_yaml_decimal_integer(s) {
        return Some("integer");
    }
    // Float constants (.nan, .inf)
    if s.eq_ignore_ascii_case(".nan") {
        return Some("float");
    }
    let float_inf = s
        .strip_prefix('+')
        .or_else(|| s.strip_prefix('-'))
        .unwrap_or(s);
    if float_inf.eq_ignore_ascii_case(".inf") {
        return Some("float");
    }
    // Numbers with decimal point or scientific notation
    if (s.contains('.') || s.contains('e') || s.contains('E'))
        && s.replace('_', "").parse::<f64>().is_ok()
    {
        return Some("float");
    }
    None
}

/// Returns whether a scalar uses the accepted decimal YAML-integer spelling.
///
/// YAML permits `_` as a digit separator. Keeping this check separate from
/// Rust's number parser avoids accidentally treating an integral-looking float
/// (for example `30.0` or `3e1`) as an integer during deserialization.
fn is_yaml_decimal_integer(s: &str) -> bool {
    let digits = s
        .strip_prefix('+')
        .or_else(|| s.strip_prefix('-'))
        .unwrap_or(s);
    !digits.is_empty()
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && digits
            .split('_')
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
}

struct YamlStream<'a> {
    parser: Parser<'a, saphyr_parser::StrInput<'a>>,
    peeked: Option<(Event<'a>, Span)>,
}

impl<'a> YamlStream<'a> {
    fn new(yaml: &'a str) -> Self {
        Self {
            parser: Parser::new_from_str(yaml),
            peeked: None,
        }
    }

    fn next(&mut self) -> Result<Option<(Event<'a>, Span)>, ConfigError> {
        if let Some(event) = self.peeked.take() {
            return Ok(Some(event));
        }
        match self.parser.next_event() {
            Some(Ok(event)) => Ok(Some(event)),
            Some(Err(scan_err)) => {
                let marker = scan_err.marker();
                let info = scan_err.info();
                let kind = if info.contains("unknown anchor") {
                    ConfigErrorKind::ProhibitedAlias
                } else {
                    ConfigErrorKind::SyntaxError {
                        message: info.to_string(),
                    }
                };
                Err(ConfigError::new(kind, Some(marker_to_location(marker))))
            }
            None => Ok(None),
        }
    }

    fn peek(&mut self) -> Result<Option<&(Event<'a>, Span)>, ConfigError> {
        if self.peeked.is_none() {
            self.peeked = self.next()?;
        }
        Ok(self.peeked.as_ref())
    }
}

/// Parses a YAML string into a `SpannedNode` while enforcing the restricted v0.1 subset.
pub fn parse_spanned_node(yaml: &str) -> Result<SpannedNode, ConfigError> {
    let mut stream = YamlStream::new(yaml);

    // Stream must begin with StreamStart
    match stream.next()? {
        Some((Event::StreamStart, _)) => {}
        Some((_, span)) => {
            return Err(ConfigError::new(
                ConfigErrorKind::SyntaxError {
                    message: "expected stream start".to_string(),
                },
                Some(marker_to_location(&span.start)),
            ));
        }
        None => {
            return Err(ConfigError::new(ConfigErrorKind::EmptyDocument, None));
        }
    }

    let root_node = match stream.next()? {
        Some((Event::DocumentStart(_), _)) => {
            let node = parse_node(&mut stream, "")?;
            match stream.next()? {
                Some((Event::DocumentEnd, _)) => {}
                Some((_, span)) => {
                    return Err(ConfigError::new(
                        ConfigErrorKind::SyntaxError {
                            message: "expected document end".to_string(),
                        },
                        Some(marker_to_location(&span.start)),
                    ));
                }
                None => {}
            }
            node
        }
        Some((Event::StreamEnd, _)) => {
            return Err(ConfigError::new(ConfigErrorKind::EmptyDocument, None));
        }
        Some((_, span)) => {
            return Err(ConfigError::new(
                ConfigErrorKind::SyntaxError {
                    message: "expected document start".to_string(),
                },
                Some(marker_to_location(&span.start)),
            ));
        }
        None => {
            return Err(ConfigError::new(ConfigErrorKind::EmptyDocument, None));
        }
    };

    // Check for any subsequent documents
    if let Some((event, span)) = stream.next()? {
        if event != Event::StreamEnd {
            return Err(ConfigError::new(
                ConfigErrorKind::MultipleDocuments,
                Some(marker_to_location(&span.start)),
            ));
        }
    }

    Ok(root_node)
}

fn parse_node<'a>(
    stream: &mut YamlStream<'a>,
    current_path: &str,
) -> Result<SpannedNode, ConfigError> {
    let (event, span) = match stream.next()? {
        Some(ev) => ev,
        None => {
            return Err(ConfigError::new(
                ConfigErrorKind::SyntaxError {
                    message: "unexpected end of stream".to_string(),
                },
                None,
            ));
        }
    };

    let location = marker_to_location(&span.start);

    match event {
        Event::Alias(_) => Err(ConfigError::new(
            ConfigErrorKind::ProhibitedAlias,
            Some(location),
        )),
        Event::Scalar(value, style, anchor_id, tag) => {
            if anchor_id != 0 {
                return Err(ConfigError::new(
                    ConfigErrorKind::ProhibitedAnchor,
                    Some(location),
                ));
            }
            if let Some(tag) = tag {
                return Err(ConfigError::new(
                    ConfigErrorKind::ProhibitedTag {
                        tag: format!("{}{}", tag.handle, tag.suffix),
                    },
                    Some(location),
                ));
            }
            Ok(SpannedNode::Scalar {
                value: value.into_owned(),
                style,
                location,
            })
        }
        Event::SequenceStart(anchor_id, tag) => {
            if anchor_id != 0 {
                return Err(ConfigError::new(
                    ConfigErrorKind::ProhibitedAnchor,
                    Some(location),
                ));
            }
            if let Some(tag) = tag {
                return Err(ConfigError::new(
                    ConfigErrorKind::ProhibitedTag {
                        tag: format!("{}{}", tag.handle, tag.suffix),
                    },
                    Some(location),
                ));
            }
            let mut items = Vec::new();
            loop {
                match stream.peek()? {
                    Some((Event::SequenceEnd, _)) => {
                        stream.next()?; // consume SequenceEnd
                        break;
                    }
                    Some(_) => {
                        let item_path = if current_path.is_empty() {
                            format!("[{}]", items.len())
                        } else {
                            format!("{current_path}[{}]", items.len())
                        };
                        let item = parse_node(stream, &item_path)?;
                        items.push(item);
                    }
                    None => {
                        return Err(ConfigError::new(
                            ConfigErrorKind::SyntaxError {
                                message: "unclosed sequence".to_string(),
                            },
                            Some(location),
                        ));
                    }
                }
            }
            Ok(SpannedNode::Sequence { items, location })
        }
        Event::MappingStart(anchor_id, tag) => {
            if anchor_id != 0 {
                return Err(ConfigError::new(
                    ConfigErrorKind::ProhibitedAnchor,
                    Some(location),
                ));
            }
            if let Some(tag) = tag {
                return Err(ConfigError::new(
                    ConfigErrorKind::ProhibitedTag {
                        tag: format!("{}{}", tag.handle, tag.suffix),
                    },
                    Some(location),
                ));
            }
            let mut entries = Vec::new();
            let mut seen_keys = HashSet::new();

            loop {
                match stream.peek()? {
                    Some((Event::MappingEnd, _)) => {
                        stream.next()?; // consume MappingEnd
                        break;
                    }
                    None => {
                        return Err(ConfigError::new(
                            ConfigErrorKind::SyntaxError {
                                message: "unclosed mapping".to_string(),
                            },
                            Some(location),
                        ));
                    }
                    Some(_) => {
                        let (key_event, key_span) = stream.next()?.unwrap();
                        let key_loc = marker_to_location(&key_span.start);

                        let key_str = match key_event {
                            Event::Alias(_) => {
                                return Err(ConfigError::new(
                                    ConfigErrorKind::ProhibitedAlias,
                                    Some(key_loc),
                                ));
                            }
                            Event::SequenceStart(_, _) => {
                                return Err(ConfigError::new(
                                    ConfigErrorKind::NonStringKey {
                                        found: "sequence".to_string(),
                                    },
                                    Some(key_loc),
                                ));
                            }
                            Event::MappingStart(_, _) => {
                                return Err(ConfigError::new(
                                    ConfigErrorKind::NonStringKey {
                                        found: "mapping".to_string(),
                                    },
                                    Some(key_loc),
                                ));
                            }
                            Event::Scalar(key_val, key_style, key_anchor, key_tag) => {
                                if key_anchor != 0 {
                                    return Err(ConfigError::new(
                                        ConfigErrorKind::ProhibitedAnchor,
                                        Some(key_loc),
                                    ));
                                }
                                if let Some(tag) = key_tag {
                                    return Err(ConfigError::new(
                                        ConfigErrorKind::ProhibitedTag {
                                            tag: format!("{}{}", tag.handle, tag.suffix),
                                        },
                                        Some(key_loc),
                                    ));
                                }
                                if key_val == "<<" {
                                    return Err(ConfigError::new(
                                        ConfigErrorKind::ProhibitedMergeKey,
                                        Some(key_loc),
                                    ));
                                }
                                if key_style == ScalarStyle::Plain {
                                    if let Some(found_type) = is_plain_non_string(&key_val) {
                                        return Err(ConfigError::new(
                                            ConfigErrorKind::NonStringKey {
                                                found: found_type.to_string(),
                                            },
                                            Some(key_loc),
                                        ));
                                    }
                                }
                                key_val.into_owned()
                            }
                            _ => {
                                return Err(ConfigError::new(
                                    ConfigErrorKind::SyntaxError {
                                        message: "unexpected event while expecting mapping key"
                                            .to_string(),
                                    },
                                    Some(key_loc),
                                ));
                            }
                        };

                        let entry_path = if current_path.is_empty() {
                            key_str.clone()
                        } else {
                            format!("{current_path}.{key_str}")
                        };

                        if !seen_keys.insert(key_str.clone()) {
                            return Err(ConfigError::new(
                                ConfigErrorKind::DuplicateKey { path: entry_path },
                                Some(key_loc),
                            ));
                        }

                        let val_node = parse_node(stream, &entry_path)?;
                        entries.push((key_str, val_node, key_loc));
                    }
                }
            }
            Ok(SpannedNode::Mapping { entries, location })
        }
        _ => Err(ConfigError::new(
            ConfigErrorKind::SyntaxError {
                message: "unexpected YAML event".to_string(),
            },
            Some(location),
        )),
    }
}

/// Parses a YAML configuration string into a `RawConfig` structure.
pub fn parse_yaml_str(yaml: &str) -> Result<RawConfig, ConfigError> {
    let root = parse_spanned_node(yaml)?;
    deserialize_raw_config(&root)
}

fn deserialize_raw_config(node: &SpannedNode) -> Result<RawConfig, ConfigError> {
    let entries = match node {
        SpannedNode::Mapping { entries, .. } => entries,
        _ => {
            return Err(ConfigError::new(
                ConfigErrorKind::InvalidType {
                    path: String::new(),
                    expected: "mapping",
                    found: node.type_name().to_string(),
                },
                Some(node.location()),
            ));
        }
    };

    let mut model_list = None;
    let mut general_settings = None;

    for (key, val, loc) in entries {
        match key.as_str() {
            "model_list" => {
                model_list = Some(parse_model_list(val, "model_list")?);
            }
            "general_settings" => {
                general_settings = Some(parse_general_settings(val, "general_settings")?);
            }
            _ => {
                return Err(ConfigError::new(
                    ConfigErrorKind::UnknownKey {
                        path: String::new(),
                        key: key.clone(),
                    },
                    Some(*loc),
                ));
            }
        }
    }

    Ok(RawConfig {
        model_list: model_list.unwrap_or_default(),
        general_settings,
    })
}

fn parse_model_list(node: &SpannedNode, path: &str) -> Result<Vec<RawModelEntry>, ConfigError> {
    let items = match node {
        SpannedNode::Sequence { items, .. } => items,
        _ => {
            return Err(ConfigError::new(
                ConfigErrorKind::InvalidType {
                    path: path.to_string(),
                    expected: "sequence",
                    found: node.type_name().to_string(),
                },
                Some(node.location()),
            ));
        }
    };

    let mut model_entries = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let item_path = format!("{path}[{i}]");
        model_entries.push(parse_model_entry(item, &item_path)?);
    }
    Ok(model_entries)
}

fn parse_model_entry(node: &SpannedNode, path: &str) -> Result<RawModelEntry, ConfigError> {
    let entries = match node {
        SpannedNode::Mapping { entries, .. } => entries,
        _ => {
            return Err(ConfigError::new(
                ConfigErrorKind::InvalidType {
                    path: path.to_string(),
                    expected: "mapping",
                    found: node.type_name().to_string(),
                },
                Some(node.location()),
            ));
        }
    };

    let mut model_name = None;
    let mut litellm_params = None;

    for (key, val, loc) in entries {
        match key.as_str() {
            "model_name" => {
                let p = format!("{path}.model_name");
                model_name = Some(parse_string(val, &p)?);
            }
            "litellm_params" => {
                let p = format!("{path}.litellm_params");
                litellm_params = Some(parse_litellm_params(val, &p)?);
            }
            _ => {
                return Err(ConfigError::new(
                    ConfigErrorKind::UnknownKey {
                        path: path.to_string(),
                        key: key.clone(),
                    },
                    Some(*loc),
                ));
            }
        }
    }

    let model_name = model_name.ok_or_else(|| {
        ConfigError::new(
            ConfigErrorKind::MissingField {
                path: path.to_string(),
                field: "model_name",
            },
            Some(node.location()),
        )
    })?;

    let litellm_params = litellm_params.ok_or_else(|| {
        ConfigError::new(
            ConfigErrorKind::MissingField {
                path: path.to_string(),
                field: "litellm_params",
            },
            Some(node.location()),
        )
    })?;

    Ok(RawModelEntry {
        model_name,
        litellm_params,
    })
}

fn parse_litellm_params(node: &SpannedNode, path: &str) -> Result<RawLiteLlmParams, ConfigError> {
    let entries = match node {
        SpannedNode::Mapping { entries, .. } => entries,
        _ => {
            return Err(ConfigError::new(
                ConfigErrorKind::InvalidType {
                    path: path.to_string(),
                    expected: "mapping",
                    found: node.type_name().to_string(),
                },
                Some(node.location()),
            ));
        }
    };

    let mut model = None;
    let mut api_key = None;
    let mut api_base = None;
    let mut timeout = None;

    for (key, val, loc) in entries {
        match key.as_str() {
            "model" => {
                let p = format!("{path}.model");
                model = Some(parse_string(val, &p)?);
            }
            "api_key" => {
                let p = format!("{path}.api_key");
                api_key = parse_optional_string(val, &p)?;
            }
            "api_base" => {
                let p = format!("{path}.api_base");
                api_base = parse_optional_string(val, &p)?;
            }
            "timeout" => {
                let p = format!("{path}.timeout");
                timeout = parse_optional_number(val, &p)?;
            }
            _ => {
                return Err(ConfigError::new(
                    ConfigErrorKind::UnknownKey {
                        path: path.to_string(),
                        key: key.clone(),
                    },
                    Some(*loc),
                ));
            }
        }
    }

    let model = model.ok_or_else(|| {
        ConfigError::new(
            ConfigErrorKind::MissingField {
                path: path.to_string(),
                field: "model",
            },
            Some(node.location()),
        )
    })?;

    Ok(RawLiteLlmParams {
        model,
        api_key,
        api_base,
        timeout,
    })
}

fn parse_general_settings(
    node: &SpannedNode,
    path: &str,
) -> Result<RawGeneralSettings, ConfigError> {
    let entries = match node {
        SpannedNode::Mapping { entries, .. } => entries,
        _ => {
            return Err(ConfigError::new(
                ConfigErrorKind::InvalidType {
                    path: path.to_string(),
                    expected: "mapping",
                    found: node.type_name().to_string(),
                },
                Some(node.location()),
            ));
        }
    };

    let mut master_key = None;
    let mut request_timeout = None;
    let mut overall_timeout = None;
    let mut max_in_flight = None;

    for (key, val, loc) in entries {
        match key.as_str() {
            "master_key" => {
                let p = format!("{path}.master_key");
                master_key = parse_optional_string(val, &p)?;
            }
            "request_timeout" => {
                let p = format!("{path}.request_timeout");
                request_timeout = parse_optional_number(val, &p)?;
            }
            "overall_timeout" => {
                let p = format!("{path}.overall_timeout");
                overall_timeout = parse_optional_number(val, &p)?;
            }
            "max_in_flight" => {
                let p = format!("{path}.max_in_flight");
                max_in_flight = parse_optional_integer(val, &p)?;
            }
            _ => {
                return Err(ConfigError::new(
                    ConfigErrorKind::UnknownKey {
                        path: path.to_string(),
                        key: key.clone(),
                    },
                    Some(*loc),
                ));
            }
        }
    }

    Ok(RawGeneralSettings {
        master_key,
        request_timeout,
        overall_timeout,
        max_in_flight,
    })
}

fn parse_string(node: &SpannedNode, path: &str) -> Result<String, ConfigError> {
    match node {
        SpannedNode::Scalar { value, style, .. } => {
            if *style == ScalarStyle::Plain {
                if let Some(t) = is_plain_non_string(value) {
                    return Err(ConfigError::new(
                        ConfigErrorKind::InvalidType {
                            path: path.to_string(),
                            expected: "string",
                            found: t.to_string(),
                        },
                        Some(node.location()),
                    ));
                }
            }
            Ok(value.clone())
        }
        _ => Err(ConfigError::new(
            ConfigErrorKind::InvalidType {
                path: path.to_string(),
                expected: "string",
                found: node.type_name().to_string(),
            },
            Some(node.location()),
        )),
    }
}

fn parse_optional_string(node: &SpannedNode, path: &str) -> Result<Option<String>, ConfigError> {
    match node {
        SpannedNode::Scalar { value, style, .. } => {
            if *style == ScalarStyle::Plain
                && (value == "~" || value.eq_ignore_ascii_case("null") || value.is_empty())
            {
                Ok(None)
            } else if *style == ScalarStyle::Plain && is_plain_non_string(value).is_some() {
                let t = is_plain_non_string(value).unwrap();
                Err(ConfigError::new(
                    ConfigErrorKind::InvalidType {
                        path: path.to_string(),
                        expected: "string",
                        found: t.to_string(),
                    },
                    Some(node.location()),
                ))
            } else {
                Ok(Some(value.clone()))
            }
        }
        _ => Err(ConfigError::new(
            ConfigErrorKind::InvalidType {
                path: path.to_string(),
                expected: "string",
                found: node.type_name().to_string(),
            },
            Some(node.location()),
        )),
    }
}

fn parse_optional_number(node: &SpannedNode, path: &str) -> Result<Option<f64>, ConfigError> {
    match node {
        SpannedNode::Scalar { value, style, .. } => {
            if *style == ScalarStyle::Plain && is_yaml_decimal_integer(value) {
                // `_` is legal YAML integer syntax but not Rust numeric syntax.
                let num = value.replace('_', "").parse::<f64>().map_err(|_| {
                    ConfigError::new(
                        ConfigErrorKind::InvalidType {
                            path: path.to_string(),
                            expected: "integer",
                            found: value.clone(),
                        },
                        Some(node.location()),
                    )
                })?;
                if num.is_finite() {
                    Ok(Some(num))
                } else {
                    Err(ConfigError::new(
                        ConfigErrorKind::InvalidType {
                            path: path.to_string(),
                            expected: "integer",
                            found: value.clone(),
                        },
                        Some(node.location()),
                    ))
                }
            } else {
                Err(ConfigError::new(
                    ConfigErrorKind::InvalidType {
                        path: path.to_string(),
                        expected: "integer",
                        found: value.clone(),
                    },
                    Some(node.location()),
                ))
            }
        }
        _ => Err(ConfigError::new(
            ConfigErrorKind::InvalidType {
                path: path.to_string(),
                expected: "number",
                found: node.type_name().to_string(),
            },
            Some(node.location()),
        )),
    }
}

fn parse_optional_integer(node: &SpannedNode, path: &str) -> Result<Option<usize>, ConfigError> {
    match node {
        SpannedNode::Scalar { value, style, .. } => {
            if *style == ScalarStyle::Plain && is_yaml_decimal_integer(value) {
                let normalized = value.replace('_', "");
                if let Ok(num) = normalized.parse::<usize>() {
                    Ok(Some(num))
                } else {
                    Err(ConfigError::new(
                        ConfigErrorKind::InvalidType {
                            path: path.to_string(),
                            expected: "integer",
                            found: value.clone(),
                        },
                        Some(node.location()),
                    ))
                }
            } else {
                Err(ConfigError::new(
                    ConfigErrorKind::InvalidType {
                        path: path.to_string(),
                        expected: "integer",
                        found: value.clone(),
                    },
                    Some(node.location()),
                ))
            }
        }
        _ => Err(ConfigError::new(
            ConfigErrorKind::InvalidType {
                path: path.to_string(),
                expected: "integer",
                found: node.type_name().to_string(),
            },
            Some(node.location()),
        )),
    }
}
