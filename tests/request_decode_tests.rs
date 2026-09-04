use nano_llm::{decode_chat_request, DecodeError, JsonValue};

fn valid_request(extra: &str) -> Vec<u8> {
    format!(r#"{{"model":"test","messages":[{{"role":"user","content":"hello"}}]{extra}}}"#)
        .into_bytes()
}

fn validation(error: DecodeError) -> (Option<String>, String) {
    match error {
        DecodeError::Validation { param, message } => (param, message),
        other => panic!("expected validation error, got {other:?}"),
    }
}

#[test]
fn byte_level_invalid_json_cases_are_distinct_from_validation() {
    let cases: Vec<Vec<u8>> = vec![
        vec![b'{', 0xff, b'}'],
        br#"{"model":"test","messages":[}"#.to_vec(),
        br#"[1]"#.to_vec(),
        br#"{"model":"test","messages":[]} trailing"#.to_vec(),
    ];

    for bytes in cases {
        let error = decode_chat_request(&bytes).expect_err("must reject invalid JSON body");
        assert!(matches!(error, DecodeError::InvalidJson { .. }));
        assert_eq!(error.code(), "invalid_json");
        assert_eq!(error.param(), None);
    }
}

#[test]
fn duplicate_members_report_the_top_level_owner_and_precise_path() {
    let top = decode_chat_request(
        br#"{"model":"one","model":"two","messages":[{"role":"user","content":"hi"}]}"#,
    )
    .expect_err("top-level duplicate must fail");
    let (param, message) = validation(top);
    assert_eq!(param.as_deref(), Some("model"));
    assert!(message.contains("model"));

    let nested = decode_chat_request(
        br#"{"model":"test","messages":[{"role":"user","content":"one","content":"two"}]}"#,
    )
    .expect_err("nested duplicate must fail");
    let (param, message) = validation(nested);
    assert_eq!(param.as_deref(), Some("messages"));
    assert!(message.contains("messages[0].content"));
}

#[test]
fn parameters_is_opaque_but_duplicate_aware() {
    let accepted = decode_chat_request(&valid_request(
        r#", "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"value":{"type":"string","vendor_keyword":true}},"arbitrary":17}}}]"#,
    ))
    .expect("schema members must remain opaque");
    let tools = accepted
        .fields
        .iter()
        .find(|(key, _)| key == "tools")
        .unwrap();
    assert!(matches!(tools.1, JsonValue::Array(_)));

    let duplicate = decode_chat_request(&valid_request(
        r#", "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","type":"array"}}}]"#,
    ))
    .expect_err("schema duplicates must not be silently lost");
    let (param, message) = validation(duplicate);
    assert_eq!(param.as_deref(), Some("tools"));
    assert!(message.contains("tools[0].function.parameters.type"));
}

#[test]
fn unknown_members_are_rejected_at_each_gateway_defined_depth() {
    for (body, expected_param, expected_path) in [
        (
            valid_request(",\"unexpected\":true"),
            "unexpected",
            "unexpected",
        ),
        (
            br#"{"model":"test","messages":[{"role":"user","content":"hi","extra":true}]}"#
                .to_vec(),
            "messages",
            "messages[0]",
        ),
        (
            valid_request(r#", "stream_options":{"include_usage":true,"extra":true}"#),
            "stream_options",
            "stream_options",
        ),
        (
            valid_request(
                r#", "tool_choice":{"type":"function","function":{"name":"x","extra":true}}"#,
            ),
            "tool_choice",
            "tool_choice.function",
        ),
    ] {
        let error = decode_chat_request(&body).expect_err("unknown member must fail");
        let (param, message) = validation(error);
        assert_eq!(param.as_deref(), Some(expected_param));
        assert!(message.contains(expected_path), "{message}");
    }
}

#[test]
fn decoder_accepts_complete_nested_canonical_shapes() {
    let body = br#"{
        "model":"test",
        "messages":[
          {"role":"system","content":"rules"},
          {"role":"user","content":"question"},
          {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
          {"role":"tool","tool_call_id":"call_1","content":"answer"}
        ],
        "stream":true,
        "stream_options":{"include_usage":true},
        "tools":[{"type":"function","function":{"name":"lookup","description":"","parameters":{"type":"object"}}}],
        "tool_choice":{"type":"function","function":{"name":"lookup"}}
    }"#;
    decode_chat_request(body).expect("complete canonical shape should decode");
}

#[test]
fn top_level_scalar_shapes_are_closed_without_numeric_range_validation() {
    decode_chat_request(&valid_request(
        r#", "max_tokens":12,"max_completion_tokens":13,"temperature":0.5,"top_p":1,"stop":["END","STOP"]"#,
    ))
    .expect("numeric fields and stop shapes should decode");

    for (extra, param) in [
        (r#", "max_tokens":"12""#, "max_tokens"),
        (r#", "temperature":null"#, "temperature"),
        (r#", "stop":["END", 1]"#, "stop"),
    ] {
        let (actual, _) = validation(decode_chat_request(&valid_request(extra)).unwrap_err());
        assert_eq!(actual.as_deref(), Some(param));
    }
}
