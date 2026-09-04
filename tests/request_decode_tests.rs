use nano_llm::{
    decode_chat_request, decode_chat_request_for_route, DecodeError, JsonValue, ProviderKind,
    RuntimeRoute, RuntimeTarget,
};

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
fn generation_controls_normalize_and_enforce_boundaries() {
    let request = decode_chat_request(&valid_request(
        r#", "max_completion_tokens":2147483647,"temperature":0.0,"top_p":1e0,"stop":"END""#,
    ))
    .expect("boundary values should decode");
    assert_eq!(request.max_tokens, Some(2_147_483_647));
    assert_eq!(request.stop, Some(vec!["END".to_string()]));
    assert!(!request.stream);
    assert!(!request.include_usage);
    decode_chat_request(&valid_request(&format!(
        r#", "stop":["{}","b","c","d"]"#,
        "a".repeat(256)
    )))
    .expect("four 256-byte stop strings are within the portable boundary");

    for (extra, param) in [
        (
            r#", "max_tokens":1,"max_completion_tokens":2"#,
            "max_tokens",
        ),
        (r#", "max_tokens":0"#, "max_tokens"),
        (r#", "max_tokens":1.0"#, "max_tokens"),
        (r#", "max_tokens":2147483648"#, "max_tokens"),
        (
            r#", "max_completion_tokens":false"#,
            "max_completion_tokens",
        ),
        (r#", "max_completion_tokens":null"#, "max_completion_tokens"),
        (r#", "temperature":-0.01"#, "temperature"),
        (r#", "temperature":1.01"#, "temperature"),
        (r#", "top_p":"1""#, "top_p"),
        (r#", "stop":"""#, "stop"),
        (&format!(r#", "stop":"{}""#, "a".repeat(257)), "stop"),
        (r#", "stop":["a","b","c","d","e"]"#, "stop"),
    ] {
        let (actual, _) = validation(decode_chat_request(&valid_request(extra)).unwrap_err());
        assert_eq!(actual.as_deref(), Some(param), "{extra}");
    }

    for (extra, param) in [
        (r#", "max_tokens":"12""#, "max_tokens"),
        (r#", "temperature":null"#, "temperature"),
        (r#", "stop":["END", 1]"#, "stop"),
    ] {
        let (actual, _) = validation(decode_chat_request(&valid_request(extra)).unwrap_err());
        assert_eq!(actual.as_deref(), Some(param));
    }
}

#[test]
fn instructions_and_conversation_turns_follow_the_portable_state_machine() {
    let request = decode_chat_request(
        br#"{
      "model":"test",
      "messages":[
        {"role":"system","content":" keep "},
        {"role":"developer","content":"rules\n"},
        {"role":"user","content":"question"}
      ]
    }"#,
    )
    .expect("leading instructions followed by a user turn are valid");
    assert_eq!(request.instruction.as_deref(), Some(" keep \n\nrules\n"));

    for body in [
        r#"{"model":"test","messages":[]}"#,
        r#"{"model":"test","messages":[{"role":"assistant","content":"no"}]}"#,
        r#"{"model":"test","messages":[{"role":"tool","tool_call_id":"x","content":"no"}]}"#,
        r#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]}"#,
        r#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}"#,
        r#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"assistant","content":"c"}]}"#,
        r#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"tool","tool_call_id":"x","content":"r"}]}"#,
        r#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"system","content":"late"}]}"#,
        r#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"assistant","content":null,"tool_calls":[{"id":"x","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"x","content":"r"},{"role":"user","content":"again"}]}"#,
        r#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"assistant","content":null,"tool_calls":[{"id":"x","type":"function","function":{"name":"f","arguments":"{}"}}]}]}"#,
        r#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"assistant","content":null}]}"#,
    ] {
        let (param, _) = validation(decode_chat_request(body.as_bytes()).unwrap_err());
        assert_eq!(param.as_deref(), Some("messages"), "{body}");
    }

    decode_chat_request(br#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"assistant","content":""},{"role":"user","content":"b"}]}"#)
        .expect("empty assistant text is valid and must be followed by a user turn");
    decode_chat_request(br#"{"model":"test","messages":[{"role":"user","content":"a"},{"role":"assistant","content":null,"tool_calls":[{"id":"x","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"x","content":"r"}]}"#)
        .expect("a complete tool-result group is a terminal user-side turn");
}

#[test]
fn stream_options_and_anthropic_output_limit_are_route_aware() {
    let stream = decode_chat_request(&valid_request(
        r#", "stream":true,"stream_options":{"include_usage":true}"#,
    ))
    .expect("stream options are valid for a stream");
    assert!(stream.stream && stream.include_usage);
    let (param, _) = validation(
        decode_chat_request(&valid_request(
            r#", "stream_options":{"include_usage":true}"#,
        ))
        .unwrap_err(),
    );
    assert_eq!(param.as_deref(), Some("stream_options"));

    let route = RuntimeRoute {
        model_name: "test".into(),
        targets: vec![RuntimeTarget {
            model: "anthropic/claude".into(),
            provider: ProviderKind::Anthropic,
            model_suffix: "claude".into(),
            api_key: None,
            api_base: "https://example.test".into(),
            timeout: 30,
            explicit_timeout: None,
        }],
    };
    let (param, _) =
        validation(decode_chat_request_for_route(&valid_request(""), &route).unwrap_err());
    assert_eq!(param.as_deref(), Some("max_tokens"));
    decode_chat_request_for_route(&valid_request(r#", "max_tokens":1"#), &route)
        .expect("an Anthropic route accepts one output-token alias");
}
