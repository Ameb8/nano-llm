use nano_llm::{
    build_response, decode_chat_request, normalize_response, normalize_usage, safety_response,
    AssistantDelta, FinishReason, NativeChoice, NativeResponse, NativeTerminal, NativeToolCall,
    ResponseError, ResponseMetadata, StreamAssembler, ToolCallDelta,
};

fn request(extra: &str) -> nano_llm::CanonicalRequest {
    decode_chat_request(
        format!(
            r#"{{"model":"public-model","messages":[{{"role":"user","content":"hi"}}]{extra}}}"#
        )
        .as_bytes(),
    )
    .unwrap()
}

#[test]
fn buffered_responses_are_gateway_owned_and_usage_is_all_or_nothing() {
    let req = request("");
    let metadata = ResponseMetadata::for_model(&req.model);
    let response = build_response(
        &req,
        metadata.clone(),
        Some("hello".into()),
        vec![],
        NativeTerminal::Stop,
        normalize_usage(Some(3), Some(5)),
    )
    .unwrap();
    assert!(response.id.starts_with("chatcmpl-"));
    assert_eq!(response.object, "chat.completion");
    assert_eq!(response.model, "public-model");
    assert_eq!(response.choices.len(), 1);
    assert_eq!(response.choices[0].index, 0);
    assert_eq!(response.usage.unwrap().total_tokens, 8);
    assert!(normalize_usage(Some(u64::MAX as u128), Some(1)).is_none());
    assert!(normalize_usage(None, Some(1)).is_none());
    assert!(normalize_usage(Some(u64::MAX as u128 + 1), Some(0)).is_none());

    let malformed = build_response(
        &req,
        ResponseMetadata::for_model(&req.model),
        Some("still valid".into()),
        vec![],
        NativeTerminal::Stop,
        Some(nano_llm::Usage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 99,
        }),
    )
    .unwrap();
    assert!(malformed.usage.is_none());
}

#[test]
fn calls_are_constrained_and_missing_or_colliding_ids_are_generated() {
    let req = request(
        r#", "tools":[{"type":"function","function":{"name":"weather"}}],"tool_choice":"required""#,
    );
    let response = build_response(
        &req,
        ResponseMetadata::for_model(&req.model),
        None,
        vec![
            NativeToolCall {
                id: None,
                name: "weather".into(),
                arguments: "not-json-is-opaque".into(),
            },
            NativeToolCall {
                id: Some("".into()),
                name: "weather".into(),
                arguments: "{}".into(),
            },
        ],
        NativeTerminal::Stop,
        None,
    )
    .unwrap();
    let calls = response.choices[0].message.tool_calls.as_ref().unwrap();
    assert!(calls.iter().all(|call| call.id.starts_with("call_")));
    assert_ne!(calls[0].id, calls[1].id);
    assert_eq!(response.choices[0].finish_reason, FinishReason::ToolCalls);

    let no_calls = build_response(
        &req,
        ResponseMetadata::for_model(&req.model),
        Some("no".into()),
        vec![],
        NativeTerminal::Stop,
        None,
    );
    assert!(matches!(no_calls, Err(ResponseError::InvalidResponse(_))));
    let undeclared = build_response(
        &req,
        ResponseMetadata::for_model(&req.model),
        None,
        vec![NativeToolCall {
            id: Some("x".into()),
            name: "other".into(),
            arguments: "{}".into(),
        }],
        NativeTerminal::ToolCalls,
        None,
    );
    assert!(matches!(undeclared, Err(ResponseError::InvalidResponse(_))));
}

#[test]
fn safety_and_terminal_mappings_are_strict() {
    let req = request("");
    let safety = safety_response(&req, ResponseMetadata::for_model(&req.model), None).unwrap();
    assert_eq!(safety.choices[0].finish_reason, FinishReason::ContentFilter);
    assert_eq!(safety.choices[0].message.content, None);
    assert!(matches!(
        build_response(
            &req,
            ResponseMetadata::for_model(&req.model),
            Some("x".into()),
            vec![],
            NativeTerminal::Overloaded,
            None
        ),
        Err(ResponseError::Overloaded)
    ));
    assert!(matches!(
        build_response(
            &req,
            ResponseMetadata::for_model(&req.model),
            None,
            vec![],
            NativeTerminal::Unknown,
            None
        ),
        Err(ResponseError::InvalidResponse(_))
    ));
    let unknown = build_response(
        &req,
        ResponseMetadata::for_model(&req.model),
        Some("x".into()),
        vec![],
        NativeTerminal::Unknown,
        None,
    )
    .unwrap();
    assert_eq!(unknown.choices[0].finish_reason, FinishReason::Stop);
}

#[test]
fn native_choice_cardinality_and_index_are_validated_before_normalization() {
    let req = request("");
    for choices in [
        vec![],
        vec![NativeChoice {
            index: 1,
            content: Some("one".into()),
            calls: vec![],
            terminal: NativeTerminal::Stop,
        }],
        vec![
            NativeChoice {
                index: 0,
                content: Some("one".into()),
                calls: vec![],
                terminal: NativeTerminal::Stop,
            },
            NativeChoice {
                index: 1,
                content: Some("two".into()),
                calls: vec![],
                terminal: NativeTerminal::Stop,
            },
        ],
    ] {
        assert!(matches!(
            normalize_response(
                &req,
                ResponseMetadata::for_model("upstream-model"),
                NativeResponse {
                    choices,
                    usage: None
                },
            ),
            Err(ResponseError::InvalidResponse(_))
        ));
    }

    let response = normalize_response(
        &req,
        ResponseMetadata::for_model("upstream-model"),
        NativeResponse {
            choices: vec![NativeChoice {
                index: 0,
                content: Some("one".into()),
                calls: vec![],
                terminal: NativeTerminal::Stop,
            }],
            usage: None,
        },
    )
    .unwrap();
    assert_eq!(response.model, req.model);
}

#[test]
fn stream_assembly_keeps_metadata_and_call_ids_stable_until_one_terminal_chunk() {
    let req =
        request(r#", "stream":true,"tools":[{"type":"function","function":{"name":"weather"}}]"#);
    let metadata = ResponseMetadata::for_model(&req.model);
    let mut stream = StreamAssembler::new(&req, metadata.clone());
    let first = stream
        .push(
            0,
            AssistantDelta {
                role: None,
                content: None,
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: None,
                    r#type: Some("function"),
                    name: Some("wea".into()),
                    arguments: Some("{".into()),
                }],
            },
            None,
        )
        .unwrap()
        .unwrap();
    let id = first.choices[0].delta.tool_calls[0].id.clone().unwrap();
    assert_eq!(first.id, metadata.id());
    assert_eq!(first.choices[0].delta.role, Some("assistant"));
    let terminal = stream
        .push(
            0,
            AssistantDelta {
                role: None,
                content: None,
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: Some(id.clone()),
                    r#type: None,
                    name: Some("ther".into()),
                    arguments: Some("}".into()),
                }],
            },
            Some(NativeTerminal::Stop),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        terminal.choices[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    stream.finish().unwrap();
    assert!(stream
        .usage_chunk(normalize_usage(Some(1), Some(2)).unwrap())
        .unwrap()
        .is_none());
    assert!(matches!(
        stream.push(0, AssistantDelta::default(), None),
        Err(ResponseError::InvalidResponse(_))
    ));
}

#[test]
fn stream_rejects_nonsequential_indices_and_missing_terminal() {
    let req =
        request(r#", "stream":true,"tools":[{"type":"function","function":{"name":"weather"}}]"#);
    let mut stream = StreamAssembler::new(&req, ResponseMetadata::for_model(&req.model));
    assert!(matches!(
        stream.push(
            0,
            AssistantDelta {
                role: None,
                content: None,
                tool_calls: vec![ToolCallDelta {
                    index: 1,
                    id: None,
                    r#type: None,
                    name: None,
                    arguments: None
                }]
            },
            None
        ),
        Err(ResponseError::InvalidResponse(_))
    ));
    let mut no_terminal = StreamAssembler::new(&req, ResponseMetadata::for_model(&req.model));
    no_terminal
        .push(
            0,
            AssistantDelta {
                role: None,
                content: Some("text".into()),
                tool_calls: vec![],
            },
            None,
        )
        .unwrap();
    assert!(matches!(
        no_terminal.finish(),
        Err(ResponseError::InvalidResponse(_))
    ));
}

#[test]
fn stream_ignores_empty_events_and_only_emits_one_requested_usage_chunk() {
    let req = request(r#", "stream":true,"stream_options":{"include_usage":true}"#);
    let metadata = ResponseMetadata::for_model(&req.model);
    let mut stream = StreamAssembler::new(&req, metadata.clone());
    assert!(stream
        .push(0, AssistantDelta::default(), None)
        .unwrap()
        .is_none());
    let first = stream
        .push(
            0,
            AssistantDelta::default(),
            Some(NativeTerminal::ContentFilter),
        )
        .unwrap()
        .unwrap();
    assert_eq!(first.id, metadata.id());
    assert_eq!(first.created, metadata.created());
    assert_eq!(first.choices[0].delta.role, Some("assistant"));
    assert_eq!(
        first.choices[0].finish_reason,
        Some(FinishReason::ContentFilter)
    );
    let usage = stream
        .usage_chunk(normalize_usage(Some(1), Some(2)).unwrap())
        .unwrap()
        .unwrap();
    assert!(usage.choices.is_empty());
    assert_eq!(usage.usage.unwrap().total_tokens, 3);
    assert!(stream
        .usage_chunk(normalize_usage(Some(1), Some(2)).unwrap())
        .unwrap()
        .is_none());
}

#[test]
fn generated_stream_ids_ignore_later_native_ids_but_preserved_ones_cannot_change() {
    let req =
        request(r#", "stream":true,"tools":[{"type":"function","function":{"name":"weather"}}]"#);
    let mut generated = StreamAssembler::new(&req, ResponseMetadata::for_model(&req.model));
    let initial = generated
        .push(
            0,
            AssistantDelta {
                role: None,
                content: None,
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: None,
                    r#type: None,
                    name: Some("weather".into()),
                    arguments: None,
                }],
            },
            None,
        )
        .unwrap()
        .unwrap();
    let generated_id = initial.choices[0].delta.tool_calls[0].id.clone().unwrap();
    let later = generated
        .push(
            0,
            AssistantDelta {
                role: None,
                content: None,
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: Some("native-late".into()),
                    r#type: None,
                    name: None,
                    arguments: Some("not json".into()),
                }],
            },
            Some(NativeTerminal::Stop),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        later.choices[0].delta.tool_calls[0].id.as_deref(),
        Some(generated_id.as_str())
    );

    let mut preserved = StreamAssembler::new(&req, ResponseMetadata::for_model(&req.model));
    preserved
        .push(
            0,
            AssistantDelta {
                role: None,
                content: None,
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: Some("native".into()),
                    r#type: None,
                    name: Some("weather".into()),
                    arguments: None,
                }],
            },
            None,
        )
        .unwrap();
    assert!(matches!(
        preserved.push(
            0,
            AssistantDelta {
                role: None,
                content: None,
                tool_calls: vec![ToolCallDelta {
                    index: 0,
                    id: Some("changed".into()),
                    r#type: None,
                    name: None,
                    arguments: None
                }]
            },
            None
        ),
        Err(ResponseError::InvalidResponse(_))
    ));
}
