use nano_llm::config::{
    parse_yaml_str, ConfigErrorKind, RawConfig, RawGeneralSettings, RawLiteLlmParams, RawModelEntry,
};

#[test]
fn test_valid_minimal_config() {
    let yaml = r#"
model_list:
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
"#;
    let config = parse_yaml_str(yaml).expect("should parse valid minimal config");
    assert_eq!(
        config,
        RawConfig {
            model_list: vec![RawModelEntry {
                model_name: "default".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: None,
                    api_base: None,
                    timeout: None,
                },
            }],
            general_settings: None,
        }
    );
}

#[test]
fn test_valid_full_config_with_source_order_retained() {
    let yaml = r#"
model_list:
  - model_name: fast
    litellm_params:
      model: mistral/mistral-small-latest
      api_key: os.environ/MISTRAL_API_KEY
      api_base: https://api.mistral.ai/v1
      timeout: 15.5
  - model_name: fast
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEMINI_API_KEY
      timeout: 20
  - model_name: quality
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  request_timeout: 30
  overall_timeout: 120.0
  max_in_flight: 64
"#;
    let config = parse_yaml_str(yaml).expect("should parse valid full config");
    assert_eq!(config.model_list.len(), 3);
    assert_eq!(config.model_list[0].model_name, "fast");
    assert_eq!(
        config.model_list[0].litellm_params.model,
        "mistral/mistral-small-latest"
    );
    assert_eq!(
        config.model_list[0].litellm_params.api_key.as_deref(),
        Some("os.environ/MISTRAL_API_KEY")
    );
    assert_eq!(
        config.model_list[0].litellm_params.api_base.as_deref(),
        Some("https://api.mistral.ai/v1")
    );
    assert_eq!(config.model_list[0].litellm_params.timeout, Some(15.5));

    assert_eq!(config.model_list[1].model_name, "fast");
    assert_eq!(
        config.model_list[1].litellm_params.model,
        "gemini/gemini-2.5-flash"
    );
    assert_eq!(config.model_list[1].litellm_params.timeout, Some(20.0));

    assert_eq!(config.model_list[2].model_name, "quality");
    assert_eq!(
        config.model_list[2].litellm_params.model,
        "anthropic/claude-3-5-sonnet-20241022"
    );

    let gs = config.general_settings.expect("general_settings present");
    assert_eq!(
        gs,
        RawGeneralSettings {
            master_key: Some("os.environ/LITELLM_MASTER_KEY".to_string()),
            request_timeout: Some(30.0),
            overall_timeout: Some(120.0),
            max_in_flight: Some(64),
        }
    );
}

#[test]
fn test_valid_optional_null_and_tilde_fields() {
    let yaml = r#"
model_list:
  - model_name: custom
    litellm_params:
      model: openai_compatible/custom-model
      api_key: null
      api_base: ~
      timeout: null
general_settings:
  master_key: null
  request_timeout: ~
"#;
    let config = parse_yaml_str(yaml).expect("should parse config with null/tilde optionals");
    assert_eq!(config.model_list[0].litellm_params.api_key, None);
    assert_eq!(config.model_list[0].litellm_params.api_base, None);
    assert_eq!(config.model_list[0].litellm_params.timeout, None);
    let gs = config.general_settings.expect("general_settings present");
    assert_eq!(gs.master_key, None);
    assert_eq!(gs.request_timeout, None);
}

// ---------------------------------------------------------------------------
// Prohibited construct tests: Anchors & Aliases
// ---------------------------------------------------------------------------

#[test]
fn test_rejects_scalar_anchor() {
    let yaml = r#"
model_list:
  - model_name: &anchor default
    litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::ProhibitedAnchor);
    assert_eq!(err.location.unwrap().line, 3);
}

#[test]
fn test_rejects_mapping_anchor() {
    let yaml = r#"
base_params: &base
  model: openai/gpt-4o
model_list:
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::ProhibitedAnchor);
}

#[test]
fn test_rejects_sequence_anchor() {
    let yaml = r#"
model_list: &models
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::ProhibitedAnchor);
}

#[test]
fn test_rejects_alias_in_value() {
    let yaml = r#"
model_list:
  - model_name: default
    litellm_params: *base
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::ProhibitedAlias);
}

#[test]
fn test_rejects_alias_in_key() {
    let yaml = r#"
*base:
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::ProhibitedAlias);
}

// ---------------------------------------------------------------------------
// Prohibited construct tests: Merge keys
// ---------------------------------------------------------------------------

#[test]
fn test_rejects_merge_key() {
    let yaml = r#"
model_list:
  - model_name: default
    litellm_params:
      <<: { model: openai/gpt-4o }
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::ProhibitedMergeKey);
}

// ---------------------------------------------------------------------------
// Prohibited construct tests: Tags
// ---------------------------------------------------------------------------

#[test]
fn test_rejects_custom_tag() {
    let yaml = r#"
model_list: !custom
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert!(matches!(err.kind, ConfigErrorKind::ProhibitedTag { .. }));
    let msg = err.to_string();
    assert!(msg.contains("YAML tags") && msg.contains("!custom"));
}

#[test]
fn test_rejects_core_tag() {
    let yaml = r#"
model_list:
  - model_name: !!str default
    litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert!(matches!(err.kind, ConfigErrorKind::ProhibitedTag { .. }));
}

// ---------------------------------------------------------------------------
// Prohibited construct tests: Non-string mapping keys
// ---------------------------------------------------------------------------

#[test]
fn test_rejects_integer_mapping_key() {
    let yaml = r#"
123:
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::NonStringKey {
            found: "integer".to_string()
        }
    );
}

#[test]
fn test_rejects_boolean_mapping_key() {
    let yaml = r#"
model_list:
  - model_name: default
    litellm_params:
      true: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::NonStringKey {
            found: "boolean".to_string()
        }
    );
}

#[test]
fn test_rejects_null_mapping_key() {
    let yaml = r#"
general_settings:
  null: 10
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::NonStringKey {
            found: "null".to_string()
        }
    );
}

#[test]
fn test_rejects_sequence_mapping_key() {
    let yaml = r#"
? [1, 2]
: model_list
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::NonStringKey {
            found: "sequence".to_string()
        }
    );
}

#[test]
fn test_rejects_mapping_mapping_key() {
    let yaml = r#"
? { key: val }
: model_list
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::NonStringKey {
            found: "mapping".to_string()
        }
    );
}

// ---------------------------------------------------------------------------
// Prohibited construct tests: Multiple documents & empty documents
// ---------------------------------------------------------------------------

#[test]
fn test_rejects_multiple_documents() {
    let yaml = r#"
model_list:
  - model_name: doc1
    litellm_params:
      model: openai/gpt-4o
---
model_list:
  - model_name: doc2
    litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::MultipleDocuments);
    assert_eq!(err.location.unwrap().line, 6);
}

#[test]
fn test_rejects_empty_input() {
    let err = parse_yaml_str("").unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::EmptyDocument);
}

#[test]
fn test_rejects_comments_only_input() {
    let err = parse_yaml_str("# Just a comment\n# Another comment\n").unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::EmptyDocument);
}

#[test]
fn test_rejects_whitespace_only_input() {
    let err = parse_yaml_str("   \n\n\t  \n").unwrap_err();
    assert_eq!(err.kind, ConfigErrorKind::EmptyDocument);
}

// ---------------------------------------------------------------------------
// Duplicate key diagnostics at every depth
// ---------------------------------------------------------------------------

#[test]
fn test_duplicate_key_top_level() {
    let yaml = r#"
model_list: []
model_list: []
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::DuplicateKey {
            path: "model_list".to_string()
        }
    );
    assert_eq!(err.location.unwrap().line, 3);
}

#[test]
fn test_duplicate_key_general_settings() {
    let yaml = r#"
model_list: []
general_settings:
  request_timeout: 30
  request_timeout: 60
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::DuplicateKey {
            path: "general_settings.request_timeout".to_string()
        }
    );
    assert_eq!(err.location.unwrap().line, 5);
}

#[test]
fn test_duplicate_key_model_list_entry() {
    let yaml = r#"
model_list:
  - model_name: primary
    model_name: secondary
    litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::DuplicateKey {
            path: "model_list[0].model_name".to_string()
        }
    );
    assert_eq!(err.location.unwrap().line, 4);
}

#[test]
fn test_duplicate_key_litellm_params() {
    let yaml = r#"
model_list:
  - model_name: primary
    litellm_params:
      model: openai/gpt-4o
  - model_name: secondary
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/KEY1
      api_key: os.environ/KEY2
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::DuplicateKey {
            path: "model_list[1].litellm_params.api_key".to_string()
        }
    );
    assert_eq!(err.location.unwrap().line, 10);
}

// ---------------------------------------------------------------------------
// Unknown key diagnostics at top level, general_settings, and litellm_params
// ---------------------------------------------------------------------------

#[test]
fn test_unknown_key_top_level() {
    let yaml = r#"
model_list: []
unknown_top_level: true
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::UnknownKey {
            path: String::new(),
            key: "unknown_top_level".to_string(),
        }
    );
    let msg = err.to_string();
    assert!(msg.contains("unknown top-level key 'unknown_top_level'"));
}

#[test]
fn test_unknown_key_general_settings() {
    let yaml = r#"
general_settings:
  fallbacks: ["other_group"]
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::UnknownKey {
            path: "general_settings".to_string(),
            key: "fallbacks".to_string(),
        }
    );
    let msg = err.to_string();
    assert!(msg.contains("unknown key 'fallbacks' at 'general_settings'"));
}

#[test]
fn test_unknown_key_model_list_item() {
    let yaml = r#"
model_list:
  - model_name: fast
    litellm_params:
      model: openai/gpt-4o
    extra_field: 123
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::UnknownKey {
            path: "model_list[0]".to_string(),
            key: "extra_field".to_string(),
        }
    );
}

#[test]
fn test_unknown_key_litellm_params() {
    let yaml = r#"
model_list:
  - model_name: fast
    litellm_params:
      model: openai/gpt-4o
      temperature: 0.7
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::UnknownKey {
            path: "model_list[0].litellm_params".to_string(),
            key: "temperature".to_string(),
        }
    );
    let msg = err.to_string();
    assert!(msg.contains("unknown key 'temperature' at 'model_list[0].litellm_params'"));
}

// ---------------------------------------------------------------------------
// Type mismatch diagnostics
// ---------------------------------------------------------------------------

#[test]
fn test_type_mismatch_root_not_mapping() {
    let yaml = r#"
- item1
- item2
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert!(matches!(
        err.kind,
        ConfigErrorKind::InvalidType {
            expected: "mapping",
            found,
            ..
        } if found == "sequence"
    ));
}

#[test]
fn test_type_mismatch_model_list_not_sequence() {
    let yaml = r#"
model_list: "not a sequence"
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::InvalidType {
            path: "model_list".to_string(),
            expected: "sequence",
            found: "string".to_string(),
        }
    );
}

#[test]
fn test_type_mismatch_litellm_params_not_mapping() {
    let yaml = r#"
model_list:
  - model_name: default
    litellm_params: "not a mapping"
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::InvalidType {
            path: "model_list[0].litellm_params".to_string(),
            expected: "mapping",
            found: "string".to_string(),
        }
    );
}

#[test]
fn test_type_mismatch_timeout_not_number() {
    let yaml = r#"
model_list:
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
      timeout: "slow"
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::InvalidType {
            path: "model_list[0].litellm_params.timeout".to_string(),
            expected: "number",
            found: "slow".to_string(),
        }
    );
}

#[test]
fn test_type_mismatch_max_in_flight_not_integer() {
    let yaml = r#"
general_settings:
  max_in_flight: 64.5
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::InvalidType {
            path: "general_settings.max_in_flight".to_string(),
            expected: "integer",
            found: "64.5".to_string(),
        }
    );
}

// ---------------------------------------------------------------------------
// Missing required fields
// ---------------------------------------------------------------------------

#[test]
fn test_missing_model_name() {
    let yaml = r#"
model_list:
  - litellm_params:
      model: openai/gpt-4o
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::MissingField {
            path: "model_list[0]".to_string(),
            field: "model_name",
        }
    );
}

#[test]
fn test_missing_litellm_params() {
    let yaml = r#"
model_list:
  - model_name: default
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::MissingField {
            path: "model_list[0]".to_string(),
            field: "litellm_params",
        }
    );
}

#[test]
fn test_missing_model_in_litellm_params() {
    let yaml = r#"
model_list:
  - model_name: default
    litellm_params:
      api_key: os.environ/KEY
"#;
    let err = parse_yaml_str(yaml).unwrap_err();
    assert_eq!(
        err.kind,
        ConfigErrorKind::MissingField {
            path: "model_list[0].litellm_params".to_string(),
            field: "model",
        }
    );
}
