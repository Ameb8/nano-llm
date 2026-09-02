use nano_llm::config::{build_runtime_config, parse_yaml_str, ConfigErrorKind, ProviderKind};
use std::collections::HashMap;

fn mock_env() -> HashMap<String, String> {
    let mut env = HashMap::new();
    env.insert("M_KEY".to_string(), "master_secret".to_string());
    env.insert("OAI_KEY".to_string(), "openai_secret".to_string());
    env.insert("MIS_KEY".to_string(), "mistral_secret".to_string());
    env.insert("ANT_KEY".to_string(), "anthropic_secret".to_string());
    env.insert("GEM_KEY".to_string(), "gemini_secret".to_string());
    env
}

#[test]
fn test_route_and_public_model_ordering() {
    let env = mock_env();
    let yaml = r#"
model_list:
  - model_name: alpha
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OAI_KEY
  - model_name: beta
    litellm_params:
      model: anthropic/claude-3-5-sonnet
      api_key: os.environ/ANT_KEY
  - model_name: alpha
    litellm_params:
      model: mistral/mistral-large
      api_key: os.environ/MIS_KEY
  - model_name: gamma
    litellm_params:
      model: gemini/gemini-2.5-flash
      api_key: os.environ/GEM_KEY
  - model_name: beta
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OAI_KEY
  - model_name: alpha
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OAI_KEY

general_settings:
  master_key: os.environ/M_KEY
"#;

    let raw = parse_yaml_str(yaml).expect("parse yaml");
    let runtime = build_runtime_config(raw, false, &env).expect("build runtime config");

    // Public model ordering preserves first appearance: alpha, beta, gamma
    assert_eq!(runtime.routes.len(), 3);
    assert_eq!(runtime.routes[0].model_name, "alpha");
    assert_eq!(runtime.routes[1].model_name, "beta");
    assert_eq!(runtime.routes[2].model_name, "gamma");

    // alpha route preserves target file order and keeps duplicate identical targets
    let alpha = &runtime.routes[0];
    assert_eq!(alpha.targets.len(), 3);
    assert_eq!(alpha.targets[0].model, "openai/gpt-4o");
    assert_eq!(alpha.targets[1].model, "mistral/mistral-large");
    assert_eq!(alpha.targets[2].model, "openai/gpt-4o");

    // beta route preserves target file order
    let beta = &runtime.routes[1];
    assert_eq!(beta.targets.len(), 2);
    assert_eq!(beta.targets[0].model, "anthropic/claude-3-5-sonnet");
    assert_eq!(beta.targets[1].model, "openai/gpt-4o-mini");

    // gamma route has 1 target
    let gamma = &runtime.routes[2];
    assert_eq!(gamma.targets.len(), 1);
    assert_eq!(gamma.targets[0].model, "gemini/gemini-2.5-flash");
}

#[test]
fn test_all_defaults_present_in_runtime_config() {
    let env = mock_env();
    let yaml = r#"
model_list:
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OAI_KEY
  - model_name: custom
    litellm_params:
      model: openai_compatible/llama-3
      api_base: https://custom-llm.internal:8000/v1/

general_settings:
  master_key: os.environ/M_KEY
"#;

    let raw = parse_yaml_str(yaml).expect("parse yaml");
    let runtime = build_runtime_config(raw, false, &env).expect("build runtime config");

    // General settings defaults
    assert_eq!(runtime.general_settings.request_timeout, 30);
    assert_eq!(runtime.general_settings.overall_timeout, 120);
    assert_eq!(runtime.general_settings.max_in_flight, 64);

    // Target defaults
    let t1 = &runtime.routes[0].targets[0];
    assert_eq!(t1.provider, ProviderKind::OpenAi);
    assert_eq!(t1.api_base, "https://api.openai.com/v1"); // default base
    assert_eq!(t1.timeout, 30); // defaulted from general_settings.request_timeout
    assert_eq!(t1.explicit_timeout, None);

    let t2 = &runtime.routes[1].targets[0];
    assert_eq!(t2.provider, ProviderKind::OpenAiCompatible);
    assert_eq!(t2.api_base, "https://custom-llm.internal:8000/v1"); // normalized
    assert_eq!(t2.timeout, 30); // defaulted from general_settings.request_timeout
    assert_eq!(t2.api_key, None);
}

#[test]
fn test_empty_model_list_rejected() {
    let env = mock_env();
    let yaml = r#"
model_list: []
general_settings:
  master_key: os.environ/M_KEY
"#;

    let raw = parse_yaml_str(yaml).expect("parse yaml");
    let err =
        build_runtime_config(raw, false, &env).expect_err("empty model_list should be rejected");
    assert!(matches!(err.kind, ConfigErrorKind::EmptyModelList { .. }));
}

#[test]
fn test_invalid_model_names_rejected() {
    let env = mock_env();
    let long_name = "a".repeat(129);
    let invalid_names = [
        "",
        "_leading_underscore",
        "-leading_hyphen",
        "/leading_slash",
        "has spaces in name",
        "has@symbol",
        "has$dollar",
        long_name.as_str(),
    ];

    for name in invalid_names {
        let yaml = format!(
            r#"
model_list:
  - model_name: "{name}"
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OAI_KEY
general_settings:
  master_key: os.environ/M_KEY
"#
        );

        if let Ok(raw) = parse_yaml_str(&yaml) {
            let err = build_runtime_config(raw, false, &env)
                .expect_err("expected invalid model_name to fail");
            assert!(
                matches!(err.kind, ConfigErrorKind::InvalidModelName { .. }),
                "expected InvalidModelName for '{name}', got {:?}",
                err.kind
            );
        }
    }
}

#[test]
fn test_invalid_timeouts_rejected() {
    let env = mock_env();
    let invalid_cases = [
        ("timeout: 0", "target timeout 0"),
        ("timeout: 86401", "target timeout 86401"),
        ("timeout: 10.5", "target float timeout"),
    ];

    for (field_line, desc) in invalid_cases {
        let yaml = format!(
            r#"
model_list:
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OAI_KEY
      {field_line}
general_settings:
  master_key: os.environ/M_KEY
"#
        );

        let raw = parse_yaml_str(&yaml).expect("parse yaml");
        let err =
            build_runtime_config(raw, false, &env).expect_err("expected invalid timeout to fail");
        assert!(
            matches!(err.kind, ConfigErrorKind::InvalidTimeout { .. }),
            "expected InvalidTimeout for {desc}, got {:?}",
            err.kind
        );
    }
}
