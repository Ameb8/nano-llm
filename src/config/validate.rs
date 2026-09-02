use crate::config::error::{ConfigError, ConfigErrorKind};
use crate::config::raw::RawConfig;
use crate::config::runtime::{
    ProviderKind, RuntimeConfig, RuntimeGeneralSettings, RuntimeRoute, RuntimeTarget,
};
use crate::config::secrets::{resolve_secret, EnvProvider};

/// Builds the immutable runtime configuration from parsed raw configuration and CLI flags.
pub fn build_runtime_config<E: EnvProvider>(
    raw: RawConfig,
    no_auth: bool,
    env: &E,
) -> Result<RuntimeConfig, ConfigError> {
    // 1. General settings validation & resolution
    let general_settings = if let Some(raw_gs) = raw.general_settings {
        let master_key = if let Some(ref mk_ref) = raw_gs.master_key {
            Some(resolve_secret(
                mk_ref,
                "general_settings.master_key",
                true,
                env,
            )?)
        } else if !no_auth {
            return Err(ConfigError::new(
                ConfigErrorKind::MissingMasterKey {
                    path: "general_settings.master_key".to_string(),
                },
                None,
            ));
        } else {
            None
        };

        let request_timeout = if let Some(rt) = raw_gs.request_timeout {
            if rt.fract() != 0.0 || !(1.0..=86400.0).contains(&rt) {
                return Err(ConfigError::new(
                    ConfigErrorKind::InvalidTimeout {
                        path: "general_settings.request_timeout".to_string(),
                        reason: "request_timeout must be an integer between 1 and 86400 seconds"
                            .to_string(),
                    },
                    None,
                ));
            }
            rt as u64
        } else {
            30
        };

        let overall_timeout = if let Some(ot) = raw_gs.overall_timeout {
            if ot.fract() != 0.0 || !(1.0..=86400.0).contains(&ot) {
                return Err(ConfigError::new(
                    ConfigErrorKind::InvalidTimeout {
                        path: "general_settings.overall_timeout".to_string(),
                        reason: "overall_timeout must be an integer between 1 and 86400 seconds"
                            .to_string(),
                    },
                    None,
                ));
            }
            ot as u64
        } else {
            120
        };

        let max_in_flight = if let Some(mif) = raw_gs.max_in_flight {
            if !(1..=65535).contains(&mif) {
                return Err(ConfigError::new(
                    ConfigErrorKind::InvalidMaxInFlight {
                        path: "general_settings.max_in_flight".to_string(),
                        reason: "max_in_flight must be an integer between 1 and 65535".to_string(),
                    },
                    None,
                ));
            }
            mif
        } else {
            64
        };

        RuntimeGeneralSettings {
            master_key,
            request_timeout,
            overall_timeout,
            max_in_flight,
        }
    } else if !no_auth {
        return Err(ConfigError::new(
            ConfigErrorKind::MissingGeneralSettings {
                path: "general_settings".to_string(),
            },
            None,
        ));
    } else {
        RuntimeGeneralSettings::default()
    };

    // 2. Model list validation & route grouping
    if raw.model_list.is_empty() {
        return Err(ConfigError::new(
            ConfigErrorKind::EmptyModelList {
                path: "model_list".to_string(),
            },
            None,
        ));
    }

    let mut routes: Vec<RuntimeRoute> = Vec::new();

    for (idx, entry) in raw.model_list.into_iter().enumerate() {
        // Validate model_name
        let model_name_path = format!("model_list[{idx}].model_name");
        validate_model_name(&entry.model_name, &model_name_path)?;

        // Validate litellm_params.model
        let model_path = format!("model_list[{idx}].litellm_params.model");
        let (provider, suffix) = validate_target_model(&entry.litellm_params.model, &model_path)?;
        let suffix = suffix.to_string();

        // Validate api_key
        let api_key_path = format!("model_list[{idx}].litellm_params.api_key");
        let api_key = match entry.litellm_params.api_key {
            Some(ref key_ref) => Some(resolve_secret(
                key_ref,
                &api_key_path,
                provider.is_http_header_transport(),
                env,
            )?),
            None => {
                if provider.is_branded() {
                    return Err(ConfigError::new(
                        ConfigErrorKind::MissingApiKey {
                            path: format!("model_list[{idx}].litellm_params"),
                            provider: provider.name().to_string(),
                        },
                        None,
                    ));
                }
                None
            }
        };

        // Validate api_base
        let api_base_path = format!("model_list[{idx}].litellm_params.api_base");
        let api_base = match entry.litellm_params.api_base {
            Some(ref base_str) => validate_and_normalize_api_base(
                base_str,
                provider,
                api_key.is_some(),
                &api_base_path,
            )?,
            None => match provider.default_api_base() {
                Some(default_base) => default_base.to_string(),
                None => {
                    return Err(ConfigError::new(
                        ConfigErrorKind::MissingApiBase {
                            path: format!("model_list[{idx}].litellm_params"),
                            provider: provider.name().to_string(),
                        },
                        None,
                    ));
                }
            },
        };

        // Validate timeout
        let timeout_path = format!("model_list[{idx}].litellm_params.timeout");
        let explicit_timeout = if let Some(t) = entry.litellm_params.timeout {
            if t.fract() != 0.0 || !(1.0..=86400.0).contains(&t) {
                return Err(ConfigError::new(
                    ConfigErrorKind::InvalidTimeout {
                        path: timeout_path,
                        reason: "timeout must be an integer between 1 and 86400 seconds"
                            .to_string(),
                    },
                    None,
                ));
            }
            Some(t as u64)
        } else {
            None
        };

        let effective_timeout = explicit_timeout.unwrap_or(general_settings.request_timeout);

        let target = RuntimeTarget {
            model: entry.litellm_params.model,
            provider,
            model_suffix: suffix,
            api_key,
            api_base,
            timeout: effective_timeout,
            explicit_timeout,
        };

        // Group into routes preserving first appearance order of public model names
        // and preserving target file order (no deduplication or load balancing).
        if let Some(existing_route) = routes.iter_mut().find(|r| r.model_name == entry.model_name) {
            existing_route.targets.push(target);
        } else {
            routes.push(RuntimeRoute {
                model_name: entry.model_name,
                targets: vec![target],
            });
        }
    }

    Ok(RuntimeConfig {
        general_settings,
        routes,
    })
}

fn validate_model_name(name: &str, path: &str) -> Result<(), ConfigError> {
    if name.is_empty() || name.len() > 128 {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidModelName {
                path: path.to_string(),
                model_name: name.to_string(),
                reason: "model_name length must be between 1 and 128 characters".to_string(),
            },
            None,
        ));
    }

    let first_char = name.chars().next().unwrap();
    if !first_char.is_ascii_alphanumeric() {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidModelName {
                path: path.to_string(),
                model_name: name.to_string(),
                reason: "model_name must start with an ASCII alphanumeric character".to_string(),
            },
            None,
        ));
    }

    if !name.chars().all(|c| {
        c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == ':' || c == '/' || c == '-'
    }) {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidModelName {
                path: path.to_string(),
                model_name: name.to_string(),
                reason: "model_name must contain only [A-Za-z0-9._:/-]".to_string(),
            },
            None,
        ));
    }

    Ok(())
}

fn validate_target_model<'a>(
    model: &'a str,
    path: &str,
) -> Result<(ProviderKind, &'a str), ConfigError> {
    let (prefix, suffix) = model.split_once('/').ok_or_else(|| {
        ConfigError::new(
            ConfigErrorKind::InvalidTargetModel {
                path: path.to_string(),
                model: model.to_string(),
                reason: "model must be in the format '<provider>/<model_suffix>'".to_string(),
            },
            None,
        )
    })?;

    let provider = match prefix {
        "openai" => ProviderKind::OpenAi,
        "mistral" => ProviderKind::Mistral,
        "deepseek" => ProviderKind::DeepSeek,
        "openai_compatible" => ProviderKind::OpenAiCompatible,
        "anthropic" => ProviderKind::Anthropic,
        "gemini" => ProviderKind::Gemini,
        _ => {
            return Err(ConfigError::new(
                ConfigErrorKind::InvalidTargetModel {
                    path: path.to_string(),
                    model: model.to_string(),
                    reason: format!("unknown provider prefix '{prefix}'"),
                },
                None,
            ));
        }
    };

    if suffix.is_empty() {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidTargetModel {
                path: path.to_string(),
                model: model.to_string(),
                reason: "model suffix must not be empty".to_string(),
            },
            None,
        ));
    }

    if suffix.len() > 256 {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidTargetModel {
                path: path.to_string(),
                model: model.to_string(),
                reason: "model suffix exceeds maximum length of 256 bytes".to_string(),
            },
            None,
        ));
    }

    if suffix.chars().any(|c| c.is_ascii_control()) {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidTargetModel {
                path: path.to_string(),
                model: model.to_string(),
                reason: "model suffix contains ASCII control characters".to_string(),
            },
            None,
        ));
    }

    if provider == ProviderKind::Gemini {
        if suffix.len() > 128 {
            return Err(ConfigError::new(
                ConfigErrorKind::InvalidTargetModel {
                    path: path.to_string(),
                    model: model.to_string(),
                    reason: "Gemini model suffix must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
                        .to_string(),
                },
                None,
            ));
        }
        let first_char = suffix.chars().next().unwrap();
        if !first_char.is_ascii_alphanumeric()
            || !suffix
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
        {
            return Err(ConfigError::new(
                ConfigErrorKind::InvalidTargetModel {
                    path: path.to_string(),
                    model: model.to_string(),
                    reason: "Gemini model suffix must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
                        .to_string(),
                },
                None,
            ));
        }
    }

    Ok((provider, suffix))
}

fn validate_and_normalize_api_base(
    base: &str,
    provider: ProviderKind,
    has_api_key: bool,
    path: &str,
) -> Result<String, ConfigError> {
    let (is_https, rest) = if let Some(r) = base.strip_prefix("https://") {
        (true, r)
    } else if let Some(r) = base.strip_prefix("http://") {
        (false, r)
    } else {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidApiBase {
                path: path.to_string(),
                reason: "api_base must be an absolute http or https URL".to_string(),
            },
            None,
        ));
    };

    if provider.is_branded() && !is_https {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidApiBase {
                path: path.to_string(),
                reason: "branded provider requires https api_base".to_string(),
            },
            None,
        ));
    }

    if has_api_key && !is_https {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidApiBase {
                path: path.to_string(),
                reason: "target with api_key requires https api_base".to_string(),
            },
            None,
        ));
    }

    if base.contains('#') {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidApiBase {
                path: path.to_string(),
                reason: "api_base must not contain a fragment ('#')".to_string(),
            },
            None,
        ));
    }

    if base.contains('?') {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidApiBase {
                path: path.to_string(),
                reason: "api_base must not contain a query string ('?')".to_string(),
            },
            None,
        ));
    }

    let (host_port, _) = match rest.split_once('/') {
        Some((h, p)) => (h, p),
        None => (rest, ""),
    };

    if host_port.contains('@') {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidApiBase {
                path: path.to_string(),
                reason: "api_base must not contain userinfo ('@')".to_string(),
            },
            None,
        ));
    }

    if host_port.is_empty() {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidApiBase {
                path: path.to_string(),
                reason: "api_base must contain a host".to_string(),
            },
            None,
        ));
    }

    if host_port
        .chars()
        .any(|c| c.is_whitespace() || c.is_ascii_control())
    {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidApiBase {
                path: path.to_string(),
                reason: "api_base contains invalid host characters".to_string(),
            },
            None,
        ));
    }

    let normalized = base.trim_end_matches('/').to_string();
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::raw::{RawGeneralSettings, RawLiteLlmParams, RawModelEntry};
    use std::collections::HashMap;

    fn mock_env() -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("MASTER_KEY".to_string(), "test-master-key".to_string());
        env.insert("OPENAI_KEY".to_string(), "test-openai-key".to_string());
        env.insert("MISTRAL_KEY".to_string(), "test-mistral-key".to_string());
        env.insert(
            "ANTHROPIC_KEY".to_string(),
            "test-anthropic-key".to_string(),
        );
        env.insert("GEMINI_KEY".to_string(), "test-gemini-key".to_string());
        env
    }

    #[test]
    fn test_route_grouping_preserves_order_and_duplicates() {
        let env = mock_env();
        let raw = RawConfig {
            general_settings: Some(RawGeneralSettings {
                master_key: Some("os.environ/MASTER_KEY".to_string()),
                request_timeout: None,
                overall_timeout: None,
                max_in_flight: None,
            }),
            model_list: vec![
                RawModelEntry {
                    model_name: "default".to_string(),
                    litellm_params: RawLiteLlmParams {
                        model: "mistral/mistral-small".to_string(),
                        api_key: Some("os.environ/MISTRAL_KEY".to_string()),
                        api_base: None,
                        timeout: None,
                    },
                },
                RawModelEntry {
                    model_name: "fast".to_string(),
                    litellm_params: RawLiteLlmParams {
                        model: "openai/gpt-4o-mini".to_string(),
                        api_key: Some("os.environ/OPENAI_KEY".to_string()),
                        api_base: None,
                        timeout: Some(10.0),
                    },
                },
                RawModelEntry {
                    model_name: "default".to_string(), // 2nd target for "default"
                    litellm_params: RawLiteLlmParams {
                        model: "gemini/gemini-2.5-flash".to_string(),
                        api_key: Some("os.environ/GEMINI_KEY".to_string()),
                        api_base: None,
                        timeout: None,
                    },
                },
                RawModelEntry {
                    model_name: "default".to_string(), // repeated identical target (3rd target)
                    litellm_params: RawLiteLlmParams {
                        model: "mistral/mistral-small".to_string(),
                        api_key: Some("os.environ/MISTRAL_KEY".to_string()),
                        api_base: None,
                        timeout: None,
                    },
                },
            ],
        };

        let runtime = build_runtime_config(raw, false, &env).expect("should build successfully");
        assert_eq!(runtime.routes.len(), 2);
        assert_eq!(runtime.routes[0].model_name, "default");
        assert_eq!(runtime.routes[1].model_name, "fast");

        // "default" route must have 3 targets in exact file order
        assert_eq!(runtime.routes[0].targets.len(), 3);
        assert_eq!(runtime.routes[0].targets[0].model, "mistral/mistral-small");
        assert_eq!(runtime.routes[0].targets[0].provider, ProviderKind::Mistral);
        assert_eq!(runtime.routes[0].targets[0].timeout, 30); // default
        assert_eq!(
            runtime.routes[0].targets[1].model,
            "gemini/gemini-2.5-flash"
        );
        assert_eq!(runtime.routes[0].targets[1].provider, ProviderKind::Gemini);
        assert_eq!(runtime.routes[0].targets[2].model, "mistral/mistral-small");

        // "fast" route must have 1 target
        assert_eq!(runtime.routes[1].targets.len(), 1);
        assert_eq!(runtime.routes[1].targets[0].model, "openai/gpt-4o-mini");
        assert_eq!(runtime.routes[1].targets[0].timeout, 10);
    }

    #[test]
    fn test_case_sensitive_model_names() {
        let env = mock_env();
        let raw = RawConfig {
            general_settings: Some(RawGeneralSettings {
                master_key: Some("os.environ/MASTER_KEY".to_string()),
                request_timeout: None,
                overall_timeout: None,
                max_in_flight: None,
            }),
            model_list: vec![
                RawModelEntry {
                    model_name: "gpt-4".to_string(),
                    litellm_params: RawLiteLlmParams {
                        model: "openai/gpt-4".to_string(),
                        api_key: Some("os.environ/OPENAI_KEY".to_string()),
                        api_base: None,
                        timeout: None,
                    },
                },
                RawModelEntry {
                    model_name: "GPT-4".to_string(),
                    litellm_params: RawLiteLlmParams {
                        model: "openai/gpt-4".to_string(),
                        api_key: Some("os.environ/OPENAI_KEY".to_string()),
                        api_base: None,
                        timeout: None,
                    },
                },
            ],
        };

        let runtime = build_runtime_config(raw, false, &env).expect("should build successfully");
        assert_eq!(runtime.routes.len(), 2);
        assert_eq!(runtime.routes[0].model_name, "gpt-4");
        assert_eq!(runtime.routes[1].model_name, "GPT-4");
    }

    #[test]
    fn test_defaults_in_runtime_config_with_no_auth_omitted_general_settings() {
        let env = mock_env();
        let raw = RawConfig {
            general_settings: None,
            model_list: vec![RawModelEntry {
                model_name: "local".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "openai_compatible/custom-model".to_string(),
                    api_key: None,
                    api_base: Some("http://localhost:8000/v1/".to_string()),
                    timeout: None,
                },
            }],
        };

        let runtime = build_runtime_config(raw, true, &env).expect("should build successfully");
        assert_eq!(runtime.general_settings.master_key, None);
        assert_eq!(runtime.general_settings.request_timeout, 30);
        assert_eq!(runtime.general_settings.overall_timeout, 120);
        assert_eq!(runtime.general_settings.max_in_flight, 64);

        let target = &runtime.routes[0].targets[0];
        assert_eq!(target.timeout, 30);
        assert_eq!(target.api_base, "http://localhost:8000/v1"); // normalized
        assert_eq!(target.api_key, None);
    }

    #[test]
    fn test_authenticated_mode_requires_general_settings() {
        let env = mock_env();
        let raw = RawConfig {
            general_settings: None,
            model_list: vec![RawModelEntry {
                model_name: "default".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("os.environ/OPENAI_KEY".to_string()),
                    api_base: None,
                    timeout: None,
                },
            }],
        };

        let err = build_runtime_config(raw, false, &env)
            .expect_err("should require general_settings in authenticated mode");
        assert!(matches!(
            err.kind,
            ConfigErrorKind::MissingGeneralSettings { .. }
        ));
    }

    #[test]
    fn test_authenticated_mode_requires_master_key() {
        let env = mock_env();
        let raw = RawConfig {
            general_settings: Some(RawGeneralSettings {
                master_key: None,
                request_timeout: None,
                overall_timeout: None,
                max_in_flight: None,
            }),
            model_list: vec![RawModelEntry {
                model_name: "default".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("os.environ/OPENAI_KEY".to_string()),
                    api_base: None,
                    timeout: None,
                },
            }],
        };

        let err = build_runtime_config(raw, false, &env)
            .expect_err("should require master_key in authenticated mode");
        assert!(matches!(err.kind, ConfigErrorKind::MissingMasterKey { .. }));
    }

    #[test]
    fn test_no_auth_with_master_key_resolves_and_validates() {
        let mut env = mock_env();
        env.remove("MASTER_KEY"); // Missing env var

        let raw = RawConfig {
            general_settings: Some(RawGeneralSettings {
                master_key: Some("os.environ/MASTER_KEY".to_string()),
                request_timeout: None,
                overall_timeout: None,
                max_in_flight: None,
            }),
            model_list: vec![RawModelEntry {
                model_name: "default".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "openai_compatible/custom".to_string(),
                    api_key: None,
                    api_base: Some("http://localhost:8000".to_string()),
                    timeout: None,
                },
            }],
        };

        let err = build_runtime_config(raw, true, &env)
            .expect_err("master_key present under --no-auth must still resolve and validate");
        assert!(matches!(err.kind, ConfigErrorKind::MissingEnvVar { .. }));
    }

    #[test]
    fn test_branded_provider_requires_api_key() {
        let env = mock_env();
        let raw = RawConfig {
            general_settings: None,
            model_list: vec![RawModelEntry {
                model_name: "default".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "anthropic/claude-3-5-sonnet".to_string(),
                    api_key: None, // Missing
                    api_base: None,
                    timeout: None,
                },
            }],
        };

        let err = build_runtime_config(raw, true, &env)
            .expect_err("branded provider must require api_key");
        assert!(matches!(err.kind, ConfigErrorKind::MissingApiKey { .. }));
    }

    #[test]
    fn test_branded_provider_requires_https_api_base() {
        let env = mock_env();
        let raw = RawConfig {
            general_settings: None,
            model_list: vec![RawModelEntry {
                model_name: "default".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("os.environ/OPENAI_KEY".to_string()),
                    api_base: Some("http://api.openai.com/v1".to_string()), // http plaintext!
                    timeout: None,
                },
            }],
        };

        let err = build_runtime_config(raw, true, &env)
            .expect_err("branded provider must reject http plaintext api_base");
        assert!(matches!(err.kind, ConfigErrorKind::InvalidApiBase { .. }));
    }

    #[test]
    fn test_keyed_openai_compatible_requires_https_api_base() {
        let env = mock_env();
        let raw = RawConfig {
            general_settings: None,
            model_list: vec![RawModelEntry {
                model_name: "default".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "openai_compatible/model".to_string(),
                    api_key: Some("os.environ/OPENAI_KEY".to_string()),
                    api_base: Some("http://custom.example.com".to_string()), // http with key!
                    timeout: None,
                },
            }],
        };

        let err = build_runtime_config(raw, true, &env)
            .expect_err("keyed target must reject http plaintext api_base");
        assert!(matches!(err.kind, ConfigErrorKind::InvalidApiBase { .. }));
    }

    #[test]
    fn test_gemini_suffix_validation() {
        let env = mock_env();
        // Valid gemini suffix
        let raw_valid = RawConfig {
            general_settings: None,
            model_list: vec![RawModelEntry {
                model_name: "default".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "gemini/gemini-2.5-flash_preview-001".to_string(),
                    api_key: Some("os.environ/GEMINI_KEY".to_string()),
                    api_base: None,
                    timeout: None,
                },
            }],
        };
        assert!(build_runtime_config(raw_valid, true, &env).is_ok());

        // Invalid gemini suffix (starts with hyphen)
        let raw_invalid = RawConfig {
            general_settings: None,
            model_list: vec![RawModelEntry {
                model_name: "default".to_string(),
                litellm_params: RawLiteLlmParams {
                    model: "gemini/-invalid".to_string(),
                    api_key: Some("os.environ/GEMINI_KEY".to_string()),
                    api_base: None,
                    timeout: None,
                },
            }],
        };
        let err = build_runtime_config(raw_invalid, true, &env)
            .expect_err("Gemini suffix starting with hyphen must fail");
        assert!(matches!(
            err.kind,
            ConfigErrorKind::InvalidTargetModel { .. }
        ));
    }
}
