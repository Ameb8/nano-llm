use crate::config::error::ConfigError;
use crate::config::yaml::parse_yaml_str;

/// Raw deserialized representation of the top-level configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RawConfig {
    pub model_list: Vec<RawModelEntry>,
    pub general_settings: Option<RawGeneralSettings>,
}

impl RawConfig {
    /// Parse and validate a YAML string against the restricted v0.1 subset.
    pub fn parse_str(yaml: &str) -> Result<Self, ConfigError> {
        parse_yaml_str(yaml)
    }
}

/// Raw entry within `model_list`.
#[derive(Debug, Clone, PartialEq)]
pub struct RawModelEntry {
    pub model_name: String,
    pub litellm_params: RawLiteLlmParams,
}

/// Raw LiteLLM provider parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct RawLiteLlmParams {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub timeout: Option<f64>,
}

/// Raw general settings section.
#[derive(Debug, Clone, PartialEq)]
pub struct RawGeneralSettings {
    pub master_key: Option<String>,
    pub request_timeout: Option<f64>,
    pub overall_timeout: Option<f64>,
    pub max_in_flight: Option<usize>,
}
