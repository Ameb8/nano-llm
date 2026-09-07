use crate::config::secrets::SecretString;
use std::fmt;

/// The supported provider adapter types and branded presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    OpenAi,
    Mistral,
    DeepSeek,
    OpenAiCompatible,
    Anthropic,
    Gemini,
}

impl ProviderKind {
    /// Returns true if this is a branded cloud preset.
    pub fn is_branded(&self) -> bool {
        !matches!(self, Self::OpenAiCompatible)
    }

    /// Returns the preset's default API base URL, if one exists.
    pub fn default_api_base(&self) -> Option<&'static str> {
        match self {
            Self::OpenAi => Some("https://api.openai.com/v1"),
            Self::Mistral => Some("https://api.mistral.ai/v1"),
            Self::DeepSeek => Some("https://api.deepseek.com"),
            Self::Anthropic => Some("https://api.anthropic.com/v1"),
            Self::Gemini => Some("https://generativelanguage.googleapis.com/v1beta"),
            Self::OpenAiCompatible => None,
        }
    }

    /// Returns true if credentials are transported via HTTP headers.
    pub fn is_http_header_transport(&self) -> bool {
        !matches!(self, Self::Gemini)
    }

    /// Returns the provider identifier string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Mistral => "mistral",
            Self::DeepSeek => "deepseek",
            Self::OpenAiCompatible => "openai_compatible",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        }
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A fully resolved and validated fallback target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeTarget {
    /// Full model specification string (e.g. `openai/gpt-4o`).
    pub model: String,
    /// Provider adapter kind.
    pub provider: ProviderKind,
    /// Upstream model identifier suffix.
    pub model_suffix: String,
    /// Resolved API key, if present.
    pub api_key: Option<SecretString>,
    /// Normalized provider base URL.
    pub api_base: String,
    /// Effective request timeout in seconds (target override or global default).
    pub timeout: u64,
    /// Explicit target timeout in seconds, if specified.
    pub explicit_timeout: Option<u64>,
}

/// An ordered fallback route associated with a public model name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeRoute {
    /// Case-sensitive public model name.
    pub model_name: String,
    /// Ordered fallback targets preserving configuration file order.
    pub targets: Vec<RuntimeTarget>,
}

/// Fully resolved and defaulted general settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeGeneralSettings {
    /// Resolved gateway master key for inbound authentication.
    pub master_key: Option<SecretString>,
    /// Global default per-request timeout in seconds (default 30).
    pub request_timeout: u64,
    /// Overall fallback chain timeout in seconds (default 120).
    pub overall_timeout: u64,
    /// Process-wide concurrent request capacity cap (default 64).
    pub max_in_flight: usize,
}

impl Default for RuntimeGeneralSettings {
    fn default() -> Self {
        Self {
            master_key: None,
            request_timeout: 30,
            overall_timeout: 120,
            max_in_flight: 64,
        }
    }
}

/// The complete immutable runtime configuration model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// General settings.
    pub general_settings: RuntimeGeneralSettings,
    /// Ordered fallback routes preserving first appearance order of public model names.
    pub routes: Vec<RuntimeRoute>,
}

impl RuntimeConfig {
    /// Look up an ordered fallback route by its case-sensitive public model name.
    pub fn get_route(&self, model_name: &str) -> Option<&RuntimeRoute> {
        self.routes.iter().find(|r| r.model_name == model_name)
    }

    /// Format the resolved routing table with all secret values redacted.
    pub fn route_table_display(&self) -> String {
        let mut out = String::new();
        out.push_str("nano-llm resolved route table:\n");
        out.push_str("General Settings:\n");
        match &self.general_settings.master_key {
            Some(_) => out.push_str("  master_key: [REDACTED]\n"),
            None => out.push_str("  master_key: (none)\n"),
        }
        out.push_str(&format!(
            "  request_timeout: {}s\n",
            self.general_settings.request_timeout
        ));
        out.push_str(&format!(
            "  overall_timeout: {}s\n",
            self.general_settings.overall_timeout
        ));
        out.push_str(&format!(
            "  max_in_flight: {}\n\n",
            self.general_settings.max_in_flight
        ));

        let total_targets: usize = self.routes.iter().map(|r| r.targets.len()).sum();
        out.push_str(&format!(
            "Routes ({} public models, {} total targets):\n",
            self.routes.len(),
            total_targets
        ));

        for route in &self.routes {
            let target_plural = if route.targets.len() == 1 {
                "target"
            } else {
                "targets"
            };
            out.push_str(&format!(
                "  - model: {} ({} {})\n",
                route.model_name,
                route.targets.len(),
                target_plural
            ));
            for (idx, target) in route.targets.iter().enumerate() {
                out.push_str(&format!("    {}. {}\n", idx + 1, target.model));
                out.push_str(&format!("       provider: {}\n", target.provider));
                out.push_str(&format!("       api_base: {}\n", target.api_base));
                match &target.api_key {
                    Some(_) => out.push_str("       api_key: [REDACTED]\n"),
                    None => out.push_str("       api_key: (none)\n"),
                }
                out.push_str(&format!("       timeout: {}s\n", target.timeout));
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_table_display_redacts_secrets() {
        let config = RuntimeConfig {
            general_settings: RuntimeGeneralSettings {
                master_key: Some(SecretString::new("master-secret-value".to_string())),
                request_timeout: 30,
                overall_timeout: 120,
                max_in_flight: 64,
            },
            routes: vec![RuntimeRoute {
                model_name: "default".to_string(),
                targets: vec![RuntimeTarget {
                    model: "openai/gpt-4o".to_string(),
                    provider: ProviderKind::OpenAi,
                    model_suffix: "gpt-4o".to_string(),
                    api_key: Some(SecretString::new("target-secret-key".to_string())),
                    api_base: "https://api.openai.com/v1".to_string(),
                    timeout: 30,
                    explicit_timeout: None,
                }],
            }],
        };

        let display = config.route_table_display();
        assert!(!display.contains("master-secret-value"));
        assert!(!display.contains("target-secret-key"));
        assert!(display.contains("master_key: [REDACTED]"));
        assert!(display.contains("api_key: [REDACTED]"));
        assert!(display.contains("openai/gpt-4o"));
    }
}
