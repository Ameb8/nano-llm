use crate::config::error::{ConfigError, ConfigErrorKind};
use std::collections::HashMap;
use std::fmt;

/// An environment variable provider abstraction for deterministic testing.
pub trait EnvProvider {
    fn get_var(&self, key: &str) -> Option<String>;

    /// Looks up a variable while preserving a non-UTF-8 environment value as a
    /// distinct error. Custom test providers only need to implement `get_var`.
    fn get_utf8_var(&self, key: &str) -> Result<Option<String>, EnvLookupError> {
        Ok(self.get_var(key))
    }
}

/// The only lookup failure that must be distinguished from an absent variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvLookupError {
    NotUtf8,
}

/// Real system environment variable provider.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemEnv;

impl EnvProvider for SystemEnv {
    fn get_var(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }

    fn get_utf8_var(&self, key: &str) -> Result<Option<String>, EnvLookupError> {
        match std::env::var(key) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(EnvLookupError::NotUtf8),
        }
    }
}

impl EnvProvider for HashMap<String, String> {
    fn get_var(&self, key: &str) -> Option<String> {
        self.get(key).cloned()
    }
}

/// A wrapper for sensitive string values that prevents accidental exposure in logs and display.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(secret: String) -> Self {
        Self(secret)
    }

    /// Exposes the underlying secret value for authorized cryptographic/HTTP use.
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[REDACTED]")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[REDACTED]")
    }
}

/// Resolves a secret reference of the form `os.environ/VAR_NAME` and validates its content.
pub fn resolve_secret<E: EnvProvider>(
    raw_reference: &str,
    path: &str,
    is_http_header_transport: bool,
    env: &E,
) -> Result<SecretString, ConfigError> {
    let var_name = match raw_reference.strip_prefix("os.environ/") {
        Some(name) => name,
        None => {
            return Err(ConfigError::new(
                ConfigErrorKind::InvalidSecretReference {
                    path: path.to_string(),
                    reason: "inline secret literals are prohibited; must use 'os.environ/VAR_NAME'"
                        .to_string(),
                },
                None,
            ));
        }
    };

    if !is_valid_environment_name(var_name) {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidSecretReference {
                path: path.to_string(),
                reason: "environment variable name must match [A-Za-z_][A-Za-z0-9_]*".to_string(),
            },
            None,
        ));
    }

    let secret_val = env
        .get_utf8_var(var_name)
        .map_err(|EnvLookupError::NotUtf8| {
            ConfigError::new(
                ConfigErrorKind::InvalidSecretValue {
                    path: path.to_string(),
                    var_name: var_name.to_string(),
                    reason: "resolved secret value is not valid UTF-8".to_string(),
                },
                None,
            )
        })?
        .ok_or_else(|| {
            ConfigError::new(
                ConfigErrorKind::MissingEnvVar {
                    path: path.to_string(),
                    var_name: var_name.to_string(),
                },
                None,
            )
        })?;

    if secret_val.is_empty() {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidSecretValue {
                path: path.to_string(),
                var_name: var_name.to_string(),
                reason: "resolved secret value must not be empty".to_string(),
            },
            None,
        ));
    }

    if secret_val.chars().any(|c| c.is_ascii_control()) {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidSecretValue {
                path: path.to_string(),
                var_name: var_name.to_string(),
                reason: "resolved secret value contains ASCII control characters".to_string(),
            },
            None,
        ));
    }

    if is_http_header_transport && !secret_val.is_ascii() {
        return Err(ConfigError::new(
            ConfigErrorKind::InvalidSecretValue {
                path: path.to_string(),
                var_name: var_name.to_string(),
                reason: "resolved secret value cannot be encoded in HTTP header credential transport (contains non-ASCII characters)".to_string(),
            },
            None,
        ));
    }

    Ok(SecretString::new(secret_val))
}

fn is_valid_environment_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_redaction_debug_and_display() {
        let secret = SecretString::new("super-secret-12345".to_string());
        assert_eq!(format!("{secret}"), "[REDACTED]");
        assert_eq!(format!("{secret:?}"), "[REDACTED]");
        assert_eq!(secret.expose_secret(), "super-secret-12345");
    }

    #[test]
    fn test_resolve_secret_success() {
        let mut env = HashMap::new();
        env.insert("MY_KEY".to_string(), "sk-valid-key-123".to_string());

        let resolved = resolve_secret("os.environ/MY_KEY", "test.path", true, &env)
            .expect("should resolve successfully");
        assert_eq!(resolved.expose_secret(), "sk-valid-key-123");
    }

    #[test]
    fn test_resolve_secret_inline_literal_rejected() {
        let env = HashMap::new();
        let err = resolve_secret("sk-inline-literal", "test.path", true, &env)
            .expect_err("should reject inline secret");
        assert!(matches!(
            err.kind,
            ConfigErrorKind::InvalidSecretReference { .. }
        ));
    }

    #[test]
    fn test_resolve_secret_empty_var_name_rejected() {
        let env = HashMap::new();
        let err = resolve_secret("os.environ/", "test.path", true, &env)
            .expect_err("should reject empty var name");
        assert!(matches!(
            err.kind,
            ConfigErrorKind::InvalidSecretReference { .. }
        ));
    }

    #[test]
    fn test_resolve_secret_rejects_non_exact_environment_reference() {
        let env = HashMap::new();
        for reference in [
            "os.environ/KEY/extra",
            "os.environ/1KEY",
            "os.environ/KEY.NAME",
        ] {
            let err = resolve_secret(reference, "test.path", true, &env)
                .expect_err("invalid environment reference must fail");
            assert!(matches!(
                err.kind,
                ConfigErrorKind::InvalidSecretReference { .. }
            ));
        }
    }

    #[test]
    fn test_resolve_secret_missing_env_var() {
        let env = HashMap::new();
        let err = resolve_secret("os.environ/NON_EXISTENT", "test.path", true, &env)
            .expect_err("should fail on missing env var");
        match err.kind {
            ConfigErrorKind::MissingEnvVar { var_name, path } => {
                assert_eq!(var_name, "NON_EXISTENT");
                assert_eq!(path, "test.path");
            }
            _ => panic!("unexpected error kind"),
        }
    }

    #[test]
    fn test_resolve_secret_empty_value_rejected() {
        let mut env = HashMap::new();
        env.insert("EMPTY_VAR".to_string(), "".to_string());
        let err = resolve_secret("os.environ/EMPTY_VAR", "test.path", true, &env)
            .expect_err("should reject empty secret");
        assert!(matches!(
            err.kind,
            ConfigErrorKind::InvalidSecretValue { .. }
        ));
    }

    #[test]
    fn test_resolve_secret_control_chars_rejected() {
        let mut env = HashMap::new();
        env.insert("CONTROL_VAR".to_string(), "bad\nsecret".to_string());
        let err = resolve_secret("os.environ/CONTROL_VAR", "test.path", true, &env)
            .expect_err("should reject control chars");
        assert!(matches!(
            err.kind,
            ConfigErrorKind::InvalidSecretValue { .. }
        ));
    }

    #[test]
    fn test_resolve_secret_non_ascii_header_transport_rejected() {
        let mut env = HashMap::new();
        env.insert("UNICODE_VAR".to_string(), "key-with-🔑".to_string());
        let err = resolve_secret("os.environ/UNICODE_VAR", "test.path", true, &env)
            .expect_err("should reject non-ascii for header transport");
        assert!(matches!(
            err.kind,
            ConfigErrorKind::InvalidSecretValue { .. }
        ));
    }

    #[test]
    fn test_resolve_secret_non_ascii_query_transport_accepted() {
        let mut env = HashMap::new();
        env.insert("UNICODE_VAR".to_string(), "gemini-key-🔑".to_string());
        let res = resolve_secret("os.environ/UNICODE_VAR", "test.path", false, &env)
            .expect("should allow non-ascii for query transport (Gemini)");
        assert_eq!(res.expose_secret(), "gemini-key-🔑");
    }

    #[test]
    fn test_secret_failures_name_the_path_and_variable_without_leaking_value() {
        let secret = "do-not-expose-this-secret";
        let mut env = HashMap::new();
        env.insert("SECRET_KEY".to_string(), format!("{secret}\n"));

        let err = resolve_secret(
            "os.environ/SECRET_KEY",
            "model_list[0].litellm_params.api_key",
            true,
            &env,
        )
        .expect_err("control character must be rejected");
        let diagnostic = err.to_string();
        assert!(diagnostic.contains("model_list[0].litellm_params.api_key"));
        assert!(diagnostic.contains("SECRET_KEY"));
        assert!(!diagnostic.contains(secret));

        let inline = resolve_secret(
            "do-not-expose-this-secret",
            "general_settings.master_key",
            true,
            &env,
        )
        .expect_err("inline secret must be rejected");
        assert!(!inline.to_string().contains(secret));
    }
}
