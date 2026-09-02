use std::fmt;

/// Represents a 1-indexed source location in a configuration file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigLocation {
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for ConfigLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

/// The specific category of configuration syntax or validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigErrorKind {
    /// The YAML document was empty or contained only comments/whitespace.
    EmptyDocument,
    /// More than one YAML document was present in the configuration stream.
    MultipleDocuments,
    /// A YAML anchor (`&anchor`) was encountered, which is prohibited.
    ProhibitedAnchor,
    /// A YAML alias (`*alias`) was encountered, which is prohibited.
    ProhibitedAlias,
    /// A YAML merge key (`<<`) was encountered, which is prohibited.
    ProhibitedMergeKey,
    /// A YAML tag (`!tag`, `!!type`) was encountered, which is prohibited.
    ProhibitedTag { tag: String },
    /// A non-string key was used in a mapping (e.g. sequence, mapping, int, bool, null).
    NonStringKey { found: String },
    /// A duplicate key was defined in a mapping at the given path.
    DuplicateKey { path: String },
    /// An unknown key was present in a mapping at the given path.
    UnknownKey { path: String, key: String },
    /// A field value had an unexpected type.
    InvalidType {
        path: String,
        expected: &'static str,
        found: String,
    },
    /// A required field was missing at the given path.
    MissingField { path: String, field: &'static str },
    /// A lower-level YAML scan/syntax error.
    SyntaxError { message: String },
    /// `model_list` is empty.
    EmptyModelList { path: String },
    /// `model_name` failed validation.
    InvalidModelName {
        path: String,
        model_name: String,
        reason: String,
    },
    /// Target `model` failed validation.
    InvalidTargetModel {
        path: String,
        model: String,
        reason: String,
    },
    /// A branded provider is missing its required `api_key`.
    MissingApiKey { path: String, provider: String },
    /// `openai_compatible/` provider is missing its required `api_base`.
    MissingApiBase { path: String, provider: String },
    /// `api_base` URL is invalid or violates protocol rules.
    InvalidApiBase { path: String, reason: String },
    /// Timeout is out of range or not an integer whole seconds.
    InvalidTimeout { path: String, reason: String },
    /// Max in flight is out of range or not an integer.
    InvalidMaxInFlight { path: String, reason: String },
    /// A secret field is not an `os.environ/VAR_NAME` reference.
    InvalidSecretReference { path: String, reason: String },
    /// An environment variable referenced by a secret field was not found.
    MissingEnvVar { path: String, var_name: String },
    /// A resolved secret value failed validation (empty, control chars, header encoding).
    InvalidSecretValue {
        path: String,
        var_name: String,
        reason: String,
    },
    /// `general_settings` is missing during authenticated operation.
    MissingGeneralSettings { path: String },
    /// `general_settings.master_key` is missing during authenticated operation.
    MissingMasterKey { path: String },
}

/// A configuration parsing or validation error with optional source location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    pub kind: ConfigErrorKind,
    pub location: Option<ConfigLocation>,
}

impl ConfigError {
    pub fn new(kind: ConfigErrorKind, location: Option<ConfigLocation>) -> Self {
        Self { kind, location }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let loc_suffix = match &self.location {
            Some(loc) => format!(" at {}", loc),
            None => String::new(),
        };

        match &self.kind {
            ConfigErrorKind::EmptyDocument => {
                write!(f, "configuration must contain exactly one YAML document (input is empty){loc_suffix}")
            }
            ConfigErrorKind::MultipleDocuments => {
                write!(f, "multiple YAML documents are prohibited{loc_suffix}")
            }
            ConfigErrorKind::ProhibitedAnchor => {
                write!(f, "YAML anchors ('&...') are prohibited{loc_suffix}")
            }
            ConfigErrorKind::ProhibitedAlias => {
                write!(f, "YAML aliases ('*...') are prohibited{loc_suffix}")
            }
            ConfigErrorKind::ProhibitedMergeKey => {
                write!(f, "YAML merge keys ('<<') are prohibited{loc_suffix}")
            }
            ConfigErrorKind::ProhibitedTag { tag } => {
                write!(f, "YAML tags ('{tag}') are prohibited{loc_suffix}")
            }
            ConfigErrorKind::NonStringKey { found } => {
                write!(
                    f,
                    "non-string mapping key ({found}) is prohibited{loc_suffix}"
                )
            }
            ConfigErrorKind::DuplicateKey { path } => {
                write!(f, "duplicate key '{path}'{loc_suffix}")
            }
            ConfigErrorKind::UnknownKey { path, key } => {
                if path.is_empty() {
                    write!(f, "unknown top-level key '{key}'{loc_suffix}")
                } else {
                    write!(f, "unknown key '{key}' at '{path}'{loc_suffix}")
                }
            }
            ConfigErrorKind::InvalidType {
                path,
                expected,
                found,
            } => {
                if path.is_empty() {
                    write!(f, "expected {expected}, found {found}{loc_suffix}")
                } else {
                    write!(
                        f,
                        "expected {expected} at '{path}', found {found}{loc_suffix}"
                    )
                }
            }
            ConfigErrorKind::MissingField { path, field } => {
                if path.is_empty() {
                    write!(f, "missing required field '{field}'{loc_suffix}")
                } else {
                    write!(
                        f,
                        "missing required field '{field}' at '{path}'{loc_suffix}"
                    )
                }
            }
            ConfigErrorKind::SyntaxError { message } => {
                write!(f, "YAML syntax error: {message}{loc_suffix}")
            }
            ConfigErrorKind::EmptyModelList { path } => {
                write!(f, "{path} must contain at least one entry{loc_suffix}")
            }
            ConfigErrorKind::InvalidModelName {
                path,
                model_name,
                reason,
            } => {
                write!(
                    f,
                    "invalid model_name '{model_name}' at '{path}': {reason}{loc_suffix}"
                )
            }
            ConfigErrorKind::InvalidTargetModel {
                path,
                model,
                reason,
            } => {
                write!(
                    f,
                    "invalid model '{model}' at '{path}': {reason}{loc_suffix}"
                )
            }
            ConfigErrorKind::MissingApiKey { path, provider } => {
                write!(
                    f,
                    "provider '{provider}' at '{path}' requires an api_key{loc_suffix}"
                )
            }
            ConfigErrorKind::MissingApiBase { path, provider } => {
                write!(
                    f,
                    "provider '{provider}' at '{path}' requires an api_base{loc_suffix}"
                )
            }
            ConfigErrorKind::InvalidApiBase { path, reason } => {
                write!(f, "invalid api_base at '{path}': {reason}{loc_suffix}")
            }
            ConfigErrorKind::InvalidTimeout { path, reason } => {
                write!(f, "invalid timeout at '{path}': {reason}{loc_suffix}")
            }
            ConfigErrorKind::InvalidMaxInFlight { path, reason } => {
                write!(f, "invalid max_in_flight at '{path}': {reason}{loc_suffix}")
            }
            ConfigErrorKind::InvalidSecretReference { path, reason } => {
                write!(
                    f,
                    "invalid secret reference at '{path}': {reason}{loc_suffix}"
                )
            }
            ConfigErrorKind::MissingEnvVar { path, var_name } => {
                write!(
                    f,
                    "missing environment variable '{var_name}' referenced at '{path}'{loc_suffix}"
                )
            }
            ConfigErrorKind::InvalidSecretValue {
                path,
                var_name,
                reason,
            } => {
                write!(
                    f,
                    "invalid secret for environment variable '{var_name}' at '{path}': {reason}{loc_suffix}"
                )
            }
            ConfigErrorKind::MissingGeneralSettings { path } => {
                write!(
                    f,
                    "general_settings is required during authenticated operation at '{path}'{loc_suffix}"
                )
            }
            ConfigErrorKind::MissingMasterKey { path } => {
                write!(
                    f,
                    "master_key is required during authenticated operation at '{path}'{loc_suffix}"
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}
