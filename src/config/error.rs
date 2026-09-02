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
    MissingField {
        path: String,
        field: &'static str,
    },
    /// A lower-level YAML scan/syntax error.
    SyntaxError { message: String },
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
                write!(f, "non-string mapping key ({found}) is prohibited{loc_suffix}")
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
            ConfigErrorKind::InvalidType { path, expected, found } => {
                if path.is_empty() {
                    write!(f, "expected {expected}, found {found}{loc_suffix}")
                } else {
                    write!(f, "expected {expected} at '{path}', found {found}{loc_suffix}")
                }
            }
            ConfigErrorKind::MissingField { path, field } => {
                if path.is_empty() {
                    write!(f, "missing required field '{field}'{loc_suffix}")
                } else {
                    write!(f, "missing required field '{field}' at '{path}'{loc_suffix}")
                }
            }
            ConfigErrorKind::SyntaxError { message } => {
                write!(f, "YAML syntax error: {message}{loc_suffix}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}
