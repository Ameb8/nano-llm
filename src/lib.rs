pub mod cli;
pub mod config;
pub mod request;

pub use cli::{Cli, CliError, DEFAULT_BIND_ADDR, IS_DEV_BUILD};
pub use config::{
    build_runtime_config, parse_yaml_str, ConfigError, ConfigErrorKind, ConfigLocation,
    EnvLookupError, EnvProvider, ProviderKind, RawConfig, RuntimeConfig, RuntimeGeneralSettings,
    RuntimeRoute, RuntimeTarget, SecretString, SystemEnv,
};
pub use request::{decode_chat_request, CanonicalRequest, DecodeError, JsonValue};

/// Returns the package version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns true if this binary is a development build.
pub fn is_dev_build() -> bool {
    IS_DEV_BUILD && version().contains("-dev")
}
