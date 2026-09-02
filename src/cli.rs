use clap::Parser;
use std::fmt;
use std::net::SocketAddr;
use std::path::PathBuf;

/// Default socket address for nano-llm.
pub const DEFAULT_BIND_ADDR: &str = "127.0.0.1:4000";

/// Development build indicator.
pub const IS_DEV_BUILD: bool = true;

/// nano-llm command-line interface.
#[derive(Debug, Clone, Parser, PartialEq, Eq)]
#[command(
    name = "nano-llm",
    about = "A lightweight, single-binary LLM routing gateway",
    version = env!("CARGO_PKG_VERSION")
)]
pub struct Cli {
    /// Path to the YAML configuration file
    #[arg(long, value_name = "PATH", required = true)]
    pub config: PathBuf,

    /// Socket address to bind to
    #[arg(long, value_name = "ADDRESS", default_value = DEFAULT_BIND_ADDR)]
    pub bind: SocketAddr,

    /// Disable inbound bearer token authentication (loopback bind only)
    #[arg(long)]
    pub no_auth: bool,

    /// Validate configuration, print redacted routing table, and exit
    #[arg(long)]
    pub validate: bool,
}

/// Errors that can occur during CLI argument parsing and validation.
#[derive(Debug)]
pub enum CliError {
    /// Error during argument parsing by clap.
    Clap(clap::Error),
    /// `--no-auth` specified with a non-loopback bind address.
    NoAuthNonLoopback(SocketAddr),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clap(err) => write!(f, "{err}"),
            Self::NoAuthNonLoopback(addr) => {
                write!(
                    f,
                    "error: --no-auth is for local development only and requires a loopback bind address (got {addr})"
                )
            }
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Clap(err) => Some(err),
            Self::NoAuthNonLoopback(_) => None,
        }
    }
}

impl From<clap::Error> for CliError {
    fn from(err: clap::Error) -> Self {
        Self::Clap(err)
    }
}

impl Cli {
    /// Validate semantic rules that depend on multiple CLI flags.
    pub fn validate_rules(&self) -> Result<(), CliError> {
        if self.no_auth && !self.bind.ip().is_loopback() {
            return Err(CliError::NoAuthNonLoopback(self.bind));
        }
        Ok(())
    }

    /// Parse CLI arguments from an iterator and validate semantic rules.
    pub fn parse_and_validate<I, T>(itr: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let cli = Self::try_parse_from(itr)?;
        cli.validate_rules()?;
        Ok(cli)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr, SocketAddrV4, SocketAddrV6};

    #[test]
    fn test_valid_minimal_args() {
        let args = ["nano-llm", "--config", "config.yaml"];
        let cli = Cli::parse_and_validate(args).expect("should parse successfully");
        assert_eq!(cli.config, PathBuf::from("config.yaml"));
        assert_eq!(
            cli.bind,
            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 4000))
        );
        assert!(!cli.no_auth);
        assert!(!cli.validate);
    }

    #[test]
    fn test_custom_bind_ipv4() {
        let args = [
            "nano-llm",
            "--config",
            "config.yaml",
            "--bind",
            "0.0.0.0:8080",
        ];
        let cli = Cli::parse_and_validate(args).expect("should parse successfully");
        assert_eq!(
            cli.bind,
            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(0, 0, 0, 0), 8080))
        );
    }

    #[test]
    fn test_custom_bind_ipv6() {
        let args = [
            "nano-llm",
            "--config",
            "config.yaml",
            "--bind",
            "[::1]:4000",
        ];
        let cli = Cli::parse_and_validate(args).expect("should parse successfully");
        assert_eq!(
            cli.bind,
            SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::LOCALHOST, 4000, 0, 0))
        );
    }

    #[test]
    fn test_validate_flag() {
        let args = ["nano-llm", "--config", "config.yaml", "--validate"];
        let cli = Cli::parse_and_validate(args).expect("should parse successfully");
        assert!(cli.validate);
    }

    #[test]
    fn test_no_auth_on_loopback_default_bind() {
        let args = ["nano-llm", "--config", "config.yaml", "--no-auth"];
        let cli = Cli::parse_and_validate(args).expect("should parse successfully");
        assert!(cli.no_auth);
        assert!(cli.bind.ip().is_loopback());
    }

    #[test]
    fn test_no_auth_on_loopback_ipv4() {
        let args = [
            "nano-llm",
            "--config",
            "config.yaml",
            "--bind",
            "127.0.0.2:5000",
            "--no-auth",
        ];
        let cli = Cli::parse_and_validate(args).expect("should parse successfully");
        assert!(cli.no_auth);
        assert_eq!(
            cli.bind,
            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 2), 5000))
        );
    }

    #[test]
    fn test_no_auth_on_loopback_ipv6() {
        let args = [
            "nano-llm",
            "--config",
            "config.yaml",
            "--bind",
            "[::1]:5000",
            "--no-auth",
        ];
        let cli = Cli::parse_and_validate(args).expect("should parse successfully");
        assert!(cli.no_auth);
        assert!(cli.bind.ip().is_loopback());
    }

    #[test]
    fn test_no_auth_rejected_on_non_loopback_ipv4() {
        let args = [
            "nano-llm",
            "--config",
            "config.yaml",
            "--bind",
            "0.0.0.0:4000",
            "--no-auth",
        ];
        let result = Cli::parse_and_validate(args);
        assert!(result.is_err());
        match result.unwrap_err() {
            CliError::NoAuthNonLoopback(addr) => {
                assert_eq!(
                    addr,
                    SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(0, 0, 0, 0), 4000))
                );
            }
            other => panic!("expected NoAuthNonLoopback, got {other:?}"),
        }
    }

    #[test]
    fn test_no_auth_rejected_on_routable_ip() {
        let args = [
            "nano-llm",
            "--config",
            "config.yaml",
            "--bind",
            "192.168.1.100:4000",
            "--no-auth",
        ];
        let result = Cli::parse_and_validate(args);
        assert!(result.is_err());
        match result.unwrap_err() {
            CliError::NoAuthNonLoopback(addr) => {
                assert_eq!(
                    addr,
                    SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(192, 168, 1, 100), 4000))
                );
            }
            other => panic!("expected NoAuthNonLoopback, got {other:?}"),
        }
    }

    #[test]
    fn test_missing_config_fails() {
        let args = ["nano-llm"];
        let result = Cli::parse_and_validate(args);
        assert!(result.is_err());
        match result.unwrap_err() {
            CliError::Clap(err) => {
                assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
            }
            other => panic!("expected MissingRequiredArgument, got {other:?}"),
        }
    }

    #[test]
    fn test_unknown_flags_fail() {
        let unknown_flags = [
            "--timeout",
            "--model",
            "--api-key",
            "--provider",
            "--port",
            "--host",
            "--master-key",
            "--route",
        ];

        for flag in unknown_flags {
            let args = ["nano-llm", "--config", "config.yaml", flag, "value"];
            let result = Cli::parse_and_validate(args);
            assert!(result.is_err(), "flag {flag} should be rejected as unknown");
            match result.unwrap_err() {
                CliError::Clap(err) => {
                    assert_eq!(
                        err.kind(),
                        clap::error::ErrorKind::UnknownArgument,
                        "flag {flag} should produce UnknownArgument error"
                    );
                }
                other => panic!("expected UnknownArgument for {flag}, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_invalid_bind_address_fails() {
        let invalid_binds = [
            "not-an-ip",
            "127.0.0.1",            // missing port
            ":4000",                // missing ip
            "999.999.999.999:4000", // invalid ip
            "127.0.0.1:99999",      // invalid port
        ];

        for bind in invalid_binds {
            let args = ["nano-llm", "--config", "config.yaml", "--bind", bind];
            let result = Cli::parse_and_validate(args);
            assert!(
                result.is_err(),
                "bind address {bind} should be rejected as invalid"
            );
            match result.unwrap_err() {
                CliError::Clap(err) => {
                    assert_eq!(
                        err.kind(),
                        clap::error::ErrorKind::ValueValidation,
                        "bind {bind} should produce ValueValidation error"
                    );
                }
                other => panic!("expected ValueValidation for {bind}, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_dev_build_identity() {
        let version = env!("CARGO_PKG_VERSION");
        assert!(
            version.contains("-dev"),
            "development build package version must contain -dev suffix (got {version})"
        );
        assert_eq!(version, "0.1.0-dev");
        assert!(crate::is_dev_build());
    }
}
