use nano_llm::cli::{Cli, CliError};
use nano_llm::config::{build_runtime_config, parse_yaml_str, SystemEnv};
use nano_llm::{app, serve};
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_ansi(false)
        .with_writer(std::io::stdout)
        .init();
    let cli = match Cli::parse_and_validate(std::env::args_os()) {
        Ok(cli) => cli,
        Err(CliError::Clap(err)) => {
            err.exit();
        }
        Err(CliError::NoAuthNonLoopback(addr)) => {
            eprintln!(
                "error: --no-auth is for local development only and requires a loopback bind address (got {addr})"
            );
            return ExitCode::from(2);
        }
    };

    let config_content = match fs::read_to_string(&cli.config) {
        Ok(content) => content,
        Err(err) => {
            eprintln!(
                "error: failed to read configuration file '{}': {err}",
                cli.config.display()
            );
            return ExitCode::from(1);
        }
    };

    let raw_config = match parse_yaml_str(&config_content) {
        Ok(raw) => raw,
        Err(err) => {
            eprintln!("error: configuration error: {err}");
            return ExitCode::from(1);
        }
    };

    let runtime_config = match build_runtime_config(raw_config, cli.no_auth, &SystemEnv) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("error: configuration error: {err}");
            return ExitCode::from(1);
        }
    };

    if cli.validate {
        print!("{}", runtime_config.route_table_display());
        ExitCode::SUCCESS
    } else {
        if cli.no_auth {
            eprintln!("WARNING: running with --no-auth; inbound authentication is disabled");
        }
        match serve(app(runtime_config, cli.no_auth), cli.bind) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("error: failed to serve HTTP on {}: {error}", cli.bind);
                ExitCode::from(1)
            }
        }
    }
}
