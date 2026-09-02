use nano_llm::cli::{Cli, CliError};
use std::process::ExitCode;

fn main() -> ExitCode {
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

    if cli.validate {
        // Validation mode: semantic config loading/validation is implemented in subsequent milestones.
        ExitCode::SUCCESS
    } else {
        // Server mode: HTTP listener is implemented in subsequent milestones.
        ExitCode::SUCCESS
    }
}
