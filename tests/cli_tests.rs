use std::process::Command;

fn bin_path() -> &'static str {
    env!("CARGO_BIN_EXE_nano-llm")
}

#[test]
fn test_cli_version_flag_identifies_dev_build() {
    let output = Command::new(bin_path())
        .arg("--version")
        .output()
        .expect("failed to execute binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("0.1.0-dev"),
        "version output must identify as development build (got: {stdout})"
    );
}

#[test]
fn test_cli_help_flag() {
    let output = Command::new(bin_path())
        .arg("--help")
        .output()
        .expect("failed to execute binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--config"));
    assert!(stdout.contains("--bind"));
    assert!(stdout.contains("--no-auth"));
    assert!(stdout.contains("--validate"));
}

#[test]
fn test_cli_missing_config_fails() {
    let output = Command::new(bin_path())
        .output()
        .expect("failed to execute binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--config"));
}

#[test]
fn test_cli_unknown_flag_fails() {
    let output = Command::new(bin_path())
        .args(["--config", "config.yaml", "--unknown-flag"])
        .output()
        .expect("failed to execute binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unexpected argument") || stderr.contains("unknown"));
}

#[test]
fn test_cli_no_auth_with_non_loopback_bind_fails() {
    let output = Command::new(bin_path())
        .args([
            "--config",
            "config.yaml",
            "--bind",
            "0.0.0.0:4000",
            "--no-auth",
        ])
        .output()
        .expect("failed to execute binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--no-auth") && stderr.contains("loopback"),
        "stderr should explain no-auth loopback requirement (got: {stderr})"
    );
}

#[test]
fn test_cli_valid_args_success() {
    let output = Command::new(bin_path())
        .args(["--config", "config.yaml", "--validate"])
        .output()
        .expect("failed to execute binary");

    assert!(
        output.status.success(),
        "valid args with --validate should exit successfully"
    );
}

#[test]
fn test_cli_invalid_bind_address_fails() {
    let output = Command::new(bin_path())
        .args(["--config", "config.yaml", "--bind", "invalid-ip"])
        .output()
        .expect("failed to execute binary");

    assert!(!output.status.success());
}
