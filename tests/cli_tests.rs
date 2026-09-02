use std::fs;
use std::process::Command;

fn bin_path() -> &'static str {
    env!("CARGO_BIN_EXE_nano-llm")
}

fn write_temp_config(content: &str) -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    let filename = format!(
        "nano-llm-test-{}.yaml",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    dir.push(filename);
    fs::write(&dir, content).expect("failed to write temp config file");
    dir
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
fn test_cli_no_auth_with_non_loopback_bind_under_validate_fails() {
    let output = Command::new(bin_path())
        .args([
            "--config",
            "config.yaml",
            "--bind",
            "192.168.1.1:4000",
            "--no-auth",
            "--validate",
        ])
        .output()
        .expect("failed to execute binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--no-auth") && stderr.contains("loopback"),
        "stderr should reject non-loopback bind under --validate --no-auth (got: {stderr})"
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

#[test]
fn test_cli_validate_success_with_redaction() {
    let config_yaml = r#"
model_list:
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/CLI_TEST_OPENAI_KEY
  - model_name: default
    litellm_params:
      model: anthropic/claude-3-5-sonnet
      api_key: os.environ/CLI_TEST_ANTHROPIC_KEY
      timeout: 15

general_settings:
  master_key: os.environ/CLI_TEST_MASTER_KEY
"#;

    let config_path = write_temp_config(config_yaml);

    let output = Command::new(bin_path())
        .args(["--config", config_path.to_str().unwrap(), "--validate"])
        .env("CLI_TEST_MASTER_KEY", "super_secret_master_key_12345")
        .env("CLI_TEST_OPENAI_KEY", "super_secret_openai_key_67890")
        .env("CLI_TEST_ANTHROPIC_KEY", "super_secret_anthropic_key_abcde")
        .output()
        .expect("failed to execute binary");

    let _ = fs::remove_file(&config_path);

    assert!(output.status.success(), "validation mode should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Verify resolved route table details
    assert!(stdout.contains("nano-llm resolved route table:"));
    assert!(stdout.contains("master_key: [REDACTED]"));
    assert!(stdout.contains("openai/gpt-4o"));
    assert!(stdout.contains("anthropic/claude-3-5-sonnet"));
    assert!(stdout.contains("api_key: [REDACTED]"));
    assert!(stdout.contains("timeout: 15s"));

    // Verify secrets are NEVER present in stdout or stderr
    assert!(!stdout.contains("super_secret_master_key_12345"));
    assert!(!stdout.contains("super_secret_openai_key_67890"));
    assert!(!stdout.contains("super_secret_anthropic_key_abcde"));
    assert!(!stderr.contains("super_secret_master_key_12345"));
    assert!(!stderr.contains("super_secret_openai_key_67890"));
    assert!(!stderr.contains("super_secret_anthropic_key_abcde"));
}

#[test]
fn test_cli_no_auth_serving_warning() {
    let config_yaml = r#"
model_list:
  - model_name: local
    litellm_params:
      model: openai_compatible/custom
      api_base: http://localhost:8000/v1
"#;

    let config_path = write_temp_config(config_yaml);

    let output = Command::new(bin_path())
        .args(["--config", config_path.to_str().unwrap(), "--no-auth"])
        .output()
        .expect("failed to execute binary");

    let _ = fs::remove_file(&config_path);

    assert!(output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("WARNING: running with --no-auth; inbound authentication is disabled"),
        "stderr should contain loud warning when serving with --no-auth (got: {stderr})"
    );
}

#[test]
fn test_cli_validate_no_auth_no_serving_warning() {
    let config_yaml = r#"
model_list:
  - model_name: local
    litellm_params:
      model: openai_compatible/custom
      api_base: http://localhost:8000/v1
"#;

    let config_path = write_temp_config(config_yaml);

    let output = Command::new(bin_path())
        .args([
            "--config",
            config_path.to_str().unwrap(),
            "--no-auth",
            "--validate",
        ])
        .output()
        .expect("failed to execute binary");

    let _ = fs::remove_file(&config_path);

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(stdout.contains("master_key: (none)"));
    assert!(
        !stderr.contains("WARNING: running with --no-auth; inbound authentication is disabled"),
        "validation mode should not emit serving warning on stderr (got: {stderr})"
    );
}

#[test]
fn test_cli_validate_missing_env_var_fails_cleanly() {
    let config_yaml = r#"
model_list:
  - model_name: default
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/TOTALLY_MISSING_VAR_12345

general_settings:
  master_key: os.environ/ALSO_MISSING_KEY
"#;

    let config_path = write_temp_config(config_yaml);

    let output = Command::new(bin_path())
        .args(["--config", config_path.to_str().unwrap(), "--validate"])
        .output()
        .expect("failed to execute binary");

    let _ = fs::remove_file(&config_path);

    assert!(
        !output.status.success(),
        "validation should fail for missing env vars"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing environment variable"));
}
