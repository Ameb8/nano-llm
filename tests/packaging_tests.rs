use std::fs;
use std::path::Path;

fn repository_file(path: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|error| panic!("failed to read {path}: {error}"))
}

#[test]
fn scratch_image_has_only_the_binary_and_plain_http_entrypoint() {
    let dockerfile = repository_file("Dockerfile");
    let workflow = repository_file(".github/workflows/release-artifacts.yml");
    assert!(
        dockerfile.contains("FROM scratch AS artifact\nCOPY --from=build /out/nano-llm /nano-llm")
    );
    assert!(dockerfile.contains("USER 65532:65532"));
    assert!(dockerfile.contains("ENTRYPOINT [\"/nano-llm\", \"--config\", \"/etc/nano-llm/config.yaml\", \"--bind\", \"0.0.0.0:4000\"]"));
    assert!(!dockerfile.contains("/bin/sh"));
    assert!(!dockerfile.contains("HEALTHCHECK"));
    assert!(workflow.contains("tar -tf - nano-llm"));
    assert!(!workflow.contains("tar -t | sort"));
}

#[test]
fn release_builds_both_static_linux_architectures() {
    let dockerfile = repository_file("Dockerfile");
    let verifier = repository_file("scripts/verify-static-artifact.sh");
    assert!(dockerfile.contains("x86_64-unknown-linux-musl"));
    assert!(dockerfile.contains("aarch64-unknown-linux-musl"));
    assert!(dockerfile.contains("apt-get install --yes --no-install-recommends musl-tools"));
    assert!(dockerfile.contains("CC=musl-gcc"));
    assert!(dockerfile.contains("target-feature=+crt-static"));
    assert!(dockerfile.contains("cargo build --locked --release"));
    assert!(verifier.contains("static-pie linked"));
    assert!(verifier.contains("Requesting program interpreter"));
    assert!(verifier.contains("(NEEDED)"));
    assert!(verifier.contains("--help"));
}

#[test]
fn executable_matrix_ci_does_not_assume_task_is_preinstalled() {
    let workflow = repository_file(".github/workflows/release-artifacts.yml");
    assert!(!workflow.contains("run: task executable-matrix"));
    assert!(workflow.contains(
        "run: cargo test --features executable-test-tls --test executable_matrix_tests -- --test-threads=1"
    ));
}

#[test]
fn release_smoke_exercises_the_public_gateway_surface_and_canary_scan() {
    let smoke = repository_file("scripts/smoke-release-artifact.sh");
    assert!(smoke.contains("/health"));
    assert!(smoke.contains("/v1/models"));
    assert!(smoke.contains("/v1/chat/completions"));
    assert!(smoke.contains("data: [DONE]"));
    assert!(smoke.contains("\"finish_reason\":\"stop\""));
    assert!(smoke.contains("release-master-canary"));
    assert!(smoke.contains("request-body-canary"));
    assert!(smoke.contains("qemu-aarch64"));
    assert!(smoke.contains("qemu-x86_64"));
    assert!(smoke.contains("runner_label=binfmt"));
}

#[test]
fn bundled_verified_outbound_tls_stays_separate_from_inbound_http() {
    let providers = repository_file("src/providers.rs");
    let readme = repository_file("README.md");
    assert!(providers.contains("TransportTlsVerification::BundledRoots"));
    assert!(providers.contains("follow_redirects: false"));
    assert!(readme.contains("does not provide inbound TLS"));
    assert!(readme.contains("outbound provider HTTPS"));
}
