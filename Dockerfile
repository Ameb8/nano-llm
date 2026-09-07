# syntax=docker/dockerfile:1.7
ARG RUST_VERSION=1.94.0
FROM --platform=$TARGETPLATFORM rust:${RUST_VERSION}-slim-bookworm AS build

ARG TARGETARCH
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN case "$TARGETARCH" in \
      amd64) rust_target=x86_64-unknown-linux-musl ;; \
      arm64) rust_target=aarch64-unknown-linux-musl ;; \
      *) echo "unsupported Linux architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && rustup target add "$rust_target" \
    && SOURCE_DATE_EPOCH=0 RUSTFLAGS='-C target-feature=+crt-static -C strip=symbols --remap-path-prefix=/src=.' cargo build --locked --release --target "$rust_target" \
    && mkdir /out \
    && cp "target/$rust_target/release/nano-llm" /out/nano-llm

FROM scratch AS artifact
COPY --from=build /out/nano-llm /nano-llm

# Operators mount the config and pass only the secret environment values it references.
FROM scratch
COPY --from=build /out/nano-llm /nano-llm
USER 65532:65532
EXPOSE 4000
ENTRYPOINT ["/nano-llm", "--config", "/etc/nano-llm/config.yaml", "--bind", "0.0.0.0:4000"]
