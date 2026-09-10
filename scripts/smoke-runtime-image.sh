#!/usr/bin/env bash
# Verify the packaged scratch image and exercise its health endpoint.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <image>" >&2
    exit 2
fi

image=$1
filesystem_container=''
runtime_container=''
workdir=''
cleanup() {
    [[ -z "$filesystem_container" ]] || docker rm -f "$filesystem_container" >/dev/null 2>&1 || true
    [[ -z "$runtime_container" ]] || docker rm -f "$runtime_container" >/dev/null 2>&1 || true
    [[ -z "$workdir" ]] || rm -rf "$workdir"
}
trap cleanup EXIT INT TERM

test "$(docker image inspect "$image" --format '{{.Config.Entrypoint}}')" = '[/nano-llm --config /etc/nano-llm/config.yaml --bind 0.0.0.0:4000]'
test "$(docker image inspect "$image" --format '{{.Config.User}}')" = '65532:65532'
filesystem_container=$(docker create "$image")
# Docker injects runtime files such as /etc/hosts into a created container.
# Verify the image payload without comparing the exported filesystem verbatim.
docker export "$filesystem_container" | tar -tf - nano-llm >/dev/null
docker rm "$filesystem_container" >/dev/null
filesystem_container=''

if docker run --rm "$image" >/dev/null 2>&1; then
    echo 'image unexpectedly started without its required config mount' >&2
    exit 1
fi

workdir=$(mktemp -d)
printf '%s\n' \
    'model_list:' \
    '  - model_name: local' \
    '    litellm_params:' \
    '      model: openai_compatible/test' \
    '      api_base: http://127.0.0.1:1/v1' \
    'general_settings:' \
    '  master_key: os.environ/NANO_LLM_MASTER_KEY' \
    >"$workdir/config.yaml"

if docker run --rm -v "$workdir/config.yaml:/etc/nano-llm/config.yaml:ro" "$image" >/dev/null 2>&1; then
    echo 'image unexpectedly started without the required secret variable' >&2
    exit 1
fi

runtime_container="nano-llm-smoke-$$"
docker run -d --name "$runtime_container" -p 127.0.0.1::4000 \
    -v "$workdir/config.yaml:/etc/nano-llm/config.yaml:ro" \
    -e NANO_LLM_MASTER_KEY=test-key "$image" >/dev/null
endpoint="http://$(docker port "$runtime_container" 4000/tcp)/health"
for _ in {1..50}; do
    if curl -fsS "$endpoint"; then
        printf '\nruntime_image_smoke=passed image=%s\n' "$image"
        exit 0
    fi
    sleep .1
done
docker logs "$runtime_container"
exit 1
