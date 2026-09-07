#!/usr/bin/env bash
# Send one authenticated chat-completions request to a running nano-llm server.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <model> <message>" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
env_file="$repo_root/.env"

if [[ ! -f "$env_file" ]]; then
    echo "missing $env_file (expected LITELLM_MASTER_KEY)" >&2
    exit 1
fi

# Export values loaded from .env so the bearer token is available to curl.
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
    echo "LITELLM_MASTER_KEY is unset or empty in $env_file" >&2
    exit 1
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

model=$1
message=$2
base_url=${NANO_LLM_BASE_URL:-http://127.0.0.1:4000}
base_url=${base_url%/}

payload=$(jq -n --arg model "$model" --arg message "$message" \
    '{model: $model, messages: [{role: "user", content: $message}]}')

curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -H 'Content-Type: application/json' \
    --data "$payload" \
    "$base_url/v1/chat/completions"
