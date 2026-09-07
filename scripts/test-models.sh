#!/usr/bin/env bash
# Verify that a running nano-llm server exposes its authenticated model list.
set -euo pipefail

if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
    echo "LITELLM_MASTER_KEY is unset or empty" >&2
    exit 1
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

base_url=${NANO_LLM_BASE_URL:-http://127.0.0.1:4000}
base_url=${base_url%/}

response=$(curl --fail-with-body --silent --show-error \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    "$base_url/v1/models")

jq -e '
    .object == "list" and
    (.data | type) == "array" and
    (.data | all(.[]; (.id | type) == "string"))
' <<<"$response" >/dev/null

printf '%s\n' "$response"
