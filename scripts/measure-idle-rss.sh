#!/usr/bin/env sh
# Measure the resident set size of a running release binary on Linux.
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <binary> <minimal-config>" >&2
    exit 2
fi

binary=$1
config=$2
"$binary" --config "$config" --no-auth --bind 127.0.0.1:40138 >/dev/null 2>&1 &
pid=$!
cleanup() {
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# The process has no request activity; this is intentionally an idle sample.
sleep 1
rss_kib=$(ps -o rss= -p "$pid" | tr -d '[:space:]')
if [ -z "$rss_kib" ]; then
    echo "error: nano-llm exited before idle RSS could be sampled" >&2
    exit 1
fi
printf 'idle_rss_kib=%s\n' "$rss_kib"
