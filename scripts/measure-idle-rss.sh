#!/usr/bin/env sh
# Measure the resident set size of a running release binary on Linux.
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <binary> <minimal-config>" >&2
    exit 2
fi

binary=$1
config=$2
test -x "$binary"
test -f "$config"
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
# The release target is expressed as 20 MiB, not decimal MB: 20 MiB is
# 20 * 1024 KiB.  Keep both values in the evidence output to avoid silently
# changing units during qualification.
awk -v rss_kib="$rss_kib" 'BEGIN { printf "idle_rss_mib=%.2f\nidle_rss_under_20_mib=%s\n", rss_kib / 1024, (rss_kib < 20 * 1024 ? "true" : "false") }'
