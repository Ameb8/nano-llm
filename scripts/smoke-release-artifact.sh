#!/usr/bin/env sh
# Exercise a release artifact through its public HTTP surface against a
# controlled keyless OpenAI-compatible peer.  This is intentionally a host
# test harness: the packaged runtime remains scratch and has no shell, CA
# files, or provider CLI.
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <x86_64|aarch64> <binary>" >&2
    exit 2
fi
architecture=$1
binary=$2
case "$architecture" in
    x86_64) host_machine=x86_64; runner='qemu-x86_64-static qemu-x86_64' ;;
    aarch64) host_machine=aarch64; runner='qemu-aarch64-static qemu-aarch64' ;;
    *) echo "unsupported architecture: $architecture" >&2; exit 2 ;;
esac
test -x "$binary"
command -v curl >/dev/null
command -v python3 >/dev/null

workdir=$(mktemp -d)
gateway_pid=''
fixture_pid=''
cleanup() {
    [ -z "$gateway_pid" ] || kill "$gateway_pid" 2>/dev/null || true
    [ -z "$fixture_pid" ] || kill "$fixture_pid" 2>/dev/null || true
    wait "$gateway_pid" 2>/dev/null || true
    wait "$fixture_pid" 2>/dev/null || true
    rm -rf "$workdir"
}
trap cleanup EXIT INT TERM

python3 - "$workdir/fixture-port" <<'PY' &
import http.server, json, sys
class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        if self.path != "/v1/chat/completions":
            self.send_error(404); return
        if request.get("stream"):
            body = ('data: {"id":"fixture","object":"chat.completion.chunk","created":0,'
                    '"model":"private","choices":[{"index":0,"delta":{"content":"increment"},"finish_reason":"stop"}]}\n\n'
                    'data: [DONE]\n\n').encode()
            self.send_response(200); self.send_header("content-type", "text/event-stream")
        else:
            body = b'{"id":"fixture","object":"chat.completion","created":0,"model":"private","choices":[{"index":0,"message":{"role":"assistant","content":"controlled reply"},"finish_reason":"stop"}]}'
            self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_): pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
open(sys.argv[1], "w").write(str(server.server_address[1]))
server.serve_forever()
PY
fixture_pid=$!
for _ in $(seq 1 50); do [ -s "$workdir/fixture-port" ] && break; sleep .1; done
test -s "$workdir/fixture-port"
fixture_port=$(cat "$workdir/fixture-port")
cat > "$workdir/config.yaml" <<EOF
model_list:
  - model_name: qualification
    litellm_params:
      model: openai_compatible/controlled
      api_base: http://127.0.0.1:$fixture_port/v1
general_settings:
  master_key: os.environ/NANO_LLM_SMOKE_MASTER
EOF
export NANO_LLM_SMOKE_MASTER=release-master-canary
runner_label=native
if [ "$(uname -m)" = "$host_machine" ]; then
    "$binary" --config "$workdir/config.yaml" --bind 127.0.0.1:40138 >"$workdir/gateway.log" 2>&1 &
else
    if "$binary" --help >/dev/null 2>&1; then
        runner=''
        runner_label=binfmt
    else
        selected_runner=''
        for candidate in $runner; do
            if command -v "$candidate" >/dev/null 2>&1; then
                selected_runner=$candidate
                break
            fi
        done
        [ -n "$selected_runner" ] || { echo "no compatible runner for $architecture" >&2; exit 1; }
        runner=$selected_runner
        runner_label=$selected_runner
    fi
    if [ -n "$runner" ]; then
        "$runner" "$binary" --config "$workdir/config.yaml" --bind 127.0.0.1:40138 >"$workdir/gateway.log" 2>&1 &
    else
        "$binary" --config "$workdir/config.yaml" --bind 127.0.0.1:40138 >"$workdir/gateway.log" 2>&1 &
    fi
fi
gateway_pid=$!
for _ in $(seq 1 50); do curl -fsS http://127.0.0.1:40138/health >"$workdir/health" 2>/dev/null && break; sleep .1; done
test "$(cat "$workdir/health")" = '{"status":"ok"}'
curl -fsS -H "Authorization: Bearer $NANO_LLM_SMOKE_MASTER" http://127.0.0.1:40138/v1/models >"$workdir/models"
grep -q '"id":"qualification"' "$workdir/models"
curl -fsS -H "Authorization: Bearer $NANO_LLM_SMOKE_MASTER" -H 'Content-Type: application/json' -d '{"model":"qualification","messages":[{"role":"user","content":"request-body-canary"}]}' http://127.0.0.1:40138/v1/chat/completions >"$workdir/chat"
grep -q 'controlled reply' "$workdir/chat"
curl -fsS -N -H "Authorization: Bearer $NANO_LLM_SMOKE_MASTER" -H 'Content-Type: application/json' -d '{"model":"qualification","stream":true,"messages":[{"role":"user","content":"stream-body-canary"}]}' http://127.0.0.1:40138/v1/chat/completions >"$workdir/stream"
grep -q 'increment' "$workdir/stream"
grep -qx 'data: \[DONE\]' "$workdir/stream"
! grep -R -F -e "$NANO_LLM_SMOKE_MASTER" -e request-body-canary -e stream-body-canary "$workdir/gateway.log" "$workdir/health" "$workdir/models" "$workdir/chat" "$workdir/stream"
printf 'release_artifact_smoke=passed architecture=%s runner=%s\n' "$architecture" "$runner_label"
