#!/bin/bash
# gws-mcp OAuth login wrapper.
# Launches workspace-mcp in stdio mode, sends an MCP tool call to trigger
# Google's OAuth flow (opens browser), and waits for the per-user token
# file to appear in the credentials directory.
set -euo pipefail

TOKEN_DIR="$HOME/.nanoclaw/.gws-mcp-tokens"
mkdir -p "$TOKEN_DIR"
export WORKSPACE_MCP_CREDENTIALS_DIR="$TOKEN_DIR"

# Load OAuth client credentials from project .env if not already exported.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
if [[ -f "$ENV_FILE" && ( -z "${GOOGLE_OAUTH_CLIENT_ID:-}" || -z "${GOOGLE_OAUTH_CLIENT_SECRET:-}" ) ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${GOOGLE_OAUTH_CLIENT_ID:-}" || -z "${GOOGLE_OAUTH_CLIENT_SECRET:-}" ]]; then
  echo "ERROR: GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET must be set." >&2
  echo >&2
  echo "To create a Desktop OAuth client:" >&2
  echo "  1. https://console.cloud.google.com/apis/credentials" >&2
  echo "  2. Create Credentials -> OAuth client ID -> Desktop app" >&2
  echo "  3. Add the client ID and secret to $ENV_FILE" >&2
  exit 1
fi

if ! command -v uvx >/dev/null 2>&1; then
  echo "ERROR: uvx not found. Install uv: https://astral.sh/uv" >&2
  exit 1
fi

echo "Starting workspace-mcp OAuth flow"
echo "  Credentials dir: $TOKEN_DIR"
echo

FIFO="$(mktemp -u)"
mkfifo "$FIFO"
SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  rm -f "$FIFO" || true
}
trap cleanup EXIT

before_count() {
  ls "$TOKEN_DIR"/*.json 2>/dev/null | wc -l | tr -d ' '
}
BEFORE="$(before_count)"

uvx workspace-mcp \
  --single-user \
  --tool-tier extended \
  --tools gmail calendar drive docs sheets slides \
  <"$FIFO" &
SERVER_PID=$!

exec 3>"$FIFO"
sleep 2

printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"nanoclaw-gws-login","version":"1.0"}}}' >&3
printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/initialized"}' >&3
sleep 1
printf '%s\n' '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_calendars","arguments":{}}}' >&3

echo "Browser should open for Google consent."
echo "Waiting for token file (up to 5 minutes)..."
echo

for _ in $(seq 1 300); do
  if [[ "$(before_count)" -gt "$BEFORE" ]]; then
    echo "Token saved:"
    ls -1 "$TOKEN_DIR"/*.json
    exec 3>&-
    exit 0
  fi
  sleep 1
done

echo "ERROR: timeout waiting for OAuth completion." >&2
echo "If the browser did not open, check workspace-mcp stderr above." >&2
exit 1
