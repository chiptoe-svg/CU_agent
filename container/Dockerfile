# NanoClaw Agent Container
# Runs Claude Agent SDK in isolated Linux VM with browser automation

FROM node:22-slim

# Install system dependencies for Chromium
RUN apt-get update && apt-get install -y \
    chromium \
    fonts-liberation \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    libgbm1 \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libcups2 \
    libdrm2 \
    libxshmfence1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Chromium path for agent-browser
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium

# Install Python for Google ADK (Gemini runtime)
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

# Install uv (provides uvx — used to launch workspace-mcp for Google Workspace)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Install shared tools globally. SDK CLIs are added by /add-agentSDK-* skills.
RUN npm install -g agent-browser @anthropic-ai/claude-code @openai/codex

# Install Google Workspace CLI (gws)
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then GWS_ARCH="aarch64"; else GWS_ARCH="x86_64"; fi && \
    curl -fsSL "https://github.com/googleworkspace/cli/releases/latest/download/google-workspace-cli-${GWS_ARCH}-unknown-linux-gnu.tar.gz" \
    | tar xz --strip-components=0 -C /usr/local/bin/ ./gws && chmod +x /usr/local/bin/gws

# Create app directory
WORKDIR /app

# Copy package files first for better caching
COPY agent-runner/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY agent-runner/ ./

# Install ADK in a virtual environment (avoids system package conflicts)
RUN python3 -m venv /opt/adk-venv && \
    /opt/adk-venv/bin/pip install --no-cache-dir -r adk/requirements.txt
ENV PATH="/opt/adk-venv/bin:$PATH"

# Build TypeScript
RUN npm run build

# Disable Codex's internal Landlock sandbox — Docker is already the sandbox.
# Wrap the binary to inject --dangerously-bypass-approvals-and-sandbox and
# strip --sandbox/--approval flags that the SDK adds (they conflict with bypass).
RUN for CODEX_BIN in $(find /usr/local/lib/node_modules/@openai/codex /app/node_modules/@openai/codex-linux-* -name "codex" -type f 2>/dev/null); do \
      mv "$CODEX_BIN" "${CODEX_BIN}.real" && \
      printf '#!/bin/sh\n# Strip --sandbox and -a/--ask-for-approval flags, add bypass\nARGS=""\nSKIP_NEXT=""\nfor arg in "$@"; do\n  if [ -n "$SKIP_NEXT" ]; then SKIP_NEXT=""; continue; fi\n  case "$arg" in\n    --sandbox|--ask-for-approval|-a) SKIP_NEXT=1; continue ;;\n    --dangerously-bypass-approvals-and-sandbox) continue ;;\n  esac\n  ARGS="$ARGS \\"$arg\\""\ndone\neval exec "%s.real" --dangerously-bypass-approvals-and-sandbox $ARGS\n' "$CODEX_BIN" > "$CODEX_BIN" && \
      chmod +x "$CODEX_BIN"; \
    done

# Create workspace directories
RUN mkdir -p /workspace/group /workspace/global /workspace/extra /workspace/ipc/messages /workspace/ipc/tasks /workspace/ipc/input

# Pre-compile TypeScript at build time (used as fallback if source hasn't changed)
RUN cd /app && npx tsc --outDir /tmp/dist-prebuilt 2>&1 >&2 && \
    ln -s /app/node_modules /tmp/dist-prebuilt/node_modules

# Create entrypoint script
# Container input (prompt, group info) is passed via stdin JSON.
# Credentials are injected by the host's credential proxy — never passed here.
# Follow-up messages arrive via IPC files in /workspace/ipc/input/
#
# Optimization: skip recompilation if mounted source matches the pre-built image.
# Compares mtimes of /app/src/index.ts (mounted, writable) vs the pre-built dist.
# Only recompiles when the host has newer agent-runner source.
RUN printf '#!/bin/bash\nset -e\ncd /app\nSRC_MTIME=$(stat -c %%Y /app/src/index.ts 2>/dev/null || stat -f %%m /app/src/index.ts 2>/dev/null || echo 0)\nBUILT_MTIME=$(stat -c %%Y /tmp/dist-prebuilt/index.js 2>/dev/null || stat -f %%m /tmp/dist-prebuilt/index.js 2>/dev/null || echo 0)\nif [ "$SRC_MTIME" -gt "$BUILT_MTIME" ]; then\n  npx tsc --outDir /tmp/dist 2>&1 >&2\n  ln -sf /app/node_modules /tmp/dist/node_modules\nelse\n  cp -a /tmp/dist-prebuilt /tmp/dist\nfi\ncat > /tmp/input.json\nnode /tmp/dist/index.js < /tmp/input.json\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set ownership to node user (non-root) for writable directories
RUN chown -R node:node /workspace && chmod 777 /home/node

# Switch to non-root user (required for --dangerously-skip-permissions)
USER node

# Set working directory to group workspace
WORKDIR /workspace/group

# Entry point reads JSON from stdin, outputs JSON to stdout
ENTRYPOINT ["/app/entrypoint.sh"]
