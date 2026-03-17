#!/usr/bin/env bash
set -euo pipefail

# ─── Config ───────────────────────────────────────────────
ENV_NAME="whisper_streaming"
PYTHON_VER="3.10"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8000
WEB_PORT=8080

# ─── Colors ───────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[START]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── Cleanup on exit ─────────────────────────────────────
cleanup() {
    log "Shutting down..."
    [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null && log "Backend server stopped."
    [[ -n "${WEB_PID:-}" ]]     && kill "$WEB_PID"     2>/dev/null && log "Web server stopped."
    wait 2>/dev/null
    log "Done."
}
trap cleanup EXIT INT TERM

# ─── 1. Check conda ──────────────────────────────────────
if ! command -v conda &>/dev/null; then
    err "conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Make sure conda activate works in scripts
eval "$(conda shell.bash hook)"

# ─── 2. Create / activate conda env ──────────────────────
if ! conda env list | grep -q "^${ENV_NAME} "; then
    log "Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VER})..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
fi

log "Activating environment '${ENV_NAME}'..."
conda activate "$ENV_NAME"

# ─── 3. Install dependencies ─────────────────────────────
cd "$PROJECT_DIR"

if ! python -c "import uvicorn" 2>/dev/null; then
    log "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install numpy sounddevice
    log "Dependencies installed."
else
    log "Dependencies already installed, skipping."
fi

# ─── 4. Kill old processes on target ports ────────────────
kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        warn "Killing existing process(es) on port $port (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

kill_port "$BACKEND_PORT"
kill_port "$WEB_PORT"

# ─── 5. Start backend ASR server ─────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log "Starting backend server (port ${BACKEND_PORT})..."
uvicorn server:app --host 0.0.0.0 --port "$BACKEND_PORT" &
BACKEND_PID=$!

# Wait for backend to be ready (may take 3-6 min to load Whisper + Translator)
log "Waiting for backend to initialize (loading Whisper + HY-MT translator)..."
for i in $(seq 1 180); do
    if curl -sf "http://localhost:${BACKEND_PORT}/docs" >/dev/null 2>&1; then
        log "Backend server ready (PID: ${BACKEND_PID})."
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        err "Backend server crashed during startup. Check logs above."
        exit 1
    fi
    sleep 2
done

# ─── 6. Start web server (proxy + UI) ────────────────────
log "Starting web server (port ${WEB_PORT})..."
python web_server.py &
WEB_PID=$!
sleep 2

if ! kill -0 "$WEB_PID" 2>/dev/null; then
    err "Web server failed to start."
    exit 1
fi

log "Web server ready (PID: ${WEB_PID})."

# ─── 7. Print summary ────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Whisper Stream is running!${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "  Web UI:       ${YELLOW}http://localhost:${WEB_PORT}${NC}"
echo -e "  Backend API:  ${YELLOW}http://localhost:${BACKEND_PORT}${NC}"
echo -e "  WebSocket:    ${YELLOW}ws://localhost:${BACKEND_PORT}/v1/streaming${NC}"
echo -e ""
echo -e "  Press ${RED}Ctrl+C${NC} to stop all servers."
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo ""

# ─── 8. Wait for processes ───────────────────────────────
wait
