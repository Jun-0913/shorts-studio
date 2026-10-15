#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
studio_python="${PYTHON_BIN:-python3.11}"
if [ -x /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg ]; then
  export PATH="/opt/homebrew/opt/ffmpeg-full/bin:$PATH"
fi
command -v "$studio_python" >/dev/null
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
ffmpeg -hide_banner -filters 2>/dev/null | grep -E '[[:space:]]ass[[:space:]]' >/dev/null || {
  printf '%s\n' 'FFmpeg needs the ass filter. On macOS: brew install ffmpeg-full' >&2
  exit 1
}
"$studio_python" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
if [ "${1:-}" = "--full" ]; then
  .venv/bin/python -m pip install -r requirements-runtime.txt
fi
printf '%s\n' 'Ready. Run: scripts/start-studio.sh'
