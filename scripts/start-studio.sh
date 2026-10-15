#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg ]; then
  export PATH="/opt/homebrew/opt/ffmpeg-full/bin:$PATH"
fi
export PATH="$PWD/.venv/bin:$PATH"
export STUDIO_ASR_BACKEND="${STUDIO_ASR_BACKEND:-faster-whisper}"
exec .venv/bin/python apps/shorts-lab/server.py "$@"
