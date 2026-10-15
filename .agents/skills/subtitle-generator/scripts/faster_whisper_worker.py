#!/usr/bin/env python3
"""Private ASR worker; the caller owns admission and the inherited media lease."""

import json
from pathlib import Path
import sys

import transcribe


def main() -> int:
    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected the private result path")
        request = json.load(sys.stdin)
        request["audio_path"] = Path(request["audio_path"])
        request["cache_dir"] = Path(request["cache_dir"])
        result = transcribe._run_faster_whisper(**request)
        transcribe.write_json_atomic(Path(sys.argv[1]), result)
        return 0
    except (OSError, ValueError, KeyError, TypeError, transcribe.SubtitleGeneratorError) as error:
        print(f"faster-whisper worker: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
