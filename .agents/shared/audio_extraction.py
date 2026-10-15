"""Canonical decoded audio for source probes and their exact SHA verification.

Decode from the beginning before seeking on the output timeline. Input seeking
can alter codec preroll, timestamps, and resampler state (notably for Opus), so
it must not be mixed with output seeking when comparing exact PCM. Every caller
uses the same microsecond interval and mono 16 kHz signed-16-bit conversion.
Only the destination muxer differs between a WAV, raw PCM, and a SHA-256 hash.
"""

from __future__ import annotations

import math
from pathlib import Path


SAMPLE_RATE = 16_000
CHANNELS = 1


def pcm_command(
    path: Path,
    *,
    ffmpeg: Path | str = "ffmpeg",
    start: float | None = None,
    end: float | None = None,
) -> list[str]:
    """Build canonical PCM output options; callers append a muxer/destination."""
    first = 0.0 if start is None else start
    if isinstance(first, bool) or not isinstance(first, (int, float)) or not math.isfinite(first) or first < 0:
        raise ValueError("Audio interval start must be a finite nonnegative number.")
    if end is not None and (
        isinstance(end, bool) or not isinstance(end, (int, float))
        or not math.isfinite(end) or end <= first
    ):
        raise ValueError("Audio interval end must be finite and greater than start.")
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-i", str(path)]
    if first > 0:
        command += ["-ss", f"{first:.6f}"]
    if end is not None:
        command += ["-t", f"{end - first:.6f}"]
    return command + ["-map", "0:a:0", "-vn", "-ac", str(CHANNELS),
                      "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le"]


def fingerprint_command(path: Path, **kwargs) -> list[str]:
    """Hash the exact same decoded samples written by a WAV producer."""
    return pcm_command(path, **kwargs) + ["-f", "hash", "-hash", "sha256", "-"]


def main() -> int:
    """Produce review WAVs without duplicating the extraction command."""
    import argparse
    import os
    import subprocess
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("The probe output must differ from the original media.")
    if args.output.suffix.lower() != ".wav":
        parser.error("The probe output must be a .wav file.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".wav", dir=args.output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        command = pcm_command(args.source, ffmpeg=args.ffmpeg, start=args.start, end=args.end)
        subprocess.run(command + [str(temporary)], check=True)
        temporary.replace(args.output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Could not extract canonical probe audio: {error}\n")
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
