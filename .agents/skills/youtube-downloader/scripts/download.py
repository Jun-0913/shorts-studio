#!/usr/bin/env python3
"""Download one YouTube video and emit a verified pipeline handoff manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


FORMAT_SELECTOR = "bestvideo+bestaudio/best"
OUTPUT_TEMPLATE = "%(id)s/source.%(ext)s"
MERGE_CONTAINERS = "mp4/mkv"
MANIFEST_SCHEMA_VERSION = 1
PO_TOKEN_PROVIDER_VERSION = "1.3.1"


class DownloaderError(RuntimeError):
    """An expected, user-actionable downloader failure."""


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start.resolve().parents[4]


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "downloads"
DEFAULT_PLUGIN_DIR = REPO_ROOT / ".tools" / "yt-dlp-plugins"
DEFAULT_PROVIDER_SERVER = (
    REPO_ROOT / ".tools" / "bgutil-ytdlp-pot-provider" / "server"
)
DEFAULT_YT_DLP_BIN = Path(
    shutil.which("yt-dlp")
    or REPO_ROOT / ".tools" / "yt-dlp-nightly" / "yt-dlp_macos"
)


def validate_youtube_url(value: str) -> str:
    """Accept one HTTPS YouTube video URL and reject playlist/channel URLs."""
    if not value or any(character in value for character in "\r\n\x00"):
        raise DownloaderError("A single YouTube URL is required.")

    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")
    ):
        raise DownloaderError("Only HTTPS youtube.com or youtu.be URLs are supported.")

    if host == "youtu.be":
        if not parsed.path.strip("/"):
            raise DownloaderError("The youtu.be URL does not contain a video id.")
        return value

    path = parsed.path.rstrip("/") or "/"
    if path == "/watch":
        if not parse_qs(parsed.query).get("v"):
            raise DownloaderError("The YouTube watch URL does not contain a video id.")
        return value
    if any(path.startswith(prefix) for prefix in ("/shorts/", "/live/", "/embed/")):
        return value
    raise DownloaderError("The URL must point directly to one YouTube video.")


def resolve_executable(value: str, install_hint: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    else:
        discovered = shutil.which(value)
        if discovered:
            return Path(discovered).resolve()
    raise DownloaderError(f"Required executable '{value}' was not found. {install_hint}")


def resolve_po_token_provider() -> tuple[Path, Path]:
    plugin = DEFAULT_PLUGIN_DIR / "bgutil-ytdlp-pot-provider.zip"
    generator = DEFAULT_PROVIDER_SERVER / "src" / "generate_once.ts"
    if plugin.is_file() and generator.is_file():
        return DEFAULT_PLUGIN_DIR.resolve(), DEFAULT_PROVIDER_SERVER.resolve()
    setup = ".agents/skills/youtube-downloader/scripts/setup-provider.sh"
    raise DownloaderError(f"Project-local PO token provider is missing. Run: bash {setup}")


def build_command(
    *,
    url: str,
    output_dir: Path,
    path_trace: Path,
    yt_dlp: Path,
    ffmpeg: Path,
    youtube_client: str,
    plugin_dir: Path | None = None,
    provider_server: Path | None = None,
    cookies_from_browser: str | None = None,
    force: bool = False,
) -> list[str]:
    command = [
        str(yt_dlp),
        "--no-playlist",
        "--abort-on-error",
        "--force-ipv4",
        "--format",
        FORMAT_SELECTOR,
        "--merge-output-format",
        MERGE_CONTAINERS,
        "--paths",
        str(output_dir),
        "--output",
        OUTPUT_TEMPLATE,
        "--write-info-json",
        "--concurrent-fragments",
        "4",
        "--retries",
        "10",
        "--fragment-retries",
        "10",
        "--continue",
        "--newline",
        "--ffmpeg-location",
        str(ffmpeg.parent),
        "--extractor-args",
        f"youtube:player_client={youtube_client}",
        "--print-to-file",
        "after_move:%(filepath)j",
        str(path_trace),
    ]
    if youtube_client == "default":
        index = command.index("--extractor-args")
        del command[index:index+2]
    if shutil.which("node"):
        command.extend(["--js-runtimes", "node"])
    if plugin_dir and provider_server:
        insert_at = command.index("--print-to-file")
        command[insert_at:insert_at] = [
            "--plugin-dirs",
            str(plugin_dir),
            "--extractor-args",
            f"youtubepot-bgutilscript:server_home={provider_server}",
        ]
    command.append("--force-overwrites" if force else "--no-overwrites")
    if cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    command.extend(["--", url])
    return command


def decode_output_path(path_trace: Path, output_dir: Path) -> Path:
    try:
        lines = [line.strip() for line in path_trace.read_text().splitlines() if line.strip()]
    except OSError as error:
        raise DownloaderError("yt-dlp did not report the downloaded file path.") from error
    if not lines:
        raise DownloaderError("yt-dlp completed without reporting a downloaded file path.")
    try:
        decoded = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise DownloaderError("yt-dlp returned an invalid downloaded file path.") from error
    if not isinstance(decoded, str) or not decoded:
        raise DownloaderError("yt-dlp returned an empty downloaded file path.")

    output_path = Path(decoded).expanduser().resolve()
    output_root = output_dir.expanduser().resolve()
    try:
        output_path.relative_to(output_root)
    except ValueError as error:
        raise DownloaderError("yt-dlp reported a file outside the requested output directory.") from error
    if not output_path.is_file():
        raise DownloaderError(f"Downloaded media file is missing: {output_path}")
    return output_path


def load_source_metadata(output_path: Path) -> tuple[Path, dict[str, Any]]:
    expected = output_path.with_name("source.info.json")
    candidates = [expected] if expected.is_file() else sorted(output_path.parent.glob("*.info.json"))
    if len(candidates) != 1:
        raise DownloaderError("Expected exactly one yt-dlp info JSON file beside the media.")
    try:
        value = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DownloaderError("The yt-dlp info JSON file could not be read.") from error
    if not isinstance(value, dict):
        raise DownloaderError("The yt-dlp info JSON root must be an object.")
    return candidates[0], value


def run_json(command: list[str], label: str) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        value = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise DownloaderError(f"{label} failed for the downloaded media.") from error
    if not isinstance(value, dict):
        raise DownloaderError(f"{label} returned an unexpected response.")
    return value


def tool_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return (result.stdout or result.stderr).splitlines()[0].strip() or "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_max(values: Iterable[Any]) -> int | float | None:
    numeric = [value for value in values if isinstance(value, (int, float))]
    return max(numeric) if numeric else None


def format_summary(source: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "format_id",
        "format",
        "ext",
        "protocol",
        "vcodec",
        "acodec",
        "width",
        "height",
        "fps",
        "dynamic_range",
        "tbr",
        "vbr",
        "abr",
        "asr",
        "audio_channels",
        "language",
        "filesize",
        "filesize_approx",
    )
    return {key: source.get(key) for key in allowed if source.get(key) is not None}


def selected_formats(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    formats = metadata.get("requested_formats")
    if not isinstance(formats, list):
        formats = metadata.get("requested_downloads")
    if not isinstance(formats, list):
        formats = [metadata]
    return [format_summary(item) for item in formats if isinstance(item, dict)]


def available_quality(metadata: dict[str, Any]) -> dict[str, Any]:
    formats = metadata.get("formats")
    if not isinstance(formats, list):
        return {}
    video = [
        item
        for item in formats
        if isinstance(item, dict) and item.get("vcodec") not in (None, "none")
    ]
    audio = [
        item
        for item in formats
        if isinstance(item, dict) and item.get("acodec") not in (None, "none")
    ]
    return {
        "max_height": numeric_max(item.get("height") for item in video),
        "max_fps": numeric_max(item.get("fps") for item in video),
        "max_reported_audio_bitrate_kbps": numeric_max(item.get("abr") for item in audio),
    }


def stream_summary(stream: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "index",
        "codec_name",
        "codec_long_name",
        "profile",
        "codec_type",
        "width",
        "height",
        "pix_fmt",
        "r_frame_rate",
        "avg_frame_rate",
        "sample_rate",
        "channels",
        "channel_layout",
        "bit_rate",
        "duration",
    )
    return {key: stream.get(key) for key in allowed if stream.get(key) is not None}


def create_manifest(
    *,
    source_url: str,
    output_path: Path,
    info_path: Path,
    metadata: dict[str, Any],
    probe: dict[str, Any],
    digest: str,
    versions: dict[str, str],
    youtube_client: str,
    browser_cookies_used: bool,
) -> dict[str, Any]:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        streams = []
    video_streams = [
        stream_summary(item)
        for item in streams
        if isinstance(item, dict) and item.get("codec_type") == "video"
    ]
    audio_streams = [
        stream_summary(item)
        for item in streams
        if isinstance(item, dict) and item.get("codec_type") == "audio"
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "agent": "youtube-downloader",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "requested_url": source_url,
            "webpage_url": metadata.get("webpage_url") or source_url,
            "extractor": metadata.get("extractor_key") or metadata.get("extractor"),
            "video_id": metadata.get("id"),
            "title": metadata.get("title"),
            "channel": metadata.get("channel") or metadata.get("uploader"),
            "channel_id": metadata.get("channel_id") or metadata.get("uploader_id"),
            "duration_seconds": metadata.get("duration"),
            "upload_date": metadata.get("upload_date"),
        },
        "selection": {
            "selector": FORMAT_SELECTOR,
            "merge_containers": MERGE_CONTAINERS,
            "reencoded": False,
            "selected_formats": selected_formats(metadata),
            "available_quality_summary": available_quality(metadata),
        },
        "output": {
            "media_path": str(output_path.resolve()),
            "metadata_path": str(info_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": digest,
            "container": (probe.get("format") or {}).get("format_name"),
            "video_streams": video_streams,
            "audio_streams": audio_streams,
        },
        "verification": {
            "has_video": bool(video_streams),
            "has_audio": bool(audio_streams),
            "sha256_verified": sha256_file(output_path) == digest,
        },
        "tools": versions,
        "proof_of_origin": {
            "provider": (
                "bgutil-ytdlp-pot-provider" if youtube_client == "mweb" else None
            ),
            "provider_version": (
                PO_TOKEN_PROVIDER_VERSION if youtube_client == "mweb" else None
            ),
            "youtube_client": youtube_client,
            "browser_cookies_used": browser_cookies_used,
        },
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def download(args: argparse.Namespace) -> Path:
    url = validate_youtube_url(args.url)
    install_hint = (
        "Install FFmpeg and run scripts/bootstrap.sh --full; install Node.js for YouTube challenges."
    )
    yt_dlp = resolve_executable(args.yt_dlp_bin, install_hint)
    ffmpeg = resolve_executable(args.ffmpeg_bin, install_hint)
    ffprobe = resolve_executable(args.ffprobe_bin, install_hint)
    try:
        plugin_dir, provider_server = resolve_po_token_provider()
    except DownloaderError:
        plugin_dir = provider_server = None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = [("default", None, None), ("web_embedded", None, None)]
    if plugin_dir and provider_server:
        strategies.append(("mweb", plugin_dir, provider_server))
    failures: list[str] = []
    output_path: Path | None = None
    selected_client = ""
    print(f"Downloading one video into {output_dir}")
    for youtube_client, strategy_plugins, strategy_provider in strategies:
        descriptor, trace_name = tempfile.mkstemp(
            prefix=".youtube-downloader-", suffix=".paths.jsonl", dir=output_dir
        )
        os.close(descriptor)
        path_trace = Path(trace_name)
        try:
            command = build_command(
                url=url,
                output_dir=output_dir,
                path_trace=path_trace,
                yt_dlp=yt_dlp,
                ffmpeg=ffmpeg,
                youtube_client=youtube_client,
                plugin_dir=strategy_plugins,
                provider_server=strategy_provider,
                cookies_from_browser=args.cookies_from_browser,
                force=args.force,
            )
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                output_path = decode_output_path(path_trace, output_dir)
                selected_client = youtube_client
                break
            failures.append(f"{youtube_client}: exit {result.returncode}")
        finally:
            path_trace.unlink(missing_ok=True)
    if output_path is None:
        raise DownloaderError(
            f"All YouTube access strategies failed ({', '.join(failures)})."
        )

    info_path, metadata = load_source_metadata(output_path)
    probe = run_json(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(output_path),
        ],
        "ffprobe",
    )
    digest = sha256_file(output_path)
    versions = {
        "yt_dlp": tool_version([str(yt_dlp), "--version"]),
        "ffmpeg": tool_version([str(ffmpeg), "-version"]),
        "ffprobe": tool_version([str(ffprobe), "-version"]),
    }
    manifest = create_manifest(
        source_url=url,
        output_path=output_path,
        info_path=info_path,
        metadata=metadata,
        probe=probe,
        digest=digest,
        versions=versions,
        youtube_client=selected_client,
        browser_cookies_used=bool(args.cookies_from_browser),
    )
    verification = manifest["verification"]
    if not all(verification.values()):
        raise DownloaderError(f"Downloaded media verification failed: {verification}")
    manifest_path = output_path.parent / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(json.dumps({"media": str(output_path), "manifest": str(manifest_path)}))
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one YouTube video at maximum source quality."
    )
    parser.add_argument("url", help="One HTTPS YouTube video URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Download root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--cookies-from-browser",
        help="Optional yt-dlp browser[:profile] cookie source for authorized access",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing files")
    parser.add_argument(
        "--yt-dlp-bin",
        default=os.environ.get("YT_DLP_BIN", str(DEFAULT_YT_DLP_BIN)),
    )
    parser.add_argument("--ffmpeg-bin", default=os.environ.get("FFMPEG_BIN", "ffmpeg"))
    parser.add_argument("--ffprobe-bin", default=os.environ.get("FFPROBE_BIN", "ffprobe"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        download(parse_args(argv))
    except DownloaderError as error:
        print(f"youtube-downloader: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("youtube-downloader: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
