#!/usr/bin/env python3
"""Reuse verified source-language YouTube captions before Whisper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlparse


SCHEMA_VERSION = 1
TRANSCRIPT_SCHEMA_VERSION = 1
WHISPER_REQUIRED_EXIT = 3
SOURCE_DURATION_TOLERANCE_SECONDS = 2.0


class CaptionIngestorError(RuntimeError):
    """An expected, user-actionable caption ingestor failure."""


class CaptionUnavailable(RuntimeError):
    """A verified source exists, but no acceptable caption transcript does."""

    def __init__(self, message: str, *, attempts: Optional[list[dict[str, str]]] = None):
        super().__init__(message)
        self.attempts = attempts or []


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start.resolve().parents[4]


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "subtitles"
DEFAULT_YT_DLP_BIN = Path(
    shutil.which("yt-dlp")
    or REPO_ROOT / ".tools" / "yt-dlp-nightly" / "yt-dlp_macos"
)
DEFAULT_PLUGIN_DIR = REPO_ROOT / ".tools" / "yt-dlp-plugins"
DEFAULT_PROVIDER_SERVER = (
    REPO_ROOT / ".tools" / "bgutil-ytdlp-pot-provider" / "server"
)


@dataclass(frozen=True)
class SourceVideo:
    manifest_path: Path
    media_path: Path
    metadata_path: Path
    video_id: str
    title: Optional[str]
    webpage_url: str
    source_language: str
    duration_seconds: Optional[float]
    sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CaptionCandidate:
    track_type: str
    language: str
    name: Optional[str]
    ext: str


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CaptionIngestorError(f"{label} could not be read: {path}") from error
    if not isinstance(value, dict):
        raise CaptionIngestorError(f"{label} must contain a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_identifier(value: str) -> str:
    # A leading hyphen is valid in a YouTube video ID and is part of its
    # identity.  Remove only leading/trailing dots so the value remains a safe
    # single path component without rewriting the source ID.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".")
    return cleaned or "video"


def resolve_manifest_file(
    manifest_path: Path,
    declared: Any,
    *,
    label: str,
    sibling_glob: str,
    excluded_names: set[str],
) -> Path:
    candidate = Path(declared).expanduser() if isinstance(declared, str) else None
    if candidate and not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if not candidate or not candidate.is_file():
        candidates = [
            path
            for path in manifest_path.parent.glob(sibling_glob)
            if path.name not in excluded_names and path.is_file()
        ]
        if len(candidates) != 1:
            raise CaptionIngestorError(f"Downloader {label} is missing beside the manifest.")
        candidate = candidates[0]

    resolved = candidate.resolve()
    if resolved.parent != manifest_path.parent.resolve():
        raise CaptionIngestorError(
            f"Downloader manifest points to {label} outside its artifact directory."
        )
    return resolved


def normalize_language(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("_", "-")
    if not re.fullmatch(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*", normalized):
        return None
    return normalized


def language_base(value: str) -> str:
    return value.split("-", 1)[0].lower()


def infer_source_language(metadata: dict[str, Any]) -> Optional[str]:
    for key in ("original_language", "language"):
        language = normalize_language(metadata.get(key))
        if language:
            return language

    captions = metadata.get("automatic_captions")
    if isinstance(captions, dict):
        originals = {
            key[:-5]
            for key, formats in captions.items()
            if isinstance(key, str)
            and key.lower().endswith("-orig")
            and first_json3_format(formats) is not None
        }
        if len(originals) == 1:
            return normalize_language(originals.pop())
    return None


def load_source(value: str) -> SourceVideo:
    manifest_path = Path(value).expanduser().resolve()
    if not manifest_path.is_file():
        raise CaptionIngestorError(f"Downloader manifest does not exist: {manifest_path}")

    manifest = read_json(manifest_path, "Downloader manifest")
    if manifest.get("agent") != "youtube-downloader":
        raise CaptionIngestorError("Input must be a youtube-downloader manifest.")
    verification = manifest.get("verification")
    required_checks = ("has_video", "has_audio", "sha256_verified")
    if not isinstance(verification, dict) or not all(
        verification.get(key) is True for key in required_checks
    ):
        raise CaptionIngestorError(
            "Downloader manifest verification must pass before caption selection."
        )

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise CaptionIngestorError("Downloader manifest is missing output metadata.")
    media_path = resolve_manifest_file(
        manifest_path,
        output.get("media_path"),
        label="media file",
        sibling_glob="source.*",
        excluded_names={"source.info.json"},
    )
    metadata_path = resolve_manifest_file(
        manifest_path,
        output.get("metadata_path"),
        label="metadata file",
        sibling_glob="*.info.json",
        excluded_names=set(),
    )

    expected_digest = output.get("sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_digest
    ):
        raise CaptionIngestorError("Downloader manifest has no valid source SHA-256.")
    actual_digest = sha256_file(media_path)
    if actual_digest.lower() != expected_digest.lower():
        raise CaptionIngestorError("Downloader media SHA-256 no longer matches manifest.")

    metadata = read_json(metadata_path, "yt-dlp metadata")
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    video_id = source.get("video_id") or metadata.get("id")
    if not isinstance(video_id, str) or not video_id:
        raise CaptionIngestorError("Downloader metadata has no video id.")
    webpage_url = source.get("webpage_url") or metadata.get("webpage_url")
    parsed_url = urlparse(webpage_url) if isinstance(webpage_url, str) else None
    hostname = (parsed_url.hostname or "").lower() if parsed_url else ""
    if (
        not parsed_url
        or parsed_url.scheme != "https"
        or not (
            hostname == "youtu.be"
            or hostname == "youtube.com"
            or hostname.endswith(".youtube.com")
        )
    ):
        raise CaptionIngestorError("Downloader metadata has no HTTPS YouTube webpage URL.")
    source_language = infer_source_language(metadata)
    if not source_language:
        raise CaptionIngestorError("The video's source language could not be determined.")
    duration = source.get("duration_seconds")
    if not isinstance(duration, (int, float)):
        duration = metadata.get("duration")

    return SourceVideo(
        manifest_path=manifest_path,
        media_path=media_path,
        metadata_path=metadata_path,
        video_id=safe_identifier(video_id),
        title=(source.get("title") if isinstance(source.get("title"), str) else None),
        webpage_url=webpage_url,
        source_language=source_language,
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
        sha256=actual_digest,
        metadata=metadata,
    )


def is_machine_translation_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return any(
        key == "tlang"
        for key, _ in parse_qsl(urlparse(value).query, keep_blank_values=True)
    )


def first_json3_format(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, list):
        return None
    for item in value:
        if (
            isinstance(item, dict)
            and item.get("ext") == "json3"
            and not is_machine_translation_url(item.get("url"))
        ):
            return item
    return None


def ordered_language_keys(tracks: dict[str, Any], source_language: str) -> list[str]:
    exact = source_language.lower()
    base = language_base(source_language)
    keys = [key for key in tracks if isinstance(key, str)]
    ranked: list[str] = []
    for target in (exact, base):
        ranked.extend(key for key in keys if key.lower() == target and key not in ranked)
    ranked.extend(
        key
        for key in keys
        if language_base(key) == base and not key.lower().endswith("-orig") and key not in ranked
    )
    return ranked


def count_machine_translated_json3(metadata: dict[str, Any]) -> int:
    count = 0
    for group_name in ("subtitles", "automatic_captions"):
        group = metadata.get(group_name)
        if not isinstance(group, dict):
            continue
        for formats in group.values():
            if not isinstance(formats, list):
                continue
            count += sum(
                1
                for item in formats
                if isinstance(item, dict)
                and item.get("ext") == "json3"
                and is_machine_translation_url(item.get("url"))
            )
    return count


def select_caption(metadata: dict[str, Any], source_language: str) -> CaptionCandidate:
    manual = metadata.get("subtitles")
    if isinstance(manual, dict):
        for key in ordered_language_keys(manual, source_language):
            item = first_json3_format(manual.get(key))
            if item:
                return CaptionCandidate(
                    track_type="uploader",
                    language=key,
                    name=item.get("name") if isinstance(item.get("name"), str) else None,
                    ext="json3",
                )

    automatic = metadata.get("automatic_captions")
    if isinstance(automatic, dict):
        base = language_base(source_language)
        keys = [key for key in automatic if isinstance(key, str)]
        original_keys = [
            key
            for key in keys
            if key.lower() == f"{base}-orig"
            or (key.lower().endswith("-orig") and language_base(key) == base)
        ]
        fallback_keys = ordered_language_keys(automatic, source_language)
        for key in original_keys + [key for key in fallback_keys if key not in original_keys]:
            item = first_json3_format(automatic.get(key))
            if item:
                return CaptionCandidate(
                    track_type="automatic-original",
                    language=key,
                    name=item.get("name") if isinstance(item.get("name"), str) else None,
                    ext="json3",
                )

    raise CaptionUnavailable(
        f"No uploader or original automatic JSON3 track matches {source_language}."
    )


def resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    else:
        discovered = shutil.which(value)
        if discovered:
            return Path(discovered).resolve()
    raise CaptionUnavailable(
        f"Required yt-dlp executable was not found: {value}. Run the downloader setup first."
    )


def provider_paths() -> Optional[tuple[Path, Path]]:
    plugin = DEFAULT_PLUGIN_DIR / "bgutil-ytdlp-pot-provider.zip"
    generator = DEFAULT_PROVIDER_SERVER / "src" / "generate_once.ts"
    if plugin.is_file() and generator.is_file():
        return DEFAULT_PLUGIN_DIR.resolve(), DEFAULT_PROVIDER_SERVER.resolve()
    return None


def build_caption_command(
    *,
    yt_dlp: Path,
    url: str,
    output_dir: Path,
    candidate: CaptionCandidate,
    youtube_client: str,
    provider: Optional[tuple[Path, Path]] = None,
) -> list[str]:
    command = [
        str(yt_dlp),
        "--no-playlist",
        "--abort-on-error",
        "--force-ipv4",
        "--skip-download",
        "--write-subs" if candidate.track_type == "uploader" else "--write-auto-subs",
        "--sub-langs",
        candidate.language,
        "--sub-format",
        "json3",
        "--paths",
        str(output_dir),
        "--output",
        "caption.%(ext)s",
        "--retries",
        "3",
        "--extractor-args",
        f"youtube:player_client={youtube_client}",
    ]
    if provider:
        plugin_dir, provider_server = provider
        command.extend(
            [
                "--plugin-dirs",
                str(plugin_dir),
                "--extractor-args",
                f"youtubepot-bgutilscript:server_home={provider_server}",
            ]
        )
    command.extend(["--", url])
    return command


def compact_error(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    message = lines[-1] if lines else "yt-dlp failed without an error message."
    message = re.sub(r"https?://\S+", "[url]", message)
    return message[:500]


def download_caption(
    *, source: SourceVideo, candidate: CaptionCandidate, yt_dlp: Path
) -> tuple[bytes, dict[str, Any], str, list[dict[str, str]]]:
    attempts: list[dict[str, str]] = []
    clients: list[tuple[str, Optional[tuple[Path, Path]]]] = [("web_embedded", None)]
    provider = provider_paths()
    if provider:
        clients.append(("mweb", provider))

    for youtube_client, provider_config in clients:
        with tempfile.TemporaryDirectory(prefix="studio-caption-") as temporary_name:
            temporary_dir = Path(temporary_name)
            command = build_caption_command(
                yt_dlp=yt_dlp,
                url=source.webpage_url,
                output_dir=temporary_dir,
                candidate=candidate,
                youtube_client=youtube_client,
                provider=provider_config,
            )
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False)
            except OSError as error:
                attempts.append({"youtube_client": youtube_client, "result": str(error)})
                continue
            if result.returncode != 0:
                attempts.append(
                    {
                        "youtube_client": youtube_client,
                        "result": compact_error(result.stderr or result.stdout),
                    }
                )
                continue

            captions = sorted(temporary_dir.glob("*.json3"))
            if len(captions) != 1:
                attempts.append(
                    {
                        "youtube_client": youtube_client,
                        "result": f"Expected one JSON3 file, found {len(captions)}.",
                    }
                )
                continue
            raw = captions[0].read_bytes()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                attempts.append(
                    {"youtube_client": youtube_client, "result": "Downloaded JSON3 was invalid."}
                )
                continue
            if not isinstance(payload, dict):
                attempts.append(
                    {"youtube_client": youtube_client, "result": "JSON3 root was not an object."}
                )
                continue
            attempts.append({"youtube_client": youtube_client, "result": "downloaded"})
            return raw, payload, youtube_client, attempts

    reason = attempts[-1]["result"] if attempts else "No caption download client was available."
    raise CaptionUnavailable(f"Caption download failed: {reason}", attempts=attempts)


def milliseconds(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    return None


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"^>>\s*", "", text).strip()


def content_event(raw_event: Any, source_index: int) -> Optional[dict[str, Any]]:
    if not isinstance(raw_event, dict):
        return None
    start_ms = milliseconds(raw_event.get("tStartMs"))
    raw_segments = raw_event.get("segs")
    if start_ms is None or not isinstance(raw_segments, list):
        return None
    joined = clean_text("".join(
        segment.get("utf8", "") for segment in raw_segments if isinstance(segment, dict)
    ))
    if not joined:
        return None
    return {
        "source_index": source_index,
        "start_ms": start_ms,
        "duration_ms": milliseconds(raw_event.get("dDurationMs")),
        "text": joined,
        "speaker_change": any(
            segment.get("isSpeakerChange") is True
            for segment in raw_segments
            if isinstance(segment, dict)
        ),
        "raw_segments": raw_segments,
    }


def normalize_json3(
    payload: dict[str, Any], *, language: str, automatic: bool
) -> tuple[dict[str, Any], dict[str, bool]]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise CaptionUnavailable("YouTube JSON3 has no events array.")
    raw_content_events = [
        event
        for index, raw_event in enumerate(raw_events)
        if (event := content_event(raw_event, index)) is not None
    ]
    if not raw_content_events:
        raise CaptionUnavailable("YouTube JSON3 has no timed text events.")

    events: list[dict[str, Any]] = []
    collapsed_same_start_source_indexes: list[int] = []
    for event in raw_content_events:
        if events and event["start_ms"] == events[-1]["start_ms"]:
            collapsed_same_start_source_indexes.append(events[-1]["source_index"])
            events[-1] = event
        else:
            events.append(event)

    source_event_order_monotonic = all(
        raw_content_events[index]["start_ms"]
        >= raw_content_events[index - 1]["start_ms"]
        for index in range(1, len(raw_content_events))
    )
    segments: list[dict[str, Any]] = []
    word_sequence = 0
    timestamps_monotonic = True
    timestamps_valid = True
    timestamps_non_overlapping = True
    previous_start_ms: Optional[float] = None
    previous_end_ms: Optional[float] = None

    for index, event in enumerate(events):
        start_ms = event["start_ms"]
        next_start_ms = events[index + 1]["start_ms"] if index + 1 < len(events) else None
        declared_end_ms = (
            start_ms + event["duration_ms"] if event["duration_ms"] is not None else None
        )
        if declared_end_ms is None:
            end_ms = next_start_ms if next_start_ms is not None else start_ms + 1000.0
        elif next_start_ms is None:
            end_ms = declared_end_ms
        else:
            end_ms = min(declared_end_ms, next_start_ms)

        if end_ms <= start_ms:
            timestamps_valid = False
            end_ms = start_ms
        if previous_end_ms is not None and start_ms + 1e-6 < previous_end_ms:
            timestamps_non_overlapping = False
        if previous_start_ms is not None and start_ms + 1e-6 < previous_start_ms:
            timestamps_monotonic = False

        timed_words: list[tuple[float, str]] = []
        previous_word_start = start_ms
        for raw_segment in event["raw_segments"]:
            if not isinstance(raw_segment, dict):
                continue
            word_text = clean_text(raw_segment.get("utf8"))
            if not word_text:
                continue
            offset_ms = milliseconds(raw_segment.get("tOffsetMs"))
            word_start_ms = start_ms + (offset_ms or 0.0)
            if word_start_ms < previous_word_start:
                timestamps_monotonic = False
            previous_word_start = word_start_ms
            timed_words.append((word_start_ms, word_text))

        words: list[dict[str, Any]] = []
        for word_index, (word_start_ms, word_text) in enumerate(timed_words):
            next_word_start_ms = (
                timed_words[word_index + 1][0]
                if word_index + 1 < len(timed_words)
                else end_ms
            )
            word_end_ms = min(max(next_word_start_ms, word_start_ms), end_ms)
            word_start = round(word_start_ms / 1000, 3)
            word_end = round(word_end_ms / 1000, 3)
            if (word_start_ms < start_ms or word_start_ms > end_ms
                    or word_end_ms <= word_start_ms or word_end <= word_start):
                timestamps_valid = False
            word_sequence += 1
            words.append(
                {
                    "id": f"word-{word_sequence:06d}",
                    "start": word_start,
                    "end": word_end,
                    "text": word_text,
                }
            )

        segment = {
            "id": f"seg-{index + 1:06d}",
            "start": round(start_ms / 1000, 3),
            "end": round(end_ms / 1000, 3),
            "text": event["text"],
            "words": words,
        }
        if event["speaker_change"]:
            segment["speaker_change"] = True
        segments.append(segment)
        previous_start_ms = start_ms
        previous_end_ms = end_ms

    transcript_text = " ".join(segment["text"] for segment in segments).strip()
    segments_without_words = [segment["id"] for segment in segments if not segment["words"]]
    verification = {
        "has_text": bool(transcript_text),
        "has_segments": bool(segments),
        "has_word_timestamps": word_sequence > 0,
        "timestamps_monotonic": timestamps_monotonic,
        "timestamps_valid": timestamps_valid,
        "timestamps_non_overlapping": timestamps_non_overlapping,
        "source_event_order_monotonic": source_event_order_monotonic,
    }
    return {
        "language": language,
        "text": transcript_text,
        "segment_count": len(segments),
        "word_count": word_sequence,
        "quality": {
            "source": "youtube-captions",
            "automatic": automatic,
            "raw_content_event_count": len(raw_content_events),
            "normalized_content_event_count": len(events),
            "same_start_rollup_event_count": len(
                collapsed_same_start_source_indexes
            ),
            "same_start_rollup_source_indexes": (
                collapsed_same_start_source_indexes
            ),
            "low_confidence_threshold": None,
            "low_confidence_word_count": 0,
            "low_confidence_word_ids": [],
            "segments_without_word_timestamps_count": len(segments_without_words),
            "segments_without_word_timestamps": segments_without_words,
            "word_timestamp_segment_coverage": round(
                (len(segments) - len(segments_without_words)) / len(segments), 6
            ),
            "review_required": automatic or bool(segments_without_words),
        },
        "segments": segments,
    }, verification


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def yt_dlp_version(yt_dlp: Optional[Path]) -> str:
    if not yt_dlp:
        return "unavailable"
    try:
        result = subprocess.run(
            [str(yt_dlp), "--version"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return (result.stdout or result.stderr).splitlines()[0].strip() or "unknown"


def source_summary(source: SourceVideo) -> dict[str, Any]:
    return {
        "video_id": source.video_id,
        "title": source.title,
        "media_path": str(source.media_path),
        "metadata_path": str(source.metadata_path),
        "downloader_manifest_path": str(source.manifest_path),
        "sha256": source.sha256,
        "duration_seconds": source.duration_seconds,
        "source_language": source.source_language,
    }


def selection_document(
    *,
    source: SourceVideo,
    status: str,
    candidate: Optional[CaptionCandidate],
    reason: Optional[str],
    attempts: list[dict[str, str]],
    yt_dlp: Optional[Path],
    youtube_client: Optional[str] = None,
    raw_caption_path: Optional[Path] = None,
    raw_caption_sha256: Optional[str] = None,
) -> dict[str, Any]:
    selected = None
    if candidate:
        selected = {
            "track_type": candidate.track_type,
            "language": candidate.language,
            "name": candidate.name,
            "format": candidate.ext,
            "youtube_client": youtube_client,
            "raw_caption_path": str(raw_caption_path) if raw_caption_path else None,
            "raw_caption_sha256": raw_caption_sha256,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "agent": "youtube-caption-ingestor",
        "stage": "source-selection",
        "created_at": utc_now(),
        "status": status,
        "source": source_summary(source),
        "policy": {
            "priority": ["uploader", "automatic-original", "whisper"],
            "machine_translated_tracks_allowed": False,
            "machine_translated_json3_tracks_rejected": count_machine_translated_json3(
                source.metadata
            ),
        },
        "selected": selected,
        "reason": reason,
        "attempts": attempts,
        "tools": {"yt_dlp": yt_dlp_version(yt_dlp)},
    }


def transcript_document(
    *,
    source: SourceVideo,
    candidate: CaptionCandidate,
    transcript: dict[str, Any],
    transcript_verification: dict[str, bool],
    yt_dlp: Path,
    youtube_client: str,
    raw_caption_path: Path,
    raw_caption_sha256: str,
    selection_path: Path,
) -> dict[str, Any]:
    transcript_segments = transcript.get("segments")
    last_caption_end = None
    if isinstance(transcript_segments, list) and transcript_segments:
        last_segment = transcript_segments[-1]
        if isinstance(last_segment, dict) and isinstance(last_segment.get("end"), (int, float)):
            last_caption_end = float(last_segment["end"])
    within_source_duration = (
        source.duration_seconds is None
        or (
            last_caption_end is not None
            and last_caption_end
            <= source.duration_seconds + SOURCE_DURATION_TOLERANCE_SECONDS
        )
    )
    return {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "agent": "subtitle-generator",
        "stage": "transcription",
        "created_at": utc_now(),
        "source": {
            **source_summary(source),
            "clip_start_seconds": 0.0,
            "clip_end_seconds": None,
        },
        "audio": {"required": False, "path": None},
        "asr": {
            "engine": "youtube-captions",
            "whisper_used": False,
            "language_hint": source.source_language,
            "word_timestamps": True,
        },
        "caption_source": {
            "track_type": candidate.track_type,
            "language": candidate.language,
            "name": candidate.name,
            "format": candidate.ext,
            "youtube_client": youtube_client,
            "raw_caption_path": str(raw_caption_path),
            "raw_caption_sha256": raw_caption_sha256,
            "selection_path": str(selection_path),
            "source_duration_tolerance_seconds": SOURCE_DURATION_TOLERANCE_SECONDS,
        },
        "transcript": transcript,
        "verification": {
            "source_sha256_verified": True,
            "audio_contract_verified": True,
            "within_source_duration": within_source_duration,
            **transcript_verification,
        },
        "tools": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "yt_dlp": yt_dlp_version(yt_dlp),
        },
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def default_output_dir(source: SourceVideo) -> Path:
    return DEFAULT_OUTPUT_ROOT / source.video_id / "transcription"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse uploader or original automatic YouTube captions before Whisper."
    )
    parser.add_argument("manifest", help="Verified youtube-downloader manifest.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--yt-dlp",
        default=os.environ.get("YT_DLP_BIN", str(DEFAULT_YT_DLP_BIN)),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = load_source(args.manifest)
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else default_output_dir(source)
        )
        raw_caption_path = output_dir / "caption.source.json3"
        selection_path = output_dir / "source-selection.json"
        transcript_path = output_dir / "transcript.source.json"
        existing = [
            path for path in (raw_caption_path, selection_path, transcript_path) if path.exists()
        ]
        if existing and not args.force:
            raise CaptionIngestorError(
                f"Caption artifacts already exist. Use --force to replace them: {existing[0]}"
            )

        candidate: Optional[CaptionCandidate] = None
        yt_dlp: Optional[Path] = None
        raw: Optional[bytes] = None
        youtube_client: Optional[str] = None
        attempts: list[dict[str, str]] = []
        try:
            candidate = select_caption(source.metadata, source.source_language)
            yt_dlp = resolve_executable(args.yt_dlp)
            raw, payload, youtube_client, attempts = download_caption(
                source=source, candidate=candidate, yt_dlp=yt_dlp
            )
            transcript, transcript_verification = normalize_json3(
                payload,
                language=source.source_language,
                automatic=candidate.track_type == "automatic-original",
            )
            failed = [key for key, passed in transcript_verification.items() if not passed]
            if failed:
                raise CaptionUnavailable(
                    "Caption verification failed: " + ", ".join(failed)
                )
        except CaptionUnavailable as error:
            if transcript_path.exists():
                raise CaptionIngestorError(
                    "The prior verified transcript was preserved after caption fallback failed."
                ) from error
            # Keep rejected caption bytes as evidence without admitting a transcript.
            if raw is not None:
                write_bytes_atomic(raw_caption_path, raw)
            fallback = selection_document(
                source=source,
                status="whisper_required",
                candidate=candidate,
                reason=str(error),
                attempts=error.attempts or attempts,
                yt_dlp=yt_dlp,
                youtube_client=youtube_client,
                raw_caption_path=raw_caption_path if raw is not None else None,
                raw_caption_sha256=sha256_bytes(raw) if raw is not None else None,
            )
            write_json_atomic(selection_path, fallback)
            print(f"Whisper required: {selection_path}", file=sys.stderr)
            return WHISPER_REQUIRED_EXIT

        raw_digest = sha256_bytes(raw)
        selection = selection_document(
            source=source,
            status="captions_selected",
            candidate=candidate,
            reason=None,
            attempts=attempts,
            yt_dlp=yt_dlp,
            youtube_client=youtube_client,
            raw_caption_path=raw_caption_path,
            raw_caption_sha256=raw_digest,
        )
        document = transcript_document(
            source=source,
            candidate=candidate,
            transcript=transcript,
            transcript_verification=transcript_verification,
            yt_dlp=yt_dlp,
            youtube_client=youtube_client,
            raw_caption_path=raw_caption_path,
            raw_caption_sha256=raw_digest,
            selection_path=selection_path,
        )
        failed = [key for key, passed in document["verification"].items() if not passed]
        if failed:
            raise CaptionIngestorError("Transcript verification failed: " + ", ".join(failed))

        write_bytes_atomic(raw_caption_path, raw)
        write_json_atomic(selection_path, selection)
        write_json_atomic(transcript_path, document)
        print(transcript_path)
        return 0
    except CaptionIngestorError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
