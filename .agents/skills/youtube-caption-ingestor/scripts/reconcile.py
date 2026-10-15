#!/usr/bin/env python3
"""Compare YouTube primary captions with Whisper and emit a hybrid transcript."""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_REVIEW_THRESHOLD = 0.10
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
PRIMARY_ENGINES = {
    "youtube-captions",
    "mlx-whisper-chunked",
    "faster-whisper-chunked",
}
WHISPER_ENGINES = {
    "mlx-whisper",
    "mlx-whisper-chunked",
    "faster-whisper",
    "faster-whisper-chunked",
}


class ReconciliationError(RuntimeError):
    """An expected, user-actionable transcript reconciliation failure."""


@dataclass(frozen=True)
class TranscriptArtifact:
    path: Path
    sha256: str
    document: dict[str, Any]
    video_id: str
    media_sha256: str
    clip_start: float
    clip_end: Optional[float]


@dataclass(frozen=True)
class TimedToken:
    text: str
    start: float
    end: float
    segment_id: str


def validate_engine_pair(
    primary: TranscriptArtifact, verifier: TranscriptArtifact
) -> None:
    if primary.document.get("asr", {}).get("engine") not in PRIMARY_ENGINES:
        raise ReconciliationError(
            "Primary transcript must use YouTube captions or a verified "
            "chunked Whisper fallback."
        )
    if verifier.document.get("asr", {}).get("engine") not in WHISPER_ENGINES:
        raise ReconciliationError("Verifier transcript must use Whisper.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReconciliationError(f"{label} could not be read: {path}") from error
    if not isinstance(value, dict):
        raise ReconciliationError(f"{label} must contain a JSON object: {path}")
    return value


def as_float(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) else None


def load_transcript(value: str, label: str) -> TranscriptArtifact:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ReconciliationError(f"{label} does not exist: {path}")
    document = read_json(path, label)
    if document.get("stage") != "transcription":
        raise ReconciliationError(f"{label} must be a transcription artifact.")
    verification = document.get("verification")
    if not isinstance(verification, dict) or not verification or not all(
        value is True for value in verification.values()
    ):
        raise ReconciliationError(f"Every {label} verification check must pass.")
    source = document.get("source")
    if not isinstance(source, dict):
        raise ReconciliationError(f"{label} is missing source metadata.")
    video_id = source.get("video_id")
    media_sha256 = source.get("sha256")
    if (
        not isinstance(video_id, str)
        or not video_id
        or not isinstance(media_sha256, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", media_sha256)
    ):
        raise ReconciliationError(f"{label} source identity is invalid.")
    transcript = document.get("transcript")
    segments = transcript.get("segments") if isinstance(transcript, dict) else None
    if not isinstance(segments, list) or not segments:
        raise ReconciliationError(f"{label} has no transcript segments.")
    clip_start = as_float(source.get("clip_start_seconds")) or 0.0
    clip_end = as_float(source.get("clip_end_seconds"))
    return TranscriptArtifact(
        path=path,
        sha256=sha256_file(path),
        document=document,
        video_id=video_id,
        media_sha256=media_sha256.lower(),
        clip_start=clip_start,
        clip_end=clip_end,
    )


def normalize_tokens(value: str) -> list[str]:
    return TOKEN_PATTERN.findall(value.lower().replace("’", "'"))


def timed_tokens(artifact: TranscriptArtifact) -> list[TimedToken]:
    tokens: list[TimedToken] = []
    for segment in artifact.document["transcript"]["segments"]:
        segment_id = segment.get("id")
        words = segment.get("words")
        if not isinstance(segment_id, str) or not isinstance(words, list):
            continue
        for word in words:
            if not isinstance(word, dict):
                continue
            start = as_float(word.get("start"))
            end = as_float(word.get("end"))
            text = word.get("text")
            if start is None or end is None or not isinstance(text, str):
                continue
            for token in normalize_tokens(text):
                tokens.append(TimedToken(token, start, end, segment_id))
    if not tokens:
        raise ReconciliationError(f"Transcript has no comparable words: {artifact.path}")
    return tokens


def levenshtein(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_token in enumerate(right, 1):
        current = [right_index]
        for left_index, left_token in enumerate(left, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def tokens_in_window(
    tokens: list[TimedToken], start: float, end: float, *, include_end: bool = False
) -> list[TimedToken]:
    return [
        token
        for token in tokens
        if token.start >= start - 1e-6
        and (token.start < end - 1e-6 or (include_end and token.start <= end + 0.5))
    ]


def difference_time(
    primary: list[TimedToken], verifier: list[TimedToken], fallback: float
) -> tuple[float, float]:
    available = primary + verifier
    if not available:
        return fallback, fallback
    return min(token.start for token in available), max(token.end for token in available)


def compare_window(
    *,
    primary: list[TimedToken],
    verifier: list[TimedToken],
    start: float,
    end: float,
    review_threshold: float,
) -> dict[str, Any]:
    left = [token.text for token in primary]
    right = [token.text for token in verifier]
    edit_distance = levenshtein(left, right)
    disagreement_rate = edit_distance / max(1, len(left))
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    differences: list[dict[str, Any]] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        primary_part = primary[left_start:left_end]
        verifier_part = verifier[right_start:right_end]
        difference_start, difference_end = difference_time(
            primary_part, verifier_part, start
        )
        differences.append(
            {
                "type": tag,
                "start": round(difference_start, 3),
                "end": round(difference_end, 3),
                "primary_text": " ".join(token.text for token in primary_part),
                "whisper_text": " ".join(token.text for token in verifier_part),
                "primary_segment_ids": list(
                    dict.fromkeys(token.segment_id for token in primary_part)
                ),
            }
        )
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "primary_token_count": len(left),
        "whisper_token_count": len(right),
        "edit_distance": edit_distance,
        "disagreement_rate": round(disagreement_rate, 6),
        "review_required": disagreement_rate >= review_threshold,
        "differences": differences,
    }


def coverage_range(
    primary: TranscriptArtifact, verifier: TranscriptArtifact
) -> tuple[float, float]:
    primary_duration = as_float(primary.document.get("source", {}).get("duration_seconds"))
    verifier_duration = as_float(verifier.document.get("source", {}).get("duration_seconds"))
    starts = [primary.clip_start, verifier.clip_start]
    ends = [value for value in (primary.clip_end, verifier.clip_end) if value is not None]
    if not ends:
        known_durations = [value for value in (primary_duration, verifier_duration) if value]
        if not known_durations:
            raise ReconciliationError("Transcript coverage end could not be determined.")
        ends.append(min(known_durations))
    start = max(starts)
    end = min(ends)
    if end <= start:
        raise ReconciliationError("Primary and Whisper transcripts do not overlap.")
    return start, end


def build_report(
    *,
    primary: TranscriptArtifact,
    verifier: TranscriptArtifact,
    window_seconds: float,
    review_threshold: float,
) -> dict[str, Any]:
    if primary.video_id != verifier.video_id:
        raise ReconciliationError("Transcript video IDs do not match.")
    if primary.media_sha256 != verifier.media_sha256:
        raise ReconciliationError("Transcript media SHA-256 values do not match.")
    if window_seconds <= 0 or not 0 <= review_threshold <= 1:
        raise ReconciliationError("Comparison window or review threshold is invalid.")

    start, end = coverage_range(primary, verifier)
    primary_tokens = timed_tokens(primary)
    verifier_tokens = timed_tokens(verifier)
    windows: list[dict[str, Any]] = []
    cursor = start
    while cursor < end - 1e-6:
        window_end = min(end, cursor + window_seconds)
        windows.append(
            compare_window(
                primary=tokens_in_window(
                    primary_tokens, cursor, window_end, include_end=window_end == end
                ),
                verifier=tokens_in_window(
                    verifier_tokens, cursor, window_end, include_end=window_end == end
                ),
                start=cursor,
                end=window_end,
                review_threshold=review_threshold,
            )
        )
        cursor = window_end

    primary_covered = tokens_in_window(primary_tokens, start, end, include_end=True)
    verifier_covered = tokens_in_window(verifier_tokens, start, end, include_end=True)
    overall_distance = levenshtein(
        [token.text for token in primary_covered],
        [token.text for token in verifier_covered],
    )
    differences = [difference for window in windows for difference in window["differences"]]
    review_segment_ids = list(
        dict.fromkeys(
            segment_id
            for difference in differences
            for segment_id in difference["primary_segment_ids"]
        )
    )
    review_windows = [
        {key: value for key, value in window.items() if key != "differences"}
        for window in windows
        if window["review_required"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "agent": "youtube-caption-ingestor",
        "stage": "hybrid-verification",
        "created_at": utc_now(),
        "status": "review_required" if differences else "matched",
        "source": {
            "video_id": primary.video_id,
            "media_sha256": primary.media_sha256,
            "coverage_start_seconds": start,
            "coverage_end_seconds": end,
        },
        "inputs": {
            "primary": {
                "engine": primary.document.get("asr", {}).get("engine"),
                "path": str(primary.path),
                "sha256": primary.sha256,
            },
            "whisper": {
                "engine": verifier.document.get("asr", {}).get("engine"),
                "path": str(verifier.path),
                "sha256": verifier.sha256,
            },
        },
        "comparison": {
            "window_seconds": window_seconds,
            "review_threshold": review_threshold,
            "primary_token_count": len(primary_covered),
            "whisper_token_count": len(verifier_covered),
            "edit_distance": overall_distance,
            "disagreement_rate": round(
                overall_distance / max(1, len(primary_covered)), 6
            ),
            "difference_count": len(differences),
            "review_window_count": len(review_windows),
            "review_segment_count": len(review_segment_ids),
            "review_segment_ids": review_segment_ids,
        },
        "review_windows": review_windows,
        "differences": differences,
        "verification": {
            "video_id_match": True,
            "media_sha256_match": True,
            "primary_verified": True,
            "whisper_verified": True,
            "comparison_complete": True,
        },
    }


def words_in_coverage(
    words: list[dict[str, Any]], start: float, end: float, *, include_end: bool
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for word in words:
        word_start = as_float(word.get("start"))
        word_end = as_float(word.get("end"))
        if word_start is None or word_end is None:
            continue
        midpoint = (word_start + word_end) / 2
        if midpoint >= start - 1e-6 and (
            midpoint < end - 1e-6 or (include_end and midpoint <= end + 0.5)
        ):
            selected.append(copy.deepcopy(word))
    return selected


def clip_primary_transcript(
    primary: TranscriptArtifact, *, start: float, end: float
) -> dict[str, Any]:
    original = primary.document["transcript"]
    clipped_segments: list[dict[str, Any]] = []
    for raw_segment in original["segments"]:
        words = raw_segment.get("words")
        if not isinstance(words, list):
            continue
        selected = words_in_coverage(words, start, end, include_end=True)
        if not selected:
            continue
        segment = copy.deepcopy(raw_segment)
        segment["words"] = selected
        segment["start"] = selected[0]["start"]
        segment["end"] = selected[-1]["end"]
        if len(selected) != len(words):
            segment["text"] = " ".join(str(word["text"]).strip() for word in selected)
        clipped_segments.append(segment)
    if not clipped_segments:
        raise ReconciliationError("Primary transcript has no words in Whisper coverage.")
    transcript = copy.deepcopy(original)
    transcript["segments"] = clipped_segments
    transcript["segment_count"] = len(clipped_segments)
    transcript["word_count"] = sum(len(segment["words"]) for segment in clipped_segments)
    transcript["text"] = " ".join(segment["text"] for segment in clipped_segments)
    return transcript


def hybrid_document(
    *,
    primary: TranscriptArtifact,
    verifier: TranscriptArtifact,
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    start = float(report["source"]["coverage_start_seconds"])
    end = float(report["source"]["coverage_end_seconds"])
    document = copy.deepcopy(primary.document)
    document["created_at"] = utc_now()
    document["agent"] = "subtitle-generator"
    document["source"]["clip_start_seconds"] = start
    document["source"]["clip_end_seconds"] = end
    document["transcript"] = clip_primary_transcript(primary, start=start, end=end)
    base_quality = document["transcript"].get("quality")
    if not isinstance(base_quality, dict):
        base_quality = {}
        document["transcript"]["quality"] = base_quality
    review_ids = report["comparison"]["review_segment_ids"]
    base_quality["hybrid_verification"] = {
        "whisper_engine": verifier.document.get("asr", {}).get("engine"),
        "disagreement_rate": report["comparison"]["disagreement_rate"],
        "difference_count": report["comparison"]["difference_count"],
        "review_window_count": report["comparison"]["review_window_count"],
        "review_segment_count": len(review_ids),
        "review_segment_ids": review_ids,
    }
    base_quality["review_required"] = bool(
        base_quality.get("review_required") or review_ids
    )
    document["asr"] = {
        "engine": "hybrid-youtube-whisper",
        "primary_engine": primary.document.get("asr", {}).get("engine"),
        "verifier_engine": verifier.document.get("asr", {}).get("engine"),
        "whisper_used": True,
        "word_timestamps": True,
    }
    document["hybrid_verification"] = {
        "policy": "youtube-primary-whisper-verifier",
        "automatic_correction": False,
        "primary_path": str(primary.path),
        "primary_sha256": primary.sha256,
        "whisper_path": str(verifier.path),
        "whisper_sha256": verifier.sha256,
        "report_path": str(report_path),
    }
    document["verification"] = {
        "source_sha256_verified": True,
        "primary_transcript_verified": True,
        "whisper_transcript_verified": True,
        "video_id_match": True,
        "media_sha256_match": True,
        "comparison_complete": True,
        "primary_text_preserved": True,
    }
    return document


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep YouTube captions primary and flag disagreements with Whisper."
    )
    parser.add_argument("primary", help="YouTube transcript.source.json")
    parser.add_argument("whisper", help="Chunked Whisper transcript JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument(
        "--review-threshold", type=float, default=DEFAULT_REVIEW_THRESHOLD
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        primary = load_transcript(args.primary, "Primary transcript")
        verifier = load_transcript(args.whisper, "Whisper transcript")
        validate_engine_pair(primary, verifier)
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else primary.path.parent
        )
        report_path = output_dir / "hybrid-verification.json"
        transcript_path = output_dir / "transcript.hybrid.json"
        if (report_path.exists() or transcript_path.exists()) and not args.force:
            raise ReconciliationError(
                f"Hybrid transcript already exists. Use --force: {transcript_path}"
            )
        report = build_report(
            primary=primary,
            verifier=verifier,
            window_seconds=args.window_seconds,
            review_threshold=args.review_threshold,
        )
        document = hybrid_document(
            primary=primary,
            verifier=verifier,
            report=report,
            report_path=report_path,
        )
        failed = [key for key, passed in document["verification"].items() if not passed]
        if failed:
            raise ReconciliationError(
                "Hybrid transcript verification failed: " + ", ".join(failed)
            )
        write_json_atomic(report_path, report)
        write_json_atomic(transcript_path, document)
        print(transcript_path)
        return 0
    except ReconciliationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
