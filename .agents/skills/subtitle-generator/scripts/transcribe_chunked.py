#!/usr/bin/env python3
"""Transcribe bounded clips with MLX Whisper and merge their owned ranges."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import transcribe as core  # noqa: E402


CHUNK_SCHEMA_VERSION = 1
DEFAULT_CHUNK_SECONDS = 60.0
DEFAULT_OVERLAP_SECONDS = 0.0
MAX_SEGMENT_WORDS = 12
MAX_SEGMENT_SECONDS = 6.0
SEGMENT_GAP_SECONDS = 0.8


class ChunkedTranscriptionError(RuntimeError):
    """An expected, user-actionable chunked transcription failure."""


@dataclass(frozen=True)
class ChunkRange:
    index: int
    core_start: float
    core_end: float
    decode_start: float
    decode_end: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_chunk_ranges(
    *,
    duration: float,
    start: float = 0.0,
    end: Optional[float] = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> list[ChunkRange]:
    if duration <= 0:
        raise ChunkedTranscriptionError("Source duration must be positive.")
    requested_end = duration if end is None else end
    if start < 0 or start >= requested_end or requested_end > duration + 0.5:
        raise ChunkedTranscriptionError("Requested transcription range is invalid.")
    if chunk_seconds <= 0:
        raise ChunkedTranscriptionError("--chunk-seconds must be positive.")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ChunkedTranscriptionError(
            "--overlap-seconds must be nonnegative and smaller than a chunk."
        )

    ranges: list[ChunkRange] = []
    core_start = start
    index = 0
    while core_start < requested_end - 1e-6:
        core_end = min(requested_end, core_start + chunk_seconds)
        ranges.append(
            ChunkRange(
                index=index,
                core_start=round(core_start, 3),
                core_end=round(core_end, 3),
                decode_start=round(max(0.0, core_start - overlap_seconds), 3),
                decode_end=round(min(duration, core_end + overlap_seconds), 3),
            )
        )
        core_start = core_end
        index += 1
    return ranges


def chunk_metadata(value: ChunkRange) -> dict[str, Any]:
    return {
        "index": value.index,
        "core_start_seconds": value.core_start,
        "core_end_seconds": value.core_end,
        "decode_start_seconds": value.decode_start,
        "decode_end_seconds": value.decode_end,
    }


def chunk_directory(output_dir: Path, value: ChunkRange) -> Path:
    return output_dir / "chunks" / f"{value.index:04d}"


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChunkedTranscriptionError(f"{label} could not be read: {path}") from error
    if not isinstance(value, dict):
        raise ChunkedTranscriptionError(f"{label} must contain a JSON object: {path}")
    return value


def verified_chunk_document(
    path: Path,
    *,
    source: core.SourceMedia,
    chunk: ChunkRange,
    model_id: str,
    model_revision: str,
    language: str,
    prompt: Optional[str],
    backend: str = "mlx-whisper",
) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    document = read_json(path, "Chunk transcript")
    verification = document.get("verification")
    source_data = document.get("source")
    chunk_data = document.get("chunk")
    asr_data = document.get("asr")
    if (
        not isinstance(verification, dict)
        or not verification
        or not all(value is True for value in verification.values())
        or not isinstance(source_data, dict)
        or source_data.get("video_id") != source.video_id
        or source_data.get("sha256") != source.sha256
        or not isinstance(chunk_data, dict)
        or chunk_data != chunk_metadata(chunk)
        or not isinstance(asr_data, dict)
        or asr_data.get("engine") != backend
        or asr_data.get("model_id") != model_id
        or asr_data.get("model_revision") != model_revision
        or asr_data.get("language_hint") != language
        or asr_data.get("initial_prompt") != prompt
    ):
        raise ChunkedTranscriptionError(
            f"Existing chunk does not match this run. Use --force: {path}"
        )
    return document


def transcribe_chunk(
    *,
    source: core.SourceMedia,
    chunk: ChunkRange,
    output_dir: Path,
    ffmpeg: Path,
    ffprobe: Path,
    model_path: Path | str,
    model_id: str,
    model_revision: str,
    language: str,
    prompt: Optional[str],
    force: bool,
    backend: str = "mlx-whisper",
    model_cache: Path = core.DEFAULT_MODEL_CACHE,
) -> tuple[Path, dict[str, Any]]:
    directory = chunk_directory(output_dir, chunk)
    transcript_path = directory / "transcript.source.json"
    if not force:
        existing = verified_chunk_document(
            transcript_path,
            source=source,
            chunk=chunk,
            model_id=model_id,
            model_revision=model_revision,
            language=language,
            prompt=prompt,
            backend=backend,
        )
        if existing:
            return transcript_path, existing

    audio_path = directory / "audio.wav"
    core.extract_audio(
        ffmpeg=ffmpeg,
        media_path=source.media_path,
        audio_path=audio_path,
        clip_start=chunk.decode_start,
        clip_end=chunk.decode_end,
    )
    audio_probe = core.probe_audio(ffprobe, audio_path)
    raw_result = core.run_whisper_backend(
        backend=backend,
        audio_path=audio_path,
        model=model_path,
        model_revision=model_revision,
        model_cache=model_cache,
        language=language,
        prompt=prompt,
    )
    transcript, transcript_verification = core.normalize_transcript(
        raw_result, clip_start=chunk.decode_start
    )
    document = core.create_document(
        source=source,
        audio_path=audio_path,
        audio_probe=audio_probe,
        transcript=transcript,
        transcript_verification=transcript_verification,
        model_id=model_id,
        model_revision=model_revision,
        language=language,
        prompt=prompt,
        clip_start=chunk.decode_start,
        clip_end=chunk.decode_end,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        backend=backend,
    )
    document["chunk"] = chunk_metadata(chunk)
    failed = [key for key, passed in document["verification"].items() if not passed]
    if failed:
        raise ChunkedTranscriptionError(
            f"Chunk {chunk.index} verification failed: {', '.join(failed)}"
        )
    core.write_json_atomic(transcript_path, document)
    return transcript_path, document


def canonical_word(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", value.lower()))


def words_for_core(
    document: dict[str, Any], chunk: ChunkRange, *, is_last: bool
) -> list[dict[str, Any]]:
    transcript = document.get("transcript")
    segments = transcript.get("segments") if isinstance(transcript, dict) else None
    if not isinstance(segments, list):
        raise ChunkedTranscriptionError(f"Chunk {chunk.index} has no segments.")

    words: list[dict[str, Any]] = []
    for segment in segments:
        raw_words = segment.get("words") if isinstance(segment, dict) else None
        if not isinstance(raw_words, list):
            continue
        for raw_word in raw_words:
            if not isinstance(raw_word, dict):
                continue
            start = raw_word.get("start")
            end = raw_word.get("end")
            text = raw_word.get("text")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not isinstance(text, str)
                or end < start
            ):
                continue
            midpoint = (float(start) + float(end)) / 2
            in_core = midpoint >= chunk.core_start - 1e-6 and (
                midpoint < chunk.core_end - 1e-6
                or (is_last and midpoint <= chunk.core_end + 1e-6)
            )
            if in_core:
                word = {
                    key: copy.deepcopy(value)
                    for key, value in raw_word.items()
                    if key != "id"
                }
                word["chunk_index"] = chunk.index
                words.append(word)
    return words


def deduplicate_boundary_words(words: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    ordered = sorted(words, key=lambda item: (float(item["start"]), float(item["end"])))
    kept: list[dict[str, Any]] = []
    removed = 0
    for word in ordered:
        token = canonical_word(str(word["text"]))
        duplicate = False
        for previous in reversed(kept):
            delta = float(word["start"]) - float(previous["start"])
            if delta > 0.5:
                break
            if (
                previous.get("chunk_index") != word.get("chunk_index")
                and abs(delta) <= 0.5
                and canonical_word(str(previous["text"])) == token
            ):
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(word)
    return kept, removed


def should_split_segment(
    current: list[dict[str, Any]], next_word: dict[str, Any]
) -> bool:
    if not current:
        return False
    gap = float(next_word["start"]) - float(current[-1]["end"])
    duration = float(next_word["end"]) - float(current[0]["start"])
    return (
        len(current) >= MAX_SEGMENT_WORDS
        or gap > SEGMENT_GAP_SECONDS
        or duration > MAX_SEGMENT_SECONDS
    )


def segment_words(words: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        if should_split_segment(current, word):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)

    segments: list[dict[str, Any]] = []
    low_confidence_word_ids: list[str] = []
    word_sequence = 0
    for segment_index, group in enumerate(groups, 1):
        normalized_words: list[dict[str, Any]] = []
        for raw_word in group:
            word_sequence += 1
            word = {
                key: copy.deepcopy(value)
                for key, value in raw_word.items()
                if key not in {"id", "chunk_index"}
            }
            word["id"] = f"word-{word_sequence:06d}"
            word = {"id": word.pop("id"), **word}
            probability = word.get("probability")
            if isinstance(probability, (int, float)) and probability < core.LOW_CONFIDENCE_THRESHOLD:
                low_confidence_word_ids.append(word["id"])
            normalized_words.append(word)
        text = " ".join(str(word["text"]).strip() for word in normalized_words).strip()
        segments.append(
            {
                "id": f"seg-{segment_index:06d}",
                "start": round(float(normalized_words[0]["start"]), 3),
                "end": round(float(normalized_words[-1]["end"]), 3),
                "text": text,
                "words": normalized_words,
            }
        )
    return segments, low_confidence_word_ids


def merge_chunk_documents(
    *,
    source: core.SourceMedia,
    chunks: list[ChunkRange],
    documents: list[dict[str, Any]],
    requested_start: float,
    requested_end: float,
    chunk_seconds: float,
    overlap_seconds: float,
    model_id: str,
    model_revision: str,
    language: str,
    chunk_manifest_path: Path,
    backend: str = "mlx-whisper",
) -> dict[str, Any]:
    if len(chunks) != len(documents) or not chunks:
        raise ChunkedTranscriptionError("Chunk document collection is incomplete.")
    collected: list[dict[str, Any]] = []
    review_chunks: list[int] = []
    for index, (chunk, document) in enumerate(zip(chunks, documents)):
        collected.extend(
            words_for_core(document, chunk, is_last=index == len(chunks) - 1)
        )
        quality = document.get("transcript", {}).get("quality", {})
        if isinstance(quality, dict) and quality.get("review_required") is True:
            review_chunks.append(chunk.index)

    words, deduplicated_count = deduplicate_boundary_words(collected)
    invalid_timestamp_words = [
        word
        for word in words
        if float(word["start"]) < 0 or float(word["end"]) <= float(word["start"])
    ]
    words = [
        word
        for word in words
        if float(word["start"]) >= 0 and float(word["end"]) > float(word["start"])
    ]
    if not words:
        raise ChunkedTranscriptionError("Chunked Whisper produced no core-range words.")
    segments, low_confidence_ids = segment_words(words)
    timestamps_monotonic = all(
        float(words[index]["start"]) >= float(words[index - 1]["start"]) - 0.01
        for index in range(1, len(words))
    )
    timestamps_valid = all(
        float(word["start"]) >= 0 and float(word["end"]) >= float(word["start"])
        for word in words
    )
    transcript_text = " ".join(segment["text"] for segment in segments)
    verification = {
        "source_sha256_verified": source.sha256_verified,
        "all_chunks_verified": all(
            all(value is True for value in document["verification"].values())
            for document in documents
        ),
        "has_text": bool(transcript_text),
        "has_segments": bool(segments),
        "has_word_timestamps": bool(words),
        "timestamps_monotonic": timestamps_monotonic,
        "timestamps_valid": timestamps_valid,
    }
    return {
        "schema_version": core.TRANSCRIPT_SCHEMA_VERSION,
        "agent": "subtitle-generator",
        "stage": "transcription",
        "created_at": utc_now(),
        "source": {
            "video_id": source.video_id,
            "title": source.title,
            "media_path": str(source.media_path),
            "downloader_manifest_path": (
                str(source.manifest_path) if source.manifest_path else None
            ),
            "sha256": source.sha256,
            "duration_seconds": source.duration_seconds,
            "clip_start_seconds": requested_start,
            "clip_end_seconds": requested_end,
        },
        "audio": {
            "required": True,
            "chunk_count": len(chunks),
            "chunk_manifest_path": str(chunk_manifest_path),
        },
        "asr": {
            "engine": f"{backend}-chunked",
            "model_id": model_id,
            "model_revision": model_revision,
            "language_hint": language,
            "word_timestamps": True,
            "temperature": 0.0,
            "condition_on_previous_text": False,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
        },
        "transcript": {
            "language": language or next((d.get("transcript", {}).get("language") for d in documents if d.get("transcript", {}).get("language")), None),
            "text": transcript_text,
            "segment_count": len(segments),
            "word_count": len(words),
            "quality": {
                "low_confidence_threshold": core.LOW_CONFIDENCE_THRESHOLD,
                "low_confidence_word_count": len(low_confidence_ids),
                "low_confidence_word_ids": low_confidence_ids,
                "review_chunk_count": len(review_chunks),
                "review_chunks": review_chunks,
                "boundary_word_deduplication_count": deduplicated_count,
                "invalid_timestamp_word_count": len(invalid_timestamp_words),
                "review_required": bool(
                    low_confidence_ids or review_chunks or invalid_timestamp_words
                ),
            },
            "segments": segments,
        },
        "verification": verification,
        "tools": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def manifest_document(
    *,
    source: core.SourceMedia,
    chunks: list[ChunkRange],
    transcript_paths: list[Path],
    output_path: Path,
    output_sha256: str,
    chunk_seconds: float,
    overlap_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "agent": "subtitle-generator",
        "stage": "chunked-transcription-manifest",
        "created_at": utc_now(),
        "status": "complete",
        "source": {
            "video_id": source.video_id,
            "sha256": source.sha256,
            "duration_seconds": source.duration_seconds,
        },
        "chunking": {
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "chunk_count": len(chunks),
        },
        "chunks": [
            {
                **chunk_metadata(chunk),
                "transcript_path": str(path),
                "transcript_sha256": core.sha256_file(path),
            }
            for chunk, path in zip(chunks, transcript_paths)
        ],
        "output": {
            "transcript_path": str(output_path),
            "transcript_sha256": output_sha256,
        },
        "verification": {
            "source_sha256_verified": source.sha256_verified,
            "all_chunk_transcripts_present": all(path.is_file() for path in transcript_paths),
            "chunk_count_match": len(chunks) == len(transcript_paths),
            "merged_transcript_present": output_path.is_file(),
            "merged_transcript_sha256_verified": (
                output_path.is_file() and core.sha256_file(output_path) == output_sha256
            ),
        },
    }


def default_output_dir(source: core.SourceMedia) -> Path:
    return core.DEFAULT_OUTPUT_ROOT / source.video_id / "verification" / "whisper-chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded MLX Whisper chunks and merge their owned core ranges."
    )
    parser.add_argument("input", help="Verified youtube-downloader manifest.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS)
    parser.add_argument("--language", default="en")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt")
    group.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--model", default=core.DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=core.DEFAULT_MODEL_REVISION)
    parser.add_argument("--model-cache", type=Path, default=core.DEFAULT_MODEL_CACHE)
    parser.add_argument(
        "--backend",
        choices=("auto", "mlx-whisper", "faster-whisper"),
        default=os.environ.get("STUDIO_ASR_BACKEND", "auto"),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.language == "auto":
        args.language = None
    try:
        backend = core.resolve_asr_backend(args.backend)
        source = core.load_source(args.input)
        if source.duration_seconds is None:
            raise ChunkedTranscriptionError(
                "Chunked transcription requires a known source duration."
            )
        requested_end = source.duration_seconds if args.end is None else args.end
        chunks = build_chunk_ranges(
            duration=source.duration_seconds,
            start=args.start,
            end=requested_end,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else default_output_dir(source)
        )
        merged_path = output_dir / "transcript.whisper-chunked.json"
        manifest_path = output_dir / "manifest.json"
        if (merged_path.exists() or manifest_path.exists()) and not args.force:
            raise ChunkedTranscriptionError(
                f"Chunked transcript already exists. Use --force: {merged_path}"
            )

        ffmpeg = core.resolve_executable(args.ffmpeg, "Install it with: brew install ffmpeg")
        ffprobe = core.resolve_executable(args.ffprobe, "Install it with: brew install ffmpeg")
        os.environ.setdefault("HF_HOME", str(args.model_cache.expanduser().resolve()))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        model_id = core.backend_model_id(backend, args.model)
        model_revision = core.backend_model_revision(
            backend, args.model, args.model_revision
        )
        model_path = core.prepare_asr_model(
            backend,
            model_id,
            model_revision,
            args.model_cache.expanduser().resolve(),
        )
        prompt = None if args.no_prompt else (args.prompt if args.prompt is not None else source.title)
        transcript_paths: list[Path] = []
        documents: list[dict[str, Any]] = []
        for chunk in chunks:
            path, document = transcribe_chunk(
                source=source,
                chunk=chunk,
                output_dir=output_dir,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                model_path=model_path,
                model_id=model_id,
                model_revision=model_revision,
                language=args.language,
                prompt=prompt,
                force=args.force,
                backend=backend,
                model_cache=args.model_cache.expanduser().resolve(),
            )
            transcript_paths.append(path)
            documents.append(document)
            print(f"Verified Whisper chunk {chunk.index + 1}/{len(chunks)}", file=sys.stderr)

        merged = merge_chunk_documents(
            source=source,
            chunks=chunks,
            documents=documents,
            requested_start=args.start,
            requested_end=requested_end,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            model_id=model_id,
            model_revision=model_revision,
            language=args.language,
            chunk_manifest_path=manifest_path,
            backend=backend,
        )
        failed = [key for key, passed in merged["verification"].items() if not passed]
        if failed:
            raise ChunkedTranscriptionError(
                "Merged transcript verification failed: " + ", ".join(failed)
            )
        core.write_json_atomic(merged_path, merged)
        manifest = manifest_document(
            source=source,
            chunks=chunks,
            transcript_paths=transcript_paths,
            output_path=merged_path,
            output_sha256=core.sha256_file(merged_path),
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )
        manifest_failed = [
            key for key, passed in manifest["verification"].items() if not passed
        ]
        if manifest_failed:
            raise ChunkedTranscriptionError(
                "Chunk manifest verification failed: " + ", ".join(manifest_failed)
            )
        core.write_json_atomic(manifest_path, manifest)
        print(merged_path)
        return 0
    except (core.SubtitleGeneratorError, ChunkedTranscriptionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
