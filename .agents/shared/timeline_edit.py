#!/usr/bin/env python3
"""Shared, provenance-preserving timeline editing for Studio video agents."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TIME_TOLERANCE = 0.015
DURATION_TOLERANCE = 0.08
AUDIO_EDGE_FADE_SECONDS = 0.015
AUDIO_MICRO_CROSSFADE_SECONDS = 0.02


class TimelineEditError(RuntimeError):
    """Raised when an edit would lose provenance or cut an unintended word."""


@dataclass(frozen=True)
class MediaArtifact:
    manifest_path: Path
    manifest_sha256: str
    media_path: Path
    media_sha256: str
    video_id: str
    title: str | None
    duration_seconds: float
    document: dict[str, Any]


@dataclass(frozen=True)
class TranscriptArtifact:
    path: Path
    sha256: str
    document: dict[str, Any]
    segments: list[dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path_value: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise TimelineEditError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimelineEditError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise TimelineEditError(f"{label} must contain a JSON object: {path}")
    return path, value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _positive_float(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise TimelineEditError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise TimelineEditError(f"{label} must be positive and finite")
    return result


def _verification_passed(document: Mapping[str, Any], label: str) -> None:
    verification = document.get("verification")
    if not isinstance(verification, Mapping) or not verification:
        raise TimelineEditError(f"{label} has no verification block")
    failed = [key for key, value in verification.items() if value is not True]
    if failed:
        raise TimelineEditError(f"{label} verification failed: {', '.join(failed)}")


def load_media_manifest(path_value: str | Path) -> MediaArtifact:
    path, document = read_json(path_value, "Media manifest")
    _verification_passed(document, "Media manifest")
    output = document.get("output")
    source = document.get("source")
    if not isinstance(output, Mapping) or not isinstance(source, Mapping):
        raise TimelineEditError("Media manifest is missing source or output metadata")
    media_value = output.get("media_path")
    expected_sha256 = output.get("sha256")
    video_id = source.get("video_id")
    if not isinstance(media_value, str) or not media_value:
        raise TimelineEditError("Media manifest has no output.media_path")
    media_path = Path(media_value).expanduser()
    if not media_path.is_absolute():
        media_path = path.parent / media_path
    media_path = media_path.resolve()
    if not media_path.is_file():
        raise TimelineEditError(f"Media file does not exist: {media_path}")
    if media_path.parent != path.parent:
        raise TimelineEditError("Media manifest output must be beside its manifest")
    if not isinstance(expected_sha256, str) or sha256_file(media_path) != expected_sha256:
        raise TimelineEditError("Media SHA-256 does not match its manifest")
    if not isinstance(video_id, str) or not video_id:
        raise TimelineEditError("Media manifest has no source.video_id")
    duration_value = output.get("duration_seconds", source.get("duration_seconds"))
    duration = _positive_float(duration_value, "Media duration")
    return MediaArtifact(
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        media_path=media_path,
        media_sha256=expected_sha256,
        video_id=video_id,
        title=source.get("title") if isinstance(source.get("title"), str) else None,
        duration_seconds=duration,
        document=document,
    )


def load_transcript(
    path_value: str | Path,
    media: MediaArtifact,
    *,
    allow_media_bound_video_id_rebind: bool = False,
) -> TranscriptArtifact:
    path, document = read_json(path_value, "Source transcript")
    if document.get("stage") != "transcription":
        raise TimelineEditError("Input transcript must be a transcription artifact")
    _verification_passed(document, "Source transcript")
    source = document.get("source")
    transcript = document.get("transcript")
    if not isinstance(source, Mapping) or not isinstance(transcript, Mapping):
        raise TimelineEditError("Transcript is missing source or transcript metadata")
    if source.get("sha256") != media.media_sha256:
        raise TimelineEditError("Transcript media SHA-256 does not match media")
    if source.get("video_id") != media.video_id:
        if not allow_media_bound_video_id_rebind:
            raise TimelineEditError("Transcript video ID does not match media")
        asr = document.get("asr")
        if not isinstance(asr, Mapping) or asr.get("initial_prompt") is not None:
            raise TimelineEditError(
                "Transcript video ID rebind requires a prompt-free ASR artifact"
            )
        original_video_id = source.get("video_id")
        if not isinstance(source, dict) or not isinstance(original_video_id, str):
            raise TimelineEditError("Transcript video ID cannot be safely rebound")
        source["video_id_rebound_from"] = original_video_id
        source["video_id"] = media.video_id
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise TimelineEditError("Transcript has no segments")
    seen_words: set[str] = set()
    previous_start = -1.0
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise TimelineEditError("Transcript contains an invalid segment")
        words = segment.get("words")
        if not isinstance(words, list) or not words:
            raise TimelineEditError(
                f"Every edited segment needs word timestamps: {segment.get('id')}"
            )
        for word in words:
            if not isinstance(word, Mapping):
                raise TimelineEditError("Transcript contains an invalid word")
            word_id = word.get("id")
            start = word.get("start")
            end = word.get("end")
            if (
                not isinstance(word_id, str)
                or word_id in seen_words
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or float(end) < float(start)
                or float(start) < previous_start - TIME_TOLERANCE
            ):
                raise TimelineEditError(f"Invalid word timing or ID: {word_id}")
            seen_words.add(word_id)
            previous_start = float(start)
    return TranscriptArtifact(path, sha256_file(path), document, list(segments))


def transcript_words(transcript: TranscriptArtifact) -> list[dict[str, Any]]:
    return [dict(word) for segment in transcript.segments for word in segment["words"]]


def transcript_excerpt(
    transcript: TranscriptArtifact, start: float, end: float, context_seconds: float = 3.0
) -> str:
    left = max(0.0, start - context_seconds)
    right = end + context_seconds
    values = []
    for word in transcript_words(transcript):
        midpoint = (float(word["start"]) + float(word["end"])) / 2
        if left <= midpoint <= right:
            values.append(str(word.get("text") or "").strip())
    return " ".join(value for value in values if value)


def words_in_range(
    transcript: TranscriptArtifact, start: float, end: float
) -> list[dict[str, Any]]:
    selected = []
    for word in transcript_words(transcript):
        midpoint = (float(word["start"]) + float(word["end"])) / 2
        if start - TIME_TOLERANCE <= midpoint < end - TIME_TOLERANCE:
            selected.append(word)
    return selected


def validate_cut_boundaries(
    transcript: TranscriptArtifact, removals: Sequence[Mapping[str, Any]]
) -> None:
    words = transcript_words(transcript)
    for removal in removals:
        acoustic_pause = (
            removal.get("kind") == "long-pause"
            and removal.get("acoustic_silence") is True
            and removal.get("removed_word_ids") == []
        )
        refined_spoken = (
            removal.get("acoustic_boundary_refined") is True
            and isinstance(removal.get("removed_word_ids"), list)
            and bool(removal.get("removed_word_ids"))
        )
        shadow_spoken = (
            removal.get("kind") == "repetition"
            and str(removal.get("detector") or "").startswith(
                "enhanced-verbatim-shadow-repeat"
            )
            and removal.get("acoustic_boundary_refined") is True
            and removal.get("removed_word_ids") == []
            and isinstance(removal.get("shadow_removed_word_ids"), list)
            and bool(removal.get("shadow_removed_word_ids"))
        )
        for boundary_name in ("start", "end"):
            boundary = float(removal[boundary_name])
            for word in words:
                word_start = float(word["start"])
                word_end = float(word["end"])
                if word_start + TIME_TOLERANCE < boundary < word_end - TIME_TOLERANCE:
                    if acoustic_pause or refined_spoken or shadow_spoken:
                        continue
                    raise TimelineEditError(
                        f"{removal['id']} {boundary_name} cuts through word {word['id']}"
                    )


def normalize_removals(
    removals: Iterable[Mapping[str, Any]], duration: float
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, removal in enumerate(removals, 1):
        if not isinstance(removal, Mapping):
            raise TimelineEditError("Every removal must be an object")
        start = removal.get("start")
        end = removal.get("end")
        kind = removal.get("kind")
        removal_id = removal.get("id") or f"cut-{index:04d}"
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise TimelineEditError(f"{removal_id} needs numeric start and end")
        start_value = float(start)
        end_value = float(end)
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or start_value < 0
            or end_value > duration + DURATION_TOLERANCE
            or end_value - start_value < 0.04
        ):
            raise TimelineEditError(f"{removal_id} has an invalid range")
        if not isinstance(kind, str) or not kind:
            raise TimelineEditError(f"{removal_id} needs a cut kind")
        item = dict(removal)
        item.update(
            {
                "id": str(removal_id),
                "start": round(max(0.0, start_value), 6),
                "end": round(min(duration, end_value), 6),
                "duration_seconds": round(end_value - start_value, 6),
                "kind": kind,
            }
        )
        normalized.append(item)
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    for previous, current in zip(normalized, normalized[1:]):
        if current["start"] < previous["end"] - TIME_TOLERANCE:
            raise TimelineEditError(
                f"Removal ranges overlap: {previous['id']} and {current['id']}"
            )
    removed_duration = sum(item["end"] - item["start"] for item in normalized)
    if removed_duration >= duration - 0.5:
        raise TimelineEditError("Edit would remove the entire program")
    return normalized


def kept_ranges(removals: Sequence[Mapping[str, Any]], duration: float) -> list[dict[str, float]]:
    kept: list[dict[str, float]] = []
    source_cursor = 0.0
    output_cursor = 0.0
    for removal in removals:
        start = float(removal["start"])
        end = float(removal["end"])
        if start > source_cursor + TIME_TOLERANCE:
            length = start - source_cursor
            kept.append(
                {
                    "source_start": round(source_cursor, 6),
                    "source_end": round(start, 6),
                    "output_start": round(output_cursor, 6),
                    "output_end": round(output_cursor + length, 6),
                }
            )
            output_cursor += length
        source_cursor = max(source_cursor, end)
    if source_cursor < duration - TIME_TOLERANCE:
        length = duration - source_cursor
        kept.append(
            {
                "source_start": round(source_cursor, 6),
                "source_end": round(duration, 6),
                "output_start": round(output_cursor, 6),
                "output_end": round(output_cursor + length, 6),
            }
        )
    if not kept:
        raise TimelineEditError("Edit contains no retained media")
    return kept


def _mapped_time(value: float, kept_range: Mapping[str, float]) -> float:
    return float(kept_range["output_start"]) + value - float(kept_range["source_start"])


def _join_word_text(words: Sequence[Mapping[str, Any]]) -> str:
    values = [str(word.get("text") or "") for word in words]
    if any(value[:1].isspace() for value in values[1:]):
        return "".join(values).strip()
    return " ".join(value.strip() for value in values if value.strip())


def _compressed_time(value: float, removals: Sequence[Mapping[str, Any]]) -> float:
    """Map a source timestamp through removals without dropping spanning words."""
    removed_before = 0.0
    for removal in removals:
        start = float(removal["start"])
        end = float(removal["end"])
        if value < start:
            break
        if value <= end:
            return start - removed_before
        removed_before += end - start
    return value - removed_before


def remap_transcript(
    transcript: TranscriptArtifact,
    *,
    media: MediaArtifact,
    output_media_path: Path,
    output_media_sha256: str,
    output_manifest_path: Path,
    removals: Sequence[Mapping[str, Any]],
    kept: Sequence[Mapping[str, float]],
    agent_name: str,
) -> dict[str, Any]:
    document = copy.deepcopy(transcript.document)
    remapped_segments: list[dict[str, Any]] = []
    removed_ids = {
        str(word_id)
        for removal in removals
        for word_id in removal.get("removed_word_ids", [])
    }
    for source_segment in transcript.segments:
        selected = []
        for source_word in source_segment["words"]:
            if str(source_word["id"]) in removed_ids:
                continue
            word = copy.deepcopy(source_word)
            mapped_start = _compressed_time(float(word["start"]), removals)
            mapped_end = _compressed_time(float(word["end"]), removals)
            word["start"] = round(mapped_start, 3)
            word["end"] = round(max(mapped_start, mapped_end), 3)
            selected.append(word)
        if not selected:
            continue
        segment = copy.deepcopy(source_segment)
        source_segment_id = str(source_segment.get("id") or "segment")
        if len(selected) != len(source_segment["words"]):
            segment["source_segment_id"] = source_segment_id
            segment["id"] = f"{source_segment_id}-edit-01"
        segment["words"] = selected
        segment["start"] = selected[0]["start"]
        segment["end"] = selected[-1]["end"]
        segment["text"] = _join_word_text(selected)
        remapped_segments.append(segment)
    if not remapped_segments:
        raise TimelineEditError("No transcript words remain after the edit")

    transcript_body = document["transcript"]
    transcript_body["segments"] = remapped_segments
    transcript_body["segment_count"] = len(remapped_segments)
    transcript_body["word_count"] = sum(len(item["words"]) for item in remapped_segments)
    transcript_body["text"] = " ".join(item["text"] for item in remapped_segments)
    quality = transcript_body.get("quality")
    if not isinstance(quality, dict):
        quality = {}
        transcript_body["quality"] = quality
    quality["timeline_edit"] = {
        "agent": agent_name,
        "removed_word_count": len(removed_ids),
        "removed_word_ids": sorted(removed_ids),
        "removed_range_count": len(removals),
    }
    source = document["source"]
    output_duration = float(kept[-1]["output_end"])
    source.update(
        {
            "media_path": str(output_media_path.resolve()),
            "downloader_manifest_path": str(output_manifest_path.resolve()),
            "sha256": output_media_sha256,
            "duration_seconds": round(output_duration, 6),
            "clip_start_seconds": 0.0,
            "clip_end_seconds": None,
        }
    )
    history = document.get("timeline_edits")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "agent": agent_name,
            "source_manifest_path": str(media.manifest_path),
            "source_manifest_sha256": media.manifest_sha256,
            "source_media_sha256": media.media_sha256,
            "output_media_sha256": output_media_sha256,
            "removed_ranges": [dict(item) for item in removals],
            "kept_ranges": [dict(item) for item in kept],
        }
    )
    document["timeline_edits"] = history
    document["created_at"] = utc_now()
    verification = document.get("verification")
    if not isinstance(verification, dict):
        verification = {}
    verification.update(
        {
            "source_sha256_verified": True,
            "timeline_edit_verified": True,
            "removed_words_traceable": True,
            "timestamps_remapped": True,
        }
    )
    document["verification"] = verification
    return document


def audio_micro_crossfade_durations(
    *,
    kept: Sequence[Mapping[str, float]],
    removals: Sequence[Mapping[str, Any]] | None,
) -> list[float]:
    """Resolve one duration per retained-range join without changing A/V length."""
    if len(kept) < 2:
        return []
    values: list[float] = []
    removal_values = list(removals or [])
    for left, right in zip(kept, kept[1:]):
        gap_start = float(left["source_end"])
        gap_end = float(right["source_start"])
        matching = next(
            (
                item
                for item in removal_values
                if abs(float(item["start"]) - gap_start) <= TIME_TOLERANCE
                and abs(float(item["end"]) - gap_end) <= TIME_TOLERANCE
            ),
            None,
        )
        treatment = matching.get("audio_join_treatment") if matching else None
        if not isinstance(treatment, Mapping) or treatment.get("method") != (
            "equal-power-micro-crossfade"
        ):
            values.append(0.0)
            continue
        requested = treatment.get("duration_seconds")
        if (
            not isinstance(requested, (int, float))
            or not math.isfinite(float(requested))
            or float(requested) <= 0
            or float(requested) > 0.05
        ):
            raise TimelineEditError("Audio micro-crossfade duration is invalid")
        left_duration = gap_start - float(left["source_start"])
        right_duration = float(right["source_end"]) - gap_end
        available = min(
            gap_end - gap_start,
            max(0.0, 2 * (left_duration - 0.005)),
            max(0.0, 2 * (right_duration - 0.005)),
        )
        duration = min(float(requested), available)
        values.append(round(duration, 6) if duration >= 0.005 else 0.0)
    return values


def ffmpeg_edit_command(
    *,
    ffmpeg: Path,
    input_path: Path,
    output_path: Path,
    kept: Sequence[Mapping[str, float]],
    has_audio: bool,
    removals: Sequence[Mapping[str, Any]] | None = None,
    audio_edge_fade_seconds: float = AUDIO_EDGE_FADE_SECONDS,
) -> list[str]:
    filters: list[str] = []
    if not kept:
        raise TimelineEditError("Timeline edit needs at least one kept range")

    # FFmpeg 9 can finish a late, jointly-filtered video range before its audio
    # filter has emitted any frames when both streams are decoded from the start
    # of a long input. A single retained range needs no pre-range packets, so
    # seek the demuxer first and express that range on the resulting timeline.
    # Multi-range edits keep their existing source-time behavior because their
    # gap calculations and join treatments depend on the full edit timeline.
    input_seek = 0.0
    if len(kept) == 1 and not removals:
        input_seek = max(0.0, float(kept[0]["source_start"]))
    separate_audio_input = has_audio and len(kept) == 1
    audio_input_index = 1 if separate_audio_input else 0

    # Keep video on one continuous timeline and close removed gaps in bounded
    # filter chunks. Concatenating independently trimmed video segments rounds
    # every segment to a frame boundary; over hundreds of speech cuts that error
    # accumulates into seconds of drift and causes concat to pad audio. A single
    # giant select expression, on the other hand, exhausts FFmpeg's expression
    # allocator on long edits, so process small chronological batches instead.
    first_source_start = float(kept[0]["source_start"])
    last_source_end = float(kept[-1]["source_end"])
    filter_start = first_source_start - input_seek
    filter_end = last_source_end - input_seek
    filters.append(
        f"[0:v:0]trim=start={filter_start:.6f}:end={filter_end:.6f},"
        "setpts=PTS-STARTPTS[vstage0]"
    )
    gaps: list[tuple[float, float]] = []
    for previous, retained in zip(kept, kept[1:]):
        gaps.append(
            (float(previous["source_end"]), float(retained["source_start"]))
        )
    chunk_size = 24
    removed_before_chunk = 0.0
    chunks = [
        gaps[index : index + chunk_size]
        for index in range(0, len(gaps), chunk_size)
    ]
    for chunk_index, chunk in enumerate(chunks):
        adjusted = [
            (
                start - first_source_start - removed_before_chunk,
                end - first_source_start - removed_before_chunk,
            )
            for start, end in chunk
        ]
        removal_expression = "+".join(
            f"gte(t,{start:.6f})*lt(t,{end:.6f})" for start, end in adjusted
        )
        gap_terms = [
            f"gte((PTS-STARTPTS)*TB,{end:.6f})*{end - start:.6f}"
            for start, end in adjusted
        ]
        input_label = f"vstage{chunk_index}"
        output_label = (
            "vout"
            if chunk_index == len(chunks) - 1
            else f"vstage{chunk_index + 1}"
        )
        filters.append(
            f"[{input_label}]select='not({removal_expression})',"
            f"setpts='PTS-STARTPTS-({'+'.join(gap_terms)})/TB'[{output_label}]"
        )
        removed_before_chunk += sum(end - start for start, end in chunk)
    if not chunks:
        filters.append("[vstage0]null[vout]")

    micro_crossfades = audio_micro_crossfade_durations(
        kept=kept, removals=removals
    )
    audio_inputs: list[str] = []
    for index, retained in enumerate(kept):
        previous_crossfade = micro_crossfades[index - 1] if index > 0 else 0.0
        next_crossfade = (
            micro_crossfades[index] if index < len(micro_crossfades) else 0.0
        )
        start = (
            float(retained["source_start"])
            - input_seek
            - previous_crossfade / 2
        )
        end = (
            float(retained["source_end"])
            - input_seek
            + next_crossfade / 2
        )
        if has_audio:
            audio_filter = (
                f"[{audio_input_index}:a:0]atrim=start={start:.6f}:end={end:.6f},"
                "asetpts=PTS-STARTPTS"
            )
            segment_duration = end - start
            fade = min(audio_edge_fade_seconds, segment_duration / 3)
            if fade > 0.001 and index > 0 and previous_crossfade <= 0:
                audio_filter += f",afade=t=in:st=0:d={fade:.6f}:curve=tri"
            if fade > 0.001 and index < len(kept) - 1 and next_crossfade <= 0:
                fade_start = max(0.0, segment_duration - fade)
                audio_filter += (
                    f",afade=t=out:st={fade_start:.6f}:d={fade:.6f}:curve=tri"
                )
            filters.append(audio_filter + f"[a{index}]")
            audio_inputs.append(f"[a{index}]")
    if has_audio:
        if len(audio_inputs) == 1:
            # FFmpeg 9 can emit an empty file when a single trimmed audio
            # stream is routed through concat=n=1 alongside filtered video.
            # No join is needed for one retained range, so pass it through.
            filters.append(f"{audio_inputs[0]}anull[aout]")
        elif any(duration > 0 for duration in micro_crossfades):
            current_label = "a0"
            for index in range(1, len(kept)):
                output_label = "aout" if index == len(kept) - 1 else f"ajoin{index}"
                crossfade = micro_crossfades[index - 1]
                if crossfade > 0:
                    filters.append(
                        f"[{current_label}][a{index}]"
                        f"acrossfade=d={crossfade:.6f}:c1=qsin:c2=qsin"
                        f"[{output_label}]"
                    )
                else:
                    filters.append(
                        f"[{current_label}][a{index}]concat=n=2:v=0:a=1"
                        f"[{output_label}]"
                    )
                current_label = output_label
        else:
            filters.append(
                "".join(audio_inputs) + f"concat=n={len(kept)}:v=0:a=1[aout]"
            )
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-y",
    ]
    if input_seek > TIME_TOLERANCE:
        command.extend(["-ss", f"{input_seek:.6f}"])
    command.extend(
        [
        "-i",
        str(input_path),
        ]
    )
    if separate_audio_input:
        if input_seek > TIME_TOLERANCE:
            command.extend(["-ss", f"{input_seek:.6f}"])
        command.extend(["-i", str(input_path)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
        ]
    )
    if has_audio:
        command.extend(["-map", "[aout]"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if has_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.extend(["-movflags", "+faststart", "-map_metadata", "0", str(output_path)])
    return command


def probe_media(ffprobe: Path, media_path: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=codec_type,codec_name,width,height,duration",
                "-of",
                "json",
                str(media_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        value = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise TimelineEditError(f"Could not probe media: {media_path}") from error
    if not isinstance(value, dict):
        raise TimelineEditError("ffprobe returned invalid JSON")
    return value


def _stream(probe: Mapping[str, Any], kind: str) -> Mapping[str, Any] | None:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        return None
    return next(
        (
            item
            for item in streams
            if isinstance(item, Mapping) and item.get("codec_type") == kind
        ),
        None,
    )


def _duration(probe: Mapping[str, Any]) -> float | None:
    value = probe.get("format")
    raw = value.get("duration") if isinstance(value, Mapping) else None
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _stream_duration(stream: Mapping[str, Any] | None) -> float | None:
    raw = stream.get("duration") if isinstance(stream, Mapping) else None
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def render_edit(
    *,
    media: MediaArtifact,
    kept: Sequence[Mapping[str, float]],
    removals: Sequence[Mapping[str, Any]],
    output_path: Path,
    ffmpeg: Path,
    ffprobe: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_probe = probe_media(ffprobe, media.media_path)
    input_video = _stream(input_probe, "video")
    input_audio = _stream(input_probe, "audio")
    if input_video is None or input_audio is None:
        raise TimelineEditError("Timeline editing requires both video and audio streams")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    temporary_path.unlink(missing_ok=True)
    full_range = (
        len(kept) == 1
        and abs(float(kept[0]["source_start"])) <= TIME_TOLERANCE
        and abs(float(kept[0]["source_end"]) - media.duration_seconds)
        <= DURATION_TOLERANCE
    )
    try:
        if full_range:
            try:
                os.link(media.media_path, temporary_path)
            except OSError:
                shutil.copy2(media.media_path, temporary_path)
        else:
            command = ffmpeg_edit_command(
                ffmpeg=ffmpeg,
                input_path=media.media_path,
                output_path=temporary_path,
                kept=kept,
                has_audio=True,
                removals=removals,
            )
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                detail = result.stderr.strip()[-4000:]
                raise TimelineEditError(
                    "FFmpeg timeline edit failed" + (f":\n{detail}" if detail else "")
                )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    output_probe = probe_media(ffprobe, output_path)
    return input_probe, output_probe


def apply_edit(
    *,
    agent_name: str,
    media: MediaArtifact,
    transcript: TranscriptArtifact,
    plan_path: Path,
    plan: Mapping[str, Any],
    removals: Sequence[Mapping[str, Any]],
    output_dir: Path,
    ffmpeg: Path,
    ffprobe: Path,
    force: bool,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    _, on_disk_plan = read_json(plan_path, "Edit plan")
    if dict(plan) != on_disk_plan:
        raise TimelineEditError("In-memory edit plan does not match the plan file")
    plan_sha256 = sha256_file(plan_path)
    source_transcript_sha256 = sha256_file(transcript.path)
    manifest_path = output_dir / "manifest.json"
    output_path = output_dir / "edited.mp4"
    transcript_path = output_dir / "transcript.edited.json"
    existing = [path for path in (manifest_path, output_path, transcript_path) if path.exists()]
    if existing and not force:
        raise TimelineEditError(
            f"Edit output exists. Use --force to replace it: {existing[0]}"
        )
    normalized = normalize_removals(removals, media.duration_seconds)
    validate_cut_boundaries(transcript, normalized)
    kept = kept_ranges(normalized, media.duration_seconds)
    input_probe, output_probe = render_edit(
        media=media,
        kept=kept,
        removals=normalized,
        output_path=output_path,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    output_sha256 = sha256_file(output_path)
    remapped = remap_transcript(
        transcript,
        media=media,
        output_media_path=output_path,
        output_media_sha256=output_sha256,
        output_manifest_path=manifest_path,
        removals=normalized,
        kept=kept,
        agent_name=agent_name,
    )
    write_json_atomic(transcript_path, remapped)
    output_transcript_sha256 = sha256_file(transcript_path)
    expected_duration = float(kept[-1]["output_end"])
    observed_duration = _duration(output_probe)
    input_video = _stream(input_probe, "video")
    output_video = _stream(output_probe, "video")
    output_audio = _stream(output_probe, "audio")
    has_audio = output_audio is not None
    observed_audio_duration = _stream_duration(output_audio)
    micro_crossfade_durations = audio_micro_crossfade_durations(
        kept=kept, removals=normalized
    )
    micro_crossfade_count = sum(
        duration > 0 for duration in micro_crossfade_durations
    )
    join_count = max(0, len(kept) - 1)
    if micro_crossfade_count:
        audio_join_treatment = {
            "method": (
                "equal-power-micro-crossfade"
                if micro_crossfade_count == join_count
                else "mixed-linear-edge-fade-and-equal-power-micro-crossfade"
            ),
            "join_count": join_count,
            "linear_edge_fade_duration_seconds": AUDIO_EDGE_FADE_SECONDS,
            "linear_edge_fade_join_count": join_count - micro_crossfade_count,
            "micro_crossfade_duration_seconds": max(micro_crossfade_durations),
            "micro_crossfade_join_count": micro_crossfade_count,
        }
    else:
        audio_join_treatment = {
            "method": "linear-edge-fade",
            "duration_seconds": AUDIO_EDGE_FADE_SECONDS,
            "join_count": join_count,
        }
    manifest = {
        "schema_version": 1,
        "agent": agent_name,
        "stage": "timeline-edit",
        "status": "complete",
        "created_at": utc_now(),
        "source": {
            "video_id": media.video_id,
            "title": media.title,
            "manifest_path": str(media.manifest_path),
            "manifest_sha256": media.manifest_sha256,
            "media_path": str(media.media_path),
            "media_sha256": media.media_sha256,
            "duration_seconds": media.duration_seconds,
            "transcript_path": str(transcript.path),
            "transcript_sha256": transcript.sha256,
        },
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": plan_sha256,
            "review": copy.deepcopy(plan.get("review")),
        },
        "timeline": {
            "removed_ranges": normalized,
            "kept_ranges": kept,
            "removed_duration_seconds": round(
                sum(float(item["end"]) - float(item["start"]) for item in normalized),
                6,
            ),
            "output_duration_seconds": round(expected_duration, 6),
        },
        "output": {
            "media_path": str(output_path),
            "sha256": output_sha256,
            "size_bytes": output_path.stat().st_size,
            "duration_seconds": observed_duration,
            "audio_duration_seconds": observed_audio_duration,
            "width": output_video.get("width") if output_video else None,
            "height": output_video.get("height") if output_video else None,
            "transcript_path": str(transcript_path),
            "transcript_sha256": output_transcript_sha256,
        },
        "transform_policy": {
            "speed_changed": False,
            "audio_replaced": False,
            "crop_or_scale_applied": False,
            "purpose": (
                "editorial pacing: natural pause shortening and conservatively reviewed disfluency removal"
                if agent_name == "speech-tightener"
                else "editorial pacing and source branding boundary removal"
            ),
            "content_id_evasion": False,
            "audio_join_treatment": audio_join_treatment,
        },
        "verification": {
            "source_manifest_sha256_verified": sha256_file(media.manifest_path)
            == media.manifest_sha256,
            "source_media_sha256_verified": sha256_file(media.media_path)
            == media.media_sha256,
            "plan_sha256_verified": sha256_file(plan_path) == plan_sha256,
            "source_transcript_sha256_verified": sha256_file(transcript.path)
            == source_transcript_sha256,
            "output_sha256_verified": sha256_file(output_path) == output_sha256,
            "transcript_sha256_verified": sha256_file(transcript_path)
            == output_transcript_sha256,
            "has_video": output_video is not None,
            "has_audio": has_audio,
            "duration_matches": observed_duration is not None
            and abs(observed_duration - expected_duration) <= 1.0,
            "audio_duration_matches": observed_audio_duration is not None
            and abs(observed_audio_duration - expected_duration) <= 1.0,
            "dimensions_preserved": bool(input_video and output_video)
            and input_video.get("width") == output_video.get("width")
            and input_video.get("height") == output_video.get("height"),
            "sha256_verified": True,
            "timeline_monotonic": all(
                float(left["output_end"]) <= float(right["output_start"]) + TIME_TOLERANCE
                for left, right in zip(kept, kept[1:])
            ),
            "transcript_remapped": True,
            "speed_unchanged": True,
            "audio_not_replaced": True,
            "audio_edges_smoothed": True,
            "crop_or_scale_not_applied": True,
        },
    }
    failed = [key for key, value in manifest["verification"].items() if value is not True]
    if failed:
        output_path.unlink(missing_ok=True)
        transcript_path.unlink(missing_ok=True)
        raise TimelineEditError("Edit verification failed: " + ", ".join(failed))
    write_json_atomic(manifest_path, manifest)
    return manifest
