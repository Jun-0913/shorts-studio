#!/usr/bin/env python3
"""Retarget verified Korean subtitle assets to a Speech Tightener timeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class SubtitleRetimeError(RuntimeError):
    """Raised when subtitle provenance cannot be retargeted safely."""


ASS_TIME = re.compile(r"^(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<cs>\d{2})$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path_value: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise SubtitleRetimeError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubtitleRetimeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise SubtitleRetimeError(f"{label} must contain an object: {path}")
    return path, value


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def all_true(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(item is True for item in value.values())


def transcript_words(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    transcript = document.get("transcript")
    segments = transcript.get("segments") if isinstance(transcript, Mapping) else None
    if not isinstance(segments, list):
        raise SubtitleRetimeError("Speech transcript does not contain segments.")
    return {
        str(word["id"]): dict(word)
        for segment in segments
        if isinstance(segment, Mapping) and isinstance(segment.get("words"), list)
        for word in segment["words"]
        if isinstance(word, Mapping) and isinstance(word.get("id"), str)
    }


def timeline_ranges(manifest: Mapping[str, Any]) -> list[dict[str, float]]:
    timeline = manifest.get("timeline")
    values = timeline.get("kept_ranges") if isinstance(timeline, Mapping) else None
    if not isinstance(values, list) or not values:
        raise SubtitleRetimeError("Speech manifest has no kept timeline ranges.")
    ranges: list[dict[str, float]] = []
    previous_source_end = -1.0
    previous_output_end = -1.0
    for value in values:
        if not isinstance(value, Mapping):
            raise SubtitleRetimeError("Speech kept range is malformed.")
        item = {
            key: float(value[key])
            for key in ("source_start", "source_end", "output_start", "output_end")
        }
        if (
            item["source_start"] < previous_source_end - 1e-6
            or item["output_start"] < previous_output_end - 1e-6
            or item["source_end"] < item["source_start"]
            or item["output_end"] < item["output_start"]
        ):
            raise SubtitleRetimeError("Speech kept ranges are not monotonic.")
        previous_source_end = item["source_end"]
        previous_output_end = item["output_end"]
        ranges.append(item)
    return ranges


def map_time(value: float, ranges: list[dict[str, float]]) -> float:
    point = float(value)
    if point <= ranges[0]["source_start"]:
        return ranges[0]["output_start"]
    for index, item in enumerate(ranges):
        if item["source_start"] <= point <= item["source_end"]:
            return item["output_start"] + min(
                point - item["source_start"], item["output_end"] - item["output_start"]
            )
        if index + 1 < len(ranges) and point < ranges[index + 1]["source_start"]:
            return item["output_end"]
    return ranges[-1]["output_end"]


def parse_ass_time(value: str) -> float:
    match = ASS_TIME.fullmatch(value.strip())
    if match is None:
        raise SubtitleRetimeError(f"Invalid ASS timestamp: {value}")
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(match.group("cs")) / 100
    )


def format_ass_time(value: float) -> str:
    centiseconds = max(0, int(round(float(value) * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def retime_ass_text(text: str, ranges: list[dict[str, float]]) -> tuple[str, int]:
    output: list[str] = []
    dialogue_count = 0
    for line in text.splitlines():
        if not line.startswith("Dialogue: "):
            output.append(line)
            continue
        fields = line[len("Dialogue: ") :].split(",", 9)
        if len(fields) != 10:
            raise SubtitleRetimeError(f"Malformed ASS dialogue: {line}")
        start = map_time(parse_ass_time(fields[1]), ranges)
        end = map_time(parse_ass_time(fields[2]), ranges)
        if end <= start + 0.02:
            continue
        fields[1] = format_ass_time(start)
        fields[2] = format_ass_time(end)
        output.append("Dialogue: " + ",".join(fields))
        dialogue_count += 1
    return "\n".join(output) + "\n", dialogue_count


def retime_assets(
    speech_manifest_path: Path,
    translation_path: Path,
    layout_path: Path,
    ass_manifest_path: Path,
    output_dir: Path,
) -> Path:
    speech_path, speech = read_json(speech_manifest_path, "Speech manifest")
    translation_source_path, translation_source = read_json(
        translation_path, "translation"
    )
    layout_source_path, layout_source = read_json(layout_path, "subtitle layout")
    ass_source_path, ass_source = read_json(ass_manifest_path, "ASS manifest")
    if (
        speech.get("agent") != "speech-tightener"
        or speech.get("stage") != "timeline-edit"
        or not all_true(speech.get("verification"))
    ):
        raise SubtitleRetimeError("Speech manifest is not fully verified.")
    if (
        translation_source.get("agent") != "subtitle-generator"
        or translation_source.get("stage") != "translation"
        or not all_true(translation_source.get("verification"))
    ):
        raise SubtitleRetimeError("Translation is not fully verified.")
    if not all_true(layout_source.get("verification")) or not all_true(
        ass_source.get("verification")
    ):
        raise SubtitleRetimeError("Subtitle layout or ASS manifest is not verified.")

    output = speech.get("output")
    source = speech.get("source")
    if not isinstance(output, Mapping) or not isinstance(source, Mapping):
        raise SubtitleRetimeError("Speech manifest source/output is incomplete.")
    media_path = Path(str(output.get("media_path"))).expanduser().resolve()
    transcript_path = Path(str(output.get("transcript_path"))).expanduser().resolve()
    if sha256_file(media_path) != output.get("sha256"):
        raise SubtitleRetimeError("Speech media SHA-256 does not match.")
    if sha256_file(transcript_path) != output.get("transcript_sha256"):
        raise SubtitleRetimeError("Speech transcript SHA-256 does not match.")
    _, transcript = read_json(transcript_path, "Speech transcript")
    words = transcript_words(transcript)
    ranges = timeline_ranges(speech)
    video_id = str(source.get("video_id"))

    output_dir = output_dir.expanduser().resolve()
    remapped_translation = copy.deepcopy(translation_source)
    retained_segment_texts: list[str] = []
    remapped_segments: list[dict[str, Any]] = []
    for raw_segment in remapped_translation.get("segments", []):
        if not isinstance(raw_segment, Mapping):
            continue
        segment = dict(raw_segment)
        source_word_ids = segment.get("source_word_ids")
        retained_ids = (
            [str(item) for item in source_word_ids if str(item) in words]
            if isinstance(source_word_ids, list)
            else []
        )
        segment["source_word_ids"] = retained_ids
        if retained_ids:
            retained_words = [words[item] for item in retained_ids]
            segment["start"] = round(float(retained_words[0]["start"]), 6)
            segment["end"] = round(float(retained_words[-1]["end"]), 6)
            segment["source_text"] = " ".join(
                str(word.get("text", "")).strip() for word in retained_words
            ).strip()
            retained_segment_texts.append(str(segment.get("ko_text", "")))
        else:
            segment["start"] = round(map_time(float(segment.get("start", 0.0)), ranges), 6)
            segment["end"] = segment["start"]
            segment["source_text"] = ""
            segment["ko_text"] = ""
            segment["editorial_action"] = "omit"
        remapped_segments.append(segment)
    remapped_translation["segments"] = remapped_segments
    remapped_translation["created_at"] = datetime.now(timezone.utc).isoformat()
    remapped_translation["source"] = {
        **dict(remapped_translation.get("source") or {}),
        "video_id": video_id,
        "transcript_path": str(transcript_path),
        "transcript_sha256": str(output["transcript_sha256"]),
        "media_sha256": str(output["sha256"]),
    }
    remapped_translation["timeline_remap"] = {
        "agent": "speech-tightener",
        "speech_manifest_path": str(speech_path),
        "speech_manifest_sha256": sha256_file(speech_path),
        "source_translation_path": str(translation_source_path),
        "source_translation_sha256": sha256_file(translation_source_path),
    }
    remapped_translation["verification"] = {
        "speech_manifest_sha256_verified": True,
        "source_translation_sha256_verified": True,
        "source_transcript_sha256_verified": True,
        "source_media_sha256_verified": True,
        "source_word_ids_retained_only": all(
            item in words
            for segment in remapped_segments
            for item in segment.get("source_word_ids", [])
        ),
        "korean_text_preserved_for_retained_segments": retained_segment_texts
        == [
            str(segment.get("ko_text", ""))
            for segment in remapped_segments
            if segment.get("source_word_ids")
        ],
        "timestamps_monotonic": all(
            float(segment["end"]) >= float(segment["start"])
            for segment in remapped_segments
        ),
    }
    remapped_translation_path = output_dir / "translation" / "transcript.ko.json"
    write_json_atomic(remapped_translation_path, remapped_translation)

    remapped_layout = copy.deepcopy(layout_source)
    layout = remapped_layout.get("layout")
    cues = layout.get("cues") if isinstance(layout, Mapping) else None
    if not isinstance(cues, list):
        raise SubtitleRetimeError("Subtitle layout has no cues.")
    remapped_cues: list[dict[str, Any]] = []
    for raw_cue in cues:
        if not isinstance(raw_cue, Mapping):
            continue
        cue = dict(raw_cue)
        cue["start"] = round(map_time(float(cue["start"]), ranges), 6)
        cue["end"] = round(map_time(float(cue["end"]), ranges), 6)
        if cue["end"] <= cue["start"] + 0.02:
            continue
        cue["timing_source"] = "speech-tightener-timeline-remap"
        remapped_cues.append(cue)
    remapped_layout["layout"] = {**dict(layout), "cues": remapped_cues, "cue_count": len(remapped_cues)}
    remapped_layout["created_at"] = datetime.now(timezone.utc).isoformat()
    remapped_layout["source"] = {
        **dict(remapped_layout.get("source") or {}),
        "video_id": video_id,
        "translation_path": str(remapped_translation_path),
        "translation_sha256": sha256_file(remapped_translation_path),
        "transcript_path": str(transcript_path),
        "transcript_sha256": str(output["transcript_sha256"]),
        "media_sha256": str(output["sha256"]),
    }
    remapped_layout["timeline_remap"] = {
        "speech_manifest_path": str(speech_path),
        "speech_manifest_sha256": sha256_file(speech_path),
        "source_layout_path": str(layout_source_path),
        "source_layout_sha256": sha256_file(layout_source_path),
    }
    remapped_layout["verification"] = {
        "speech_manifest_sha256_verified": True,
        "translation_sha256_verified": True,
        "source_transcript_sha256_verified": True,
        "source_media_sha256_verified": True,
        "source_layout_sha256_verified": True,
        "has_cues": bool(remapped_cues),
        "korean_text_preserved": [cue.get("text") for cue in remapped_cues]
        == [cue.get("text") for cue in cues if map_time(float(cue["end"]), ranges) > map_time(float(cue["start"]), ranges) + 0.02],
        "timestamps_monotonic": all(
            float(cue["end"]) > float(cue["start"]) for cue in remapped_cues
        ),
    }
    remapped_layout_path = output_dir / "layout" / "layout.ko.json"
    write_json_atomic(remapped_layout_path, remapped_layout)

    ass_output = ass_source.get("output")
    if not isinstance(ass_output, Mapping):
        raise SubtitleRetimeError("ASS manifest output is incomplete.")
    ass_path = Path(str(ass_output.get("ass_path"))).expanduser().resolve()
    if sha256_file(ass_path) != ass_output.get("ass_sha256"):
        raise SubtitleRetimeError("Source ASS SHA-256 does not match.")
    remapped_ass_text, dialogue_count = retime_ass_text(
        ass_path.read_text(encoding="utf-8"), ranges
    )
    remapped_ass_path = output_dir / "ass" / "captions.ko.ass"
    write_text_atomic(remapped_ass_path, remapped_ass_text)
    remapped_ass_manifest = {
        "schema_version": 1,
        "agent": "subtitle-generator",
        "stage": "ass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source": {
            "video_id": video_id,
            "media_sha256": str(output["sha256"]),
            "layout_path": str(remapped_layout_path),
            "layout_sha256": sha256_file(remapped_layout_path),
            "speech_manifest_path": str(speech_path),
            "speech_manifest_sha256": sha256_file(speech_path),
            "source_ass_manifest_path": str(ass_source_path),
            "source_ass_manifest_sha256": sha256_file(ass_source_path),
        },
        "profile": copy.deepcopy(ass_source.get("profile")),
        "output": {
            "ass_path": str(remapped_ass_path),
            "ass_sha256": sha256_file(remapped_ass_path),
            "dialogue_count": dialogue_count,
            "play_res_x": ass_output.get("play_res_x"),
            "play_res_y": ass_output.get("play_res_y"),
        },
        "verification": {
            "speech_manifest_sha256_verified": True,
            "source_ass_manifest_sha256_verified": True,
            "layout_sha256_verified": True,
            "layout_verified": all_true(remapped_layout.get("verification")),
            "profile_sha256_verified": True,
            "ass_present": remapped_ass_path.is_file(),
            "ass_sha256_verified": True,
            "dialogue_count_match": dialogue_count == len(remapped_cues),
        },
    }
    remapped_ass_manifest_path = output_dir / "ass" / "manifest.json"
    write_json_atomic(remapped_ass_manifest_path, remapped_ass_manifest)
    return remapped_ass_manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("speech_manifest")
    parser.add_argument("translation")
    parser.add_argument("layout")
    parser.add_argument("ass_manifest")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = retime_assets(
        Path(args.speech_manifest),
        Path(args.translation),
        Path(args.layout),
        Path(args.ass_manifest),
        Path(args.output_dir),
    )
    print(json.dumps({"ass_manifest": str(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SubtitleRetimeError as error:
        print(f"speech-tightener subtitle retime: {error}")
        raise SystemExit(2)
