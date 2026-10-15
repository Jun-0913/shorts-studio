#!/usr/bin/env python3
"""Recover ASR-hidden adjacent repetitions with overlapping prompt-free windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SHARED_DIR = REPO_ROOT / ".agents" / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

import timeline_edit as core


DEFAULT_MODEL_ID = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MODEL_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
DEFAULT_MODEL_CACHE = REPO_ROOT / ".tools" / "subtitle-generator" / "huggingface"
SAMPLE_RATE = 16_000
MIN_WINDOW_MARGIN_SECONDS = 0.25
CLUSTER_TIME_TOLERANCE_SECONDS = 0.38


class ProbeError(RuntimeError):
    """An expected verbatim-probe failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_word(value: Any) -> str:
    return re.sub(r"(^[^a-z0-9']+|[^a-z0-9']+$)", "", str(value or "").lower())


def window_starts(duration: float, *, window_seconds: float, stride_seconds: float) -> list[float]:
    if duration <= 0 or window_seconds <= 0 or stride_seconds <= 0:
        raise ProbeError("Probe duration, window, and stride must be positive")
    if duration <= window_seconds:
        return [0.0]
    values: list[float] = []
    value = 0.0
    while value + window_seconds < duration:
        values.append(round(value, 6))
        value += stride_seconds
    final_start = max(0.0, duration - window_seconds)
    if not values or abs(values[-1] - final_start) > 0.05:
        values.append(round(final_start, 6))
    return values


def normalize_probe_words(
    result: Mapping[str, Any], *, window_id: str, window_start: float
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    sequence = 0
    for segment in result.get("segments") or []:
        if not isinstance(segment, Mapping):
            continue
        for raw_word in segment.get("words") or []:
            if not isinstance(raw_word, Mapping):
                continue
            start = raw_word.get("start")
            end = raw_word.get("end")
            text = raw_word.get("word")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            if not isinstance(text, str) or float(end) < float(start):
                continue
            sequence += 1
            word = {
                "id": f"{window_id}-word-{sequence:04d}",
                "start": round(window_start + float(start), 6),
                "end": round(window_start + float(end), 6),
                "text": text,
            }
            probability = raw_word.get("probability")
            if isinstance(probability, (int, float)):
                word["probability"] = round(float(probability), 6)
            words.append(word)
    return words


def repeated_runs(
    words: Sequence[Mapping[str, Any]],
    *,
    window_id: str,
    window_start: float,
    window_end: float,
    max_phrase_words: int = 4,
) -> list[dict[str, Any]]:
    """Return maximal adjacent repeated token or phrase runs from one window."""

    tokens = [normalized_word(word.get("text")) for word in words]
    claimed: set[int] = set()
    values: list[dict[str, Any]] = []
    for size in range(max_phrase_words, 0, -1):
        index = 0
        while index + size * 2 <= len(words):
            phrase = tokens[index : index + size]
            if not all(phrase):
                index += 1
                continue
            copies = 1
            while (
                index + (copies + 1) * size <= len(words)
                and tokens[index + copies * size : index + (copies + 1) * size]
                == phrase
            ):
                copies += 1
            if copies < 2:
                index += 1
                continue
            group_end_index = index + copies * size
            group_indices = set(range(index, group_end_index))
            if claimed & group_indices:
                index = group_end_index
                continue
            group_start = float(words[index]["start"])
            group_end = float(words[group_end_index - 1]["end"])
            if (
                group_start < window_start + MIN_WINDOW_MARGIN_SECONDS
                or group_end > window_end - MIN_WINDOW_MARGIN_SECONDS
            ):
                index = group_end_index
                continue
            retained = words[index : index + size]
            removed = words[index + size : group_end_index]
            probabilities = [
                float(word["probability"])
                for word in words[index:group_end_index]
                if isinstance(word.get("probability"), (int, float))
            ]
            values.append(
                {
                    "window_id": window_id,
                    "phrase_tokens": phrase,
                    "phrase_text": " ".join(phrase),
                    "copy_count": copies,
                    "group_start": round(group_start, 6),
                    "group_end": round(group_end, 6),
                    "removal_start": round(float(removed[0]["start"]), 6),
                    "removal_end": round(float(removed[-1]["end"]), 6),
                    "shadow_retained_word_ids": [str(word["id"]) for word in retained],
                    "shadow_removed_word_ids": [str(word["id"]) for word in removed],
                    "shadow_retained_text": " ".join(
                        str(word.get("text") or "").strip() for word in retained
                    ),
                    "shadow_removed_text": " ".join(
                        str(word.get("text") or "").strip() for word in removed
                    ),
                    "mean_word_probability": (
                        round(sum(probabilities) / len(probabilities), 6)
                        if probabilities
                        else None
                    ),
                }
            )
            claimed.update(group_indices)
            index = group_end_index
    return values


def canonical_words(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    transcript = document.get("transcript")
    if not isinstance(transcript, Mapping):
        raise ProbeError("Canonical transcript has no transcript object")
    values: list[dict[str, Any]] = []
    for segment in transcript.get("segments") or []:
        if not isinstance(segment, Mapping):
            continue
        for word in segment.get("words") or []:
            if isinstance(word, Mapping):
                values.append(dict(word))
    if not values:
        raise ProbeError("Canonical transcript has no word timestamps")
    return values


def canonical_match(
    candidate: Mapping[str, Any], words: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    phrase = list(candidate["phrase_tokens"])
    size = len(phrase)
    tokens = [normalized_word(word.get("text")) for word in words]
    group_midpoint = (float(candidate["group_start"]) + float(candidate["group_end"])) / 2
    matches: list[tuple[float, int]] = []
    for index in range(len(words) - size + 1):
        if tokens[index : index + size] != phrase:
            continue
        start = float(words[index]["start"])
        end = float(words[index + size - 1]["end"])
        distance = abs(((start + end) / 2) - group_midpoint)
        if distance <= max(1.2, float(candidate["group_end"]) - float(candidate["group_start"])):
            matches.append((distance, index))
    if not matches:
        return None
    _, index = min(matches)
    if tokens[index : index + size * 2] == phrase * 2:
        return None
    removal_start = float(candidate["removal_start"])
    removal_end = float(candidate["removal_end"])
    canonical_phrase_word_ids = [
        str(word["id"]) for word in words[index : index + size]
    ]
    canonical_phrase_ids = set(canonical_phrase_word_ids)
    collapsed = [
        str(word["id"])
        for word in words
        if str(word["id"]) in canonical_phrase_ids
        and
        max(
            0.0,
            min(removal_end, float(word["end"]))
            - max(removal_start, float(word["start"])),
        )
        > core.TIME_TOLERANCE
    ]
    if not collapsed:
        return None
    return {
        "canonical_phrase_word_ids": canonical_phrase_word_ids,
        "canonical_collapsed_word_ids": collapsed,
        "canonical_phrase_start": round(float(words[index]["start"]), 6),
        "canonical_phrase_end": round(float(words[index + size - 1]["end"]), 6),
        "canonical_following_word_start": (
            round(float(words[index + size]["start"]), 6)
            if index + size < len(words)
            else None
        ),
    }


def cluster_candidates(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clusters: list[list[Mapping[str, Any]]] = []
    for value in sorted(values, key=lambda item: float(item["removal_start"])):
        match = None
        for cluster in clusters:
            anchor = cluster[0]
            if (
                anchor.get("phrase_tokens") == value.get("phrase_tokens")
                and abs(float(anchor["removal_start"]) - float(value["removal_start"]))
                <= CLUSTER_TIME_TOLERANCE_SECONDS
            ):
                match = cluster
                break
        if match is None:
            clusters.append([value])
        else:
            match.append(value)

    output: list[dict[str, Any]] = []
    for cluster in clusters:
        representative = max(
            cluster,
            key=lambda item: (
                int(item.get("copy_count") or 0),
                float(item.get("mean_word_probability") or 0),
            ),
        )
        merged = dict(representative)
        merged["support_count"] = len({str(item["window_id"]) for item in cluster})
        merged["support_window_ids"] = sorted(
            {str(item["window_id"]) for item in cluster}
        )
        output.append(merged)
    return output


def probe(
    media_path: Path,
    transcript_path: Path,
    *,
    output_path: Path,
    window_seconds: float,
    stride_seconds: float,
    model_id: str,
    model_revision: str,
    model_cache: Path,
) -> Path:
    try:
        import mlx_whisper
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ProbeError("The MLX Whisper probe runtime is unavailable") from error

    media_path = media_path.expanduser().resolve()
    transcript_path = transcript_path.expanduser().resolve()
    if not media_path.is_file() or not transcript_path.is_file():
        raise ProbeError("Probe media and transcript must exist")
    transcript_document = json.loads(transcript_path.read_text(encoding="utf-8"))
    words = canonical_words(transcript_document)
    source = transcript_document.get("source")
    if not isinstance(source, Mapping):
        raise ProbeError("Canonical transcript has no source binding")
    expected_sha = str(source.get("sha256") or "")
    media_sha = sha256_file(media_path)
    if expected_sha != media_sha:
        raise ProbeError("Canonical transcript is not bound to the exact probe media")
    duration = float(source.get("duration_seconds") or words[-1]["end"])
    model_path = snapshot_download(
        repo_id=model_id,
        revision=model_revision,
        cache_dir=str(model_cache.expanduser().resolve()),
    )
    audio = mlx_whisper.audio.load_audio(str(media_path), sr=SAMPLE_RATE)
    windows: list[dict[str, Any]] = []
    raw_candidates: list[dict[str, Any]] = []
    for sequence, start in enumerate(
        window_starts(duration, window_seconds=window_seconds, stride_seconds=stride_seconds),
        1,
    ):
        end = min(duration, start + window_seconds)
        window_id = f"window-{sequence:04d}"
        sample_start = max(0, int(round(start * SAMPLE_RATE)))
        sample_end = min(len(audio), int(round(end * SAMPLE_RATE)))
        result = mlx_whisper.transcribe(
            np.asarray(audio[sample_start:sample_end], dtype=np.float32),
            path_or_hf_repo=str(model_path),
            language="en",
            task="transcribe",
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=None,
            verbose=False,
        )
        probe_words = normalize_probe_words(
            result, window_id=window_id, window_start=start
        )
        candidates = repeated_runs(
            probe_words,
            window_id=window_id,
            window_start=start,
            window_end=end,
        )
        raw_candidates.extend(candidates)
        windows.append(
            {
                "id": window_id,
                "start": round(start, 6),
                "end": round(end, 6),
                "text": str(result.get("text") or "").strip(),
                "words": probe_words,
                "candidate_count": len(candidates),
            }
        )
    candidates: list[dict[str, Any]] = []
    for sequence, value in enumerate(cluster_candidates(raw_candidates), 1):
        if int(value.get("support_count") or 0) < 2:
            continue
        match = canonical_match(value, words)
        if match is None:
            continue
        constrained_end = float(value["removal_end"])
        following_start = match.get("canonical_following_word_start")
        if isinstance(following_start, (int, float)):
            constrained_end = min(constrained_end, float(following_start))
        if constrained_end - float(value["removal_start"]) < 0.04:
            continue
        candidates.append(
            {
                "id": f"shadow-repeat-{sequence:04d}",
                "kind": "repetition",
                "detector": "enhanced-verbatim-shadow-repeat",
                "start": value["removal_start"],
                "end": round(constrained_end, 6),
                "original_start": value["removal_start"],
                "original_end": round(constrained_end, 6),
                "removed_word_ids": [],
                "shadow_removed_word_ids": value["shadow_removed_word_ids"],
                "shadow_retained_word_ids": value["shadow_retained_word_ids"],
                "shadow_removed_text": value["shadow_removed_text"],
                "shadow_retained_text": value["shadow_retained_text"],
                "phrase_tokens": value["phrase_tokens"],
                "copy_count": value["copy_count"],
                "support_count": value["support_count"],
                "support_window_ids": value["support_window_ids"],
                "mean_word_probability": value["mean_word_probability"],
                **match,
            }
        )
    document = {
        "schema_version": 1,
        "agent": "speech-tightener",
        "stage": "enhanced-verbatim-probe",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "media_path": str(media_path),
            "media_sha256": media_sha,
            "canonical_transcript_path": str(transcript_path),
            "canonical_transcript_sha256": sha256_file(transcript_path),
            "duration_seconds": round(duration, 6),
        },
        "asr": {
            "engine": "mlx-whisper-overlapping-windows",
            "model_id": model_id,
            "model_revision": model_revision,
            "initial_prompt": None,
            "condition_on_previous_text": False,
            "word_timestamps": True,
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
        },
        "windows": windows,
        "candidates": candidates,
        "verification": {
            "media_sha256_verified": expected_sha == media_sha,
            "canonical_transcript_sha256_recorded": True,
            "prompt_free": True,
            "complete_media_window_coverage": bool(windows)
            and float(windows[0]["start"]) == 0.0
            and float(windows[-1]["end"]) >= duration - core.TIME_TOLERANCE,
        },
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    core.write_json_atomic(output_path, document)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=6.0)
    parser.add_argument("--stride-seconds", type=float, default=2.5)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        path = probe(
            args.media,
            args.transcript,
            output_path=args.output,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            model_id=args.model,
            model_revision=args.model_revision,
            model_cache=args.model_cache,
        )
        print(path)
        return 0
    except (ProbeError, core.TimelineEditError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
