#!/usr/bin/env python3
"""Find, file-review, and remove pauses, fillers, repetitions, and false starts."""

from __future__ import annotations

import argparse
import array
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SHARED_DIR = Path(__file__).resolve().parents[3] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

import timeline_edit as core  # noqa: E402


AGENT_NAME = "speech-tightener"
ALLOWED_KINDS = {
    "long-pause",
    "filler",
    "discourse-marker",
    "repetition",
    "false-start",
    "word-search",
}
FILLERS = {"um", "umm", "uh", "uhh", "erm", "er", "hmm", "mm"}
DISCOURSE_MARKERS = (
    ("i", "mean", "you", "know"),
    ("you", "know", "i", "mean"),
    ("you", "know"),
    ("i", "mean"),
)
LITERAL_KNOWLEDGE_PREDECESSORS = {"do", "did", "does", "what", "how", "if", "whether"}
LITERAL_KNOWLEDGE_FOLLOWERS = {"how", "what", "when", "where", "who", "why", "something"}
DISCOURSE_MARKER_CROSSFADE_SECONDS = 0.02
ENHANCED_DISFLUENCY_CROSSFADE_SECONDS = 0.03
ENHANCED_DISFLUENCY_BOUNDARY_NOISE_DB = -18.0
ENHANCED_DISFLUENCY_ZERO_CROSSING_MAX_DBFS = -45.0
ENHANCED_DISFLUENCY_INWARD_TOLERANCE_SECONDS = 0.08
SPOKEN_ZERO_CROSSING_MAX_DBFS = -60.0
MAX_COLLATERAL_WORD_OVERLAP_SECONDS = 0.02
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
DETECTION_PROFILES = {"standard", "enhanced-disfluency"}
CONTRACTION_PARTS = {
    "i'll": ("i", "will"),
    "it'll": ("it", "will"),
    "we'll": ("we", "will"),
    "you'll": ("you", "will"),
    "they'll": ("they", "will"),
    "i'm": ("i", "am"),
    "it's": ("it", "is"),
    "that's": ("that", "is"),
    "we're": ("we", "are"),
    "you're": ("you", "are"),
    "they're": ("they", "are"),
}
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "timeline-edits"


def resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    discovered = shutil.which(value)
    if discovered:
        return Path(discovered).resolve()
    raise core.TimelineEditError(f"Required executable was not found: {value}")


def normalized_word(value: Any) -> str:
    match = TOKEN_PATTERN.search(str(value or "").lower().replace("’", "'"))
    return match.group(0) if match else ""


def lexical_signature(words: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    signature: list[str] = []
    for word in words:
        token = normalized_word(word.get("text"))
        if not token:
            continue
        signature.extend(CONTRACTION_PARTS.get(token, (token,)))
    return tuple(signature)


def has_trailing_comma(word: Mapping[str, Any]) -> bool:
    return str(word.get("text") or "").strip().endswith(",")


def candidate(
    *,
    candidate_id: str,
    kind: str,
    start: float,
    end: float,
    words: Sequence[Mapping[str, Any]],
    transcript: core.TranscriptArtifact,
    confidence: str,
    detector: str,
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "kind": kind,
        "start": round(start, 6),
        "end": round(end, 6),
        "duration_seconds": round(end - start, 6),
        "removed_word_ids": [str(word["id"]) for word in words],
        "source_text": " ".join(str(word.get("text") or "").strip() for word in words),
        "context": core.transcript_excerpt(transcript, start, end),
        "detector": detector,
        "confidence": confidence,
        "decision": "review",
        "meaning_preserved": None,
        "evidence_mode": "local-file-analysis",
        "review_evidence": [],
        "rationale": "",
    }


def overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return float(left["start"]) < float(right["end"]) - core.TIME_TOLERANCE and float(
        right["start"]
    ) < float(left["end"]) - core.TIME_TOLERANCE


def deduplicate_candidates(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "long-pause": 0,
        "false-start": 1,
        "word-search": 1,
        "repetition": 2,
        "filler": 3,
    }
    selected: list[dict[str, Any]] = []
    for value in sorted(
        values,
        key=lambda item: (
            priority.get(str(item["kind"]), 9),
            float(item["start"]),
            -float(item["duration_seconds"]),
        ),
    ):
        value_is_pause = value.get("kind") == "long-pause"
        if not any(
            overlaps(value, existing)
            and value_is_pause == (existing.get("kind") == "long-pause")
            for existing in selected
        ):
            selected.append(value)
    selected.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    for index, item in enumerate(selected, 1):
        item["id"] = f"speech-{index:04d}"
    return selected


def detect_enhanced_disfluencies(
    transcript: core.TranscriptArtifact,
    words: Sequence[Mapping[str, Any]],
    tokens: Sequence[str],
) -> list[dict[str, Any]]:
    """Propose conservative repair candidates that need word-level ASR timing."""

    values: list[dict[str, Any]] = []

    # Catch phrase restarts even when the second attempt uses a contraction,
    # for example "it will then get, it'll then get".
    claimed_restarts: set[tuple[int, int]] = set()
    for first_start in range(len(words)):
        for first_size in range(6, 1, -1):
            first_end = first_start + first_size
            if first_end > len(words):
                continue
            first_words = words[first_start:first_end]
            first_signature = lexical_signature(first_words)
            if len(first_signature) < 3:
                continue
            found = False
            for second_start in range(first_end, min(len(words), first_end + 5)):
                gap_seconds = float(words[second_start]["start"]) - float(
                    words[first_end - 1]["end"]
                )
                if gap_seconds > 2.5:
                    break
                for second_size in range(1, 7):
                    second_end = second_start + second_size
                    if second_end > len(words):
                        break
                    second_signature = lexical_signature(
                        words[second_start:second_end]
                    )
                    signatures_match = second_signature == first_signature
                    reordered_restart = (
                        len(second_signature) == len(first_signature)
                        and sorted(second_signature) == sorted(first_signature)
                        and bool(
                            {"i", "it", "we", "you", "they"}
                            & set(first_signature)
                        )
                    )
                    if not signatures_match and not reordered_restart:
                        continue
                    word_range = (first_start, first_end)
                    if word_range not in claimed_restarts:
                        values.append(
                            candidate(
                                candidate_id="pending",
                                kind="false-start",
                                start=float(first_words[0]["start"]),
                                end=float(first_words[-1]["end"]),
                                words=first_words,
                                transcript=transcript,
                                confidence="high",
                                detector="enhanced-phrase-restart",
                            )
                        )
                        claimed_restarts.add(word_range)
                    found = True
                    break
                if found:
                    break

    # A comma-terminated article phrase immediately followed by the same article
    # is often an abandoned noun-phrase start: "the scaling, the only place".
    articles = {"a", "an", "the"}
    for index, token in enumerate(tokens):
        if token not in articles:
            continue
        for restart_index in range(index + 2, min(len(words), index + 4)):
            if tokens[restart_index] != token or not has_trailing_comma(
                words[restart_index - 1]
            ):
                continue
            selected_words = words[index:restart_index]
            values.append(
                candidate(
                    candidate_id="pending",
                    kind="false-start",
                    start=float(selected_words[0]["start"]),
                    end=float(selected_words[-1]["end"]),
                    words=selected_words,
                    transcript=transcript,
                    confidence="medium",
                    detector="enhanced-article-phrase-restart",
                )
            )
            break

    # Detect an abandoned determiner phrase repaired by "all", such as
    # "the entire all US electricity production".
    for index in range(len(words) - 2):
        if tokens[index] != "the":
            continue
        for restart_index in range(index + 2, min(len(words), index + 4)):
            if tokens[restart_index] != "all":
                continue
            selected_words = words[index:restart_index]
            values.append(
                candidate(
                    candidate_id="pending",
                    kind="false-start",
                    start=float(selected_words[0]["start"]),
                    end=float(selected_words[-1]["end"]),
                    words=selected_words,
                    transcript=transcript,
                    confidence="medium",
                    detector="enhanced-quantifier-restart",
                )
            )
            break

    # A comma-terminated noun fragment after a conjunction can be explicitly
    # superseded by a fresh pronoun clause: "because the casting, it's ...".
    conjunctions = {"and", "because", "but", "so"}
    pronoun_restarts = {"i", "i'm", "it", "it's", "that", "that's", "we", "we're"}
    for index, token in enumerate(tokens[:-3]):
        if token not in conjunctions:
            continue
        for comma_index in range(index + 2, min(len(words) - 1, index + 5)):
            if not has_trailing_comma(words[comma_index]):
                continue
            if tokens[comma_index + 1] not in pronoun_restarts:
                continue
            selected_words = words[index + 1 : comma_index + 1]
            values.append(
                candidate(
                    candidate_id="pending",
                    kind="false-start",
                    start=float(selected_words[0]["start"]),
                    end=float(selected_words[-1]["end"]),
                    words=selected_words,
                    transcript=transcript,
                    confidence="medium",
                    detector="enhanced-clause-restart",
                )
            )
            break

    # Preserve one uncertainty qualifier while removing a duplicated search for
    # wording: "about, call it roughly" -> "roughly".
    for index, token in enumerate(tokens):
        if token != "about":
            continue
        for roughly_index in range(index + 1, min(len(words), index + 5)):
            if tokens[roughly_index] != "roughly":
                continue
            selected_words = words[index:roughly_index]
            values.append(
                candidate(
                    candidate_id="pending",
                    kind="word-search",
                    start=float(selected_words[0]["start"]),
                    end=float(selected_words[-1]["end"]),
                    words=selected_words,
                    transcript=transcript,
                    confidence="medium",
                    detector="enhanced-duplicate-uncertainty-repair",
                )
            )
            break

    for index in range(4, len(words)):
        if tokens[index] == "like" and tuple(tokens[index - 4 : index]) == (
            "on",
            "the",
            "order",
            "of",
        ):
            values.append(
                candidate(
                    candidate_id="pending",
                    kind="word-search",
                    start=float(words[index]["start"]),
                    end=float(words[index]["end"]),
                    words=[words[index]],
                    transcript=transcript,
                    confidence="high",
                    detector="enhanced-redundant-approximation-marker",
                )
            )

    # "no" is a repair only when the following clause immediately repeats the
    # negation, avoiding removal from standalone disagreement or emphasis.
    for index, token in enumerate(tokens):
        if token != "no":
            continue
        following = tuple(tokens[index + 1 : index + 4])
        repair = following[:2] == ("it's", "not") or following == (
            "it",
            "is",
            "not",
        )
        if not repair:
            continue
        values.append(
            candidate(
                candidate_id="pending",
                kind="false-start",
                start=float(words[index]["start"]),
                end=float(words[index]["end"]),
                words=[words[index]],
                transcript=transcript,
                confidence="medium",
                detector="enhanced-negation-repair",
            )
        )

    return values


def load_verbatim_probe_candidates(
    probe_path: Path,
    *,
    media: core.MediaArtifact,
    transcript: core.TranscriptArtifact,
) -> list[dict[str, Any]]:
    resolved_path, document = core.read_json(
        probe_path, "Enhanced verbatim repetition probe"
    )
    if (
        document.get("agent") != AGENT_NAME
        or document.get("stage") != "enhanced-verbatim-probe"
    ):
        raise core.TimelineEditError("Enhanced verbatim probe identity is invalid")
    source = document.get("source")
    verification = document.get("verification")
    if not isinstance(source, Mapping) or not isinstance(verification, Mapping):
        raise core.TimelineEditError("Enhanced verbatim probe is incomplete")
    if (
        source.get("media_sha256") != media.media_sha256
        or source.get("canonical_transcript_sha256") != transcript.sha256
        or verification.get("media_sha256_verified") is not True
        or verification.get("prompt_free") is not True
        or verification.get("complete_media_window_coverage") is not True
    ):
        raise core.TimelineEditError(
            "Enhanced verbatim probe is stale or not prompt-free"
        )
    candidates = document.get("candidates")
    if not isinstance(candidates, list):
        raise core.TimelineEditError("Enhanced verbatim probe candidates must be a list")
    values: list[dict[str, Any]] = []
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise core.TimelineEditError("Enhanced verbatim probe candidate is invalid")
        if (
            raw.get("kind") != "repetition"
            or raw.get("detector") != "enhanced-verbatim-shadow-repeat"
            or raw.get("removed_word_ids") != []
            or not isinstance(raw.get("shadow_removed_word_ids"), list)
            or not raw.get("shadow_removed_word_ids")
            or int(raw.get("support_count") or 0) < 2
        ):
            raise core.TimelineEditError(
                "Enhanced verbatim probe candidate lacks corroborated shadow provenance"
            )
        value = dict(raw)
        start = float(value["start"])
        end = float(value["end"])
        value.update(
            {
                "id": "pending",
                "duration_seconds": round(end - start, 6),
                "source_text": str(value.get("shadow_removed_text") or ""),
                "context": core.transcript_excerpt(transcript, start, end),
                "confidence": "high",
                "decision": "review",
                "meaning_preserved": None,
                "evidence_mode": "local-file-analysis",
                "review_evidence": [
                    f"verbatim_probe:{resolved_path}",
                    f"support_windows:{value.get('support_window_ids')}",
                    f"shadow_removed_word_ids:{value.get('shadow_removed_word_ids')}",
                    f"canonical_collapsed_word_ids:{value.get('canonical_collapsed_word_ids')}",
                ],
                "rationale": "",
                "verbatim_probe_path": str(resolved_path),
                "verbatim_probe_sha256": core.sha256_file(resolved_path),
                "verbatim_probe_media_sha256": media.media_sha256,
                "verbatim_probe_transcript_sha256": transcript.sha256,
            }
        )
        values.append(value)
    return values


def decode_mono_pcm(
    media: core.MediaArtifact,
    *,
    ffmpeg: Path,
    sample_rate: int = 8000,
) -> array.array:
    if sample_rate <= 0:
        raise core.TimelineEditError("PCM sample rate must be positive")
    with tempfile.NamedTemporaryFile(suffix=".pcm") as temporary:
        result = subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(media.media_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-f",
                "s16le",
                temporary.name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip()[-4000:]
            raise core.TimelineEditError(
                "FFmpeg PCM decode failed" + (f":\n{detail}" if detail else "")
            )
        temporary.seek(0)
        samples = array.array("h")
        samples.frombytes(temporary.read())
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise core.TimelineEditError("Decoded PCM audio is empty")
    return samples


def pcm_rms_db(
    samples: array.array,
    *,
    sample_rate: int,
    center: float,
    half_window: float = 0.006,
) -> float:
    start = max(0, int((center - half_window) * sample_rate))
    end = min(len(samples), int((center + half_window) * sample_rate))
    if end <= start:
        return -120.0
    mean_square = sum(value * value for value in samples[start:end]) / (end - start)
    rms = max(math.sqrt(mean_square), 1.0)
    return 20 * math.log10(rms / 32768)


def quietest_pcm_boundary(
    samples: array.array,
    *,
    sample_rate: int,
    start: float,
    end: float,
    step: float = 0.004,
) -> tuple[float, float, float]:
    if end < start:
        start, end = end, start
    count = max(1, int((end - start) / step) + 1)
    candidates = []
    for index in range(count):
        timestamp = min(end, start + index * step)
        candidates.append(
            (
                pcm_rms_db(
                    samples,
                    sample_rate=sample_rate,
                    center=timestamp,
                ),
                timestamp,
            )
        )
    rms_db, center = min(candidates)
    radius = max(1, int(0.004 * sample_rate))
    center_index = min(len(samples) - 1, max(0, int(center * sample_rate)))
    search_lower = max(0, int(start * sample_rate))
    search_upper = min(len(samples), int(end * sample_rate) + 1)
    lower = max(search_lower, center_index - radius)
    upper = min(search_upper, center_index + radius + 1)
    if upper <= lower:
        lower = min(len(samples) - 1, max(0, center_index))
        upper = min(len(samples), lower + 1)
    zero_index = min(range(lower, upper), key=lambda index: abs(samples[index]))
    zero_amplitude_dbfs = 20 * math.log10(
        max(abs(int(samples[zero_index])), 1) / 32768
    )
    return (
        round(zero_index / sample_rate, 6),
        round(rms_db, 3),
        round(zero_amplitude_dbfs, 3),
    )


def refine_spoken_candidates(
    media: core.MediaArtifact,
    transcript: core.TranscriptArtifact,
    candidates: Sequence[dict[str, Any]],
    *,
    ffmpeg: Path,
    boundary_noise_db: float,
    inward_tolerance: float,
    sample_rate: int = 8000,
) -> list[dict[str, Any]]:
    samples = decode_mono_pcm(media, ffmpeg=ffmpeg, sample_rate=sample_rate)
    refined: list[dict[str, Any]] = []
    spoken_kinds = {
        "filler",
        "discourse-marker",
        "repetition",
        "false-start",
        "word-search",
    }
    for source_candidate in candidates:
        value = dict(source_candidate)
        if value.get("kind") not in spoken_kinds:
            refined.append(value)
            continue
        original_start = float(value["start"])
        original_end = float(value["end"])
        declared_ids = {str(item) for item in value.get("removed_word_ids", [])}
        shadow_repetition = (
            value.get("detector") == "enhanced-verbatim-shadow-repeat"
            and value.get("kind") == "repetition"
            and value.get("removed_word_ids") == []
        )
        canonical_collapsed_ids = {
            str(item) for item in value.get("canonical_collapsed_word_ids", [])
        }
        retained_words = [
            word
            for word in core.transcript_words(transcript)
            if str(word["id"]) not in declared_ids
            and (not shadow_repetition or str(word["id"]) not in canonical_collapsed_ids)
        ]
        previous_retained = [
            word for word in retained_words if float(word["end"]) <= original_start
        ]
        following_retained = [
            word for word in retained_words if float(word["start"]) >= original_end
        ]
        start_floor = max(0.0, original_start - 0.24)
        if previous_retained:
            start_floor = max(
                start_floor,
                float(previous_retained[-1]["end"])
                - MAX_COLLATERAL_WORD_OVERLAP_SECONDS,
            )
        end_ceiling = min(media.duration_seconds, original_end + 0.24)
        if following_retained:
            end_ceiling = min(
                end_ceiling,
                float(following_retained[0]["start"])
                + MAX_COLLATERAL_WORD_OVERLAP_SECONDS,
            )
        start, start_db, start_zero_dbfs = quietest_pcm_boundary(
            samples,
            sample_rate=sample_rate,
            start=start_floor,
            end=original_start + 0.06,
        )
        end, end_db, end_zero_dbfs = quietest_pcm_boundary(
            samples,
            sample_rate=sample_rate,
            start=max(original_start + 0.04, original_end - 0.06),
            end=end_ceiling,
        )
        collateral_overlaps = [
            {
                "word_id": str(word["id"]),
                "text": str(word.get("text") or ""),
                "overlap_seconds": round(
                    max(
                        0.0,
                        min(end, float(word["end"]))
                        - max(start, float(word["start"])),
                    ),
                    6,
                ),
            }
            for word in core.transcript_words(transcript)
            if str(word["id"]) not in declared_ids
            and (not shadow_repetition or str(word["id"]) not in canonical_collapsed_ids)
            and max(
                0.0,
                min(end, float(word["end"]))
                - max(start, float(word["start"])),
            )
            > MAX_COLLATERAL_WORD_OVERLAP_SECONDS
        ]
        enhanced_speech_edge = str(value.get("detector") or "").startswith(
            "enhanced-"
        ) and value.get("kind") in {"false-start", "word-search", "repetition"}
        candidate_boundary_noise_db = (
            boundary_noise_db
            if enhanced_speech_edge
            else min(boundary_noise_db, -35.0)
        )
        candidate_zero_crossing_max_dbfs = (
            ENHANCED_DISFLUENCY_ZERO_CROSSING_MAX_DBFS
            if enhanced_speech_edge
            else SPOKEN_ZERO_CROSSING_MAX_DBFS
        )
        safe = (
            max(start_db, end_db) <= candidate_boundary_noise_db
            and max(start_zero_dbfs, end_zero_dbfs)
            <= candidate_zero_crossing_max_dbfs
            and start <= original_start + inward_tolerance
            and end >= original_end - inward_tolerance
            and end - start >= 0.04
            and not collateral_overlaps
        )
        value.update(
            {
                "original_start": round(original_start, 6),
                "original_end": round(original_end, 6),
                "original_duration_seconds": round(original_end - original_start, 6),
                "acoustic_boundary_refined": safe,
                "boundary_threshold_db": candidate_boundary_noise_db,
                "boundary_zero_crossing_threshold_dbfs": (
                    candidate_zero_crossing_max_dbfs
                ),
                "boundary_rms_db": {"start": start_db, "end": end_db},
                "boundary_zero_crossing_dbfs": {
                    "start": start_zero_dbfs,
                    "end": end_zero_dbfs,
                },
                "boundary_shift_seconds": {
                    "start": round(start - original_start, 6),
                    "end": round(end - original_end, 6),
                },
                "boundary_sample_rate_hz": sample_rate,
                "boundary_inward_tolerance_seconds": inward_tolerance,
                "collateral_word_overlaps": collateral_overlaps,
                "maximum_collateral_word_overlap_seconds": (
                    MAX_COLLATERAL_WORD_OVERLAP_SECONDS
                ),
                "canonical_collapsed_word_overlaps": (
                    sorted(canonical_collapsed_ids) if shadow_repetition else []
                ),
            }
        )
        if safe:
            value["start"] = start
            value["end"] = end
            value["duration_seconds"] = round(end - start, 6)
            value["detector"] = f"{value.get('detector')}+pcm-boundary-refinement"
            if value.get("kind") == "discourse-marker":
                value["audio_join_treatment"] = {
                    "method": "equal-power-micro-crossfade",
                    "duration_seconds": DISCOURSE_MARKER_CROSSFADE_SECONDS,
                }
            elif enhanced_speech_edge:
                value["audio_join_treatment"] = {
                    "method": "equal-power-micro-crossfade",
                    "duration_seconds": ENHANCED_DISFLUENCY_CROSSFADE_SECONDS,
                }
        else:
            value["refinement_rejection"] = (
                "Refined boundary overlaps a retained transcript word"
                if collateral_overlaps
                else "No sufficiently quiet boundary pair covered the complete ASR word span"
            )
        refined.append(value)
    return refined


def detect_acoustic_silences(
    media: core.MediaArtifact,
    *,
    ffmpeg: Path,
    noise_db: float,
    minimum_duration: float,
    retained_silence: float,
    source_start: float,
    source_end: float,
) -> list[dict[str, float]]:
    if minimum_duration <= 0 or retained_silence < 0:
        raise core.TimelineEditError("Acoustic silence thresholds are invalid")
    if retained_silence >= minimum_duration:
        raise core.TimelineEditError(
            "Retained silence must be shorter than the detected silence threshold"
        )
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-nostats",
        "-i",
        str(media.media_path),
        "-af",
        f"silencedetect=noise={noise_db:g}dB:d={minimum_duration:g}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip()[-4000:]
        raise core.TimelineEditError(
            "FFmpeg acoustic silence detection failed"
            + (f":\n{detail}" if detail else "")
        )
    start_pattern = re.compile(r"silence_start:\s*([0-9.]+)")
    end_pattern = re.compile(
        r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)"
    )
    pending_start: float | None = None
    detected: list[dict[str, float]] = []
    for line in result.stderr.splitlines():
        start_match = start_pattern.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue
        end_match = end_pattern.search(line)
        if not end_match or pending_start is None:
            continue
        silence_start = max(source_start, pending_start)
        silence_end = min(source_end, float(end_match.group(1)))
        pending_start = None
        duration = silence_end - silence_start
        if duration < minimum_duration:
            continue
        retained_each_side = retained_silence / 2
        cut_start = silence_start + retained_each_side
        cut_end = silence_end - retained_each_side
        if cut_end - cut_start < 0.08:
            continue
        detected.append(
            {
                "silence_start": round(silence_start, 6),
                "silence_end": round(silence_end, 6),
                "start": round(cut_start, 6),
                "end": round(cut_end, 6),
                "silence_duration_seconds": round(duration, 6),
                "retained_silence_seconds": round(retained_silence, 6),
            }
        )
    return detected


def detect_candidates(
    transcript: core.TranscriptArtifact,
    *,
    pause_threshold: float,
    retained_pause: float,
    acoustic_silences: Sequence[Mapping[str, float]] | None = None,
    detection_profile: str = "standard",
) -> list[dict[str, Any]]:
    if detection_profile not in DETECTION_PROFILES:
        raise core.TimelineEditError(
            f"Unsupported Speech Tightener detection profile: {detection_profile}"
        )
    if pause_threshold <= retained_pause * 2 or retained_pause < 0:
        raise core.TimelineEditError("Pause thresholds are invalid")
    words = core.transcript_words(transcript)
    tokens = [normalized_word(word.get("text")) for word in words]
    values: list[dict[str, Any]] = []

    if acoustic_silences is not None:
        for silence in acoustic_silences:
            value = candidate(
                candidate_id="pending",
                kind="long-pause",
                start=float(silence["start"]),
                end=float(silence["end"]),
                words=[],
                transcript=transcript,
                confidence="high",
                detector="ffmpeg-silencedetect",
            )
            value.update(
                {
                    "acoustic_silence": True,
                    "silence_start": float(silence["silence_start"]),
                    "silence_end": float(silence["silence_end"]),
                    "silence_duration_seconds": float(
                        silence["silence_duration_seconds"]
                    ),
                    "retained_silence_seconds": float(
                        silence["retained_silence_seconds"]
                    ),
                }
            )
            values.append(value)
    else:
        for index in range(1, len(words)):
            previous = words[index - 1]
            current = words[index]
            gap = float(current["start"]) - float(previous["end"])
            if gap >= pause_threshold:
                start = float(previous["end"]) + retained_pause
                end = float(current["start"]) - retained_pause
                if end - start >= 0.15:
                    values.append(
                        candidate(
                            candidate_id="pending",
                            kind="long-pause",
                            start=start,
                            end=end,
                            words=[],
                            transcript=transcript,
                            confidence="high",
                            detector="inter-word-gap",
                        )
                    )

    for index, word in enumerate(words):
        if tokens[index] in FILLERS:
            values.append(
                candidate(
                    candidate_id="pending",
                    kind="filler",
                    start=float(word["start"]),
                    end=float(word["end"]),
                    words=[word],
                    transcript=transcript,
                    confidence="high",
                    detector="standalone-filler-token",
                )
            )

    claimed_marker_indices: set[int] = set()
    for phrase in DISCOURSE_MARKERS:
        phrase_size = len(phrase)
        for index in range(0, len(words) - phrase_size + 1):
            if tuple(tokens[index : index + phrase_size]) != phrase:
                continue
            phrase_indices = set(range(index, index + phrase_size))
            if claimed_marker_indices & phrase_indices:
                continue
            if any(
                float(words[word_index + 1]["start"])
                - float(words[word_index]["end"])
                > 0.45
                for word_index in range(index, index + phrase_size - 1)
            ):
                continue
            selected_words = words[index : index + phrase_size]
            value = candidate(
                candidate_id="pending",
                kind="discourse-marker",
                start=float(selected_words[0]["start"]),
                end=float(selected_words[-1]["end"]),
                words=selected_words,
                transcript=transcript,
                confidence="medium",
                detector="discourse-marker-phrase",
            )
            previous_token = tokens[index - 1] if index > 0 else ""
            next_index = index + phrase_size
            next_token = tokens[next_index] if next_index < len(tokens) else ""
            literal_knowledge_risk = (
                phrase[-2:] == ("you", "know")
                and (
                    previous_token in LITERAL_KNOWLEDGE_PREDECESSORS
                    or next_token in LITERAL_KNOWLEDGE_FOLLOWERS
                )
            )
            value.update(
                {
                    "semantic_role": None,
                    "semantic_review_required": True,
                    "semantic_risk": (
                        "literal-knowledge-or-question"
                        if literal_knowledge_risk
                        else "correction-emphasis-or-discourse-link"
                    ),
                }
            )
            values.append(value)
            claimed_marker_indices.update(phrase_indices)

    for index in range(len(words) - 1):
        if tokens[index] and tokens[index] == tokens[index + 1]:
            values.append(
                candidate(
                    candidate_id="pending",
                    kind="repetition",
                    start=float(words[index]["start"]),
                    end=float(words[index]["end"]),
                    words=[words[index]],
                    transcript=transcript,
                    confidence="high",
                    detector="adjacent-token-repeat",
                )
            )

    claimed_phrase_ranges: set[tuple[int, int]] = set()
    for size in range(6, 1, -1):
        for index in range(0, len(words) - size * 2 + 1):
            first = tokens[index : index + size]
            second = tokens[index + size : index + size * 2]
            if not all(first) or first != second:
                continue
            if float(words[index + size]["start"]) - float(words[index + size - 1]["end"]) > 1.5:
                continue
            word_range = (index, index + size)
            if any(left <= index < right for left, right in claimed_phrase_ranges):
                continue
            claimed_phrase_ranges.add(word_range)
            repeated = words[index : index + size]
            values.append(
                candidate(
                    candidate_id="pending",
                    kind="repetition",
                    start=float(repeated[0]["start"]),
                    end=float(repeated[-1]["end"]),
                    words=repeated,
                    transcript=transcript,
                    confidence="high",
                    detector=f"adjacent-{size}-token-phrase-repeat",
                )
            )
    if detection_profile == "enhanced-disfluency":
        values.extend(detect_enhanced_disfluencies(transcript, words, tokens))
    return deduplicate_candidates(values)


def prepare_plan(
    *,
    media: core.MediaArtifact,
    transcript: core.TranscriptArtifact,
    output_dir: Path,
    pause_threshold: float,
    retained_pause: float,
    ffmpeg: Path,
    silence_noise_db: float,
    silence_min_duration: float,
    retained_silence: float,
    spoken_boundary_noise_db: float,
    spoken_boundary_inward_tolerance: float,
    force: bool,
    detection_profile: str = "standard",
    verbatim_probe_path: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    if plan_path.exists() and not force:
        raise core.TimelineEditError(
            f"Speech edit plan exists. Use --force to replace it: {plan_path}"
        )
    source_start = 0.0
    source_end = media.duration_seconds
    acoustic_silences = detect_acoustic_silences(
        media,
        ffmpeg=ffmpeg,
        noise_db=silence_noise_db,
        minimum_duration=silence_min_duration,
        retained_silence=retained_silence,
        source_start=source_start,
        source_end=source_end,
    )
    candidates = detect_candidates(
        transcript,
        pause_threshold=pause_threshold,
        retained_pause=retained_pause,
        acoustic_silences=acoustic_silences,
        detection_profile=detection_profile,
    )
    if detection_profile == "enhanced-disfluency":
        if verbatim_probe_path is None:
            raise core.TimelineEditError(
                "enhanced-disfluency requires a complete prompt-free verbatim probe"
            )
        candidates = deduplicate_candidates(
            [
                *candidates,
                *load_verbatim_probe_candidates(
                    verbatim_probe_path,
                    media=media,
                    transcript=transcript,
                ),
            ]
        )
    elif verbatim_probe_path is not None:
        raise core.TimelineEditError(
            "A verbatim probe is only valid with enhanced-disfluency"
        )
    candidates = refine_spoken_candidates(
        media,
        transcript,
        candidates,
        ffmpeg=ffmpeg,
        boundary_noise_db=spoken_boundary_noise_db,
        inward_tolerance=spoken_boundary_inward_tolerance,
    )
    plan = {
        "schema_version": 1,
        "agent": AGENT_NAME,
        "video_id": media.video_id,
        "source_manifest_sha256": media.manifest_sha256,
        "source_media_sha256": media.media_sha256,
        "source_transcript_sha256": transcript.sha256,
        "analysis_scope": {
            "kind": "complete-media-artifact",
            "start": round(source_start, 6),
            "end": round(min(source_end, media.duration_seconds), 6),
        },
        "detection": {
            "profile": detection_profile,
            "pause_threshold_seconds": pause_threshold,
            "retained_pause_each_side_seconds": retained_pause,
            "acoustic_silence": {
                "detector": "ffmpeg-silencedetect",
                "noise_db": silence_noise_db,
                "minimum_duration_seconds": silence_min_duration,
                "retained_total_seconds": retained_silence,
                "detected_count": len(acoustic_silences),
                "automatic_removal": True,
            },
            "spoken_boundary_refinement": {
                "method": "8kHz-mono-PCM quiet-boundary and zero-crossing search",
                "maximum_boundary_rms_db": spoken_boundary_noise_db,
                "maximum_zero_crossing_amplitude_dbfs": (
                    SPOKEN_ZERO_CROSSING_MAX_DBFS
                ),
                "maximum_inward_shift_seconds": spoken_boundary_inward_tolerance,
                "safe_candidate_count": sum(
                    item.get("acoustic_boundary_refined") is True
                    for item in candidates
                ),
                "enhanced_disfluency": {
                    "eligible_kinds": ["false-start", "word-search", "repetition"],
                    "maximum_boundary_rms_db": (
                        spoken_boundary_noise_db
                        if detection_profile == "enhanced-disfluency"
                        else None
                    ),
                    "maximum_zero_crossing_amplitude_dbfs": (
                        ENHANCED_DISFLUENCY_ZERO_CROSSING_MAX_DBFS
                    ),
                    "audio_join": {
                        "method": "equal-power-micro-crossfade",
                        "duration_seconds": ENHANCED_DISFLUENCY_CROSSFADE_SECONDS,
                    },
                    "automatic_removal": False,
                    "verbatim_probe_path": (
                        str(verbatim_probe_path.resolve())
                        if verbatim_probe_path is not None
                        else None
                    ),
                    "verbatim_shadow_repetition_count": sum(
                        item.get("detector", "").startswith(
                            "enhanced-verbatim-shadow-repeat"
                        )
                        for item in candidates
                    ),
                },
            },
            "discourse_markers": {
                "phrases": ["you know", "I mean"],
                "candidate_count": sum(
                    item.get("kind") == "discourse-marker" for item in candidates
                ),
                "semantic_review_required": True,
                "removal_role": "non-semantic-discourse-marker",
                "maximum_boundary_rms_db": -35.0,
                "audio_join": {
                    "method": "equal-power-micro-crossfade",
                    "duration_seconds": DISCOURSE_MARKER_CROSSFADE_SECONDS,
                },
            },
            "candidate_count": len(candidates),
            "automatic_removal": {
                "measured_acoustic_silence": True,
                "spoken_candidates": False,
            },
        },
        "candidates": candidates,
        "review": {
            "status": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
            "review_mode": "local-file-first",
            "model_audio_access": False,
            "human_listening_required": False,
            "final_reel_human_review_required": True,
            "notes": "Automatically shorten every measured acoustic silence to the configured retained pause. Use transcript semantics, PCM RMS and zero-crossing boundaries, and join frames to review spoken candidates. Uncertain spoken candidates stay. Computer-controlled playback state is not listening evidence; human review occurs on the rendered final Reel in Reels Review. Remove you know or I mean only after confirming a non-semantic discourse-marker role; literal knowledge, correction, emphasis, and discourse links stay. The enhanced-disfluency profile proposes phrase restarts, abandoned fragments, and word-search repairs from prompt-free word timestamps, but never authorizes spoken removal automatically.",
        },
    }
    core.write_json_atomic(plan_path, plan)
    return plan_path


def approved_removals(
    plan: Mapping[str, Any],
    *,
    media: core.MediaArtifact,
    transcript: core.TranscriptArtifact,
) -> list[dict[str, Any]]:
    if plan.get("agent") != AGENT_NAME or plan.get("video_id") != media.video_id:
        raise core.TimelineEditError("Speech edit plan identity does not match media")
    if plan.get("source_manifest_sha256") != media.manifest_sha256:
        raise core.TimelineEditError("Speech edit plan media manifest is stale")
    if plan.get("source_transcript_sha256") != transcript.sha256:
        raise core.TimelineEditError("Speech edit plan transcript is stale")
    review = plan.get("review")
    if (
        not isinstance(review, Mapping)
        or review.get("status") != "agent-reviewed"
        or not isinstance(review.get("reviewed_by"), str)
        or not review.get("reviewed_by")
    ):
        raise core.TimelineEditError("Speech edit plan must be agent-reviewed before apply")
    if (
        review.get("review_mode") != "local-file-first"
        or review.get("model_audio_access") is not False
        or review.get("human_listening_required") is not False
        or review.get("final_reel_human_review_required") is not True
    ):
        raise core.TimelineEditError(
            "Speech edit plan must use the local-file-first review contract"
        )
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        raise core.TimelineEditError("Speech edit candidates must be a list")
    analysis_scope = plan.get("analysis_scope")
    if not isinstance(analysis_scope, Mapping):
        raise core.TimelineEditError(
            "Speech edit plan has no complete-media analysis_scope"
        )
    window_start = float(analysis_scope.get("start", 0))
    window_end = float(analysis_scope.get("end", media.duration_seconds))
    if (
        analysis_scope.get("kind") != "complete-media-artifact"
        or window_start < 0
        or window_end > media.duration_seconds + core.DURATION_TOLERANCE
        or window_end - window_start < 0.5
    ):
        raise core.TimelineEditError("Speech edit analysis_scope is invalid")
    if (
        window_start > core.TIME_TOLERANCE
        or window_end < media.duration_seconds - core.TIME_TOLERANCE
    ):
        raise core.TimelineEditError(
            "Speech Tightener cannot analyze a bounded source window. Extract the "
            "Short clip first, then run Speech Tightener on that clip manifest."
        )
    selected: list[dict[str, Any]] = []
    for index, value in enumerate(candidates, 1):
        if not isinstance(value, Mapping):
            raise core.TimelineEditError(
                "Every Speech Tightener candidate must be an object"
            )
        acoustic_pause_candidate = (
            value.get("kind") == "long-pause"
            and value.get("acoustic_silence") is True
        )
        if acoustic_pause_candidate and value.get("decision") != "remove":
            raise core.TimelineEditError(
                f"{value.get('id', index)} measured acoustic silence must be "
                "automatically removed"
            )
        if value.get("decision") != "remove":
            continue
        kind = value.get("kind")
        if kind not in ALLOWED_KINDS:
            raise core.TimelineEditError(f"Unsupported speech cut kind: {kind}")
        if value.get("meaning_preserved") is not True:
            raise core.TimelineEditError(
                f"{value.get('id', index)} must explicitly confirm meaning_preserved"
            )
        evidence = value.get("review_evidence")
        if not isinstance(evidence, list) or not evidence:
            raise core.TimelineEditError(
                f"{value.get('id', index)} needs file-based review evidence"
            )
        if value.get("evidence_mode") != "local-file-analysis":
            raise core.TimelineEditError(
                f"{value.get('id', index)} needs local-file-analysis evidence_mode"
            )
        if not isinstance(value.get("rationale"), str) or not value.get("rationale"):
            raise core.TimelineEditError(f"{value.get('id', index)} needs a rationale")
        item = dict(value)
        item.setdefault("id", f"speech-{index:04d}")
        start = float(item["start"])
        end = float(item["end"])
        if start < window_start - core.TIME_TOLERANCE or end > window_end + core.TIME_TOLERANCE:
            raise core.TimelineEditError(
                f"{item['id']} falls outside the complete media artifact"
            )
        actual_words = core.words_in_range(transcript, start, end)
        actual_ids = [str(word["id"]) for word in actual_words]
        declared_ids = item.get("removed_word_ids")
        shadow_repetition = (
            kind == "repetition"
            and str(item.get("detector") or "").startswith(
                "enhanced-verbatim-shadow-repeat"
            )
            and declared_ids == []
        )
        acoustic_pause = (
            kind == "long-pause" and item.get("acoustic_silence") is True
        )
        refined_spoken = (
            kind
            in {
                "filler",
                "discourse-marker",
                "repetition",
                "false-start",
                "word-search",
            }
            and item.get("acoustic_boundary_refined") is True
        )
        spoken_cut = kind in {
            "filler",
            "discourse-marker",
            "repetition",
            "false-start",
            "word-search",
        }
        boundary_rms = item.get("boundary_rms_db")
        boundary_zero = item.get("boundary_zero_crossing_dbfs")
        boundary_zero_threshold = float(
            item.get(
                "boundary_zero_crossing_threshold_dbfs",
                SPOKEN_ZERO_CROSSING_MAX_DBFS,
            )
        )
        if spoken_cut and (
            not refined_spoken
            or not isinstance(boundary_rms, Mapping)
            or not isinstance(boundary_rms.get("start"), (int, float))
            or not isinstance(boundary_rms.get("end"), (int, float))
            or max(float(boundary_rms["start"]), float(boundary_rms["end"]))
            > float(item.get("boundary_threshold_db", -35.0))
            or not isinstance(boundary_zero, Mapping)
            or not isinstance(boundary_zero.get("start"), (int, float))
            or not isinstance(boundary_zero.get("end"), (int, float))
            or max(float(boundary_zero["start"]), float(boundary_zero["end"]))
            > boundary_zero_threshold
        ):
            raise core.TimelineEditError(
                f"{item['id']} is an unsafe spoken cut and must stay keep"
            )
        if kind == "discourse-marker":
            if item.get("semantic_role") != "non-semantic-discourse-marker":
                raise core.TimelineEditError(
                    f"{item['id']} must confirm a non-semantic discourse-marker role"
                )
            if (
                not refined_spoken
                or not isinstance(boundary_rms, Mapping)
                or not isinstance(boundary_rms.get("start"), (int, float))
                or not isinstance(boundary_rms.get("end"), (int, float))
                or max(float(boundary_rms["start"]), float(boundary_rms["end"]))
                > -35.0
            ):
                raise core.TimelineEditError(
                    f"{item['id']} discourse marker needs two boundaries at or below -35 dB"
                )
            item["audio_join_treatment"] = {
                "method": "equal-power-micro-crossfade",
                "duration_seconds": DISCOURSE_MARKER_CROSSFADE_SECONDS,
            }
        if acoustic_pause and declared_ids != []:
            raise core.TimelineEditError(
                f"{item['id']} acoustic pause must not remove transcript words"
            )
        if acoustic_pause and (
            not isinstance(item.get("silence_duration_seconds"), (int, float))
            or not isinstance(item.get("retained_silence_seconds"), (int, float))
            or float(item["silence_duration_seconds"])
            <= float(item["retained_silence_seconds"])
        ):
            raise core.TimelineEditError(
                f"{item['id']} acoustic pause needs measured silence duration"
            )
        if refined_spoken:
            original_start = float(item.get("original_start", start))
            original_end = float(item.get("original_end", end))
            original_ids = [
                str(word["id"])
                for word in core.words_in_range(
                    transcript, original_start, original_end
                )
            ]
            canonical_collapsed_ids = {
                str(word_id)
                for word_id in item.get("canonical_collapsed_word_ids", [])
            }
            if shadow_repetition:
                probe_path = Path(
                    str(item.get("verbatim_probe_path") or "")
                ).expanduser().resolve()
                shadow_ids = item.get("shadow_removed_word_ids")
                phrase_ids = {
                    str(word_id)
                    for word_id in item.get("canonical_phrase_word_ids", [])
                }
                if (
                    not probe_path.is_file()
                    or item.get("verbatim_probe_sha256")
                    != core.sha256_file(probe_path)
                    or item.get("verbatim_probe_media_sha256")
                    != media.media_sha256
                    or item.get("verbatim_probe_transcript_sha256")
                    != transcript.sha256
                    or not isinstance(shadow_ids, list)
                    or not shadow_ids
                    or not canonical_collapsed_ids
                    or not canonical_collapsed_ids.issubset(phrase_ids)
                ):
                    raise core.TimelineEditError(
                        f"{item['id']} shadow repetition provenance is stale or incomplete"
                    )
            elif not isinstance(declared_ids, list) or declared_ids != original_ids:
                raise core.TimelineEditError(
                    f"{item['id']} refined spoken cut changed its word provenance"
                )
            collateral_overlaps = [
                word
                for word in core.transcript_words(transcript)
                if str(word["id"]) not in set(declared_ids)
                and (
                    not shadow_repetition
                    or str(word["id"]) not in canonical_collapsed_ids
                )
                and max(
                    0.0,
                    min(end, float(word["end"]))
                    - max(start, float(word["start"])),
                )
                > MAX_COLLATERAL_WORD_OVERLAP_SECONDS
            ]
            if collateral_overlaps:
                raise core.TimelineEditError(
                    f"{item['id']} refined boundary overlaps retained transcript words"
                )
        elif not acoustic_pause and (
            not isinstance(declared_ids, list) or declared_ids != actual_ids
        ):
            raise core.TimelineEditError(
                f"{item['id']} removed_word_ids do not exactly match its time range"
            )
        if end - start > 30 and not item.get("long_cut_exception"):
            raise core.TimelineEditError(
                f"{item['id']} exceeds 30 seconds without long_cut_exception"
            )
        if acoustic_pause:
            item["source_text"] = ""
        elif shadow_repetition:
            item["source_text"] = str(item.get("shadow_removed_text") or "")
        elif refined_spoken:
            word_by_id = {
                str(word["id"]): word for word in core.transcript_words(transcript)
            }
            item["source_text"] = " ".join(
                str(word_by_id[word_id].get("text") or "").strip()
                for word_id in declared_ids
            )
        else:
            item["source_text"] = " ".join(
                str(word.get("text") or "").strip() for word in actual_words
            )
        selected.append(item)
    return core.normalize_removals(selected, media.duration_seconds)


def compressed_timeline_seconds(
    source_seconds: float, removals: Sequence[Mapping[str, Any]]
) -> float:
    """Map a source timestamp through the approved removal timeline."""
    removed_before = 0.0
    for removal in removals:
        start = float(removal["start"])
        end = float(removal["end"])
        if source_seconds < start:
            break
        if source_seconds <= end:
            return start - removed_before
        removed_before += end - start
    return source_seconds - removed_before


def propagate_editorial_joins(
    *,
    media: core.MediaArtifact,
    removals: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    """Carry narrative join IDs onto the tightened timeline for exact stingers."""
    raw_joins = media.document.get("plan", {}).get("editorial_joins")
    if raw_joins is None:
        return
    if not isinstance(raw_joins, list):
        raise core.TimelineEditError("Source editorial_joins must be an array")
    mapped: list[dict[str, Any]] = []
    for raw in raw_joins:
        if not isinstance(raw, Mapping):
            raise core.TimelineEditError("Source editorial join must be an object")
        source_seconds = raw.get("output_seconds")
        if not isinstance(source_seconds, (int, float)):
            raise core.TimelineEditError("Source editorial join needs output_seconds")
        item = dict(raw)
        item["source_timeline_seconds"] = round(float(source_seconds), 6)
        item["output_seconds"] = round(
            compressed_timeline_seconds(float(source_seconds), removals), 6
        )
        item["mapped_by"] = AGENT_NAME
        mapped.append(item)
    manifest_path, manifest = core.read_json(
        output_dir / "manifest.json", "Speech Tightener output manifest"
    )
    manifest["editorial_joins"] = mapped
    verification = manifest.get("verification")
    if not isinstance(verification, dict):
        raise core.TimelineEditError("Speech Tightener output verification is missing")
    verification["editorial_joins_propagated"] = True
    core.write_json_atomic(manifest_path, manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("media_manifest")
    prepare.add_argument("transcript")
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--pause-threshold", type=float, default=0.9)
    prepare.add_argument("--retained-pause", type=float, default=0.12)
    prepare.add_argument("--silence-noise-db", type=float, default=-35.0)
    prepare.add_argument("--silence-min-duration", type=float, default=0.55)
    prepare.add_argument("--retained-silence", type=float, default=0.48)
    prepare.add_argument("--spoken-boundary-noise-db", type=float)
    prepare.add_argument(
        "--detection-profile",
        choices=sorted(DETECTION_PROFILES),
        default="standard",
    )
    prepare.add_argument(
        "--verbatim-probe",
        type=Path,
        help=(
            "Use an existing prompt-free overlapping-window repetition probe. "
            "enhanced-disfluency creates one automatically when omitted."
        ),
    )
    prepare.add_argument(
        "--allow-media-bound-transcript-rebind",
        action="store_true",
        help=(
            "Accept a prompt-free transcript created from the exact media file "
            "when its safe basename video ID differs; media SHA-256 must match."
        ),
    )
    prepare.add_argument(
        "--spoken-boundary-inward-tolerance", type=float
    )
    prepare.add_argument("--ffmpeg", default="ffmpeg")
    prepare.add_argument("--force", action="store_true")

    apply = commands.add_parser("apply")
    apply.add_argument("media_manifest")
    apply.add_argument("transcript")
    apply.add_argument("plan")
    apply.add_argument("--output-dir", type=Path)
    apply.add_argument("--ffmpeg", default="ffmpeg")
    apply.add_argument("--ffprobe", default="ffprobe")
    apply.add_argument("--retranscription-language", default="en")
    apply.add_argument(
        "--allow-media-bound-transcript-rebind",
        action="store_true",
    )
    apply.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        media = core.load_media_manifest(args.media_manifest)
        transcript = core.load_transcript(
            args.transcript,
            media,
            allow_media_bound_video_id_rebind=(
                args.allow_media_bound_transcript_rebind
            ),
        )
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else DEFAULT_OUTPUT_ROOT / media.video_id / AGENT_NAME
        )
        if args.command == "prepare":
            verbatim_probe_path = (
                args.verbatim_probe.expanduser().resolve()
                if args.verbatim_probe is not None
                else None
            )
            if args.detection_profile == "enhanced-disfluency":
                verbatim_probe_path = (
                    verbatim_probe_path
                    if verbatim_probe_path is not None
                    else output_dir / "verbatim-probe.json"
                )
                if args.verbatim_probe is None and (
                    args.force or not verbatim_probe_path.is_file()
                ):
                    probe_script = Path(__file__).resolve().with_name(
                        "verbatim_probe.py"
                    )
                    probe_python = (
                        Path(sys.executable)
                    )
                    if not probe_script.is_file() or not probe_python.is_file():
                        raise core.TimelineEditError(
                            "Enhanced verbatim repetition probe runtime is unavailable"
                        )
                    probe_result = subprocess.run(
                        [
                            str(probe_python),
                            str(probe_script),
                            str(media.media_path),
                            str(transcript.path),
                            "--output",
                            str(verbatim_probe_path),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if probe_result.returncode != 0:
                        detail = probe_result.stderr.strip()[-4000:]
                        raise core.TimelineEditError(
                            "Enhanced verbatim repetition probe failed"
                            + (f":\n{detail}" if detail else "")
                        )
            spoken_boundary_noise_db = (
                args.spoken_boundary_noise_db
                if args.spoken_boundary_noise_db is not None
                else (
                    ENHANCED_DISFLUENCY_BOUNDARY_NOISE_DB
                    if args.detection_profile == "enhanced-disfluency"
                    else -35.0
                )
            )
            spoken_boundary_inward_tolerance = (
                args.spoken_boundary_inward_tolerance
                if args.spoken_boundary_inward_tolerance is not None
                else (
                    ENHANCED_DISFLUENCY_INWARD_TOLERANCE_SECONDS
                    if args.detection_profile == "enhanced-disfluency"
                    else 0.03
                )
            )
            plan_path = prepare_plan(
                media=media,
                transcript=transcript,
                output_dir=output_dir,
                pause_threshold=args.pause_threshold,
                retained_pause=args.retained_pause,
                ffmpeg=resolve_executable(args.ffmpeg),
                silence_noise_db=args.silence_noise_db,
                silence_min_duration=args.silence_min_duration,
                retained_silence=args.retained_silence,
                spoken_boundary_noise_db=spoken_boundary_noise_db,
                spoken_boundary_inward_tolerance=(
                    spoken_boundary_inward_tolerance
                ),
                force=args.force,
                detection_profile=args.detection_profile,
                verbatim_probe_path=verbatim_probe_path,
            )
            print(json.dumps({"plan": str(plan_path)}))
        else:
            plan_path, plan = core.read_json(args.plan, "Speech edit plan")
            removals = approved_removals(plan, media=media, transcript=transcript)
            ffmpeg = resolve_executable(args.ffmpeg)
            ffprobe = resolve_executable(args.ffprobe)
            core.apply_edit(
                agent_name=AGENT_NAME,
                media=media,
                transcript=transcript,
                plan_path=plan_path,
                plan=plan,
                removals=removals,
                output_dir=output_dir,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                force=args.force,
            )
            propagate_editorial_joins(
                media=media,
                removals=removals,
                output_dir=output_dir,
            )
            transcribe_script = (
                REPO_ROOT
                / ".agents"
                / "skills"
                / "subtitle-generator"
                / "scripts"
                / "transcribe.py"
            )
            transcribe_python = (
                Path(sys.executable)
            )
            if not transcribe_script.is_file() or not transcribe_python.is_file():
                raise core.TimelineEditError(
                    "Prompt-free post-edit transcription runtime is unavailable"
                )
            retranscription_dir = output_dir / "retranscription"
            retranscription_path = retranscription_dir / "transcript.source.json"
            retranscription_command = [
                str(transcribe_python),
                str(transcribe_script),
                str(output_dir / "edited.mp4"),
                "--language",
                args.retranscription_language,
                "--output-dir",
                str(retranscription_dir),
                "--no-prompt",
            ]
            if args.force:
                retranscription_command.append("--force")
            retranscription_result = subprocess.run(
                retranscription_command,
                capture_output=True,
                text=True,
                check=False,
            )
            if retranscription_result.returncode != 0:
                detail = retranscription_result.stderr.strip()[-4000:]
                raise core.TimelineEditError(
                    "Speech Tightener prompt-free retranscription failed"
                    + (f":\n{detail}" if detail else "")
                )
            qa_command = [
                sys.executable,
                str(Path(__file__).resolve().with_name("verify.py")),
                str(output_dir / "manifest.json"),
                "--output-dir",
                str(output_dir / "qa"),
                "--ffmpeg",
                str(ffmpeg),
                "--ffprobe",
                str(ffprobe),
                "--retranscript",
                str(retranscription_path),
            ]
            hybrid_verification = transcript.document.get("hybrid_verification")
            hybrid_verification = (
                hybrid_verification
                if isinstance(hybrid_verification, Mapping)
                else {}
            )
            source_retranscript_path = Path(
                str(hybrid_verification.get("whisper_path") or "")
            ).expanduser()
            if source_retranscript_path.is_file():
                qa_command.extend(
                    ["--source-retranscript", str(source_retranscript_path.resolve())]
                )
            if args.force:
                qa_command.append("--force")
            qa_result = subprocess.run(
                qa_command, capture_output=True, text=True, check=False
            )
            if qa_result.returncode != 0:
                detail = qa_result.stderr.strip()[-4000:]
                raise core.TimelineEditError(
                    "Speech Tightener local-file QA package failed"
                    + (f":\n{detail}" if detail else "")
                )
            _, qa_manifest = core.read_json(
                output_dir / "qa" / "manifest.json", "Speech Tightener QA manifest"
            )
            print(
                json.dumps(
                    {
                        "manifest": str(output_dir / "manifest.json"),
                        "qa_manifest": str(output_dir / "qa" / "manifest.json"),
                        "qa_status": qa_manifest.get("status"),
                    }
                )
            )
        return 0
    except (core.TimelineEditError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
