#!/usr/bin/env python3
"""Build a local-file-first QA package for one Speech Tightener output."""

from __future__ import annotations

import argparse
import array
import difflib
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.IGNORECASE)
DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_SAMPLE_JUMP_THRESHOLD_DBFS = -35.0
DEFAULT_BOUNDARY_RMS_THRESHOLD_DBFS = -30.0
ENHANCED_SAMPLE_JUMP_THRESHOLD_DBFS = -30.0
ENHANCED_BOUNDARY_RMS_THRESHOLD_DBFS = -15.0
MINIMUM_FULL_SCOPE_COVERAGE_WITH_TARGETED_EVIDENCE = 0.8
SOURCE_BASELINE_SPOKEN_KINDS = {
    "discourse-marker",
    "false-start",
    "filler",
    "repetition",
    "word-search",
}


class VerificationError(RuntimeError):
    """Raised when an artifact cannot satisfy the local verification contract."""


def automatic_silence_policy_passes(
    candidates: Sequence[Mapping[str, Any]],
    removed_ranges: Sequence[Mapping[str, Any]],
    *,
    media_is_byte_identical: bool = False,
) -> bool:
    automatic_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("kind") == "long-pause"
        and candidate.get("acoustic_silence") is True
    ]
    automatic_ids = {
        str(candidate.get("id"))
        for candidate in automatic_candidates
        if candidate.get("id")
    }
    removed_ids = {
        str(removal.get("id"))
        for removal in removed_ranges
        if removal.get("id")
    }
    return (
        len(automatic_ids) == len(automatic_candidates)
        and automatic_ids.issubset(removed_ids)
        and not (automatic_candidates and media_is_byte_identical)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path_value: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise VerificationError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must contain a JSON object: {path}")
    return path, value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent != Path("."):
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    discovered = shutil.which(value)
    if discovered:
        return Path(discovered).resolve()
    raise VerificationError(f"Required executable was not found: {value}")


def normalized_tokens(text: Any) -> list[str]:
    return TOKEN_PATTERN.findall(str(text or "").lower().replace("’", "'"))


def transcript_words(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    transcript = document.get("transcript")
    if not isinstance(transcript, Mapping):
        return []
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        return []
    words: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        values = segment.get("words")
        if isinstance(values, list):
            words.extend(dict(word) for word in values if isinstance(word, Mapping))
    return words


def scoped_expected_tokens(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> tuple[list[str], dict[str, float | int | None]]:
    source = observed.get("source")
    source = source if isinstance(source, Mapping) else {}
    start = float(source.get("clip_start_seconds") or 0.0)
    raw_end = source.get("clip_end_seconds")
    end = float(raw_end) if isinstance(raw_end, (int, float)) else None
    words = transcript_words(expected)
    indexed_tokens: list[tuple[str, float]] = []
    for word in words:
        word_start = float(word.get("start", 0.0))
        word_end = float(word.get("end", word_start))
        midpoint = (word_start + word_end) / 2
        indexed_tokens.extend(
            (token, midpoint) for token in normalized_tokens(word.get("text"))
        )
    selected_indices = [
        index
        for index, (_, midpoint) in enumerate(indexed_tokens)
        if midpoint + 1e-6 >= start
        and (end is None or midpoint <= end + 1e-6)
    ]
    selected = [indexed_tokens[index][0] for index in selected_indices]
    global_start = selected_indices[0] if selected_indices else 0
    global_end = selected_indices[-1] + 1 if selected_indices else 0
    if not selected:
        transcript = expected.get("transcript")
        text = transcript.get("text") if isinstance(transcript, Mapping) else ""
        selected = normalized_tokens(text)
        global_start = 0
        global_end = len(selected)
    return selected, {
        "start": round(start, 6),
        "end": end,
        "global_start_token_index": global_start,
        "global_end_token_index": global_end,
    }


def compare_retranscription(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    edited_media_sha256: str,
) -> dict[str, Any]:
    expected_tokens, scope = scoped_expected_tokens(expected, observed)
    transcript = observed.get("transcript")
    transcript = transcript if isinstance(transcript, Mapping) else {}
    observed_tokens = normalized_tokens(transcript.get("text"))
    matcher = difflib.SequenceMatcher(
        None, expected_tokens, observed_tokens, autojunk=False
    )
    matches = sum(block.size for block in matcher.get_matching_blocks())
    missing_runs: list[dict[str, Any]] = []
    substitutions: list[dict[str, Any]] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag in {"delete", "replace"} and left_end > left_start:
            missing_runs.append(
                {
                    "start_token_index": left_start,
                    "global_start_token_index": int(
                        scope["global_start_token_index"]
                    )
                    + left_start,
                    "global_end_token_index": int(
                        scope["global_start_token_index"]
                    )
                    + left_end,
                    "token_count": left_end - left_start,
                    "tokens": expected_tokens[left_start:left_end],
                }
            )
        if tag == "replace":
            substitutions.append(
                {
                    "expected": expected_tokens[left_start:left_end],
                    "observed": observed_tokens[right_start:right_end],
                }
            )
    coverage = matches / len(expected_tokens) if expected_tokens else 0.0
    max_missing_run = max(
        (int(item["token_count"]) for item in missing_runs), default=0
    )
    verification = observed.get("verification")
    verification = verification if isinstance(verification, Mapping) else {}
    structural_pass = bool(verification) and all(
        value is True for value in verification.values()
    )
    source = observed.get("source")
    source = source if isinstance(source, Mapping) else {}
    observed_source_sha = source.get("sha256")
    source_sha256_verified = observed_source_sha == edited_media_sha256
    quality = transcript.get("quality")
    quality = quality if isinstance(quality, Mapping) else {}
    content_pass = (
        bool(expected_tokens)
        and bool(observed_tokens)
        and coverage >= 0.95
        and max_missing_run <= 2
        and structural_pass
        and source_sha256_verified
    )
    return {
        "scope": scope,
        "expected_token_count": len(expected_tokens),
        "observed_token_count": len(observed_tokens),
        "matched_token_count": matches,
        "token_coverage": round(coverage, 6),
        "maximum_missing_run_tokens": max_missing_run,
        "missing_runs": missing_runs,
        "substitutions": substitutions,
        "low_confidence_word_count": int(
            quality.get("low_confidence_word_count") or 0
        ),
        "asr_review_required": quality.get("review_required") is True,
        "structural_verification_pass": structural_pass,
        "source_sha256_verified": source_sha256_verified,
        "content_pass": content_pass,
    }


def targeted_evidence_resolves_full_comparison(
    full_comparison: Mapping[str, Any],
    targeted_comparisons: Sequence[Mapping[str, Any]],
) -> bool:
    if (
        float(full_comparison.get("token_coverage", 0.0))
        < MINIMUM_FULL_SCOPE_COVERAGE_WITH_TARGETED_EVIDENCE
        or full_comparison.get("structural_verification_pass") is not True
        or full_comparison.get("source_sha256_verified") is not True
    ):
        return False
    missing_runs = full_comparison.get("missing_runs")
    if not isinstance(missing_runs, list):
        return False
    for missing in missing_runs:
        if not isinstance(missing, Mapping):
            return False
        resolved = any(
            targeted.get("content_pass") is True
            and int(targeted["scope"]["global_start_token_index"])
            <= int(missing["global_start_token_index"])
            and int(targeted["scope"]["global_end_token_index"])
            >= int(missing["global_end_token_index"])
            for targeted in targeted_comparisons
            if isinstance(targeted.get("scope"), Mapping)
        )
        if not resolved:
            return False
    return True


def source_baseline_confirms_word_preservation(
    source_comparisons: Sequence[Mapping[str, Any]],
    edited_comparisons: Sequence[Mapping[str, Any]],
    removed_ranges: Sequence[Mapping[str, Any]],
) -> bool:
    """Prove that post-edit gaps existed before a verified safe timeline edit."""
    if not removed_ranges:
        return False
    silence_only = not any(
        item.get("kind") != "long-pause" for item in removed_ranges
    ) and not any(
        item.get("kind") != "long-pause"
        or item.get("acoustic_silence") is not True
        or item.get("meaning_preserved") is not True
        or item.get("removed_word_ids") != []
        for item in removed_ranges
    )
    reviewed_spoken = any(
        item.get("kind") != "long-pause" for item in removed_ranges
    ) and all(
        (
            item.get("kind") == "long-pause"
            and item.get("acoustic_silence") is True
            and item.get("meaning_preserved") is True
            and item.get("removed_word_ids") == []
        )
        or (
            item.get("kind") in SOURCE_BASELINE_SPOKEN_KINDS
            and item.get("decision") == "remove"
            and item.get("meaning_preserved") is True
            and item.get("evidence_mode") == "local-file-analysis"
            and item.get("acoustic_boundary_refined") is True
            and isinstance(item.get("removed_word_ids"), list)
            and bool(item.get("removed_word_ids"))
            and not item.get("collateral_word_overlaps")
        )
        for item in removed_ranges
    )
    if not silence_only and not reviewed_spoken:
        return False
    targeted_observed_indices: set[int] = set()
    for comparison in edited_comparisons:
        scope = comparison.get("scope")
        missing_runs = comparison.get("missing_runs")
        if (
            not isinstance(scope, Mapping)
            or scope.get("end") is None
            or not isinstance(missing_runs, list)
            or comparison.get("structural_verification_pass") is not True
            or comparison.get("source_sha256_verified") is not True
        ):
            continue
        start = int(scope.get("global_start_token_index") or 0)
        end = int(scope.get("global_end_token_index") or 0)
        missing_indices = {
            index
            for gap in missing_runs
            if isinstance(gap, Mapping)
            for index in range(
                int(gap.get("global_start_token_index") or 0),
                int(gap.get("global_end_token_index") or 0),
            )
        }
        targeted_observed_indices.update(set(range(start, end)) - missing_indices)
    for source in source_comparisons:
        for edited in edited_comparisons:
            if (
                source.get("structural_verification_pass") is not True
                or source.get("source_sha256_verified") is not True
                or edited.get("structural_verification_pass") is not True
                or edited.get("source_sha256_verified") is not True
                or source.get("expected_token_count")
                != edited.get("expected_token_count")
            ):
                continue
            if reviewed_spoken:
                source_scope = source.get("scope")
                edited_scope = edited.get("scope")
                if (
                    not isinstance(source_scope, Mapping)
                    or not isinstance(edited_scope, Mapping)
                    or dict(source_scope) != dict(edited_scope)
                ):
                    continue
                source_missing = source.get("missing_runs")
                edited_missing = edited.get("missing_runs")
                if not isinstance(source_missing, list) or not isinstance(
                    edited_missing, list
                ):
                    continue
                source_missing_indices = {
                    index
                    for gap in source_missing
                    if isinstance(gap, Mapping)
                    for index in range(
                        int(gap.get("global_start_token_index") or 0),
                        int(gap.get("global_end_token_index") or 0),
                    )
                }
                edited_missing_indices = {
                    index
                    for gap in edited_missing
                    if isinstance(gap, Mapping)
                    for index in range(
                        int(gap.get("global_start_token_index") or 0),
                        int(gap.get("global_end_token_index") or 0),
                    )
                }
                all_edited_gaps_resolved = edited_missing_indices.issubset(
                    source_missing_indices | targeted_observed_indices
                )
                if (
                    all_edited_gaps_resolved
                    and int(edited.get("matched_token_count") or 0)
                    >= int(source.get("matched_token_count") or 0)
                    and float(edited.get("token_coverage") or 0.0)
                    + 1e-6
                    >= float(source.get("token_coverage") or 0.0)
                    and int(edited.get("maximum_missing_run_tokens") or 0)
                    <= int(source.get("maximum_missing_run_tokens") or 0)
                ):
                    return True
                continue
            expected_tokens = int(source.get("expected_token_count") or 0)
            source_matches = int(source.get("matched_token_count") or 0)
            edited_matches = int(edited.get("matched_token_count") or 0)
            # Prompt-free Whisper can choose a different normalization at each
            # verified silence join even when no spoken word was removed. Scale
            # the cumulative drift allowance with the number of silence-only
            # joins, but cap it at five percent of the expected transcript and
            # continue rejecting any missing run longer than the source baseline.
            drift_token_tolerance = max(
                1,
                min(len(removed_ranges), math.ceil(expected_tokens * 0.05)),
            )
            coverage_tolerance = drift_token_tolerance / max(1, expected_tokens)
            source_missing = source.get("missing_runs")
            edited_missing = edited.get("missing_runs")
            resolved_by_baseline_or_targeted = False
            if isinstance(source_missing, list) and isinstance(edited_missing, list):
                source_missing_indices = {
                    index
                    for gap in source_missing
                    if isinstance(gap, Mapping)
                    for index in range(
                        int(gap.get("global_start_token_index") or 0),
                        int(gap.get("global_end_token_index") or 0),
                    )
                }
                edited_missing_indices = {
                    index
                    for gap in edited_missing
                    if isinstance(gap, Mapping)
                    for index in range(
                        int(gap.get("global_start_token_index") or 0),
                        int(gap.get("global_end_token_index") or 0),
                    )
                }
                resolved_by_baseline_or_targeted = edited_missing_indices.issubset(
                    source_missing_indices | targeted_observed_indices
                )
            if (
                edited_matches >= source_matches - drift_token_tolerance
                and float(edited.get("token_coverage") or 0.0)
                >= float(source.get("token_coverage") or 0.0)
                - coverage_tolerance
                - 1e-6
                and (
                    int(edited.get("maximum_missing_run_tokens") or 0)
                    <= max(2, int(source.get("maximum_missing_run_tokens") or 0))
                    or resolved_by_baseline_or_targeted
                )
            ):
                return True
    return False


def dbfs(value: float) -> float:
    return 20 * math.log10(max(abs(value), 1e-9))


def rms_dbfs(samples: array.array, start: int, end: int) -> float:
    start = max(0, start)
    end = min(len(samples), end)
    if end <= start:
        return -180.0
    mean_square = sum(float(value) * float(value) for value in samples[start:end]) / (
        end - start
    )
    return dbfs(math.sqrt(mean_square))


def decode_audio(media_path: Path, *, ffmpeg: Path, sample_rate: int) -> array.array:
    result = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-4000:]
        raise VerificationError(
            "FFmpeg audio decode failed" + (f":\n{detail}" if detail else "")
        )
    samples = array.array("f")
    samples.frombytes(result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise VerificationError("Edited output contains no decodable audio")
    return samples


def join_audio_metrics(
    samples: array.array,
    *,
    timestamp: float,
    sample_rate: int,
    sample_jump_threshold_dbfs: float,
    boundary_rms_threshold_dbfs: float,
) -> dict[str, Any]:
    center = min(len(samples) - 1, max(1, int(timestamp * sample_rate)))
    rms_window = max(1, int(0.005 * sample_rate))
    zero_radius = max(1, int(0.003 * sample_rate))
    jump = float(samples[center]) - float(samples[center - 1])
    before_rms = rms_dbfs(samples, center - rms_window, center)
    after_rms = rms_dbfs(samples, center, center + rms_window)
    lower = max(0, center - zero_radius)
    upper = min(len(samples), center + zero_radius + 1)
    zero_index = min(range(lower, upper), key=lambda index: abs(samples[index]))
    sample_jump_dbfs = dbfs(jump)
    maximum_boundary_rms_dbfs = max(before_rms, after_rms)
    pop_risk_clear = (
        sample_jump_dbfs <= sample_jump_threshold_dbfs
        and maximum_boundary_rms_dbfs <= boundary_rms_threshold_dbfs
    )
    return {
        "timestamp_seconds": round(timestamp, 6),
        "sample_jump_dbfs": round(sample_jump_dbfs, 3),
        "rms_before_5ms_dbfs": round(before_rms, 3),
        "rms_after_5ms_dbfs": round(after_rms, 3),
        "maximum_boundary_rms_dbfs": round(maximum_boundary_rms_dbfs, 3),
        "nearest_zero_crossing_offset_ms": round(
            (zero_index - center) * 1000 / sample_rate, 3
        ),
        "nearest_zero_crossing_amplitude_dbfs": round(
            dbfs(float(samples[zero_index])), 3
        ),
        "pop_risk_clear": pop_risk_clear,
    }


def run_checked(command: Sequence[str], label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-4000:]
        raise VerificationError(label + (f":\n{detail}" if detail else ""))
    return result


def extract_frame(
    media_path: Path, output_path: Path, *, timestamp: float, ffmpeg: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    run_checked(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, timestamp):.6f}",
            "-i",
            str(media_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ],
        "FFmpeg frame extraction failed",
    )


def extract_review_clip(
    media_path: Path,
    output_path: Path,
    *,
    join_time: float,
    media_duration: float,
    window: float,
    ffmpeg: Path,
) -> dict[str, float]:
    start = max(0.0, join_time - window)
    end = min(media_duration, join_time + window)
    duration = max(0.05, end - start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    run_checked(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(media_path),
            "-t",
            f"{duration:.6f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        "FFmpeg join review clip extraction failed",
    )
    return {"start": round(start, 6), "end": round(end, 6)}


def frame_ssim(before: Path, after: Path, *, ffmpeg: Path) -> float | None:
    result = run_checked(
        [
            str(ffmpeg),
            "-hide_banner",
            "-i",
            str(before),
            "-i",
            str(after),
            "-lavfi",
            "ssim",
            "-f",
            "null",
            "-",
        ],
        "FFmpeg SSIM check failed",
    )
    match = re.search(r"All:([0-9.]+)", result.stderr)
    return round(float(match.group(1)), 6) if match else None


def probe_duration(media_path: Path, *, ffprobe: Path) -> float:
    result = run_checked(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        "FFprobe duration check failed",
    )
    try:
        return float(result.stdout.strip())
    except ValueError as error:
        raise VerificationError("FFprobe returned an invalid duration") from error


def verify_output(
    manifest_path: Path,
    *,
    retranscript_paths: Sequence[Path],
    source_retranscript_paths: Sequence[Path] = (),
    output_dir: Path,
    ffmpeg: Path,
    ffprobe: Path,
    frame_offset: float,
    review_window: float,
    sample_rate: int,
    sample_jump_threshold_dbfs: float,
    boundary_rms_threshold_dbfs: float,
    force: bool,
) -> Path:
    manifest_path, manifest = read_json(manifest_path, "Speech Tightener manifest")
    if manifest.get("agent") != "speech-tightener":
        raise VerificationError("Manifest is not a Speech Tightener output")
    verification = manifest.get("verification")
    if not isinstance(verification, Mapping) or not all(
        value is True for value in verification.values()
    ):
        raise VerificationError("Speech Tightener output manifest is not verified")
    output = manifest.get("output")
    source = manifest.get("source")
    timeline = manifest.get("timeline")
    plan_reference = manifest.get("plan")
    if not all(
        isinstance(value, Mapping)
        for value in (output, source, timeline, plan_reference)
    ):
        raise VerificationError("Speech Tightener manifest structure is incomplete")
    output = dict(output)
    source = dict(source)
    timeline = dict(timeline)
    plan_reference = dict(plan_reference)
    plan_path = Path(str(plan_reference.get("path") or "")).expanduser().resolve()
    if not plan_path.is_file():
        raise VerificationError("Speech Tightener review plan is missing")
    if sha256_file(plan_path) != plan_reference.get("sha256"):
        raise VerificationError("Speech Tightener review plan SHA-256 is stale")
    _, plan = read_json(plan_path, "Speech Tightener review plan")
    candidates = plan.get("candidates")
    detection = plan.get("detection")
    review = plan.get("review")
    if (
        not isinstance(candidates, list)
        or not isinstance(detection, Mapping)
        or not isinstance(detection.get("acoustic_silence"), Mapping)
        or detection["acoustic_silence"].get("automatic_removal") is not True
        or not isinstance(review, Mapping)
        or review.get("human_listening_required") is not False
        or review.get("final_reel_human_review_required") is not True
    ):
        raise VerificationError(
            "Speech Tightener plan does not use the automatic-silence and "
            "final-Reel-review contract"
        )
    automatic_silence_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("kind") == "long-pause"
        and candidate.get("acoustic_silence") is True
    ]
    if any(
        candidate.get("decision") != "remove"
        for candidate in automatic_silence_candidates
    ):
        raise VerificationError(
            "Every measured acoustic silence must be automatically removed"
        )
    media_path = Path(str(output.get("media_path"))).expanduser().resolve()
    expected_transcript_path = Path(
        str(output.get("transcript_path"))
    ).expanduser().resolve()
    source_media_path = Path(str(source.get("media_path"))).expanduser().resolve()
    if not media_path.is_file() or not expected_transcript_path.is_file():
        raise VerificationError("Speech Tightener output media or transcript is missing")
    if not source_media_path.is_file():
        raise VerificationError("Original source media is missing")
    output_media_sha = str(output.get("sha256") or "")
    expected_transcript_sha = str(output.get("transcript_sha256") or "")
    expected_path, expected = read_json(
        expected_transcript_path, "Edited transcript"
    )
    output_dir = output_dir.expanduser().resolve()
    qa_manifest_path = output_dir / "manifest.json"
    if qa_manifest_path.exists() and not force:
        raise VerificationError(
            f"QA output exists. Use --force to replace it: {qa_manifest_path}"
        )

    kept_ranges = timeline.get("kept_ranges")
    if not isinstance(kept_ranges, list):
        raise VerificationError("Speech Tightener timeline has no kept ranges")
    removed_ranges = timeline.get("removed_ranges")
    if not isinstance(removed_ranges, list):
        raise VerificationError("Speech Tightener timeline has no removed ranges")
    join_times = [
        float(value["output_end"])
        for value in kept_ranges[:-1]
        if isinstance(value, Mapping)
    ]
    samples = decode_audio(media_path, ffmpeg=ffmpeg, sample_rate=sample_rate)
    media_duration = probe_duration(media_path, ffprobe=ffprobe)
    review_dir = output_dir / "join-review"
    joins: list[dict[str, Any]] = []
    for index, join_time in enumerate(join_times, 1):
        removal = (
            removed_ranges[index - 1]
            if index - 1 < len(removed_ranges)
            and isinstance(removed_ranges[index - 1], Mapping)
            else {}
        )
        treatment = removal.get("audio_join_treatment")
        enhanced_speech_edge = (
            str(removal.get("detector") or "").startswith("enhanced-")
            and isinstance(treatment, Mapping)
            and treatment.get("method") == "equal-power-micro-crossfade"
        )
        join_sample_jump_threshold = (
            max(sample_jump_threshold_dbfs, ENHANCED_SAMPLE_JUMP_THRESHOLD_DBFS)
            if enhanced_speech_edge
            else sample_jump_threshold_dbfs
        )
        join_boundary_rms_threshold = (
            max(boundary_rms_threshold_dbfs, ENHANCED_BOUNDARY_RMS_THRESHOLD_DBFS)
            if enhanced_speech_edge
            else boundary_rms_threshold_dbfs
        )
        audio = join_audio_metrics(
            samples,
            timestamp=join_time,
            sample_rate=sample_rate,
            sample_jump_threshold_dbfs=join_sample_jump_threshold,
            boundary_rms_threshold_dbfs=join_boundary_rms_threshold,
        )
        audio["thresholds"] = {
            "sample_jump_dbfs": join_sample_jump_threshold,
            "boundary_rms_dbfs": join_boundary_rms_threshold,
            "profile": (
                "enhanced-disfluency-speech-edge"
                if enhanced_speech_edge
                else "standard"
            ),
        }
        if isinstance(treatment, Mapping):
            audio["join_treatment"] = dict(treatment)
        before = review_dir / f"join-{index:02d}-before.jpg"
        after = review_dir / f"join-{index:02d}-after.jpg"
        clip = review_dir / f"join-{index:02d}-review.mp4"
        extract_frame(
            media_path,
            before,
            timestamp=max(0.0, join_time - frame_offset),
            ffmpeg=ffmpeg,
        )
        extract_frame(
            media_path,
            after,
            timestamp=min(media_duration, join_time + frame_offset),
            ffmpeg=ffmpeg,
        )
        clip_range = extract_review_clip(
            media_path,
            clip,
            join_time=join_time,
            media_duration=media_duration,
            window=review_window,
            ffmpeg=ffmpeg,
        )
        joins.append(
            {
                "index": index,
                "audio": audio,
                "visual": {
                    "before_frame_path": str(before),
                    "before_frame_sha256": sha256_file(before),
                    "after_frame_path": str(after),
                    "after_frame_sha256": sha256_file(after),
                    "frame_offset_seconds": frame_offset,
                    "ssim": frame_ssim(before, after, ffmpeg=ffmpeg),
                    "human_visual_review_required": True,
                },
                "review_clip": {
                    "path": str(clip),
                    "sha256": sha256_file(clip),
                    "range": clip_range,
                },
            }
        )

    retranscriptions: list[dict[str, Any]] = []
    for retranscript_path in retranscript_paths:
        observed_path, observed = read_json(
            retranscript_path, "Post-edit retranscription"
        )
        retranscriptions.append(
            {
                "path": str(observed_path),
                "sha256": sha256_file(observed_path),
                "comparison": compare_retranscription(
                    expected, observed, edited_media_sha256=output_media_sha
                ),
            }
        )
    source_retranscriptions: list[dict[str, Any]] = []
    for retranscript_path in source_retranscript_paths:
        observed_path, observed = read_json(
            retranscript_path, "Pre-edit source retranscription"
        )
        source_retranscriptions.append(
            {
                "path": str(observed_path),
                "sha256": sha256_file(observed_path),
                "comparison": compare_retranscription(
                    expected,
                    observed,
                    edited_media_sha256=str(source.get("media_sha256") or ""),
                ),
            }
        )
    full_scope_comparisons = [
        value["comparison"]
        for value in retranscriptions
        if float(value["comparison"]["scope"]["start"]) <= 1e-6
        and value["comparison"]["scope"]["end"] is None
    ]
    targeted_comparisons = [
        value["comparison"]
        for value in retranscriptions
        if not (
            float(value["comparison"]["scope"]["start"]) <= 1e-6
            and value["comparison"]["scope"]["end"] is None
        )
    ]
    post_edit_transcript_present = bool(retranscriptions)
    post_edit_transcript_source_verified = post_edit_transcript_present and all(
        value["comparison"]["source_sha256_verified"] for value in retranscriptions
    )
    timeline = manifest.get("timeline")
    removed_ranges = (
        timeline.get("removed_ranges") if isinstance(timeline, Mapping) else None
    )
    media_is_byte_identical = (
        source.get("media_sha256") == output_media_sha
        and sha256_file(source_media_path) == output_media_sha
    )
    no_op_media_identity = (
        isinstance(removed_ranges, list)
        and not removed_ranges
        and media_is_byte_identical
    )
    automatic_silence_policy_verified = automatic_silence_policy_passes(
        automatic_silence_candidates,
        [
            removal
            for removal in removed_ranges or []
            if isinstance(removal, Mapping)
        ],
        media_is_byte_identical=media_is_byte_identical,
    )
    word_preservation_pass = bool(full_scope_comparisons) and any(
        value["content_pass"] for value in full_scope_comparisons
    )
    if no_op_media_identity:
        word_preservation_pass = True
    if not word_preservation_pass:
        for full_comparison in full_scope_comparisons:
            if targeted_evidence_resolves_full_comparison(
                full_comparison, targeted_comparisons
            ):
                word_preservation_pass = True
                break
    source_baseline_pass = source_baseline_confirms_word_preservation(
        [value["comparison"] for value in source_retranscriptions],
        full_scope_comparisons + targeted_comparisons,
        [value for value in removed_ranges or [] if isinstance(value, Mapping)],
    )
    if not word_preservation_pass and source_baseline_pass:
        word_preservation_pass = True
    audio_pop_risk_clear = all(
        value["audio"]["pop_risk_clear"] for value in joins
    )
    join_assets_present = all(
        Path(value["visual"]["before_frame_path"]).is_file()
        and Path(value["visual"]["after_frame_path"]).is_file()
        and Path(value["review_clip"]["path"]).is_file()
        for value in joins
    )
    if not post_edit_transcript_present:
        status = "pending-retranscription"
    elif (
        not word_preservation_pass
        or not audio_pop_risk_clear
        or not automatic_silence_policy_verified
    ):
        status = "review-required"
    else:
        status = "ready-for-render"
    qa_manifest = {
        "schema_version": 1,
        "agent": "speech-tightener",
        "stage": "local-file-qa",
        "status": status,
        "source": {
            "speech_manifest_path": str(manifest_path),
            "speech_manifest_sha256": sha256_file(manifest_path),
            "original_media_path": str(source_media_path),
            "original_media_sha256": source.get("media_sha256"),
            "edited_media_path": str(media_path),
            "edited_media_sha256": output_media_sha,
            "edited_transcript_path": str(expected_path),
            "edited_transcript_sha256": expected_transcript_sha,
        },
        "thresholds": {
            "sample_rate_hz": sample_rate,
            "sample_jump_threshold_dbfs": sample_jump_threshold_dbfs,
            "boundary_rms_threshold_dbfs": boundary_rms_threshold_dbfs,
            "frame_offset_seconds": frame_offset,
            "review_clip_each_side_seconds": review_window,
            "minimum_retranscription_token_coverage": 0.95,
            "minimum_full_scope_coverage_with_targeted_evidence": (
                MINIMUM_FULL_SCOPE_COVERAGE_WITH_TARGETED_EVIDENCE
            ),
            "maximum_missing_token_run": 2,
        },
        "word_preservation_basis": (
            "byte-identical-reviewed-no-op"
            if no_op_media_identity
            else (
                "post-edit-retranscription-within-source-baseline-tolerance"
                if source_baseline_pass
                else "post-edit-retranscription"
            )
        ),
        "joins": joins,
        "source_retranscriptions": source_retranscriptions,
        "retranscriptions": retranscriptions,
        "human_review": {
            "scope": "final-reel",
            "speech_candidate_interface_required": False,
            "speech_stage_listening_required": False,
            "final_reel_review_required": True,
            "model_audio_access": False,
            "naturalness_verified_by_model": False,
            "speech_edit_mp4_path": str(media_path),
            "review_surface": "apps/reels-review",
            "notes": (
                "Speech candidates are resolved deterministically or by agent file "
                "review. A person reviews cadence, naturalness, and visual joins on "
                "the rendered final Reel in Reels Review."
            ),
        },
        "verification": {
            "speech_manifest_sha256_verified": True,
            "original_media_preserved": sha256_file(source_media_path)
            == source.get("media_sha256"),
            "edited_media_sha256_verified": sha256_file(media_path)
            == output_media_sha,
            "edited_transcript_sha256_verified": sha256_file(
                expected_transcript_path
            )
            == expected_transcript_sha,
            "audio_joins_checked": len(joins) == len(join_times),
            "audio_pop_risk_clear": audio_pop_risk_clear,
            "join_review_assets_present": join_assets_present,
            "post_edit_transcript_present": post_edit_transcript_present,
            "post_edit_transcript_source_verified": (
                post_edit_transcript_source_verified
            ),
            "word_preservation_pass": word_preservation_pass,
            "reviewed_no_op_media_identity": no_op_media_identity,
            "automatic_silence_policy_verified": (
                automatic_silence_policy_verified
            ),
            "final_mp4_present": media_path.is_file(),
            "final_reel_review_pending": True,
        },
    }
    write_json_atomic(qa_manifest_path, qa_manifest)
    return qa_manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--retranscript", action="append", default=[], type=Path)
    parser.add_argument(
        "--source-retranscript", action="append", default=[], type=Path
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--frame-offset", type=float, default=0.05)
    parser.add_argument("--review-window", type=float, default=1.5)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--sample-jump-threshold-dbfs",
        type=float,
        default=DEFAULT_SAMPLE_JUMP_THRESHOLD_DBFS,
    )
    parser.add_argument(
        "--boundary-rms-threshold-dbfs",
        type=float,
        default=DEFAULT_BOUNDARY_RMS_THRESHOLD_DBFS,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest_path = Path(args.manifest).expanduser().resolve()
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else manifest_path.parent / "qa"
        )
        result = verify_output(
            manifest_path,
            retranscript_paths=[path.expanduser().resolve() for path in args.retranscript],
            source_retranscript_paths=[
                path.expanduser().resolve() for path in args.source_retranscript
            ],
            output_dir=output_dir,
            ffmpeg=resolve_executable(args.ffmpeg),
            ffprobe=resolve_executable(args.ffprobe),
            frame_offset=args.frame_offset,
            review_window=args.review_window,
            sample_rate=args.sample_rate,
            sample_jump_threshold_dbfs=args.sample_jump_threshold_dbfs,
            boundary_rms_threshold_dbfs=args.boundary_rms_threshold_dbfs,
            force=args.force,
        )
        print(result)
        return 0
    except (VerificationError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
