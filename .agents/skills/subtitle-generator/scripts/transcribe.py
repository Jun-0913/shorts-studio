#!/usr/bin/env python3
"""Create a verified word-timed transcript from a downloader handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
from typing import Any, Iterable, Optional


TRANSCRIPT_SCHEMA_VERSION = 1
DEFAULT_MODEL_ID = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MODEL_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
DEFAULT_FASTER_WHISPER_MODEL_ID = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
DEFAULT_FASTER_WHISPER_MODEL_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_COMPRESSION_THRESHOLD = 2.4


class SubtitleGeneratorError(RuntimeError):
    """An expected, user-actionable subtitle generator failure."""


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start.resolve().parents[4]


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SHARED_DIR = REPO_ROOT / ".agents" / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
import audio_extraction as AUDIO  # noqa: E402
import media_resources as RESOURCES  # noqa: E402
SAMPLE_RATE = AUDIO.SAMPLE_RATE
CHANNELS = AUDIO.CHANNELS

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "subtitles"
DEFAULT_MODEL_CACHE = Path(
    os.environ.get(
        "STUDIO_WHISPER_MODEL_CACHE",
        str(REPO_ROOT / ".tools" / "subtitle-generator" / "huggingface"),
    )
)


@dataclass(frozen=True)
class SourceMedia:
    media_path: Path
    manifest_path: Optional[Path]
    sha256: str
    sha256_verified: bool
    video_id: str
    title: Optional[str]
    duration_seconds: Optional[float]


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
    raise SubtitleGeneratorError(
        f"Required executable '{value}' was not found. {install_hint}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_identifier(value: str) -> str:
    # Preserve a leading hyphen because it may be part of the canonical
    # YouTube video ID.  Dots are the path-traversal concern here.
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".")
    return cleaned or "media"


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubtitleGeneratorError(f"{label} could not be read: {path}") from error
    if not isinstance(value, dict):
        raise SubtitleGeneratorError(f"{label} must contain a JSON object: {path}")
    return value


def resolve_manifest_media(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise SubtitleGeneratorError("Downloader manifest is missing output metadata.")

    declared = output.get("media_path")
    candidate = Path(declared).expanduser() if isinstance(declared, str) else None
    if candidate and not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if not candidate or not candidate.is_file():
        siblings = [
            path
            for path in manifest_path.parent.glob("source.*")
            if path.name not in {"source.info.json"}
        ]
        if len(siblings) != 1:
            raise SubtitleGeneratorError(
                "Downloader media file is missing beside the manifest."
            )
        candidate = siblings[0]

    resolved = candidate.resolve()
    if resolved.parent != manifest_path.parent.resolve():
        raise SubtitleGeneratorError(
            "Downloader manifest points outside its own artifact directory."
        )
    return resolved


def load_source(value: str) -> SourceMedia:
    input_path = Path(value).expanduser().resolve()
    if not input_path.is_file():
        raise SubtitleGeneratorError(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() != ".json":
        digest = sha256_file(input_path)
        return SourceMedia(
            media_path=input_path,
            manifest_path=None,
            sha256=digest,
            sha256_verified=True,
            video_id=safe_identifier(input_path.stem),
            title=None,
            duration_seconds=None,
        )

    manifest = read_json(input_path, "Downloader manifest")
    verification = manifest.get("verification")
    required_checks = ("has_video", "has_audio", "sha256_verified")
    if not isinstance(verification, dict) or not all(
        verification.get(key) is True for key in required_checks
    ):
        raise SubtitleGeneratorError(
            "Downloader manifest verification must pass before transcription."
        )

    media_path = resolve_manifest_media(input_path, manifest)
    output = manifest.get("output", {})
    expected_digest = output.get("sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_digest
    ):
        raise SubtitleGeneratorError("Downloader manifest has no valid source SHA-256.")
    actual_digest = sha256_file(media_path)
    if actual_digest.lower() != expected_digest.lower():
        raise SubtitleGeneratorError("Downloader media SHA-256 no longer matches manifest.")

    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    video_id = source.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        video_id = media_path.parent.name
    duration = output.get("duration_seconds", source.get("duration_seconds"))
    return SourceMedia(
        media_path=media_path,
        manifest_path=input_path,
        sha256=actual_digest,
        sha256_verified=True,
        video_id=safe_identifier(video_id),
        title=source.get("title") if isinstance(source.get("title"), str) else None,
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
    )


def validate_clip(
    start: Optional[float], end: Optional[float], source_duration: Optional[float]
) -> tuple[float, Optional[float]]:
    clip_start = 0.0 if start is None else start
    if clip_start < 0:
        raise SubtitleGeneratorError("--clip-start must be zero or greater.")
    if end is not None and end <= clip_start:
        raise SubtitleGeneratorError("--clip-end must be greater than --clip-start.")
    if source_duration is not None:
        if clip_start >= source_duration:
            raise SubtitleGeneratorError("--clip-start is outside the source duration.")
        if end is not None and end > source_duration + 0.5:
            raise SubtitleGeneratorError("--clip-end is outside the source duration.")
    return clip_start, end


def build_ffmpeg_command(
    *,
    ffmpeg: Path,
    media_path: Path,
    output_path: Path,
    clip_start: float,
    clip_end: Optional[float],
) -> list[str]:
    return AUDIO.pcm_command(media_path, ffmpeg=ffmpeg, start=clip_start,
                             end=clip_end) + [str(output_path)]


def extract_audio(
    *,
    ffmpeg: Path,
    media_path: Path,
    audio_path: Path,
    clip_start: float,
    clip_end: Optional[float],
) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{audio_path.stem}-", suffix=".wav", dir=audio_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    command = build_ffmpeg_command(
        ffmpeg=ffmpeg,
        media_path=media_path,
        output_path=temporary,
        clip_start=clip_start,
        clip_end=clip_end,
    )
    try:
        subprocess.run(command, check=True)
        temporary.replace(audio_path)
    except (OSError, subprocess.CalledProcessError) as error:
        temporary.unlink(missing_ok=True)
        raise SubtitleGeneratorError("ffmpeg could not extract the source audio.") from error


def probe_audio(ffprobe: Path, audio_path: Path) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        value = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise SubtitleGeneratorError("ffprobe could not verify extracted audio.") from error
    streams = value.get("streams") if isinstance(value, dict) else None
    if not isinstance(streams, list) or len(streams) != 1:
        raise SubtitleGeneratorError("Extracted audio must contain exactly one stream.")
    stream = streams[0]
    if (
        stream.get("codec_name") != "pcm_s16le"
        or stream.get("sample_rate") != str(SAMPLE_RATE)
        or stream.get("channels") != CHANNELS
    ):
        raise SubtitleGeneratorError("Extracted audio does not match the ASR contract.")
    duration = value.get("format", {}).get("duration")
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate")),
        "channels": stream.get("channels"),
        "duration_seconds": round(float(duration), 3),
    }


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def optional_metrics(segment: dict[str, Any]) -> dict[str, float]:
    names = ("avg_logprob", "compression_ratio", "no_speech_prob", "temperature")
    return {
        name: round(number, 6)
        for name in names
        if (number := as_float(segment.get(name))) is not None
    }


def normalize_transcript(
    result: dict[str, Any], *, clip_start: float
) -> tuple[dict[str, Any], dict[str, bool]]:
    raw_segments = result.get("segments")
    if not isinstance(raw_segments, Iterable) or isinstance(
        raw_segments, (str, bytes, dict)
    ):
        raise SubtitleGeneratorError("MLX Whisper returned no segment collection.")

    segments: list[dict[str, Any]] = []
    word_sequence = 0
    previous_start = clip_start
    previous_word_start = clip_start
    timestamps_monotonic = True
    timestamps_valid = True
    low_confidence_word_ids: list[str] = []
    segments_without_word_ids: list[str] = []
    high_compression_segment_ids: list[str] = []
    fallback_temperature_segment_ids: list[str] = []
    timestamp_issues: list[dict[str, Any]] = []
    segment_boundary_adjustments: list[dict[str, Any]] = []
    segment_sequence = 0

    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            raise SubtitleGeneratorError("MLX Whisper returned an invalid segment.")
        segment_text = str(raw_segment.get("text") or "").strip()
        raw_words = raw_segment.get("words")
        if not segment_text and (not isinstance(raw_words, list) or not raw_words):
            continue
        segment_sequence += 1
        segment_id = f"seg-{segment_sequence:06d}"
        start = as_float(raw_segment.get("start"))
        end = as_float(raw_segment.get("end"))
        if start is None or end is None:
            raise SubtitleGeneratorError("MLX Whisper segment is missing timestamps.")
        start += clip_start
        end += clip_start
        if start < 0 or end < start:
            timestamps_valid = False
            timestamp_issues.append(
                {
                    "type": "invalid_segment_range",
                    "segment_id": segment_id,
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )

        words: list[dict[str, Any]] = []
        word_starts: list[float] = []
        word_ends: list[float] = []
        if not isinstance(raw_words, list) or not raw_words:
            segments_without_word_ids.append(segment_id)
            raw_words = []
        for raw_word in raw_words:
            if not isinstance(raw_word, dict):
                timestamps_valid = False
                timestamp_issues.append(
                    {"type": "invalid_word", "segment_id": segment_id}
                )
                continue
            word_start = as_float(raw_word.get("start"))
            word_end = as_float(raw_word.get("end"))
            word_text = raw_word.get("word")
            if word_start is None or word_end is None or not isinstance(word_text, str):
                timestamps_valid = False
                timestamp_issues.append(
                    {"type": "missing_word_fields", "segment_id": segment_id}
                )
                continue
            word_start += clip_start
            word_end += clip_start
            if word_start < 0 or word_end < word_start:
                timestamps_valid = False
                timestamp_issues.append(
                    {
                        "type": "invalid_word_range",
                        "segment_id": segment_id,
                        "word_start": round(word_start, 3),
                        "word_end": round(word_end, 3),
                        "word": word_text,
                    }
                )
            if word_start < previous_word_start - 0.01:
                timestamps_monotonic = False
                timestamp_issues.append(
                    {
                        "type": "non_monotonic_word_start",
                        "segment_id": segment_id,
                        "previous_word_start": round(previous_word_start, 3),
                        "word_start": round(word_start, 3),
                        "word": word_text,
                    }
                )
            previous_word_start = word_start
            word_starts.append(word_start)
            word_ends.append(word_end)
            word_sequence += 1
            word = {
                "id": f"word-{word_sequence:06d}",
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "text": word_text,
            }
            probability = as_float(raw_word.get("probability"))
            if probability is not None:
                word["probability"] = round(probability, 6)
                if probability < LOW_CONFIDENCE_THRESHOLD:
                    low_confidence_word_ids.append(word["id"])
            words.append(word)

        original_start = start
        original_end = end
        if word_starts:
            start = min(start, min(word_starts))
            end = max(end, max(word_ends))
        if start != original_start or end != original_end:
            segment_boundary_adjustments.append(
                {
                    "segment_id": segment_id,
                    "original_start": round(original_start, 3),
                    "original_end": round(original_end, 3),
                    "adjusted_start": round(start, 3),
                    "adjusted_end": round(end, 3),
                }
            )
        if start < previous_start - 0.01:
            timestamps_monotonic = False
            timestamp_issues.append(
                {
                    "type": "non_monotonic_segment_start",
                    "segment_id": segment_id,
                    "previous_segment_start": round(previous_start, 3),
                    "segment_start": round(start, 3),
                }
            )
        previous_start = start

        normalized = {
            "id": segment_id,
            "start": round(start, 3),
            "end": round(end, 3),
            "text": segment_text,
            "words": words,
        }
        normalized.update(optional_metrics(raw_segment))
        compression_ratio = as_float(raw_segment.get("compression_ratio"))
        if (
            compression_ratio is not None
            and compression_ratio > HIGH_COMPRESSION_THRESHOLD
        ):
            high_compression_segment_ids.append(segment_id)
        temperature = as_float(raw_segment.get("temperature"))
        if temperature is not None and temperature > 0:
            fallback_temperature_segment_ids.append(segment_id)
        segments.append(normalized)

    transcript_text = str(result.get("text") or "").strip()
    verification = {
        "has_text": bool(transcript_text),
        "has_segments": bool(segments),
        "has_word_timestamps": word_sequence > 0,
        "timestamps_monotonic": timestamps_monotonic,
        "timestamps_valid": timestamps_valid,
    }
    return {
        "language": result.get("language"),
        "text": transcript_text,
        "segment_count": len(segments),
        "word_count": word_sequence,
        "quality": {
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "low_confidence_word_count": len(low_confidence_word_ids),
            "low_confidence_word_ids": low_confidence_word_ids,
            "segments_without_word_timestamps_count": len(
                segments_without_word_ids
            ),
            "segments_without_word_timestamps": segments_without_word_ids,
            "word_timestamp_segment_coverage": round(
                (len(segments) - len(segments_without_word_ids)) / len(segments), 6
            )
            if segments
            else 0.0,
            "high_compression_threshold": HIGH_COMPRESSION_THRESHOLD,
            "high_compression_segment_count": len(high_compression_segment_ids),
            "high_compression_segments": high_compression_segment_ids,
            "fallback_temperature_segment_count": len(
                fallback_temperature_segment_ids
            ),
            "fallback_temperature_segments": fallback_temperature_segment_ids,
            "timestamp_issue_count": len(timestamp_issues),
            "timestamp_issues": timestamp_issues,
            "segment_boundary_adjustment_count": len(
                segment_boundary_adjustments
            ),
            "segment_boundary_adjustments": segment_boundary_adjustments,
            "review_required": bool(
                low_confidence_word_ids
                or segments_without_word_ids
                or high_compression_segment_ids
                or fallback_temperature_segment_ids
                or timestamp_issues
            ),
        },
        "segments": segments,
    }, verification


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def download_model(model_id: str, revision: str, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SubtitleGeneratorError(
            "MLX runtime is missing. Run setup-mlx.sh before transcription."
        ) from error

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    except Exception as error:
        raise SubtitleGeneratorError("The pinned MLX Whisper model could not be loaded.") from error
    return Path(model_path).resolve()


def run_mlx_whisper(
    *, audio_path: Path, model_path: Path, language: str, prompt: Optional[str]
) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ImportError as error:
        raise SubtitleGeneratorError(
            "MLX runtime is missing. Run setup-mlx.sh before transcription."
        ) from error
    try:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=str(model_path),
            language=language,
            task="transcribe",
            word_timestamps=True,
            hallucination_silence_threshold=2.0,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            verbose=False,
        )
    except Exception as error:
        raise SubtitleGeneratorError("MLX Whisper transcription failed.") from error
    if not isinstance(result, dict):
        raise SubtitleGeneratorError("MLX Whisper returned an unexpected result.")
    return result


def resolve_asr_backend(value: str) -> str:
    if value == "auto":
        return (
            "mlx-whisper"
            if sys.platform == "darwin" and platform.machine() == "arm64"
            else "faster-whisper"
        )
    if value not in {"mlx-whisper", "faster-whisper"}:
        raise SubtitleGeneratorError(f"Unsupported ASR backend: {value}")
    if value == "mlx-whisper" and (
        sys.platform != "darwin" or platform.machine() != "arm64"
    ):
        raise SubtitleGeneratorError("The MLX backend requires an Apple Silicon Mac.")
    return value


def backend_model_id(backend: str, requested: str) -> str:
    if backend == "faster-whisper" and requested == DEFAULT_MODEL_ID:
        return DEFAULT_FASTER_WHISPER_MODEL_ID
    return requested


def backend_model_revision(backend: str, requested_model: str, requested_revision: str) -> str:
    if backend == "faster-whisper" and requested_model == DEFAULT_MODEL_ID:
        return DEFAULT_FASTER_WHISPER_MODEL_REVISION
    return requested_revision


def prepare_asr_model(
    backend: str, model_id: str, model_revision: str, cache_dir: Path
) -> Path | str:
    if backend == "mlx-whisper":
        return download_model(model_id, model_revision, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return model_id


def run_faster_whisper(
    *,
    audio_path: Path,
    model_id: str,
    model_revision: str,
    cache_dir: Path,
    language: str,
    prompt: Optional[str],
) -> dict[str, Any]:
    # Native model/VAD allocations can survive a function return. Keep them
    # outside the chunk coordinator, which must next admit an FFmpeg job.
    # The lease remains held until the worker has exited and been reaped.
    with RESOURCES.heavy_job("asr") as descriptor:
        with tempfile.TemporaryDirectory(prefix="studio-asr-") as directory:
            output = Path(directory) / "result.json"
            request = {
                "audio_path": str(audio_path.resolve()),
                "model_id": model_id,
                "model_revision": model_revision,
                "cache_dir": str(cache_dir.resolve()),
                "language": language,
                "prompt": prompt,
            }
            worker = Path(__file__).with_name("faster_whisper_worker.py")
            try:
                child = subprocess.Popen(
                    [sys.executable, str(worker), str(output)],
                    stdin=subprocess.PIPE,
                    text=True,
                    pass_fds=() if descriptor is None else (descriptor,),
                    preexec_fn=RESOURCES.encoder_parent_guard(),
                )
                try:
                    child.communicate(json.dumps(request))
                finally:
                    RESOURCES.stop_child(child)
                if child.returncode != 0:
                    raise SubtitleGeneratorError(
                        f"faster-whisper worker exited with {child.returncode}."
                    )
                result = json.loads(output.read_text(encoding="utf-8"))
                if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
                    raise SubtitleGeneratorError("faster-whisper worker returned an invalid result.")
                return result
            except (OSError, ValueError) as error:
                raise SubtitleGeneratorError("faster-whisper worker could not complete.") from error


def _run_faster_whisper(
    *, audio_path: Path, model_id: str, model_revision: str, cache_dir: Path,
    language: str, prompt: Optional[str],
) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise SubtitleGeneratorError(
            "faster-whisper is required on Linux. Install requirements-runtime.txt."
        ) from error
    try:
        model = WhisperModel(
            model_id,
            device=os.environ.get("STUDIO_WHISPER_DEVICE", "cpu"),
            compute_type=os.environ.get("STUDIO_WHISPER_COMPUTE_TYPE", "int8"),
            download_root=str(cache_dir),
            revision=model_revision,
        )
        segment_iterator, info = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            word_timestamps=True,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            vad_filter=True,
        )
        segments: list[dict[str, Any]] = []
        for segment in segment_iterator:
            segments.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text,
                    "avg_logprob": getattr(segment, "avg_logprob", None),
                    "compression_ratio": getattr(segment, "compression_ratio", None),
                    "no_speech_prob": getattr(segment, "no_speech_prob", None),
                    "temperature": getattr(segment, "temperature", 0.0),
                    "words": [
                        {
                            "start": float(word.start),
                            "end": float(word.end),
                            "word": word.word,
                            "probability": float(word.probability),
                        }
                        for word in (segment.words or [])
                    ],
                }
            )
    except Exception as error:
        raise SubtitleGeneratorError("faster-whisper transcription failed.") from error
    return {
        "language": getattr(info, "language", language),
        "text": " ".join(str(segment["text"]).strip() for segment in segments).strip(),
        "segments": segments,
    }


def run_whisper_backend(
    *,
    backend: str,
    audio_path: Path,
    model: Path | str,
    model_revision: str,
    model_cache: Path,
    language: str,
    prompt: Optional[str],
) -> dict[str, Any]:
    if backend == "mlx-whisper":
        return run_mlx_whisper(
            audio_path=audio_path,
            model_path=Path(model),
            language=language,
            prompt=prompt,
        )
    return run_faster_whisper(
        audio_path=audio_path,
        model_id=str(model),
        model_revision=model_revision,
        cache_dir=model_cache,
        language=language,
        prompt=prompt,
    )


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


def clip_directory_name(start: float, end: Optional[float]) -> Optional[str]:
    if start == 0 and end is None:
        return None
    end_label = "end" if end is None else f"{end:010.3f}"
    return f"{start:010.3f}-{end_label}"


def resolve_initial_prompt(
    source: SourceMedia, *, explicit_prompt: Optional[str], no_prompt: bool
) -> Optional[str]:
    if no_prompt:
        return None
    return explicit_prompt if explicit_prompt is not None else source.title


def default_output_dir(
    source: SourceMedia, clip_start: float, clip_end: Optional[float]
) -> Path:
    base = DEFAULT_OUTPUT_ROOT / source.video_id
    sample = clip_directory_name(clip_start, clip_end)
    return base / "transcription" if sample is None else base / "samples" / sample


def create_document(
    *,
    source: SourceMedia,
    audio_path: Path,
    audio_probe: dict[str, Any],
    transcript: dict[str, Any],
    transcript_verification: dict[str, bool],
    model_id: str,
    model_revision: str,
    language: str,
    prompt: Optional[str],
    clip_start: float,
    clip_end: Optional[float],
    ffmpeg: Path,
    ffprobe: Path,
    backend: str = "mlx-whisper",
) -> dict[str, Any]:
    verification = {
        "source_sha256_verified": source.sha256_verified,
        "audio_contract_verified": (
            audio_probe.get("codec") == "pcm_s16le"
            and audio_probe.get("sample_rate") == SAMPLE_RATE
            and audio_probe.get("channels") == CHANNELS
        ),
        **transcript_verification,
    }
    return {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "agent": "subtitle-generator",
        "stage": "transcription",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "video_id": source.video_id,
            "title": source.title,
            "media_path": str(source.media_path),
            "downloader_manifest_path": (
                str(source.manifest_path) if source.manifest_path else None
            ),
            "sha256": source.sha256,
            "duration_seconds": source.duration_seconds,
            "clip_start_seconds": clip_start,
            "clip_end_seconds": clip_end,
        },
        "audio": {
            "path": str(audio_path),
            "bytes": audio_path.stat().st_size,
            "sha256": sha256_file(audio_path),
            **audio_probe,
        },
        "asr": {
            "engine": backend,
            "mlx_whisper_version": (
                package_version("mlx-whisper") if backend == "mlx-whisper" else None
            ),
            "mlx_version": package_version("mlx") if backend == "mlx-whisper" else None,
            "faster_whisper_version": (
                package_version("faster-whisper")
                if backend == "faster-whisper"
                else None
            ),
            "model_id": model_id,
            "model_revision": model_revision,
            "language_hint": language,
            "initial_prompt": prompt,
            "word_timestamps": True,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        },
        "transcript": transcript,
        "verification": verification,
        "tools": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "ffmpeg": str(ffmpeg),
            "ffprobe": str(ffprobe),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a word-timed MLX Whisper transcript from one media file."
    )
    parser.add_argument("input", help="Downloader manifest.json or a local media file")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--clip-start", type=float)
    parser.add_argument("--clip-end", type=float)
    parser.add_argument("--language", default="en")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable both an explicit prompt and the manifest-title fallback.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
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
        backend = resolve_asr_backend(args.backend)
        source = load_source(args.input)
        clip_start, clip_end = validate_clip(
            args.clip_start, args.clip_end, source.duration_seconds
        )
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else default_output_dir(source, clip_start, clip_end)
        )
        transcript_path = output_dir / "transcript.source.json"
        audio_path = output_dir / "audio.wav"
        if transcript_path.exists() and not args.force:
            raise SubtitleGeneratorError(
                f"Transcript already exists. Use --force to replace it: {transcript_path}"
            )

        ffmpeg = resolve_executable(args.ffmpeg, "Install it with: brew install ffmpeg")
        ffprobe = resolve_executable(args.ffprobe, "Install it with: brew install ffmpeg")
        extract_audio(
            ffmpeg=ffmpeg,
            media_path=source.media_path,
            audio_path=audio_path,
            clip_start=clip_start,
            clip_end=clip_end,
        )
        audio_probe = probe_audio(ffprobe, audio_path)

        os.environ.setdefault("HF_HOME", str(args.model_cache.expanduser().resolve()))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        model_id = backend_model_id(backend, args.model)
        model_revision = backend_model_revision(
            backend, args.model, args.model_revision
        )
        model = prepare_asr_model(
            backend,
            model_id,
            model_revision,
            args.model_cache.expanduser().resolve(),
        )
        prompt = resolve_initial_prompt(
            source,
            explicit_prompt=args.prompt,
            no_prompt=args.no_prompt,
        )
        raw_result = run_whisper_backend(
            backend=backend,
            audio_path=audio_path,
            model=model,
            model_revision=model_revision,
            model_cache=args.model_cache.expanduser().resolve(),
            language=args.language,
            prompt=prompt,
        )
        transcript, transcript_verification = normalize_transcript(
            raw_result, clip_start=clip_start
        )
        document = create_document(
            source=source,
            audio_path=audio_path,
            audio_probe=audio_probe,
            transcript=transcript,
            transcript_verification=transcript_verification,
            model_id=model_id,
            model_revision=model_revision,
            language=args.language,
            prompt=prompt,
            clip_start=clip_start,
            clip_end=clip_end,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            backend=backend,
        )
        failed = [key for key, passed in document["verification"].items() if not passed]
        if failed:
            timestamp_issues = transcript["quality"]["timestamp_issues"]
            detail = (
                f"; first timestamp issue: {json.dumps(timestamp_issues[0], ensure_ascii=False)}"
                if timestamp_issues
                else ""
            )
            raise SubtitleGeneratorError(
                "Transcript verification failed: " + ", ".join(failed) + detail
            )
        write_json_atomic(transcript_path, document)
        print(transcript_path)
        return 0
    except RESOURCES.ResourceError as error:
        print(f"media-resources: {error}", file=sys.stderr)
        return 75
    except SubtitleGeneratorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
