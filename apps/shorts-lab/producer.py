#!/usr/bin/env python3
"""Produce one source's 1–4 short-form videos."""
from __future__ import annotations

import argparse
import copy
import re
import fcntl
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import editor
import media
from store import Store
from worker import yt_dlp

APP = Path(__file__).resolve().parent
REPO = APP.parents[1]
ASR_PYTHON = Path(sys.executable)
SKILLS = REPO / ".agents/skills"


def script(path, *args, log, python=None, allowed=(0,)):
    return media.run([python or sys.executable, SKILLS / path, *args], log, allowed)


def source(store, job, attempt):
    if job["source_url"].startswith("local:"):
        manifest = store.root / "sources" / job["video_id"] / "manifest.json"
        artifact = media.timeline.load_media_manifest(manifest)
        metadata = media.read(manifest.parent / "source.info.json")
        return artifact, metadata
    cache = store.root / "sources"
    cache.mkdir(exist_ok=True)
    with (cache / f"{job['video_id']}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paths = [cache / job["video_id"] / "manifest.json", REPO / "artifacts/downloads" / job["video_id"] / "manifest.json"]
        manifest = next((p for p in paths if p.is_file()), paths[0])
        if not manifest.is_file():
            if shutil.disk_usage(cache).free < store.minimum_free_bytes:
                raise RuntimeError("Insufficient storage")
            probe = subprocess.run([yt_dlp(), "--no-playlist", "--skip-download", "--dump-single-json", "--extractor-args", "youtube:lang=ko", "--socket-timeout", "25", job["source_url"]],
                                   text=True, capture_output=True, timeout=180, check=True)
            metadata = json.loads(probe.stdout)
            if metadata.get("id") != job["video_id"] or metadata.get("is_live") or not 10 <= float(metadata.get("duration") or 0) <= 7200:
                raise ValueError("Only completed videos between 10 seconds and two hours are supported")
            script("youtube-downloader/scripts/download.py", job["source_url"], "--output-dir", cache, "--yt-dlp-bin", yt_dlp(), log=attempt / "download.log")
        artifact = media.timeline.load_media_manifest(manifest)
        if artifact.video_id != job["video_id"] or not 10 <= artifact.duration_seconds <= 7200:
            raise ValueError("Source identity/duration is invalid")
        metadata = media.read(manifest.parent / "source.info.json")
        return artifact, metadata


def navigation(source, metadata, folder):
    folder.mkdir(exist_ok=True)
    cached = REPO / "artifacts/subtitles" / source.video_id / "transcription/transcript.source.json"
    if cached.is_file():
        data = media.read(cached)
        try:
            media.timeline.load_transcript(cached, source)
            if not all(data.get("verification", {}).values()):
                raise ValueError("Unverified cached transcript")
        except (ValueError, media.timeline.TimelineEditError):
            pass
        else:
            media.write(folder / "reused-navigation.json", {"path": str(cached), "sha256": media.timeline.sha256_file(cached)})
            return data, cached, data["transcript"]["language"]
    exit_code = 3
    if not metadata.get("local_file"):
        exit_code = script("youtube-caption-ingestor/scripts/ingest.py", source.manifest_path, "--output-dir", folder,
                           "--yt-dlp", yt_dlp(), log=folder / "ingest.log", allowed=(0, 1, 3))
    transcript = folder / "transcript.source.json"
    # The canonical ingestor may put files inside transcription/ for a root.
    if not transcript.is_file():
        transcript = folder / "transcription/transcript.source.json"
    language = str(metadata.get("language") or "").split("-")[0]
    if not language:
        language = "auto"
    if exit_code != 0:
        fallback = folder / "whisper"
        script("subtitle-generator/scripts/transcribe_chunked.py", source.manifest_path, "--output-dir", fallback,
               "--language", language, "--no-prompt", "--chunk-seconds", "60", "--overlap-seconds", "0", log=folder / "whisper.log", python=ASR_PYTHON)
        transcript = fallback / "transcript.whisper-chunked.json"
    data = media.read(transcript)
    if not editor.transcript_segments(data) or (data.get("verification") and not all(data["verification"].values())):
        raise ValueError("Navigation transcript is not verified")
    language = data.get("transcript", data).get("language") or language
    return data, transcript, language


def speech_stage(extracted_manifest, language, folder):
    transcribe = folder / "whisper"
    transcript = transcribe / "transcript.source.json"
    if not transcript.is_file():
        script("subtitle-generator/scripts/transcribe.py", extracted_manifest, "--output-dir", transcribe, "--language", language,
               "--no-prompt", python=ASR_PYTHON, log=folder / "transcribe.log")
    speech = folder / "speech"
    script("speech-tightener/scripts/edit.py", "prepare", extracted_manifest, transcript, "--output-dir", speech, log=folder / "speech-prepare.log")
    plan = media.read(speech / "plan.json")
    # Never authorize semantic deletions in this unattended test tool. The
    # canonical detector/reviewer owns measured acoustic silence shortening.
    keep = [str(c["id"]) for c in plan["candidates"] if not (c.get("kind") == "long-pause" and c.get("acoustic_silence") is True)]
    arguments = []
    for candidate in keep:
        arguments += ["--keep", candidate]
    script("speech-tightener/scripts/review_plan.py", speech / "plan.json", "--reviewed-by", "shorts-lab-conservative-file-policy", *arguments,
           log=folder / "speech-review.log")
    script("speech-tightener/scripts/edit.py", "apply", extracted_manifest, transcript, speech / "plan.json", "--output-dir", speech,
           "--retranscription-language", language, log=folder / "speech-apply.log")
    qa = media.read(speech / "qa/manifest.json")
    if qa.get("status") != "ready-for-render":
        return preserve_unedited_speech(extracted_manifest, transcript, folder)
    return media.timeline.load_media_manifest(speech / "manifest.json")



def boundary_transcript(document):
    """Expose sentence endings inside coarse ASR segments without dropping words."""
    result = copy.deepcopy(document)
    payload = result.get("transcript", result)
    rows = []
    for segment in payload["segments"]:
        words = segment.get("words", [])
        if not words:
            rows.append(segment)
            continue
        groups, current = [], []
        for index, word in enumerate(words):
            current.append(word)
            terminal = bool(re.search(r"[.!?。！？][\"'”’)]*$", str(word.get("text", "")).strip()))
            gap = index+1 < len(words) and float(words[index+1]["start"])-float(word["end"]) >= .65
            if terminal or gap:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        for index, group in enumerate(groups):
            rows.append({**segment, "id": f"{segment.get('id', 'segment')}-boundary-{index}",
                         "start": group[0]["start"], "end": group[-1]["end"],
                         "text": " ".join(str(word["text"]).strip() for word in group), "words": group})
    payload["segments"] = rows
    payload["segment_count"] = len(rows)
    return result


def refine_boundaries(original, clip, language, folder):
    initial = folder / "boundary-pass"
    initial.mkdir()
    extended = {**clip, "start": max(0, clip["start"]-2), "end": min(original.duration_seconds, clip["end"]+2)}
    manifest = media.extract(original, extended, initial / "source")
    transcript = initial / "whisper/transcript.source.json"
    script("subtitle-generator/scripts/transcribe.py", manifest, "--output-dir", transcript.parent,
           "--language", language, "--no-prompt", python=ASR_PYTHON, log=initial / "transcribe.log")
    document = boundary_transcript(media.read(transcript))
    media.write(initial / "review-transcript.json", document)
    review = editor.boundaries(document, initial / "editor", {"title": clip["title_lines"], "reason": clip["reason"]})
    extraction = media.read(manifest)["source"]
    limits, start, end = reviewed_cut_ranges(document, review, extraction["start_seconds"], extraction["end_seconds"])
    media.write(initial / "review.json", {**review, "source_start": start, "source_end": end})
    clip.update(start=start, end=end, boundary_limits=limits)
    return media.extract(original, clip, folder / "source")


def reviewed_cut_ranges(transcript, review, source_start, source_end):
    """Bound acoustic snapping by neighboring words, not by coarse caption time."""
    segments = editor.transcript_segments(transcript)
    first, last = review["first_segment"], review["last_segment"]
    first_words, last_words = segments[first].get("words", []), segments[last].get("words", [])
    onset = source_start + float(first_words[0]["start"] if first_words else segments[first]["start"])
    release = source_start + float(last_words[-1]["end"] if last_words else segments[last]["end"])
    before = source_start + float(segments[first-1]["end"]) if first else source_start
    after = source_start + float(segments[last+1]["start"]) if last+1 < len(segments) else source_end
    before += min(.025, max(0., onset-before)/2) if first else 0.
    after -= min(.025, max(0., after-release)/2) if last+1 < len(segments) else 0.
    start_range = [max(source_start, before, onset-.18), onset]
    end_range = [release, min(source_end, after, release+.16)]
    if start_range[0] > start_range[1] or end_range[0] > end_range[1]:
        raise ValueError("Neighboring speech overlaps the requested boundary")
    start = max(start_range[0], onset-.08)
    end = min(end_range[1], release+.08)
    return {"start": start_range, "end": end_range}, start, end


def preserve_unedited_speech(extracted_manifest, transcript, folder):
    """Reject an uncertain edit and retain every original audio/video byte."""
    source = media.timeline.load_media_manifest(extracted_manifest)
    media.timeline.load_transcript(transcript, source)
    rejected = folder / "speech-rejected"
    (folder / "speech").rename(rejected)
    speech = folder / "speech"
    (speech / "qa").mkdir(parents=True)
    output = speech / "edited.mp4"
    shutil.copyfile(source.media_path, output)
    if media.timeline.sha256_file(output) != source.media_sha256:
        raise ValueError("Unedited fallback media is not byte-identical")
    shutil.copyfile(transcript, speech / "transcript.edited.json")
    manifest = {"schema_version": 1, "agent": "shorts-lab", "stage": "unedited-speech-fallback",
        "source": {"video_id": source.video_id, "title": source.title, "manifest_path": str(source.manifest_path),
                   "manifest_sha256": source.manifest_sha256, "media_sha256": source.media_sha256},
        "output": {"media_path": str(output), "sha256": source.media_sha256, "duration_seconds": source.duration_seconds},
        "verification": {"original_media_byte_identical": True, "source_transcript_verified": True}}
    media.write(speech / "manifest.json", manifest)
    media.write(speech / "qa/manifest.json", {"agent": "shorts-lab", "status": "ready-for-render",
        "basis": "byte-identical original clip; uncertain silence edit rejected", "edited": False,
        "rejected_edit_qa": str(rejected / "qa/manifest.json"), "human_review": "pending",
        "verification": manifest["verification"]})
    return media.timeline.load_media_manifest(speech / "manifest.json")


def verify_boundaries(speech, folder, source_context):
    post_asr = folder / "speech/retranscription/transcript.source.json"
    if not post_asr.is_file():
        # The unedited fallback is byte-identical to its verified input ASR.
        post_asr = folder / "speech/transcript.edited.json"
    media.timeline.load_transcript(post_asr, speech, allow_media_bound_video_id_rebind=True)
    review = editor.audit_boundaries(media.read(post_asr), folder / "editor", source_context)
    receipt = {"status": "passed", "media_sha256": speech.media_sha256,
        "transcript_sha256": media.timeline.sha256_file(post_asr), "transcript_path": str(post_asr),
        "review": review, "human_review": "pending"}
    media.write(folder / "boundary-qa.json", receipt)
    return receipt


def produce(store, job_id, attempt_number):
    job = store.job(job_id)
    if job["status"] != "running" or job["attempts"] != attempt_number:
        raise ValueError("Stale production attempt")
    folder = store.root / "jobs" / job_id / f"attempt-{attempt_number}"
    folder.mkdir(parents=True, exist_ok=True)
    stage = "source"
    try:
        store.progress(job_id, stage, "원본과 영상 정보를 준비하고 있습니다")
        original, metadata = source(store, job, folder)
        stage = "transcript"
        store.progress(job_id, stage, "원본의 전체 대화에서 이야기를 찾고 있습니다", title=metadata.get("title"))
        transcript, transcript_path, language = navigation(original, metadata, folder / "navigation")
        metadata["introductory_captions"] = [s["text"] for s in editor.transcript_segments(transcript) if float(s["start"]) < 120]
        stage = "selection"
        store.progress(job_id, stage, f"독립적으로 볼 수 있는 이야기를 최대 {job['max_clips']}개 고르고 있습니다")
        selected = editor.select(transcript, metadata, job["max_clips"], folder / "editor")
        media.write(folder / "selection.json", {"source_manifest": str(original.manifest_path), "transcript_sha256": media.timeline.sha256_file(transcript_path), "clips": selected})
        failures = []
        for index, clip in enumerate(selected, 1):
            clip_id = f"clip-{index:02d}"
            base = folder / clip_id
            base.mkdir()
            try:
                stage = "speech"
                store.progress(job_id, stage, f"{index}/{len(selected)} · 발화와 자막을 확인하고 있습니다")
                extracted = refine_boundaries(original, clip, language, base)
                speech = speech_stage(extracted, language, base)
                contact = media.contact_sheet(speech.media_path, speech.duration_seconds, base)
                source_context = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in editor.transcript_segments(transcript)
                                  if float(s["end"]) >= clip["start"]-2 and float(s["start"]) <= clip["end"]+2]
                verify_boundaries(speech, base, source_context)
                edited = editor.captions(media.read(base / "speech/transcript.edited.json"), source_context, clip, metadata, base / "editor", contact)
                stage = "render"
                store.progress(job_id, stage, f"{index}/{len(selected)} · {' / '.join(edited['title_lines'])}")
                rendered = media.render(base, edited, clip, job["video_id"])
                stage = "verify"
                store.progress(job_id, stage, f"{index}/{len(selected)} · 파일·음성·자막 검증 완료")
                store.add_clip(job_id, {"id": f"a{attempt_number}-{clip_id}", "title": " / ".join(edited["title_lines"]),
                    "duration": rendered["output"]["duration_seconds"], "sha256": rendered["output"]["sha256"],
                    "source_start": clip["start"], "source_end": clip["end"],
                    "review_notes": edited["review_notes"],
                    "media": str((base / "render/reel.mp4").relative_to(store.root)),
                    "poster": str((base / "render/frame-0.jpg").relative_to(store.root)),
                    "manifest": str((base / "render/manifest.json").relative_to(store.root))})
            except Exception:
                (base / "failure.log").write_text(traceback.format_exc())
                failures.append(clip_id)
        if failures:
            media.write(folder / "error.json", {"message": f"{len(failures)}개 후보가 검증을 통과하지 못했습니다. 완료된 결과는 확인할 수 있습니다."})
            return 2
        return 0
    except Exception:
        messages = {"source": "원본을 가져오지 못했습니다. 10초 이상 2시간 이내의 음성이 있는 영상을 확인해주세요.",
                    "transcript": "원본 전사를 준비하지 못했습니다. 잠시 뒤 다시 시도해주세요.",
                    "selection": "내용 선정이 완료되지 않았습니다. 잠시 뒤 다시 시도해주세요."}
        media.write(folder / "error.json", {"message": messages.get(stage, "제작 중 오류가 발생했습니다. 다시 시도해주세요.")})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--attempt", type=int, required=True)
    args = p.parse_args()
    raise SystemExit(produce(Store(args.root), args.job, args.attempt))
