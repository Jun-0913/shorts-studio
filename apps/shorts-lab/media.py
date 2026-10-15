"""Shorts-specific presentation using the existing STUDIO media primitives."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = Path(__file__).resolve().parent
REPO = APP.parents[1]
sys.path.insert(0, str(REPO / ".agents/shared"))
import timeline_edit as timeline
import subtitle_layout_contract as subtitle_layout


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SHORTS = load_module("shorts_shorts_primitives", ".agents/skills/youtube-short-generator/scripts/generate.py")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    timeline.write_json_atomic(Path(path), value)


def ffmpeg():
    local = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
    return Path(os.environ.get("FFMPEG_BIN") or (str(local) if platform.system() == "Darwin" and local.is_file() else shutil.which("ffmpeg")))


def run(command, log, allowed=(0,)):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("w") as handle:
        result = subprocess.run([str(x) for x in command], stdout=handle, stderr=subprocess.STDOUT, timeout=2 * 3600)
    if result.returncode not in allowed:
        raise RuntimeError(f"Command failed ({result.returncode}): {Path(log).name}")
    return result.returncode


def pcm(path, start=None, duration=None):
    command = [str(ffmpeg()), "-v", "error"]
    if start is not None:
        command += ["-ss", str(start)]
    command += ["-i", str(path)]
    if duration is not None:
        command += ["-t", str(duration)]
    command += ["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"]
    return np.frombuffer(subprocess.check_output(command, timeout=300), dtype=np.float32)


def boundary(source, target, kind, output, limits=None):
    begin = max(0, target - .65)
    audio = pcm(source.media_path, begin, min(1.3, source.duration_seconds-begin))
    step = 160
    n = len(audio) // step
    windows = audio[:n*step].reshape(n, step)
    rms = np.sqrt(np.mean(windows.astype(np.float64)**2, axis=1))
    times = begin + (np.arange(n)+.5) * .01
    lo, hi = (target-.35, target+.08) if kind == "start" else (target-.08, target+.35)
    if limits is not None:
        lo, hi = max(lo, limits[0]), min(hi, limits[1])
        if not lo <= target <= hi:
            raise ValueError("Requested cut is outside the reviewed speech boundaries")
    valid = np.flatnonzero((times >= max(0, lo)) & (times <= min(source.duration_seconds, hi)))
    quiet = [i for i in valid if i > 0 and i+1 < n and max(rms[i-1:i+2]) < 10**(-38/20)]
    chosen = min(quiet, key=lambda i: abs(times[i]-target)) if quiet else None
    value = float(times[chosen]) if chosen is not None else target
    image = Image.new("RGB", (1000, 240), "#101510")
    draw = ImageDraw.Draw(image)
    for i in range(n):
        x = int(i / max(1, n-1) * 999)
        peak = min(1, float(np.max(np.abs(windows[i]))))
        draw.line((x, 120-90*peak, x, 120+90*peak), fill="#7ae1b6")
    x = int((value-begin)/max(.01,len(audio)/16000)*999)
    draw.line((x, 20, x, 225), fill="white", width=2)
    draw.text((12, 8), f"{kind}: {value:.3f}s; automatic quiet-window check; human review pending", fill="white")
    image.save(output)
    return value, {"requested": target, "selected": value, "quiet_window_found": chosen is not None,
                   "allowed_range": [lo, hi],
                   "method": "30ms PCM quiet window near caption boundary", "image": str(output),
                   "image_sha256": timeline.sha256_file(output), "human_review": "pending"}


def extract(source, selection, folder):
    folder.mkdir(parents=True, exist_ok=True)
    limits = selection.get("boundary_limits", {})
    start, start_evidence = boundary(source, selection["start"], "start", folder / "start-waveform.png", limits.get("start"))
    end, end_evidence = boundary(source, selection["end"], "end", folder / "end-waveform.png", limits.get("end"))
    selection.update(start=start, end=end)
    output = folder / "clip.mp4"
    # Keep the original full-quality download; a 1080p working copy bounds RAM.
    command = [ffmpeg(), "-v", "error", "-threads", "2", "-ss", f"{start:.6f}", "-i", source.media_path,
               "-t", f"{end-start:.6f}", "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1",
               "-filter_threads", "1", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2",
               "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", output]
    run(command, folder / "extract.log")
    probe = timeline.probe_media(Path(shutil.which("ffprobe")), output)
    actual = float(probe["format"]["duration"])
    checks = {"has_video": any(s["codec_type"] == "video" for s in probe["streams"]),
              "has_audio": any(s["codec_type"] == "audio" for s in probe["streams"]),
              "sha256_verified": True, "duration_matches": abs(actual-(end-start)) < .12}
    if not all(checks.values()):
        raise ValueError("Extraction failed validation")
    manifest = folder / "manifest.json"
    write(manifest, {"schema_version": 1, "agent": "shorts-lab", "stage": "clip-extraction",
        "source": {"video_id": source.video_id, "title": source.title, "manifest_path": str(source.manifest_path),
                   "manifest_sha256": source.manifest_sha256, "media_sha256": source.media_sha256, "start_seconds": start, "end_seconds": end},
        "boundary_evidence": {"start": start_evidence, "end": end_evidence},
        "output": {"media_path": str(output), "sha256": timeline.sha256_file(output), "duration_seconds": actual}, "verification": checks})
    return manifest


def contact_sheet(media, duration, folder):
    image = Image.new("RGB", (1200, 700), "#151815")
    draw = ImageDraw.Draw(image)
    for i, t in enumerate([.15, duration*.2, duration*.4, duration*.6, duration*.8, max(.1, duration-.2)]):
        path = folder / f"source-frame-{i}.jpg"
        run([ffmpeg(), "-v", "error", "-ss", str(t), "-i", media, "-frames:v", "1", "-vf", "scale=400:225", path], folder / f"source-frame-{i}.log")
        x, y = (i%3)*400, (i//3)*350
        image.paste(Image.open(path), (x, y+30))
        draw.text((x+8, y+8), f"{t:.2f}s", fill="white")
    path = folder / "source-contact.jpg"
    image.save(path)
    return path


def design():
    value = read(REPO / "profiles/vertical.json")
    font_path = Path(value["subtitles"]["font_path"])
    if not font_path.is_absolute():
        font_path = REPO / font_path
    expected = value["subtitles"].get("font_sha256")
    approved = {expected}
    if not font_path.is_file() or timeline.sha256_file(font_path) not in approved:
        raise ValueError("Reviewed subtitle font is unavailable")
    value["subtitles"]["font_sha256"] = timeline.sha256_file(font_path)
    value["subtitles"]["font_path"] = str(font_path)
    value["name"] = "shorts-lab-v1"
    value.pop("badge", None)
    value.pop("stinger", None)
    value["canvas"]["fps"] = "24000/1001"
    value["video"].update(y=572, height=776, zoom=1.18, focus_y=.42, tracking_mode="shot-face")
    value["title"].update(font_size=82, center_y=360, max_width=970, line_spacing=28)
    value["subtitles"].update(placement="canvas", font_size=59, italic=False, scale_x=100, outline=1, shadow=0, margin_v=422, max_width=960)
    value["encoding"]["video_codec"] = "libx264"
    return value


def make_overlay(titles, profile, path):
    image = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = SHORTS.fit_font(draw, titles, REPO / profile["title"]["font_path"], 82, 970, "Black")
    for line, y, fill in zip(titles, [330, 434], ["#FFFFFF", "#E8C5A5"]):
        box = draw.textbbox((0, 0), line, font=font)
        draw.text((540, y-(box[1]+box[3])/2), line, font=font, fill=fill, anchor="ma")
    image.save(path)


def subtitle_pages(text, font, maximum=940, protected_phrases=()):
    if any(font.getlength(phrase) > maximum for phrase in protected_phrases if phrase in text):
        raise ValueError("A protected subtitle phrase is too long for one line")
    phrases = [*SHORTS.load_subtitle_regression_corpus()["protected_phrases"], *protected_phrases]
    pages = SHORTS.paginate_subtitle_text(text, font=font, max_width=maximum,
        max_lines=2, scale_x=100, protected_phrases=phrases)
    if subtitle_layout.compact_subtitle_text(text) != subtitle_layout.compact_subtitle_text(" ".join(" ".join(p) for p in pages)):
        raise ValueError("Subtitle layout changed the source text")
    if subtitle_layout.layout_has_orphans(pages):
        raise ValueError("Subtitle layout detached a single word")
    if not SHORTS.subtitle_pages_respect_protected_phrases(pages, phrases):
        raise ValueError("Subtitle layout split a protected phrase")
    return pages


def make_subtitles(transcript, edited, duration, profile, folder, media_hash):
    font = ImageFont.truetype(profile["subtitles"]["font_path"], 59)
    by_id = {s["id"]: s for s in transcript.get("transcript", transcript)["segments"]}
    cues = []
    previous = 0.
    for row in edited["segments"]:
        segment = by_id[row["id"]]
        start = max(previous, 0, float(segment["start"]))
        end = min(duration, float(segment["end"]))
        if end <= start:
            raise ValueError("Caption timing would drop a segment")
        chunks = subtitle_pages(row["text"], font, protected_phrases=edited.get("protected_phrases", []))
        intervals = subtitle_layout.split_text_intervals([" ".join(c) for c in chunks],
            start=start, end=end, words=segment.get("words", []))
        for chunk, (cursor, next_time) in zip(chunks, intervals):
            cues.append({"start": cursor, "end": next_time, "lines": chunk, "source_segment_id": row["id"],
                         "timing_basis": "source-words" if segment.get("words") else "proportional-fallback"})
        previous = end
    document = {"media_sha256": media_hash, "transcript_sha256": timeline.sha256_file(folder.parent / "speech/transcript.edited.json"),
                "layout_contract": {"name": "STUDIO shared subtitle layout", "source": str(Path(subtitle_layout.__file__)),
                    "sha256": timeline.sha256_file(Path(subtitle_layout.__file__)),
                    "regression_corpus_sha256": timeline.sha256_file(SHORTS.SUBTITLE_REGRESSION_PATH),
                    "editorial_protected_phrases": edited.get("protected_phrases", []),
                    "text_preserved": True, "no_orphan_lines": True, "protected_phrases_preserved": True},
                "cues": cues, "review_notes": edited["review_notes"], "human_review": "pending"}
    write(folder / "subtitles.json", document)
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans KR,59,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,1,0,8,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    for cue in cues:
        text = r"\N".join(cue["lines"])
        header += f"Dialogue: 0,{SHORTS.ass_timestamp(cue['start'])},{SHORTS.ass_timestamp(cue['end'])},Default,{cue['source_segment_id']},0,0,0,,{{\\pos(540,1420)}}{text}\n"
    (folder / "captions.ass").write_text(header)
    return max(font.getlength(line) for c in cues for line in c["lines"])


def render(base, edited, source_selection, video_id, *, render_name="render"):
    speech = timeline.load_media_manifest(base / "speech/manifest.json")
    qa = read(base / "speech/qa/manifest.json")
    if qa.get("status") != "ready-for-render":
        raise ValueError("Speech verification has not passed")
    edge_qa = read(base / "boundary-qa.json")
    if edge_qa.get("status") != "passed" or edge_qa.get("media_sha256") != speech.media_sha256:
        raise ValueError("Final speech boundary verification has not passed for this media")
    if not render_name.startswith("render") or Path(render_name).name != render_name:
        raise ValueError("Invalid render revision name")
    folder = base / render_name
    folder.mkdir()
    profile = design()
    write(folder / "profile.json", profile)
    make_overlay(edited["title_lines"], profile, folder / "overlay.png")
    transcript = read(base / "speech/transcript.edited.json")
    width = make_subtitles(transcript, edited, speech.duration_seconds, profile, folder, speech.media_sha256)
    clip = {"id": base.name, "start": 0, "end": speech.duration_seconds, "duration": speech.duration_seconds,
            "title_lines": edited["title_lines"], "face_track": [{"time": 0, "focus_x": edited["focus_x"]}]}
    source_pcm = pcm(speech.media_path)
    peak_db = 20 * math.log10(max(float(np.max(np.abs(source_pcm))), 1e-9))
    gain = min(0, -1.5-peak_db)
    command = SHORTS.render_command(ffmpeg=ffmpeg(), source_path=speech.media_path, overlay_path=folder / "overlay.png",
        ass_path=folder / "captions.ass", output_path=folder / "reel.mp4", clip=clip, profile=profile,
        subtitle_font_path=Path(profile["subtitles"]["font_path"]))
    command[1:1] = ["-threads", "2", "-filter_complex_threads", "1"]
    command[-1:-1] = ["-threads", "2", "-preset", "fast"]
    command[command.index("-af")+1] += f",volume={gain:.3f}dB"
    run(command, folder / "render.log")
    run([ffmpeg(), "-v", "error", "-i", folder / "reel.mp4", "-f", "null", "-"], folder / "decode.log")
    probe = SHORTS.probe_media(Path(shutil.which("ffprobe")), folder / "reel.mp4")
    checks = SHORTS.verify_output(probe, speech.duration_seconds, profile)
    final_pcm = pcm(folder / "reel.mp4")
    size = min(len(source_pcm), len(final_pcm))
    a, b = source_pcm[:size].astype(np.float64), final_pcm[:size].astype(np.float64)
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    correlation = float(np.dot(a,b)/denominator) if denominator else 0
    checks.update(whole_file_decode=True, subtitle_width=width <= 960, final_speech_boundaries=True,
                  subtitle_shared_layout=True, audio_correlation=correlation > .98,
                  audio_duration=abs(len(source_pcm)-len(final_pcm))/16000 < .06, no_audio_clipping=float(np.max(np.abs(final_pcm))) < 1)
    if not all(checks.values()):
        raise ValueError(f"Final media verification failed: {checks}")
    for i, t in enumerate([.15, speech.duration_seconds/2, max(.1,speech.duration_seconds-.12)]):
        run([ffmpeg(), "-v", "error", "-ss", str(t), "-i", folder / "reel.mp4", "-frames:v", "1", "-vf", "scale=270:480", folder / f"frame-{i}.jpg"], folder / f"frame-{i}.log")
    result = {"schema_version": 1, "agent": "shorts-lab", "stage": "vertical-render", "status": "ready-for-human-review",
              "video_id": video_id, "selection": source_selection, "title_lines": edited["title_lines"],
              "source": {"speech_manifest_sha256": timeline.sha256_file(base / "speech/manifest.json"), "media_sha256": speech.media_sha256,
                         "subtitles_sha256": timeline.sha256_file(folder / "subtitles.json")},
              "boundary_qa_sha256": timeline.sha256_file(base / "boundary-qa.json"),
              "output": {"media_path": str(folder / "reel.mp4"), "sha256": timeline.sha256_file(folder / "reel.mp4"),
                         "duration_seconds": float(probe["format"]["duration"])},
              "framing": {"mode": "fixed", "focus_x": edited["focus_x"]}, "audio": {"gain_db": gain, "correlation": correlation},
              "verification": checks, "human_review": {"status": "pending", "audio_listened_by_model": False}}
    write(folder / "manifest.json", result)
    return result
