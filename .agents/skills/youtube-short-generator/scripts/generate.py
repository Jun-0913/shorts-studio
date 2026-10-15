from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from PIL import Image, ImageColor, ImageDraw, ImageFont
SHARED_DIR = Path(__file__).resolve().parents[3] / 'shared'
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
from subtitle_layout_contract import aligned_boundary_time, canonical_subtitle_word as shared_canonical_subtitle_word, layout_has_orphans, layout_respects_protected_phrases, load_regression_corpus, normalize_subtitle_text as shared_normalize_subtitle_text, protected_boundary_indices as shared_protected_boundary_indices, protected_word_matches as shared_protected_word_matches, semantic_boundary_score
SCHEMA_VERSION = 1
MAX_UPLOAD_TITLE_LENGTH = 100
CLIP_ID_PATTERN = re.compile('[a-z0-9]+(?:-[a-z0-9]+)*')
EDITORIAL_EVIDENCE_FIELDS = ('hook_delivery_seconds', 'opening_evidence', 'early_payoff', 'duration_reason')
SUBTITLE_REGRESSION_PATH = SHARED_DIR / 'references' / 'subtitle-layout-regressions.json'

class ShortGeneratorError(RuntimeError):
    """An expected, user-actionable Shorts generation failure."""

def load_subtitle_regression_corpus() -> dict[str, Any]:
    try:
        return load_regression_corpus(SUBTITLE_REGRESSION_PATH)
    except ValueError as error:
        raise ShortGeneratorError(str(error)) from error

def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShortGeneratorError(f'{label} must be a finite number.')
    result = float(value)
    if not math.isfinite(result):
        raise ShortGeneratorError(f'{label} must be a finite number.')
    return result

def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShortGeneratorError(f'{label} must be a positive integer.')
    return value

def profile_number(section: dict[str, Any], key: str, label: str) -> float:
    return finite_number(section.get(key), f'{label}.{key}')

def profile_dimensions(profile: dict[str, Any]) -> dict[str, int | float | str | bool]:
    canvas = profile.get('canvas')
    video = profile.get('video')
    title = profile.get('title')
    subtitles = profile.get('subtitles')
    encoding = profile.get('encoding')
    if not all((isinstance(value, dict) for value in (canvas, video, title, subtitles, encoding))):
        raise ShortGeneratorError('Profile sections are incomplete.')
    assert isinstance(canvas, dict)
    assert isinstance(video, dict)
    assert isinstance(title, dict)
    assert isinstance(subtitles, dict)
    assert isinstance(encoding, dict)
    tracking_mode = str(video.get('tracking_mode', 'face')).strip().lower()
    if tracking_mode not in {'face', 'locked', 'smooth-face', 'shot-face'}:
        raise ShortGeneratorError("video.tracking_mode must be 'face', 'locked', 'smooth-face', or 'shot-face'.")
    subtitle_placement = str(subtitles.get('placement', 'video')).strip().lower()
    if subtitle_placement not in {'video', 'canvas'}:
        raise ShortGeneratorError("subtitles.placement must be 'video' or 'canvas'.")
    tracking = video.get('tracking', {})
    if not isinstance(tracking, dict):
        raise ShortGeneratorError('video.tracking must be an object.')
    tracking_sample_fps = finite_number(tracking.get('sample_fps', 6.0), 'video.tracking.sample_fps')
    tracking_output_fps = finite_number(tracking.get('output_fps', 2.0), 'video.tracking.output_fps')
    tracking_dead_zone = finite_number(tracking.get('dead_zone', 0.03), 'video.tracking.dead_zone')
    tracking_smoothing_seconds = finite_number(tracking.get('smoothing_seconds', 0.8), 'video.tracking.smoothing_seconds')
    tracking_max_speed = finite_number(tracking.get('max_speed_per_second', 0.1), 'video.tracking.max_speed_per_second')
    tracking_scene_threshold = finite_number(tracking.get('scene_threshold', 0.05), 'video.tracking.scene_threshold')
    if not 1 <= tracking_sample_fps <= 30:
        raise ShortGeneratorError('video.tracking.sample_fps must be between 1 and 30.')
    if not 1 <= tracking_output_fps <= tracking_sample_fps:
        raise ShortGeneratorError('video.tracking.output_fps must be between 1 and sample_fps.')
    if not 0 <= tracking_dead_zone <= 0.2:
        raise ShortGeneratorError('video.tracking.dead_zone must be between 0 and 0.2.')
    if tracking_smoothing_seconds <= 0 or tracking_max_speed <= 0:
        raise ShortGeneratorError('video.tracking smoothing and speed values must be positive.')
    if not 0 < tracking_scene_threshold < 1:
        raise ShortGeneratorError('video.tracking.scene_threshold must be between 0 and 1.')
    values: dict[str, int | float | str | bool] = {'canvas_width': positive_integer(canvas.get('width'), 'canvas.width'), 'canvas_height': positive_integer(canvas.get('height'), 'canvas.height'), 'background': str(canvas.get('background', '#000000')), 'fps': str(canvas.get('fps', '30')), 'video_x': int(profile_number(video, 'x', 'video')), 'video_y': int(profile_number(video, 'y', 'video')), 'video_width': positive_integer(video.get('width'), 'video.width'), 'video_height': positive_integer(video.get('height'), 'video.height'), 'video_focus_x': profile_number(video, 'focus_x', 'video'), 'video_focus_y': profile_number(video, 'focus_y', 'video'), 'video_zoom': profile_number(video, 'zoom', 'video'), 'video_tracking_mode': tracking_mode, 'tracking_sample_fps': tracking_sample_fps, 'tracking_output_fps': tracking_output_fps, 'tracking_dead_zone': tracking_dead_zone, 'tracking_smoothing_seconds': tracking_smoothing_seconds, 'tracking_max_speed': tracking_max_speed, 'tracking_scene_threshold': tracking_scene_threshold, 'title_font_path': str(title.get('font_path', '')), 'title_font_size': positive_integer(title.get('font_size'), 'title.font_size'), 'title_font_variation': str(title.get('font_variation', 'Black')), 'title_fill': str(title.get('fill', '#FFFFFF')), 'title_center_x': int(profile_number(title, 'center_x', 'title')), 'title_center_y': int(profile_number(title, 'center_y', 'title')), 'title_line_spacing': int(profile_number(title, 'line_spacing', 'title')), 'title_max_width': positive_integer(title.get('max_width'), 'title.max_width'), 'subtitle_font_name': str(subtitles.get('font_name', 'Noto Sans KR')), 'subtitle_font_path': str(subtitles.get('font_path', '')), 'subtitle_font_sha256': str(subtitles.get('font_sha256', '')), 'subtitle_placement': subtitle_placement, 'subtitle_font_size': positive_integer(subtitles.get('font_size'), 'subtitles.font_size'), 'subtitle_max_width': positive_integer(subtitles.get('max_width'), 'subtitles.max_width'), 'subtitle_max_lines': positive_integer(subtitles.get('max_lines'), 'subtitles.max_lines'), 'subtitle_allow_orphans': bool(subtitles.get('allow_orphans', False)), 'subtitle_primary_color': str(subtitles.get('primary_color', '&H00FFFFFF')), 'subtitle_outline_color': str(subtitles.get('outline_color', '&H00141414')), 'subtitle_back_color': str(subtitles.get('back_color', '&H80000000')), 'subtitle_bold': bool(subtitles.get('bold', True)), 'subtitle_italic': bool(subtitles.get('italic', True)), 'subtitle_scale_x': int(profile_number(subtitles, 'scale_x', 'subtitles')), 'subtitle_scale_y': int(profile_number(subtitles, 'scale_y', 'subtitles')), 'subtitle_outline': int(profile_number(subtitles, 'outline', 'subtitles')), 'subtitle_shadow': int(profile_number(subtitles, 'shadow', 'subtitles')), 'subtitle_margin_l': int(profile_number(subtitles, 'margin_l', 'subtitles')), 'subtitle_margin_r': int(profile_number(subtitles, 'margin_r', 'subtitles')), 'subtitle_margin_v': int(profile_number(subtitles, 'margin_v', 'subtitles')), 'video_codec': str(encoding.get('video_codec', 'libx264')), 'video_bitrate': str(encoding.get('video_bitrate', '8M')), 'video_b_frames': encoding.get('b_frames'), 'audio_codec': str(encoding.get('audio_codec', 'aac')), 'audio_bitrate': str(encoding.get('audio_bitrate', '192k'))}
    return values

def fit_font(draw: ImageDraw.ImageDraw, lines: list[str], font_path: Path, initial_size: int, max_width: int, variation: str) -> ImageFont.FreeTypeFont:
    size = initial_size
    while size >= 48:
        font = ImageFont.truetype(str(font_path), size=size)
        try:
            font.set_variation_by_name(variation)
        except (AttributeError, OSError) as error:
            raise ShortGeneratorError(f'Title font has no requested variation: {variation}') from error
        widest = max((draw.textbbox((0, 0), line, font=font)[2] for line in lines))
        if widest <= max_width:
            return font
        size -= 2
    raise ShortGeneratorError('Title cannot fit the configured maximum width.')

def ass_timestamp(value: float) -> str:
    centiseconds = max(0, int(round(value * 100)))
    (hours, remainder) = divmod(centiseconds, 360000)
    (minutes, remainder) = divmod(remainder, 6000)
    (seconds, fraction) = divmod(remainder, 100)
    return f'{hours}:{minutes:02d}:{seconds:02d}.{fraction:02d}'

def subtitle_text_width(value: str, font: ImageFont.FreeTypeFont, scale_x: int) -> float:
    return float(font.getlength(value)) * scale_x / 100

def protected_boundary_indices(words: list[str], protected_phrases: list[str] | tuple[str, ...]) -> set[int]:
    return shared_protected_boundary_indices(words, protected_phrases)

def subtitle_pages_respect_protected_phrases(pages: list[list[str]], protected_phrases: list[str] | tuple[str, ...]) -> bool:
    return layout_respects_protected_phrases(pages, protected_phrases)

def split_oversize_words(words: list[str], *, font: ImageFont.FreeTypeFont, max_width: int, scale_x: int) -> list[str]:
    resolved: list[str] = []
    for word in words:
        if subtitle_text_width(word, font, scale_x) <= max_width:
            resolved.append(word)
            continue
        chunk = ''
        for character in word:
            candidate = f'{chunk}{character}'
            if chunk and subtitle_text_width(candidate, font, scale_x) > max_width:
                resolved.append(chunk)
                chunk = character
            else:
                chunk = candidate
        if chunk:
            resolved.append(chunk)
    return resolved

def semantic_boundary_penalty(words: list[str], boundary: int) -> float:
    """Prefer Korean phrase and punctuation boundaries over visual-only balance."""
    if boundary <= 0 or boundary >= len(words):
        return 0.0
    left = words[boundary - 1].strip()
    if re.search('[.!?。！？…][\\"\'”’)]*$', left):
        return 0.0
    if re.search('[,;:，；：][\\"\'”’)]*$', left):
        return 0.08
    shared_score = semantic_boundary_score(words, boundary)
    if shared_score >= 20:
        return 0.25
    if shared_score < 0:
        return 1.5
    return 0.55

def layout_subtitle_page(words: list[str], *, font: ImageFont.FreeTypeFont, max_width: int, max_lines: int, scale_x: int, protected_phrases: list[str] | tuple[str, ...]=()) -> list[str] | None:
    if not words:
        return None
    whole = ' '.join(words)
    if subtitle_text_width(whole, font, scale_x) <= max_width:
        return [whole]
    if max_lines < 2:
        return None
    protected_boundaries = protected_boundary_indices(words, protected_phrases)
    best: tuple[float, list[str]] | None = None
    for boundary in range(1, len(words)):
        if boundary in protected_boundaries:
            continue
        first = ' '.join(words[:boundary])
        second = ' '.join(words[boundary:])
        first_width = subtitle_text_width(first, font, scale_x)
        second_width = subtitle_text_width(second, font, scale_x)
        if first_width > max_width or second_width > max_width:
            continue
        orphan_penalty = max_width * 4 if len(words) >= 4 and (boundary == 1 or boundary == len(words) - 1) else 0
        phrase_penalty = semantic_boundary_penalty(words, boundary) * max_width * 0.3
        score = abs(first_width - second_width) + orphan_penalty + phrase_penalty
        if best is None or score < best[0]:
            best = (score, [first, second])
    return None if best is None else best[1]

def balanced_subtitle_word_pages(words: list[str], *, font: ImageFont.FreeTypeFont, max_width: int, max_lines: int, scale_x: int, protected_phrases: list[str] | tuple[str, ...]=()) -> list[list[str]]:
    """Partition an oversized sentence without leaving a fragment on the last page."""
    candidates: dict[int, list[int]] = {}
    for start in range(len(words)):
        candidates[start] = []
        for end in range(start + 1, len(words) + 1):
            if layout_subtitle_page(words[start:end], font=font, max_width=max_width, max_lines=max_lines, scale_x=scale_x, protected_phrases=protected_phrases) is None:
                break
            candidates[start].append(end)
    protected_boundaries = protected_boundary_indices(words, protected_phrases)
    minimum_pages = [math.inf] * (len(words) + 1)
    minimum_pages[len(words)] = 0
    for start in range(len(words) - 1, -1, -1):
        for end in candidates[start]:
            if end < len(words) and end in protected_boundaries:
                continue
            minimum_pages[start] = min(minimum_pages[start], 1 + minimum_pages[end])
    if not math.isfinite(minimum_pages[0]):
        raise ShortGeneratorError('Subtitle text cannot fit the configured width.')
    page_count = int(minimum_pages[0])
    total_load = subtitle_text_width(' '.join(words), font, scale_x)
    target_load = total_load / page_count
    memo: dict[tuple[int, int], tuple[float, list[list[str]]] | None] = {}

    def choose(start: int, pages_left: int) -> tuple[float, list[list[str]]] | None:
        key = (start, pages_left)
        if key in memo:
            return memo[key]
        if pages_left == 0:
            return (0.0, []) if start == len(words) else None
        best: tuple[float, list[list[str]]] | None = None
        for end in candidates.get(start, []):
            if end < len(words) and end in protected_boundaries:
                continue
            if minimum_pages[end] > pages_left - 1:
                continue
            remaining = choose(end, pages_left - 1)
            if remaining is None:
                continue
            page_words = words[start:end]
            load = subtitle_text_width(' '.join(page_words), font, scale_x)
            balance = ((load - target_load) / max_width) ** 2
            boundary = semantic_boundary_penalty(words, end)
            score = balance + boundary * 0.75 + remaining[0]
            if best is None or score < best[0]:
                best = (score, [page_words, *remaining[1]])
        memo[key] = best
        return best
    chosen = choose(0, page_count)
    if chosen is None:
        raise ShortGeneratorError('Subtitle pagination could not find a balanced layout.')
    return chosen[1]

def paginate_subtitle_text(value: str, *, font: ImageFont.FreeTypeFont, max_width: int, max_lines: int, scale_x: int, protected_phrases: list[str] | tuple[str, ...]=()) -> list[list[str]]:
    source_lines = [line.strip() for line in value.split('\\N') if line.strip()]
    words = split_oversize_words(' '.join(source_lines).split(), font=font, max_width=max_width, scale_x=scale_x)
    if not words:
        return [['']]
    if len(source_lines) <= max_lines and all((subtitle_text_width(line, font, scale_x) <= max_width for line in source_lines)) and (len(words) <= 2 or all((len(line.split()) > 1 for line in source_lines))) and subtitle_pages_respect_protected_phrases([source_lines], protected_phrases):
        return [source_lines]
    whole_layout = layout_subtitle_page(words, font=font, max_width=max_width, max_lines=max_lines, scale_x=scale_x, protected_phrases=protected_phrases)
    if whole_layout is not None:
        return [whole_layout]
    word_pages = balanced_subtitle_word_pages(words, font=font, max_width=max_width, max_lines=max_lines, scale_x=scale_x, protected_phrases=protected_phrases)
    pages: list[list[str]] = []
    for page_words in word_pages:
        layout = layout_subtitle_page(page_words, font=font, max_width=max_width, max_lines=max_lines, scale_x=scale_x, protected_phrases=protected_phrases)
        if layout is None:
            raise ShortGeneratorError('Subtitle pagination produced an invalid page.')
        pages.append(layout)
    return pages

def escape_filter_path(path: Path) -> str:
    value = str(path.resolve())
    for (original, replacement) in (('\\', '\\\\'), (':', '\\:'), ("'", "\\'"), (',', '\\,'), ('[', '\\['), (']', '\\]')):
        value = value.replace(original, replacement)
    return value

def frame_rate_value(value: Any, label: str='frame rate') -> float:
    """Resolve an FFmpeg integer or rational frame-rate value."""
    text = str(value).strip()
    try:
        if '/' in text:
            (numerator_text, denominator_text) = text.split('/', 1)
            numerator = float(numerator_text)
            denominator = float(denominator_text)
            result = numerator / denominator
        else:
            result = float(text)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ShortGeneratorError(f'{label} must be a positive frame rate.') from error
    if not math.isfinite(result) or result <= 0:
        raise ShortGeneratorError(f'{label} must be a positive frame rate.')
    return result

def scene_cut_switch_time(scene_cut_time: float, output_fps: float) -> float:
    """Switch midway after the prior frame so rounded cuts hit the first new frame."""
    if not math.isfinite(output_fps) or output_fps <= 0:
        raise ShortGeneratorError('Output frame rate must be positive.')
    return max(0.0, scene_cut_time - 0.5 / output_fps)

def focus_track_expression(track: list[dict[str, Any]], fallback: float, *, interpolate: bool=False, output_fps: float | None=None) -> str:
    if not track:
        return f'{fallback:.6f}'
    focus_values = [float(point['focus_x']) for point in track]
    if max(focus_values) - min(focus_values) < 1e-06:
        return f'{focus_values[0]:.6f}'
    expression = f"{float(track[-1]['focus_x']):.6f}"
    for index in range(len(track) - 2, -1, -1):
        current = track[index]
        following = track[index + 1]
        start = float(current['time'])
        boundary = float(following['time'])
        current_focus = float(current['focus_x'])
        following_focus = float(following['focus_x'])
        switch_boundary = boundary
        if following.get('scene_cut') and output_fps is not None:
            switch_boundary = scene_cut_switch_time(boundary, output_fps)
        segment = f'{current_focus:.6f}'
        if interpolate and (not following.get('scene_cut')) and (boundary > start) and (abs(following_focus - current_focus) >= 1e-06):
            progress = f'(t-{start:.3f})/{boundary - start:.3f}'
            segment = f'{current_focus:.6f}+({following_focus - current_focus:.6f})*({progress})'
        expression = f'if(lt(t,{switch_boundary:.6f}),{segment},{expression})'
    return expression

def render_command(*, ffmpeg: Path, source_path: Path, overlay_path: Path, ass_path: Path, output_path: Path, clip: dict[str, Any], profile: dict[str, Any], subtitle_font_path: Path | None=None) -> list[str]:
    values = profile_dimensions(profile)
    duration = float(clip['duration'])
    canvas_width = int(values['canvas_width'])
    canvas_height = int(values['canvas_height'])
    video_width = int(values['video_width'])
    video_height = int(values['video_height'])
    crop_ratio = video_width / video_height
    framing = clip.get('framing', {})
    focus_x = float(framing.get('focus_x', values['video_focus_x']))
    focus_y = float(framing.get('focus_y', values['video_focus_y']))
    zoom = float(framing.get('zoom', values['video_zoom']))
    if not 0 <= focus_x <= 1 or not 0 <= focus_y <= 1 or (not 1 <= zoom <= 3):
        raise ShortGeneratorError('Resolved clip framing is invalid.')
    crop_width = f'ih/{zoom:.6f}*{crop_ratio:.12f}'
    crop_height = f'ih/{zoom:.6f}'
    focus_x_expression = focus_track_expression(clip.get('face_track', []), focus_x, interpolate=values['video_tracking_mode'] == 'smooth-face', output_fps=frame_rate_value(values['fps'], 'profile canvas.fps'))
    crop_x = f'min(max(iw*({focus_x_expression})-ow/2,0),iw-ow)'
    crop_y = f'min(max(ih*{focus_y:.6f}-oh/2,0),ih-oh)'
    input_video_filter = '[0:v]setpts=PTS-STARTPTS'
    tail_hold_start = clip.get('tail_hold_start_seconds')
    if tail_hold_start is not None:
        hold_start = float(tail_hold_start)
        if not 0 < hold_start < duration:
            raise ShortGeneratorError('Resolved tail hold start is invalid.')
        input_video_filter += f',trim=end={hold_start:.6f},tpad=stop_mode=clone:stop_duration={duration - hold_start:.6f}'
    clip_filter = f"{input_video_filter},crop=w='{crop_width}':h='{crop_height}':x='{crop_x}':y='{crop_y}',scale={video_width}:{video_height},setsar=1"
    fontsdir = f":fontsdir='{escape_filter_path(subtitle_font_path.parent)}'" if subtitle_font_path is not None else ''
    if values['subtitle_placement'] == 'canvas':
        filter_value = f"{clip_filter}[clip];color=c={values['background']}:s={canvas_width}x{canvas_height}:r={values['fps']}:d={duration:.6f}[base];[base][clip]overlay={int(values['video_x'])}:{int(values['video_y'])}:shortest=1[framed];[framed]ass=filename='{escape_filter_path(ass_path)}'{fontsdir}:original_size={canvas_width}x{canvas_height}:shaping=complex[captioned];[captioned][1:v]overlay=0:0:format=auto:shortest=1[video]"
    else:
        filter_value = f"{clip_filter},ass=filename='{escape_filter_path(ass_path)}'{fontsdir}:original_size={video_width}x{video_height}:shaping=complex[clip];color=c={values['background']}:s={canvas_width}x{canvas_height}:r={values['fps']}:d={duration:.6f}[base];[base][clip]overlay={int(values['video_x'])}:{int(values['video_y'])}:shortest=1[framed];[framed][1:v]overlay=0:0:format=auto:shortest=1[video]"
    command = [str(ffmpeg), '-hide_banner', '-y', '-ss', f"{float(clip['start']):.6f}", '-t', f'{duration:.6f}', '-i', str(source_path), '-loop', '1', '-framerate', str(values['fps']), '-i', str(overlay_path), '-filter_complex', filter_value, '-map', '[video]', '-map', '0:a:0?', '-af', 'aresample=async=1:first_pts=0', '-t', f'{duration:.6f}', '-c:v', str(values['video_codec']), '-b:v', str(values['video_bitrate'])]
    if str(values['video_codec']).endswith('_videotoolbox'):
        command.extend(['-allow_sw', '1'])
    b_frames = values['video_b_frames']
    if b_frames is not None:
        if isinstance(b_frames, bool) or not isinstance(b_frames, int) or b_frames < 0:
            raise ShortGeneratorError('Profile encoding.b_frames must be a non-negative integer.')
        command.extend(['-bf', str(b_frames)])
    command.extend(['-pix_fmt', 'yuv420p', '-c:a', str(values['audio_codec']), '-b:a', str(values['audio_bitrate']), '-movflags', '+faststart', '-metadata', f"title={' / '.join(clip['title_lines'])}", str(output_path)])
    return command

def run_command(command: list[str], label: str) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise ShortGeneratorError(f'{label} could not be started.') from error
    if result.returncode != 0:
        detail = result.stderr.strip()[-5000:]
        raise ShortGeneratorError(f'{label} failed' + (f':\n{detail}' if detail else '.'))
    return result.stdout

def probe_media(ffprobe: Path, path: Path) -> dict[str, Any]:
    output = run_command([str(ffprobe), '-v', 'error', '-show_entries', 'format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,time_base,nb_frames', '-of', 'json', str(path)], 'FFprobe')
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise ShortGeneratorError('FFprobe returned invalid JSON.') from error
    if not isinstance(value, dict):
        raise ShortGeneratorError('FFprobe returned an invalid document.')
    return value

def verify_output(probe: dict[str, Any], expected_duration: float, profile: dict[str, Any]) -> dict[str, bool]:
    values = profile_dimensions(profile)
    streams = probe.get('streams')
    if not isinstance(streams, list):
        streams = []
    video_stream = next((item for item in streams if isinstance(item, dict) and item.get('codec_type') == 'video'), None)
    audio_stream = next((item for item in streams if isinstance(item, dict) and item.get('codec_type') == 'audio'), None)
    try:
        duration = float(probe.get('format', {}).get('duration'))
    except (TypeError, ValueError):
        duration = math.nan
    return {'output_present': True, 'has_video': video_stream is not None, 'has_audio': audio_stream is not None, 'dimensions_match': bool(video_stream and video_stream.get('width') == values['canvas_width'] and (video_stream.get('height') == values['canvas_height'])), 'duration_positive': math.isfinite(duration) and duration > 0, 'duration_matches': math.isfinite(duration) and abs(duration - expected_duration) <= 0.35}
