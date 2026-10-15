# Shorts Studio

A private, local tool that turns video files or YouTube links into short vertical videos. Submit a source, let the worker select complete stories and generate Korean captions, then preview and download MP4s.

- Upload MP4, MOV, MKV, WebM, AVI, M4V, MPEG, or transport-stream video, or paste one or more YouTube video links.
- Produce up to 1–4 distinct clips per source. Each clip is 10–100 seconds, usually 15–75 seconds. Sources already 30 seconds or shorter retain their complete spoken context as one clip.
- Keep the original voice, check speech boundaries, shorten verified silence conservatively, translate captions into Korean, and render 1080 × 1920 H.264/AAC video.
- No added logo, wordmark, watermark, intro, outro, channel badge, or promotional end card. Marks already present inside a source video remain part of the source.
- Durable queue, progress, retry, preview, and direct MP4 downloads. No channel discovery, posting, scheduling, dubbing, account integration, or deployment pipeline.

## Run locally

Requirements: Python 3.11, FFmpeg/FFprobe with libass, an authenticated Codex CLI, and an internet connection for YouTube and initial speech model downloads. Node.js is recommended for YouTube player challenges. macOS and Linux are supported.

```sh
scripts/bootstrap.sh --full
codex login
scripts/start-studio.sh
```

Open <http://127.0.0.1:8431>. The launcher starts both the web server and one production worker. Use `--port 8432` or `--slots 2` to change the port or parallel source count. Stop it with Ctrl-C; persisted jobs are recovered on the next start. If another worker already owns the queue, it continues to own those jobs.

Without `--full`, bootstrap installs only dependencies for offline unit tests and the synthetic render check. Full production uses `faster-whisper` on CPU by default; the first run downloads a roughly 1.5GB speech model. AI editorial calls use the Codex account and incur its normal usage. Frames and transcript excerpts are sent to the configured model for clip selection and caption editing; video source files remain local.

Inputs need an audio track, must last 10 seconds to two hours, and local uploads are limited to 2GB each. Up to ten files and ten links can be submitted together; the queue is bounded at thirty sources. The engine needs at least 20GB free disk space to start a job. Speech-oriented content works best. Music-only, silent, inaccessible, DRM-protected, or age-restricted sources may fail or yield no suitable story. A completed job is ready for human playback, not a claim of perfect editorial accuracy.

## Configuration

Optional environment variables:

| Variable | Purpose |
| --- | --- |
| `STUDIO_MODEL` | Codex model override; unset uses the CLI default |
| `STUDIO_ASR_BACKEND` | `faster-whisper` by default; `mlx-whisper` requires a separately installed Apple Silicon runtime |
| `STUDIO_WHISPER_MODEL_CACHE` | Local model cache directory |
| `STUDIO_WHISPER_DEVICE` | `cpu` by default |
| `STUDIO_WHISPER_COMPUTE_TYPE` | `int8` by default |
| `YT_DLP_BIN` | Optional downloader executable |

The server binds only to `127.0.0.1` and checks request hosts and mutation origins. It is a single-user local tool, not an authenticated internet service. `artifacts/` stores private source files, the SQLite queue, intermediate receipts, logs, and final MP4s. These, `.venv/`, and `.tools/` are ignored by Git. API credentials and browser cookies are not shipped.

YouTube access depends on the network and YouTube's current restrictions. The downloader uses ordinary extraction first and can fall back to a separately configured PO token provider. The worker does not automatically read browser cookies. A source download error appears on its job without stopping other sources.

## Layout

- `apps/shorts-lab/`: web UI, file import, queue, worker, editorial calls, and renderer orchestration.
- `.agents/shared/`: source hashes, media verification, audio and subtitle layout primitives.
- `.agents/skills/*/scripts/`: retained download, transcription, speech-edit, and render code. No assistant workflows or publishing skills are included.
- `profiles/vertical.json`: unbranded rendering settings.
- `assets/fonts/`: Noto Sans KR and its required OFL license.
- `tests/`: synthetic queue, input, HTTP, subtitle and boundary regression coverage.

## Verify

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/check_shareable.py
.venv/bin/python scripts/smoke_media.py
```

The smoke check uses synthetic media and deterministic fixtures. It exercises file import, actual FFmpeg rendering, audio fidelity, dimensions, subtitles, full-file decoding, and result persistence. It does not test AI selection or ASR. See [verification](docs/verification.md) for separately executed end-to-end checks and their limits.
