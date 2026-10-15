# Shorts Studio

This is a private, standalone video-to-short-form tool. Keep scope limited to file/YouTube input, story selection, speech and subtitle processing, rendering, preview, and download.

- Write collaboration artifacts and code comments in English.
- Keep branding, channel-specific policies, account bindings, publishing integrations, credentials, and source media out of Git.
- Do not add a logo, watermark, stinger, or end card to generated video.
- Preserve source hashes, complete utterances, transcript coverage, and audio/video quality gates.
- Run `.venv/bin/python -m unittest discover -s tests` and `scripts/check_shareable.py` after changes. Run `scripts/smoke_media.py` when media behavior changes.
- Keep the server on loopback. There is no production deployment automation.
- Runtime output belongs in ignored `artifacts/`; tests must use synthetic sources.
