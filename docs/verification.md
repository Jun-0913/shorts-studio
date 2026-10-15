# Verification

This file records checks actually run against this standalone tree. Runtime media and logs live in ignored `artifacts/`; they are not shipped with the repository.

## Scope

The retained production path covers local file or YouTube input, complete-story selection, word-timed transcription, conservative speech editing, Korean caption editing, vertical rendering, verification, preview and download. The new repository uses a fresh history. Source discovery, scheduled production, publishing adapters, account configuration, deployment, branded assets and promotional layouts are absent.

## Executed local checks

- Python unit tests: 41 passed, including concurrent queue deduplication, bounded claims, restart recovery, path and symlink containment, upload validation, source-hash checks, host/origin checks, result range requests, caption coverage, and speech-boundary validation.
- Python compilation, JavaScript syntax and shell syntax checks passed.
- Gitleaks 8.30.1 scanned the shareable source snapshot with zero findings.
- Repository scanner passed checks for credential patterns, machine-specific home paths and runtime artifacts. Branding and owner/host identifiers were audited separately; the generic scanner does not store a list of private identifiers.
- Real offline FFmpeg render: 1080 × 1920, H.264/AAC, 12 seconds. Full-file decode, expected duration, subtitle width, audio correlation, audio duration, no clipping, and result persistence passed.
- The actual rendered preview was opened and checked: no logo, watermark, badge or end card.
- Browser checks: the new interface was opened; a local MP4 was selected and submitted; the accepted job appeared; a YouTube URL was submitted and queued. Progress was read back from the durable queue.

The synthetic render does not prove ASR or AI editorial behavior. Separate real production runs completed:

| Input | Executed path | Result |
| --- | --- | --- |
| Uploaded narrated gardening MP4, outside any preset topic or channel | File import, language detection, actual speech transcription, AI selection, boundary checks, caption translation, real render | 33.343855-second Korean-captioned vertical MP4; audio correlation 0.999995; all media checks passed |
| Public YouTube video `jNQXAC9IVRw` | Real download, missing-caption fallback to actual ASR, AI selection, boundary checks, caption translation, real render | 17.580000-second vertical MP4; all media checks passed |

The uploaded-file output played in the browser at `readyState=4`, 1080 × 1920, with no media error. The actual browser-downloaded file's SHA-256 exactly matched the verified output. The test source is synthesized narration over a color test pattern; the speech transcription and editorial calls were real, not mocked. The YouTube output also played at `readyState=4`, 1080 × 1920, with no media error. Its source and resulting frames were inspected separately. That sample exposed ASR ambiguity in two phrases; the result carries review notes rather than claiming perfect transcription. Both automatic production runs are complete, with human editorial review still available.

Reproduction evidence on the validation machine: `artifacts/e2e/receipt.json`, `file-input-short.mp4`, and `youtube-input-short.mp4`. These are ignored local artifacts and are not part of the Git repository. Earlier unsuccessful attempts remain in the private job logs; the table records the final successful runs. Subsequent short-source selection changes do not affect the 34.33-second uploaded fixture.

Linux GitHub Actions runs unit tests, source scanning, and the actual FFmpeg smoke render on each push. Check the commit-matched result in the repository's Actions tab; the remote repository is private and independent, with no upstream fork association.

## Fixes found by execution

- Use the running virtual environment for all transcription subprocesses; an indirect Python symlink can lose virtual-environment dependencies.
- Install the speech runtime's explicit `requests` dependency.
- Missing YouTube source-language/caption metadata falls back to speech transcription after source verification.
- A post-edit transcript's basename-derived identifier may be rebound only when the media hash matches and the transcript is prompt-free.
- Keep all supplied speech segments for sources already 30 seconds or shorter, so subject introductions and closing reactions survive. Longer sources still use editorial excerpt selection.
- Split coarse ASR segments at measured sentence endings and pauses before boundary review, preserving every word and timestamp. This lets the review reject a neighboring sentence fragment without dropping the intended story.
- Replace the audio timestamp reset with sample-preserving resampling. The old reset lost approximately 21 milliseconds at the start; the real waveform correlation check now passes.

## Limits

The tool targets spoken video. It does not guarantee a good story from silent or music-only content, inaccessible sources, every media codec, or every language. Initial speech-model downloads and Codex authentication are required. Output remains for human review. Nothing is published to social media or deployed to a public server.
