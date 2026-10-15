#!/usr/bin/env python3
"""Record explicit local-file review decisions in one Speech Tightener plan."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReviewError(RuntimeError):
    pass


def is_automatic_acoustic_silence(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("kind") == "long-pause"
        and candidate.get("acoustic_silence") is True
    )


def read_plan(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewError(f"Could not read Speech Tightener plan: {path}") from error
    if not isinstance(value, dict) or value.get("agent") != "speech-tightener":
        raise ReviewError("Input is not a Speech Tightener plan.")
    if not isinstance(value.get("candidates"), list):
        raise ReviewError("Speech Tightener plan candidates are missing.")
    return value


def removal_rationale(candidate: dict[str, Any]) -> str:
    kind = candidate.get("kind")
    if kind == "long-pause":
        return (
            "Shorten the measured acoustic silence while retaining the configured "
            "natural pause; no transcript word is removed."
        )
    if kind == "filler":
        return (
            "Remove the complete standalone non-semantic filler at two verified "
            "quiet zero-crossing boundaries."
        )
    if kind == "repetition":
        return (
            "Remove the complete immediately repeated repair while preserving the "
            "remaining word or phrase and the speaker's claim."
        )
    if kind == "discourse-marker":
        return (
            "Remove the complete non-semantic discourse marker; the surrounding "
            "sentence remains grammatical and retains its claim, hedge, and emphasis."
        )
    return (
        "Remove the complete reviewed speech repair while preserving every claim, "
        "qualifier, number, and speaker turn."
    )


def keep_rationale(candidate: dict[str, Any]) -> str:
    rejection = str(candidate.get("refinement_rejection") or "").strip()
    overlaps = candidate.get("collateral_word_overlaps")
    if rejection and isinstance(overlaps, list) and overlaps:
        overlap = overlaps[0] if isinstance(overlaps[0], dict) else {}
        word_id = str(overlap.get("word_id") or "a retained word")
        overlap_seconds = overlap.get("overlap_seconds")
        return (
            f"Keep after local-file review: {rejection.lower()}; the proposed cut "
            f"would overlap {word_id} by {overlap_seconds}s."
        )
    if rejection:
        return f"Keep after local-file review: {rejection}."
    return (
        "Keep after local-file review: no complete meaning-preserving deletion "
        "with verified safe boundaries was found."
    )


def review_evidence(candidate: dict[str, Any], plan_path: Path) -> list[str]:
    evidence = [
        f"plan:{plan_path}",
        f"detector:{candidate.get('detector', 'unknown')}",
        f"context:{str(candidate.get('context') or '').strip()}",
    ]
    if candidate.get("acoustic_silence") is True:
        evidence.append(
            "silence:"
            f"{candidate.get('silence_duration_seconds')}s measured; "
            f"{candidate.get('retained_silence_seconds')}s retained"
        )
    if candidate.get("acoustic_boundary_refined") is True:
        evidence.extend(
            [
                f"boundary_rms_db:{candidate.get('boundary_rms_db')}",
                f"zero_crossing_dbfs:{candidate.get('boundary_zero_crossing_dbfs')}",
                f"removed_word_ids:{candidate.get('removed_word_ids')}",
            ]
        )
    if str(candidate.get("detector") or "").startswith(
        "enhanced-verbatim-shadow-repeat"
    ):
        evidence.extend(
            [
                f"verbatim_probe:{candidate.get('verbatim_probe_path')}",
                f"verbatim_probe_sha256:{candidate.get('verbatim_probe_sha256')}",
                f"support_windows:{candidate.get('support_window_ids')}",
                f"shadow_removed_word_ids:{candidate.get('shadow_removed_word_ids')}",
                f"canonical_collapsed_word_ids:{candidate.get('canonical_collapsed_word_ids')}",
            ]
        )
    if candidate.get("refinement_rejection"):
        evidence.extend(
            [
                f"refinement_rejection:{candidate.get('refinement_rejection')}",
                f"collateral_word_overlaps:{candidate.get('collateral_word_overlaps')}",
                f"boundary_rms_db:{candidate.get('boundary_rms_db')}",
            ]
        )
    return evidence


def apply_review(
    plan: dict[str, Any],
    *,
    plan_path: Path,
    remove_ids: set[str],
    keep_ids: set[str],
    reviewed_by: str,
) -> dict[str, Any]:
    candidates = plan["candidates"]
    by_id = {
        str(candidate.get("id")): candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("id")
    }
    overlap = sorted(remove_ids & keep_ids)
    if overlap:
        raise ReviewError(
            f"Candidate IDs cannot be both remove and keep: {', '.join(overlap)}"
        )
    missing = sorted((remove_ids | keep_ids) - set(by_id))
    if missing:
        raise ReviewError(f"Unknown candidate IDs: {', '.join(missing)}")
    automatic_ids = {
        candidate_id
        for candidate_id, candidate in by_id.items()
        if is_automatic_acoustic_silence(candidate)
    }
    automatic_kept = sorted(automatic_ids & keep_ids)
    if automatic_kept:
        raise ReviewError(
            "Measured acoustic silence is automatically removed and cannot be "
            "marked --keep: " + ", ".join(automatic_kept)
        )
    unreviewed = sorted(set(by_id) - automatic_ids - remove_ids - keep_ids)
    if unreviewed:
        raise ReviewError(
            "Every spoken candidate requires an explicit --remove or --keep "
            "decision: "
            + ", ".join(unreviewed)
        )

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ReviewError("Every Speech Tightener candidate must be an object.")
        candidate_id = str(candidate.get("id") or "")
        candidate["decision"] = "keep"
        candidate["meaning_preserved"] = None
        candidate["rationale"] = keep_rationale(candidate)
        candidate["review_evidence"] = review_evidence(candidate, plan_path)
        if is_automatic_acoustic_silence(candidate):
            candidate["decision"] = "remove"
            candidate["meaning_preserved"] = True
            candidate["rationale"] = removal_rationale(candidate)
            candidate["decision_source"] = "automatic-measured-acoustic-silence"
            continue
        if candidate_id in keep_ids:
            continue
        if candidate.get("kind") == "long-pause":
            if candidate.get("acoustic_silence") is not True:
                raise ReviewError(f"{candidate_id} is not a measured acoustic silence.")
        elif candidate.get("acoustic_boundary_refined") is not True:
            raise ReviewError(f"{candidate_id} has no safe spoken boundary pair.")
        candidate["decision"] = "remove"
        candidate["meaning_preserved"] = True
        candidate["rationale"] = removal_rationale(candidate)
        if candidate.get("kind") == "discourse-marker":
            candidate["semantic_role"] = "non-semantic-discourse-marker"

    review = plan.get("review")
    if not isinstance(review, dict):
        raise ReviewError("Speech Tightener review contract is missing.")
    review.update(
        {
            "status": "agent-reviewed",
            "reviewed_by": reviewed_by,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "review_mode": "local-file-first",
            "model_audio_access": False,
            "human_listening_required": False,
            "final_reel_human_review_required": True,
        }
    )
    return plan


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.review-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--remove",
        action="append",
        default=[],
        help="Explicitly reviewed candidate ID to remove (repeatable)",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Explicitly reviewed candidate ID to keep (repeatable)",
    )
    parser.add_argument("--reviewed-by", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.plan.expanduser().resolve()
    try:
        plan = read_plan(path)
        reviewed = apply_review(
            plan,
            plan_path=path,
            remove_ids=set(args.remove),
            keep_ids=set(args.keep),
            reviewed_by=str(args.reviewed_by).strip(),
        )
        write_atomic(path, reviewed)
    except ReviewError as error:
        print(f"speech-tightener-review: {error}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "plan": str(path),
                "automatic_removed_candidates": sorted(
                    str(candidate.get("id"))
                    for candidate in reviewed["candidates"]
                    if isinstance(candidate, dict)
                    and candidate.get("decision_source")
                    == "automatic-measured-acoustic-silence"
                ),
                "removed_candidates": sorted(
                    str(candidate.get("id"))
                    for candidate in reviewed["candidates"]
                    if isinstance(candidate, dict)
                    and candidate.get("decision") == "remove"
                ),
                "removed_count": sum(
                    isinstance(candidate, dict)
                    and candidate.get("decision") == "remove"
                    for candidate in reviewed["candidates"]
                ),
                "kept_candidates": sorted(
                    str(candidate.get("id"))
                    for candidate in reviewed["candidates"]
                    if isinstance(candidate, dict)
                    and candidate.get("decision") == "keep"
                ),
                "kept_count": sum(
                    isinstance(candidate, dict)
                    and candidate.get("decision") == "keep"
                    for candidate in reviewed["candidates"]
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
