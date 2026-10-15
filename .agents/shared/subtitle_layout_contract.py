"""Shared semantic and timing rules for Studio Korean subtitle layouts.

Visual capacity stays profile-specific.  These helpers define the invariant
parts of captioning: semantic break quality, protected phrase boundaries,
orphan detection, and source-word-timestamp-first interval allocation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
SENTENCE_END_RE = re.compile(r"[.!?…。！？]+(?:[\"'”’」』）)\]]*)?$")
CLAUSE_END_RE = re.compile(r"[,;:，、；：](?:[\"'”’」』）)\]]*)?$")
KOREAN_CONNECTIVE_END_RE = re.compile(
    r"[가-힣]{2,}(?:다면|라면|자면|으면|면|지만|거나|든지)$"
)
KOREAN_SOFT_CONNECTIVE_END_RE = re.compile(r"[가-힣]{2,}(?:며|면서|는데)$")
KOREAN_TOPIC_END_RE = re.compile(
    r"(?:전제|핵심|요점|이유|문제|차이|목표|결론|사실|의미|가정|원칙)(?:은|는)$"
)
KOREAN_PARTICLE_SUFFIXES = (
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "에서",
    "에게",
    "와",
    "과",
    "로",
    "으로",
    "도",
    "만",
    "까지",
    "부터",
    "처럼",
    "보다",
    "조차",
    "마저",
    "께서",
)
LINE_END_AVOID_WORDS = frozenset(
    {
        "가장",
        "그",
        "그런",
        "그리고",
        "그래서",
        "더",
        "몇",
        "모든",
        "바로",
        "수",
        "아주",
        "어떤",
        "여러",
        "우리",
        "이",
        "이런",
        "저",
        "정말",
        "첫",
        "한",
    }
)
LINE_END_AVOID_SUFFIXES = (
    "같은",
    "되는",
    "라는",
    "이라는",
    "없는",
    "있는",
    "싶은",
    "하던",
    "하는",
    "했던",
    "한",
    "할",
    "된",
)
LINE_START_PREFERRED_WORDS = frozenset(
    {
        "그런데",
        "그래서",
        "그러니까",
        "그러면",
        "그다음",
        "반면",
        "아니면",
        "하지만",
    }
)
SEMANTIC_SEQUENCE_END_WORDS = frozenset({"뒤", "후", "때", "경우", "제공해"})
WEAK_LINE_START_WORDS = frozenset({"것", "때", "중", "뒤", "수"})


def normalize_subtitle_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace(r"\N", " ")).strip()


def compact_subtitle_text(value: str) -> str:
    return re.sub(r"\s+", "", normalize_subtitle_text(value))


def canonical_subtitle_word(value: str) -> str:
    return re.sub(
        r"^[\s\"'‘’“”([{]+|[\s\"'‘’“”.,!?。！？…:;，；：)\]}]+$",
        "",
        value,
    )


def protected_word_matches(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    if not actual.startswith(expected):
        return False
    return actual[len(expected) :] in KOREAN_PARTICLE_SUFFIXES


def protected_boundary_indices(
    words: Sequence[str], protected_phrases: Sequence[str] = ()
) -> set[int]:
    canonical_words = [canonical_subtitle_word(word) for word in words]
    protected: set[int] = set()
    for phrase in protected_phrases:
        phrase_words = [
            canonical_subtitle_word(word)
            for word in normalize_subtitle_text(phrase).split()
            if canonical_subtitle_word(word)
        ]
        if len(phrase_words) < 2:
            continue
        for start in range(0, len(canonical_words) - len(phrase_words) + 1):
            if all(
                protected_word_matches(actual, expected)
                for actual, expected in zip(
                    canonical_words[start : start + len(phrase_words)],
                    phrase_words,
                )
            ):
                protected.update(range(start + 1, start + len(phrase_words)))
    for boundary in range(1, len(words)):
        if words[boundary].lstrip().startswith("(*"):
            continue
        if words[boundary - 1].rstrip().endswith(","):
            continue
        left_has_latin = bool(re.search(r"[A-Za-z]", canonical_words[boundary - 1]))
        right_has_latin = bool(re.search(r"[A-Za-z]", canonical_words[boundary]))
        left_has_hangul = bool(re.search(r"[가-힣]", canonical_words[boundary - 1]))
        if left_has_latin and right_has_latin and not left_has_hangul:
            protected.add(boundary)
    return protected


def layout_respects_protected_phrases(
    pages: Sequence[Sequence[str]], protected_phrases: Sequence[str] = ()
) -> bool:
    words: list[str] = []
    boundaries: set[int] = set()
    for page in pages:
        for line in page:
            words.extend(line.split())
            boundaries.add(len(words))
    boundaries.discard(len(words))
    return not boundaries.intersection(
        protected_boundary_indices(words, protected_phrases)
    )


def semantic_boundary_score(words: Sequence[str], boundary: int) -> int:
    """Score a break using one shared Korean semantic hierarchy.

    Higher values are better.  Hard protected boundaries are handled by the
    caller before this score is considered.
    """
    if boundary <= 0 or boundary >= len(words):
        return 0
    left = canonical_subtitle_word(words[boundary - 1])
    right = canonical_subtitle_word(words[boundary])
    raw_left = words[boundary - 1].strip()
    score = 0
    if SENTENCE_END_RE.search(raw_left):
        score = 40
    elif CLAUSE_END_RE.search(raw_left):
        score = 22
    if KOREAN_CONNECTIVE_END_RE.fullmatch(left):
        score = max(score, 30)
    elif KOREAN_SOFT_CONNECTIVE_END_RE.fullmatch(left):
        score = max(score, 24)
    elif left.endswith("고"):
        score = max(score, 8)
    if left in SEMANTIC_SEQUENCE_END_WORDS:
        score = max(score, 30)
    if KOREAN_TOPIC_END_RE.fullmatch(left):
        score = max(score, 20)
    if right in LINE_START_PREFERRED_WORDS:
        score += 5
    if left in LINE_END_AVOID_WORDS:
        score -= 20
    if left.endswith(LINE_END_AVOID_SUFFIXES):
        score -= 10
    elif len(left) >= 2 and left.endswith(("는", "던")) and score < 20:
        score -= 6
    if right in WEAK_LINE_START_WORDS:
        score -= 12
    return score


def layout_has_orphans(pages: Sequence[Sequence[str]]) -> bool:
    """Detect a detached one-word page or line in generated multiword text."""
    for page in pages:
        page_word_count = len(" ".join(page).split())
        if len(pages) > 1 and page_word_count <= 1:
            return True
        if page_word_count > 2 and any(len(line.split()) <= 1 for line in page):
            return True
    return False


def aligned_boundary_time(
    *,
    start: float,
    end: float,
    ratio: float,
    words: Sequence[Mapping[str, Any]] = (),
) -> float:
    """Prefer a source word start for a text boundary, then fall back by ratio."""
    ratio = min(1.0, max(0.0, float(ratio)))
    if ratio <= 0:
        return float(start)
    if ratio >= 1:
        return float(end)
    valid_words = [
        word
        for word in words
        if isinstance(word.get("start"), (int, float))
        and math.isfinite(float(word["start"]))
    ]
    if valid_words:
        boundary_index = min(
            len(valid_words) - 1,
            max(1, round(ratio * len(valid_words))),
        )
        boundary = float(valid_words[boundary_index]["start"])
        if float(start) < boundary < float(end):
            return boundary
    return float(start) + (float(end) - float(start)) * ratio


def split_text_intervals(
    parts: Sequence[str],
    *,
    start: float,
    end: float,
    words: Sequence[Mapping[str, Any]] = (),
) -> list[tuple[float, float]]:
    """Allocate contiguous text parts using source words when they are available."""
    if not parts:
        return []
    weights = [max(1, len(compact_subtitle_text(part))) for part in parts]
    total = sum(weights)
    boundaries = [float(start)]
    elapsed = 0
    duration = max(0.0, float(end) - float(start))
    epsilon = min(0.01, duration / (len(parts) * 2)) if duration else 0.0
    for index, weight in enumerate(weights[:-1]):
        elapsed += weight
        proportional_boundary = float(start) + duration * (elapsed / total)
        boundary = aligned_boundary_time(
            start=float(start),
            end=float(end),
            ratio=elapsed / total,
            words=words,
        )
        remaining_boundaries = len(parts) - index - 2
        minimum_boundary = boundaries[-1] + epsilon
        maximum_boundary = float(end) - epsilon * (remaining_boundaries + 1)
        if boundary < minimum_boundary or boundary > maximum_boundary:
            boundary = proportional_boundary
        boundaries.append(max(minimum_boundary, min(maximum_boundary, boundary)))
    boundaries.append(float(end))
    return list(zip(boundaries, boundaries[1:]))


def load_regression_corpus(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Subtitle layout regression corpus is unreadable: {path}") from error
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported subtitle layout regression schema_version.")
    phrases = document.get("protected_phrases")
    cases = document.get("cases")
    if (
        not isinstance(phrases, list)
        or not all(isinstance(item, str) and item.strip() for item in phrases)
        or not isinstance(cases, list)
        or not cases
    ):
        raise ValueError("Subtitle layout regression corpus is incomplete.")
    return document
