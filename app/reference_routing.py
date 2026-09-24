"""Deterministic offline routing for SIIS complaints.

This module mirrors the repo's lexical retrieval style: English stopwords,
ngram_range=(1, 2), weighted cosine similarity (0.8 primary / 0.2 secondary),
positive scores only, stable ID tie-breaks, and deep-copied plain-dict outputs.
It intentionally stays offline and never mutates the caller's inputs.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _flatten_reference(reference: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested SIIS response data without changing the loader."""
    flattened = copy.deepcopy(reference)
    siis_response = flattened.get("siis_response")
    if isinstance(siis_response, dict):
        if "title" not in flattened and isinstance(siis_response.get("title"), str):
            flattened["title"] = siis_response["title"]
        if "content" not in flattened and isinstance(siis_response.get("content"), str):
            flattened["content"] = siis_response["content"]
    return flattened


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _collect_section_texts(content: str) -> dict[str, str]:
    """Return the exact markdown section contents for every level-2 heading."""
    if not isinstance(content, str):
        return {}

    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        if re.match(r"^#{1,2}\s+", line):
            if current_heading is not None and current_lines:
                sections[current_heading] = "\n".join(current_lines).rstrip() + "\n"
            current_heading = None
            current_lines = []

            if line.startswith("## ") and not line.startswith("###"):
                candidate = line[3:].strip()
                if candidate and not candidate.startswith("#"):
                    current_heading = candidate
                    current_lines = [line]
            continue

        if current_heading is not None:
            current_lines.append(line)

    if current_heading is not None and current_lines:
        sections[current_heading] = "\n".join(current_lines).rstrip() + "\n"

    return sections


def _extract_markdown_section(text: str, heading_text: str) -> str:
    """Copy of the reviewed section helper so routing remains offline and exact."""
    if not isinstance(text, str):
        raise ValueError("Markdown content must be a string.")

    lines = text.splitlines()
    heading_pattern = re.compile(r"^(#{1,6})\s*(.+?)\s*$")
    target_index: int | None = None
    target_level: int | None = None

    for index, line in enumerate(lines):
        match = heading_pattern.match(line)
        if match is None:
            continue
        raw_heading = match.group(0).strip()
        body_heading = match.group(2).strip()
        normalized_target = heading_text.strip()
        if raw_heading == normalized_target or body_heading == normalized_target.lstrip("#").strip():
            if target_index is not None:
                raise ValueError(f"Ambiguous markdown section heading: {heading_text!r}")
            target_index = index
            target_level = len(match.group(1))

    if target_index is None or target_level is None:
        raise ValueError(f"Missing markdown section: {heading_text!r}")

    section_lines: list[str] = []
    for index in range(target_index, len(lines)):
        if index > target_index:
            candidate = heading_pattern.match(lines[index])
            if candidate is not None and len(candidate.group(1)) <= target_level:
                break
        section_lines.append(lines[index])

    return "\n".join(section_lines).rstrip() + "\n"


class ReferenceRetriever:
    """Lexical retrieval over reference original_query and title+content."""

    def __init__(self, references: list[dict[str, Any]]):
        if not isinstance(references, list):
            raise ValueError("references must be a list of dictionaries.")
        if not references:
            raise ValueError("references list cannot be empty.")

        self.references = [
            _flatten_reference(copy.deepcopy(reference))
            for reference in references
        ]
        self.primary_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
        )
        self.secondary_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
        )

        self.primary_documents = [self._primary_text(ref) for ref in self.references]
        self.secondary_documents = [self._secondary_text(ref) for ref in self.references]
        self.primary_matrix = self.primary_vectorizer.fit_transform(self.primary_documents)
        self.secondary_matrix = self.secondary_vectorizer.fit_transform(self.secondary_documents)

    @staticmethod
    def _primary_text(reference: dict[str, Any]) -> str:
        return _clean_text(reference.get("original_query"))

    @staticmethod
    def _secondary_text(reference: dict[str, Any]) -> str:
        title = _clean_text(reference.get("title"))
        content = _clean_text(reference.get("content"))
        payload = " ".join(part for part in (title, content) if part)
        return payload

    def _score_candidates(self, query: str) -> list[dict[str, Any]]:
        if not isinstance(query, str):
            return []

        cleaned = query.strip()
        if not cleaned:
            return []

        query_primary = self.primary_vectorizer.transform([cleaned])
        query_secondary = self.secondary_vectorizer.transform([cleaned])

        primary_scores = cosine_similarity(query_primary, self.primary_matrix).ravel()
        secondary_scores = cosine_similarity(query_secondary, self.secondary_matrix).ravel()

        ranked: list[dict[str, Any]] = []
        for index, reference in enumerate(self.references):
            primary_score = float(primary_scores[index])
            secondary_score = float(secondary_scores[index])
            score = 0.8 * primary_score + 0.2 * secondary_score
            if score <= 0:
                continue

            ranked.append(
                {
                    "id": str(reference.get("id", "")),
                    "score": float(score),
                    "title": _clean_text(reference.get("title")) or _clean_text(reference.get("original_query")),
                    "reference": copy.deepcopy(reference),
                }
            )

        ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return ranked

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if not isinstance(query, str):
            return []

        cleaned = query.strip()
        if not cleaned:
            return []

        return [
            {
                "id": item["id"],
                "score": float(item["score"]),
                "title": item["title"],
                "reference": copy.deepcopy(item["reference"]),
            }
            for item in self._score_candidates(cleaned)[:top_k]
        ]


@dataclass(frozen=True)
class ReferenceSelection:
    selected_id: str | None
    score: float | None
    margin: float | None
    reason: str
    candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SubsectionSelection:
    heading: str | None
    score: float | None
    margin: float | None
    reason: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    section_text: str | None = None


@dataclass(frozen=True)
class RoutingDecision:
    reference_id: str | None
    reference_score: float | None
    reference_margin: float | None
    reference_reason: str
    subsection_heading: str | None
    subsection_score: float | None
    subsection_margin: float | None
    subsection_reason: str
    subsection_candidates: list[dict[str, Any]] = field(default_factory=list)
    section_text: str | None = None
    content: str | None = None
    reference_candidates: list[dict[str, Any]] = field(default_factory=list)


def _validate_gate_inputs(
    *,
    complaint: str,
    min_score: float,
    min_margin: float,
    top_k: int,
    label: str,
) -> None:
    if isinstance(complaint, bool) or not isinstance(complaint, str) or not complaint.strip():
        raise ValueError(f"{label} must be a non-empty string.")

    for name, value, lower, upper in (
        ("min_score", min_score, 0.0, 1.0),
        ("min_margin", min_margin, 0.0, 1.0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number within [{lower}, {upper}].")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < lower or numeric > upper:
            raise ValueError(f"{name} must be a finite number within [{lower}, {upper}].")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")


def select_reference(
    *,
    complaint: str,
    references: list[dict[str, Any]],
    min_score: float = 0.5,
    min_margin: float = 0.10,
    top_k: int = 3,
) -> ReferenceSelection:
    """Select the best matching reference using the same gate semantics as link mapping."""
    if not isinstance(references, list):
        raise ValueError("references must be a list of dictionaries.")
    if not references:
        raise ValueError("references list cannot be empty.")

    _validate_gate_inputs(
        complaint=complaint,
        min_score=min_score,
        min_margin=min_margin,
        top_k=top_k,
        label="complaint",
    )

    retriever = ReferenceRetriever(references)
    ranked = retriever.search(complaint, top_k=top_k)
    if not ranked:
        return ReferenceSelection(
            selected_id=None,
            score=None,
            margin=None,
            reason="no_reference_match",
            candidates=[],
        )

    candidates = [
        {
            "id": item["id"],
            "score": float(item["score"]),
            "title": item["title"],
        }
        for item in ranked
    ]
    top1_score = float(candidates[0]["score"])

    # Deliberate conservative rule: with only one candidate there is no
    # comparative evidence, so the margin is treated as 0 and the gate fails.
    # Do not change this to margin = top1_score.
    if len(candidates) == 1:
        return ReferenceSelection(
            selected_id=None,
            score=float(top1_score),
            margin=0.0,
            reason="reference_margin_too_small",
            candidates=candidates,
        )

    second_score = float(candidates[1]["score"])
    margin = top1_score - second_score
    if margin < float(min_margin):
        return ReferenceSelection(
            selected_id=None,
            score=float(top1_score),
            margin=float(margin),
            reason="reference_margin_too_small",
            candidates=candidates,
        )

    if top1_score < float(min_score):
        return ReferenceSelection(
            selected_id=None,
            score=float(top1_score),
            margin=float(margin),
            reason="no_reference_match",
            candidates=candidates,
        )

    return ReferenceSelection(
        selected_id=str(candidates[0]["id"]),
        score=float(candidates[0]["score"]),
        margin=float(margin),
        reason="selected",
        candidates=candidates,
    )


def select_subsection(
    *,
    complaint: str,
    content: str,
    min_score: float = 0.25,
    min_margin: float = 0.05,
    top_k: int = 3,
) -> SubsectionSelection:
    """Select the strongest matching '## ' section in a reference's content."""
    if isinstance(content, bool) or not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string.")

    _validate_gate_inputs(
        complaint=complaint,
        min_score=min_score,
        min_margin=min_margin,
        top_k=top_k,
        label="complaint",
    )

    sections = _collect_section_texts(content)
    if not sections:
        return SubsectionSelection(
            heading=None,
            score=None,
            margin=None,
            reason="no_subsection_match",
            candidates=[],
            section_text=None,
        )

    primary_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )
    secondary_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
    )
    primary_documents = list(sections.keys())
    secondary_documents = list(sections.values())
    primary_matrix = primary_vectorizer.fit_transform(primary_documents)
    secondary_matrix = secondary_vectorizer.fit_transform(secondary_documents)

    query_primary = primary_vectorizer.transform([complaint])
    query_secondary = secondary_vectorizer.transform([complaint])
    primary_scores = cosine_similarity(query_primary, primary_matrix).ravel()
    secondary_scores = cosine_similarity(query_secondary, secondary_matrix).ravel()

    ranked: list[dict[str, Any]] = []
    for index, heading in enumerate(sections):
        section_text = sections[heading]
        primary_score = float(primary_scores[index])
        secondary_score = float(secondary_scores[index])
        score = 0.8 * primary_score + 0.2 * secondary_score
        ranked.append(
            {
                "heading": heading,
                "score": float(score),
                "section_text": section_text,
            }
        )

    ranked.sort(key=lambda item: (-float(item["score"]), str(item["heading"])))
    candidates = [{"heading": item["heading"], "score": float(item["score"])} for item in ranked[:top_k]]

    if not any(float(item["score"]) > 0.0 for item in candidates):
        return SubsectionSelection(
            heading=None,
            score=None,
            margin=None,
            reason="no_subsection_match",
            candidates=candidates,
            section_text=None,
        )

    top1_score = float(candidates[0]["score"])
    if len(candidates) == 1:
        return SubsectionSelection(
            heading=None,
            score=None,
            margin=0.0,
            reason="subsection_margin_too_small",
            candidates=candidates,
            section_text=None,
        )

    second_score = float(candidates[1]["score"])
    margin = top1_score - second_score
    if margin < float(min_margin):
        return SubsectionSelection(
            heading=None,
            score=None,
            margin=float(margin),
            reason="subsection_margin_too_small",
            candidates=candidates,
            section_text=None,
        )

    if top1_score < float(min_score):
        return SubsectionSelection(
            heading=None,
            score=None,
            margin=float(margin),
            reason="no_subsection_match",
            candidates=candidates,
            section_text=None,
        )

    heading = str(candidates[0]["heading"])
    section_text = _extract_markdown_section(content, heading)
    return SubsectionSelection(
        heading=heading,
        score=float(top1_score),
        margin=float(margin),
        reason="selected",
        candidates=candidates,
        section_text=section_text,
    )


def route_complaint(
    *,
    complaint: str,
    references: list[dict[str, Any]],
    min_score: float = 0.5,
    min_margin: float = 0.10,
    top_k: int = 3,
) -> RoutingDecision:
    """Route a complaint to a reference and an optional subsection."""
    if not isinstance(references, list):
        raise ValueError("references must be a list of dictionaries.")
    if not references:
        raise ValueError("references list cannot be empty.")

    _validate_gate_inputs(
        complaint=complaint,
        min_score=min_score,
        min_margin=min_margin,
        top_k=top_k,
        label="complaint",
    )

    reference_decision = select_reference(
        complaint=complaint,
        references=references,
        min_score=min_score,
        min_margin=min_margin,
        top_k=top_k,
    )
    if reference_decision.reason != "selected":
        return RoutingDecision(
            reference_id=None,
            reference_score=None,
            reference_margin=None,
            reference_reason=reference_decision.reason,
            subsection_heading=None,
            subsection_score=None,
            subsection_margin=None,
            subsection_reason="not_attempted",
            subsection_candidates=[],
            section_text=None,
            content=None,
            reference_candidates=reference_decision.candidates,
        )

    selected_reference = None
    flattened_references = [_flatten_reference(copy.deepcopy(reference)) for reference in references]
    for reference in flattened_references:
        if str(reference.get("id")) == str(reference_decision.selected_id):
            selected_reference = reference
            break
    if selected_reference is None:
        raise ValueError(f"Reference {reference_decision.selected_id!r} was not found in the supplied catalogue.")

    content = _clean_text(selected_reference.get("content")) or _clean_text(selected_reference.get("siis_response", {}).get("content"))
    subsection_decision = select_subsection(
        complaint=complaint,
        content=content,
        min_score=min_score,
        min_margin=min_margin,
        top_k=top_k,
    )

    return RoutingDecision(
        reference_id=str(reference_decision.selected_id),
        reference_score=float(reference_decision.score) if reference_decision.score is not None else None,
        reference_margin=float(reference_decision.margin) if reference_decision.margin is not None else None,
        reference_reason=reference_decision.reason,
        subsection_heading=subsection_decision.heading,
        subsection_score=float(subsection_decision.score) if subsection_decision.score is not None else None,
        subsection_margin=float(subsection_decision.margin) if subsection_decision.margin is not None else None,
        subsection_reason=subsection_decision.reason,
        subsection_candidates=subsection_decision.candidates,
        section_text=subsection_decision.section_text,
        content=content,
        reference_candidates=reference_decision.candidates,
    )
