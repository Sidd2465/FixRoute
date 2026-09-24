import json
from pathlib import Path

import pytest

from app.data_loader import load_references
from app.reference_routing import (
    ReferenceSelection,
    RoutingDecision,
    SubsectionSelection,
    route_complaint,
    select_reference,
    select_subsection,
)
from scripts.demo_reference_assembly import _extract_markdown_section


REFERENCES_PATH = Path(__file__).resolve().parent.parent / "student_kit" / "siis_responses.json"


@pytest.fixture
def all_references():
    return load_references(REFERENCES_PATH)


def test_self_routing_works_for_all_twenty_references(all_references):
    failures: list[str] = []

    for reference in all_references:
        choice = select_reference(complaint=reference["original_query"], references=all_references)
        if choice.reason != "selected" or choice.selected_id != reference["id"]:
            failure = (
                f"Reference {reference['id']} failed: complaint='{reference['original_query']}', "
                f"selected_id={choice.selected_id!r}, reason={choice.reason!r}, "
                f"candidates={[(c['id'], round(float(c['score']), 6), c['title']) for c in choice.candidates]}"
            )
            failures.append(failure)

    assert not failures, "Self-routing failed for one or more references:\n" + "\n".join(failures)


def test_no_reference_match_for_non_overlapping_complaint(all_references):
    decision = select_reference(complaint="quantum gravity reactor", references=all_references)

    assert decision.reason == "no_reference_match"
    assert decision.selected_id is None
    assert decision.score is None
    assert decision.margin is None
    assert decision.candidates == []


def test_select_subsection_prefers_clear_winner():
    content = """
## Overview
A general note.

## Battery Problems
The battery drains quickly and the phone overheats.

## Wi-Fi Problems
The Wi-Fi signal is weak and the device drops connections.
"""

    choice = select_subsection(
        complaint="The phone battery drains quickly and overheats.",
        content=content,
    )

    assert choice.reason == "selected"
    assert choice.heading == "Battery Problems"
    assert choice.section_text == _extract_markdown_section(content, "Battery Problems")
    assert choice.score is not None
    assert choice.margin is not None


def test_select_subsection_rejects_near_ties():
    content = """
## First Section
Open settings to manage network and Wi-Fi.

## Second Section
Open settings to manage network and Wi-Fi.
"""

    choice = select_subsection(
        complaint="Open settings to manage network and Wi-Fi.",
        content=content,
        min_margin=0.15,
    )

    assert choice.reason == "subsection_margin_too_small"
    assert choice.heading is None
    assert choice.score is None
    assert choice.margin is not None


def test_select_subsection_no_overlap_and_single_section_rules():
    content = """
## One Section
The device firmware is up to date.
"""

    no_overlap = select_subsection(complaint="quantum gravity reactor", content=content)
    assert no_overlap.reason == "no_subsection_match"
    assert no_overlap.heading is None
    assert no_overlap.section_text is None

    single = select_subsection(complaint="device firmware is up to date", content=content)
    assert single.reason == "subsection_margin_too_small"
    assert single.heading is None
    assert single.section_text is None


def test_route_complaint_handles_no_reference_and_selected_reference(all_references):
    no_ref = route_complaint(complaint="quantum gravity reactor", references=all_references)
    assert no_ref.reference_reason == "no_reference_match"
    assert no_ref.subsection_reason == "not_attempted"
    assert no_ref.reference_id is None
    assert no_ref.subsection_heading is None
    assert no_ref.section_text is None
    assert no_ref.content is None

    row21 = next(reference for reference in all_references if reference["id"] == "row_21")
    routed = route_complaint(complaint=row21["original_query"], references=all_references)
    assert routed.reference_id == "row_21"
    assert routed.reference_reason == "selected"
    assert routed.content == row21["siis_response"]["content"]
    assert routed.subsection_reason in {"selected", "subsection_margin_too_small", "no_subsection_match"}


def test_reference_routing_validation_rejects_blank_inputs_and_bad_gates(all_references):
    with pytest.raises(ValueError, match="non-empty string"):
        select_reference(complaint="   ", references=all_references)

    with pytest.raises(ValueError, match="empty"):
        select_reference(complaint="screen issue", references=[])

    with pytest.raises(ValueError, match="min_score"):
        select_reference(complaint="screen issue", references=all_references, min_score=1.5)

    with pytest.raises(ValueError, match="min_margin"):
        select_reference(complaint="screen issue", references=all_references, min_margin=-0.1)

    with pytest.raises(ValueError, match="top_k"):
        select_reference(complaint="screen issue", references=all_references, top_k=0)

    with pytest.raises(ValueError, match="top_k"):
        select_reference(complaint="screen issue", references=all_references, top_k=True)

    with pytest.raises(ValueError, match="non-empty string"):
        select_subsection(complaint="   ", content="## One\nExample text.")

    with pytest.raises(ValueError, match="content"):
        select_subsection(complaint="screen issue", content="")


def test_routing_is_deterministic_and_non_mutating(all_references):
    complaint = all_references[0]["original_query"]
    first = route_complaint(complaint=complaint, references=all_references)
    second = route_complaint(complaint=complaint, references=all_references)
    assert first == second

    before = json.dumps(all_references, sort_keys=True)
    route_complaint(complaint=complaint, references=all_references)
    after = json.dumps(all_references, sort_keys=True)
    assert before == after


def test_real_row_21_route_invariant(all_references):
    row21 = next(reference for reference in all_references if reference["id"] == "row_21")
    routed = route_complaint(complaint=row21["original_query"], references=all_references)

    assert routed.reference_id == "row_21"
    assert routed.reference_reason == "selected"
    assert routed.subsection_reason in {"selected", "subsection_margin_too_small", "no_subsection_match", "not_attempted"}
    assert routed.content == row21["siis_response"]["content"]

    if routed.subsection_reason == "selected":
        assert routed.section_text == _extract_markdown_section(
            routed.content,
            routed.subsection_heading,
        )
        scores = [item["score"] for item in routed.subsection_candidates]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
