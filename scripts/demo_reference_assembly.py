"""Real-data integration demonstration for the FixRoute retriever and assembler.

This script intentionally stays narrow: it loads the official data files, confirms a
real reference excerpt and deeplink candidate match, and assembles a reviewed plan
only when an explicit flag is set. It is not a general-purpose troubleshooting engine
and does not claim source-grounded correctness beyond the tested components.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_loader import load_deeplinks, load_references
from app.plan_schemas import ActionPlan, GoalPlan, StepGroupPlan
from app.response_builder import ResponseBuildError, build_response
from app.retrieval import DeeplinkRetriever


REFERENCES_PATH = PROJECT_ROOT / "student_kit" / "siis_responses.json"
DEEPLINKS_PATH = PROJECT_ROOT / "student_kit" / "deeplinks.json"


def _extract_markdown_section(text: str, heading_text: str) -> str:
    """Return the exact markdown section beginning at the given heading.

    The section starts at the matching heading and stops before the next heading
    of the same or higher level; no fixed-size slicing window is used.
    """
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


def get_row_21_excerpt() -> str:
    """Return the exact reviewed SIIS excerpt for the full-screen-gesture section."""
    references = load_references(REFERENCES_PATH)
    for entry in references:
        if entry.get("id") == "row_21":
            content = entry["siis_response"]["content"]
            heading = "## 4. Full Screen Gesture Function"
            return _extract_markdown_section(content, heading)
    raise ValueError("Reference row_21 was not found in the SIIS catalogue.")


def _reviewed_plan() -> GoalPlan:
    return GoalPlan(
        goal="Follow these steps to perform this Navigation Configuration",
        title="Navigation settings",
        score=0.0,
        actions=[
            ActionPlan(
                actionName="Configure Navigation Bar",
                description="It will switch navigation to buttons.",
                category="auto",
                stepGroups=[
                    StepGroupPlan(
                        steps=[
                            "Open Settings.",
                            "Tap Display.",
                            "Tap Navigation bar.",
                            "Select Buttons.",
                        ],
                        catalogue_id="DL-0169",
                    )
                ],
            )
        ],
    )


def assemble_demo_response(*, uses_full_screen_gestures: bool) -> dict[str, Any]:
    """Assemble the real-data demo only when the prerequisite is explicitly confirmed."""
    if not uses_full_screen_gestures:
        raise ValueError(
            "The demo requires an explicit --uses-full-screen-gestures flag because the full-screen gesture prerequisite is not confirmed."
        )

    references = load_references(REFERENCES_PATH)
    deeplinks = load_deeplinks(DEEPLINKS_PATH)
    retriever = DeeplinkRetriever(deeplinks)

    ranked = retriever.search("Open Navigation bar settings")
    candidates = [
        {"id": item["entry"]["id"], "similarity_score": float(item["similarity_score"])}
        for item in ranked
    ]
    candidate_ids = [item["id"] for item in candidates]
    if not candidate_ids:
        raise ValueError("The retriever returned no candidates for the navigation-bar query.")
    if "DL-0169" not in candidate_ids:
        raise ValueError(f"Expected DL-0169 in retriever candidates, got {candidate_ids[:5]}")

    excerpt = get_row_21_excerpt()
    if "Full Screen Gesture Function" not in excerpt:
        raise ValueError("The reviewed row_21 excerpt is missing the expected full-screen-gesture section.")

    response = build_response([_reviewed_plan()], deeplinks)
    payload = {
        "reference_id": "row_21",
        "reference_excerpt": excerpt,
        "selected_deeplink_id": "DL-0169",
        "candidate_ids": candidate_ids[:3],
        "candidate_scores": [round(item["similarity_score"], 6) for item in candidates[:3]],
        "response": response,
    }
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FixRoute real-data reference assembly demo.")
    parser.add_argument(
        "--uses-full-screen-gestures",
        action="store_true",
        help="Explicitly confirm the full-screen-gesture prerequisite before assembling the plan.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        assembled = assemble_demo_response(uses_full_screen_gestures=args.uses_full_screen_gestures)
    except (ValueError, ResponseBuildError) as exc:
        print(f"Demo assembly skipped: {exc}")
        raise SystemExit(1) from exc

    print("Human-reviewed subsection assembly test.")
    print("No automated extraction. Not a complete answer to the original complaint.")

    complaint = next(
        entry["original_query"]
        for entry in load_references(REFERENCES_PATH)
        if entry.get("id") == "row_21"
    )
    print("\nOriginal complaint:")
    print(complaint)

    if args.uses_full_screen_gestures:
        print("\nDemo assumption: the user uses full-screen gestures.")
        print("This is not established by the original complaint.")

    print("\nReference row_21 excerpt:")
    print(assembled["reference_excerpt"])

    print("\nRetriever top candidates:")
    for candidate_id, similarity_score in zip(assembled["candidate_ids"], assembled["candidate_scores"]):
        print(f"- {candidate_id}: {similarity_score}")

    print("\nAssembled response JSON:")
    print(json.dumps(assembled["response"].model_dump(mode="python"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
