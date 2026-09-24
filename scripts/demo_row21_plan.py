"""Offline demo for the row_21 extraction-to-plan flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_loader import load_deeplinks, load_references
from app.extraction_schemas import ExtractionResult
from app.link_mapping import apply_link_decision, map_links_for_group
from app.plan_from_extraction import build_extraction_plan
from app.response_builder import ResponseBuildError, build_response
from app.rule_validation import is_fatal_code, validate_response_rules

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "row21_extraction.json"
REFERENCES_PATH = PROJECT_ROOT / "student_kit" / "siis_responses.json"
DEEPLINKS_PATH = PROJECT_ROOT / "student_kit" / "deeplinks.json"

# In the final product the planner derives the query from complaint/extraction;
# it is never read from the catalogue.
ATTACH_QUERY = "navigation bar settings"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline row_21 extraction plan demo.")
    parser.add_argument("--attach-links", action="store_true", help="Optionally attach the top valid deeplink candidate for the row_21 plan.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)
    extraction = ExtractionResult.model_validate(fixture)

    references = load_references(REFERENCES_PATH)
    complaint = next(entry["original_query"] for entry in references if entry.get("id") == "row_21")

    plan = build_extraction_plan(
        extraction=extraction,
        goal_topic="Full Screen Gesture Function",
        goal_title="Full screen gestures",
        action_name="Disable Full Screen Gestures",
        action_description="It will disable full screen gestures.",
        score=0.0,
    )

    if args.attach_links:
        catalogue = load_deeplinks(DEEPLINKS_PATH)
        decision = map_links_for_group(query=ATTACH_QUERY, catalogue=catalogue)
        plan = apply_link_decision(plan=plan, decision=decision)
        try:
            assembled = build_response([plan], catalogue)
        except ResponseBuildError as exc:
            print(f"Builder rejected attached plan: {exc}")
            raise SystemExit(1) from exc
        violations = validate_response_rules(assembled)
        fatal = [v for v in violations if is_fatal_code(v.code)]
        nonfatal = [v for v in violations if v not in fatal]

        print(complaint)
        print()
        print(plan.model_dump_json(indent=2))
        print()
        print(assembled.model_dump_json(indent=2))
        print()
        print(json.dumps({
            "attached": decision.attached,
            "reason": decision.reason,
            "catalogue_id": decision.catalogue_id,
            "chosen_score": decision.chosen_score,
            "margin": decision.margin,
            "candidates": decision.candidates,
        }, indent=2))
        print()
        print(f"Fatal findings: {len(fatal)}")
        print(f"Non-fatal findings: {len(nonfatal)}")
        print()
        if decision.attached:
            print(f"Full-screen-gestures condition from the source is not established by the original complaint. Attached deeplink ID: {decision.catalogue_id}. Offline demonstration only; not a benchmark and not an approved troubleshooting plan.")
        else:
            print("Full-screen-gestures condition from the source is not established by the original complaint. No deeplinks are attached in this version. Offline demonstration only; not a benchmark and not an approved troubleshooting plan.")
        return

    try:
        assembled = build_response([plan], [])
    except ResponseBuildError as exc:
        print(f"Builder rejected manual group without catalogue_id: {exc}")
        raise SystemExit(1) from exc

    violations = validate_response_rules(assembled)
    fatal = [v for v in violations if is_fatal_code(v.code)]
    nonfatal = [v for v in violations if v not in fatal]

    print(complaint)
    print()
    print(plan.model_dump_json(indent=2))
    print()
    print(assembled.model_dump_json(indent=2))
    print()
    print(f"Fatal findings: {len(fatal)}")
    print(f"Non-fatal findings: {len(nonfatal)}")
    print()
    print("Full-screen-gestures condition from the source is not established by the original complaint. No deeplinks are attached in this version. Offline demonstration only; not a benchmark and not an approved troubleshooting plan.")


if __name__ == "__main__":
    main()
