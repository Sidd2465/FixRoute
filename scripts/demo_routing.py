"""Offline complaint routing demo using the SIIS reference catalogue."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_loader import load_references
from app.reference_routing import route_complaint, select_reference

REFERENCES_PATH = PROJECT_ROOT / "student_kit" / "siis_responses.json"


def _default_complaint(references: list[dict]) -> str:
    for entry in references:
        if entry.get("id") == "row_21":
            return str(entry.get("original_query", "")).strip()
    raise ValueError("Reference row_21 was not found in the SIIS catalogue.")


def _emit_table(rows: list[tuple[str, float]], columns: tuple[str, str]) -> None:
    if not rows:
        print(f"{columns[0]:<20} {columns[1]}")
        print("-" * 35)
        print("No rows")
        return

    header_left = columns[0]
    header_right = columns[1]
    width_left = max(len(header_left), max(len(str(item[0])) for item in rows))
    width_right = max(len(header_right), max(len(f"{item[1]:.6f}") for item in rows))
    print(f"{header_left:<{width_left}} {header_right:>{width_right}}")
    print("-" * (width_left + width_right + 1))
    for key, value in rows:
        print(f"{str(key):<{width_left}} {value:>{width_right}.6f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline complaint routing demo.")
    parser.add_argument("--complaint", help="Override the default row_21 complaint.")
    args = parser.parse_args(argv)

    references = load_references(REFERENCES_PATH)
    complaint = args.complaint if args.complaint is not None else _default_complaint(references)

    reference_decision = select_reference(complaint=complaint, references=references)
    decision = route_complaint(complaint=complaint, references=references)

    routing = {
        "reference_selection": {
            "selected_id": reference_decision.selected_id,
            "score": reference_decision.score,
            "margin": reference_decision.margin,
            "reason": reference_decision.reason,
            "candidates": reference_decision.candidates,
        },
        "subsection_selection": {
            "heading": decision.subsection_heading,
            "score": decision.subsection_score,
            "margin": decision.subsection_margin,
            "reason": decision.subsection_reason,
            "candidates": decision.subsection_candidates,
            "section_text": decision.section_text,
        },
    }

    print(f"Complaint: {complaint}")
    print()
    print("Reference top-3 candidates:")
    rows = [(item["id"], float(item["score"])) for item in reference_decision.candidates[:3]]
    _emit_table(rows, ("id", "score"))
    print()
    print("ReferenceSelection:")
    print(json.dumps(routing["reference_selection"], ensure_ascii=False, indent=2))
    print()
    print("Subsection top-3 candidates:")
    rows = [(item["heading"], float(item["score"])) for item in decision.subsection_candidates[:3]]
    _emit_table(rows, ("heading", "score"))
    print()
    print("SubsectionSelection:")
    print(json.dumps(routing["subsection_selection"], ensure_ascii=False, indent=2))
    print()
    print(f"Reference selected: {decision.reference_id}")
    print(f"Reference content length: {len(decision.content or '')}")
    preview = (decision.content or '')[:250]
    print(f"Reference preview: {preview}")
    if decision.section_text is not None:
        print("Selected section_text:")
        print(decision.section_text)
    elif decision.subsection_reason == "not_attempted":
        print(f"Subsection selection was not attempted because the reference was not selected. Reason: {decision.reference_reason}")
    else:
        print(f"Subsection selection reason: {decision.subsection_reason}")

    print()
    print("Lexical routing only. Complaints below the gate produce no_match rather than a silent guess. Offline demonstration only; not a benchmark and not an approved troubleshooting plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
