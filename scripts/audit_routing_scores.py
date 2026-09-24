"""Offline routing audit for FixRoute.

Runs every SIIS self-case (each reference's own original_query) through the
reference and subsection gates, reports score/margin distributions and a
subsection threshold calibration grid, and verifies on real data that the
section-ranking text, the routing module's section helper, and the reviewed
section helper from scripts/demo_reference_assembly all agree.

Fully offline: no API calls, no .env, no keys. Read-only on project data.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.data_loader import load_references
from app.reference_routing import (
    _collect_section_texts,
    _extract_markdown_section,
    route_complaint,
    select_subsection,
)
from scripts.demo_reference_assembly import (
    _extract_markdown_section as reviewed_extract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = PROJECT_ROOT / "student_kit" / "siis_responses.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "routing_audit.md"

DEFAULT_MIN_SCORE = 0.5
DEFAULT_MIN_MARGIN = 0.10
GRID_MIN_SCORES = (0.5, 0.3, 0.25, 0.2)
GRID_MIN_MARGINS = (0.10, 0.05, 0.02, 0.01)


def main() -> int:
    references = load_references(REFERENCES_PATH)
    if len(references) != 20:
        print(f"ERROR: expected 20 references, got {len(references)}.")
        return 1

    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    out("# FixRoute routing audit (offline)")
    out()
    out(f"References: {len(references)}")
    out(
        "Default gates: min_score=0.5, min_margin=0.10, top_k=3 "
        "(a single-candidate gate can never pass by design)"
    )
    out()

    ref_passes = 0
    sub_passes = 0
    violations: list[str] = []

    for ref in references:
        rid = str(ref["id"])
        content = ref["siis_response"]["content"]
        sections = _collect_section_texts(content)

        decision = route_complaint(
            complaint=ref["original_query"],
            references=references,
            min_score=DEFAULT_MIN_SCORE,
            min_margin=DEFAULT_MIN_MARGIN,
            top_k=3,
        )
        if decision.reference_reason == "selected":
            ref_passes += 1
        if decision.subsection_reason == "selected":
            sub_passes += 1

        out(f"## {rid} (sections={len(sections)}, content_chars={len(content)})")
        out()
        out("Reference top-3 (id, score):")
        for cand in decision.reference_candidates[:3]:
            out(f"  {str(cand['id']):<10} {float(cand['score']):.6f}")
        out(
            "Reference decision: "
            f"{decision.reference_reason} | selected={decision.reference_id} | "
            f"score={decision.reference_score} | margin={decision.reference_margin}"
        )
        out()
        out("Subsection top-3 (heading, score):")
        for cand in decision.subsection_candidates[:3]:
            out(f"  {str(cand['heading']):<45} {float(cand['score']):.6f}")
        out(
            "Subsection decision: "
            f"{decision.subsection_reason} | heading={decision.subsection_heading!r} | "
            f"score={decision.subsection_score} | margin={decision.subsection_margin}"
        )
        out()
        sections = _collect_section_texts(content)
        section_items = sections.items() if isinstance(sections, dict) else sections
        for heading, ranking_text in sections:
            try:
                module_text = _extract_markdown_section(content, heading)
            except ValueError as exc:
                violations.append(
                    f"{rid}: module helper failed for {heading!r}: {exc}"
                )
                continue
            if module_text != ranking_text:
                violations.append(
                    f"{rid}: ranking text != extracted text for {heading!r} "
                    f"(ranking_chars={len(ranking_text)}, extracted_chars={len(module_text)})"
                )
            try:
                reviewed_text = reviewed_extract(content, heading)
            except ValueError as exc:
                violations.append(
                    f"{rid}: reviewed helper failed for {heading!r}: {exc}"
                )
                continue
            if reviewed_text != module_text:
                violations.append(
                    f"{rid}: module helper drifted from reviewed helper for {heading!r}"
                )
        if not sections:
            violations.append(f"{rid}: no '## ' sections found in content")

    out("## Subsection threshold calibration grid")
    out()
    out("Self-case passes (out of 20) at each (min_score, min_margin):")
    out()
    out(
        "| min_score \\ min_margin | "
        + " | ".join(f"{m:.2f}" for m in GRID_MIN_MARGINS)
        + " |"
    )
    out("|" + "---|" * (len(GRID_MIN_MARGINS) + 1))
    grid_passers: dict[tuple[float, float], list[str]] = {}
    for s in GRID_MIN_SCORES:
        row = [f"**{s:.2f}**"]
        for m in GRID_MIN_MARGINS:
            passers: list[str] = []
            for ref in references:
                sel = select_subsection(
                    complaint=ref["original_query"],
                    content=ref["siis_response"]["content"],
                    min_score=s,
                    min_margin=m,
                    top_k=3,
                )
                if sel.reason == "selected":
                    passers.append(f"{ref['id']}:{sel.heading!r}")
            grid_passers[(s, m)] = passers
            row.append(str(len(passers)))
        out("| " + " | ".join(row) + " |")
    out()
    out("### Grid passers detail (combinations where 1-19 pass)")
    out()
    any_detail = False
    for (s, m), passers in grid_passers.items():
        if 0 < len(passers) < 20:
            any_detail = True
            out(
                f"- min_score={s:.2f}, min_margin={m:.2f} -> {len(passers)}: "
                + "; ".join(passers)
            )
    if not any_detail:
        out("- none")
    out()
    out("Single-section references can never pass any grid point (conservative rule).")
    out()

    out("## Summary")
    out()
    out(f"Reference gate passes at defaults: {ref_passes}/20")
    out(f"Subsection gate passes at defaults: {sub_passes}/20")
    out(f"Invariant violations: {len(violations)}")
    for violation in violations:
        out(f"VIOLATION: {violation}")
    out()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())