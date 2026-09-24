"""Demonstration script for the FixRoute planning layer."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from app.data_loader import load_deeplinks
from app.extraction_schemas import ExtractionResult
from app.planning import build_planned_pipeline, generate_plan_parameters
from app.planning_schemas import PlanningParameters
from app.rule_validation import is_fatal_code

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "row21_extraction.json"
DEEPLINKS_PATH = PROJECT_ROOT / "student_kit" / "deeplinks.json"

DEFAULT_COMPLAINT = (
    "My Galaxy S22 screen inputs are delayed and the touch responsiveness is laggy, "
    "causing a noticeable delay when I try to interact with the phone."
)


def _load_api_key() -> str:
    from dotenv import dotenv_values

    values = dotenv_values(PROJECT_ROOT / ".env")
    key = values.get("GEMINI_API_KEY")
    if key is None or not str(key).strip():
        raise RuntimeError("Missing GEMINI_API_KEY in the project-root .env file.")
    os.environ.setdefault("GEMINI_API_KEY", str(key).strip())
    return str(key).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo FixRoute planning pipeline.")
    parser.add_argument("--run-live", action="store_true", help="Run live Gemini call.")
    parser.add_argument("--query", default=DEFAULT_COMPLAINT, help="User complaint.")
    args = parser.parse_args()

    # Load extraction fixture
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        extraction = ExtractionResult.model_validate(json.load(f))

    # Load real catalogue
    catalogue = load_deeplinks(DEEPLINKS_PATH)

    if args.run_live:
        print("Running LIVE Gemini planning call...")
        api_key = _load_api_key()
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        start = time.perf_counter()
        try:
            params = generate_plan_parameters(args.query, extraction, client=client)
        finally:
            try:
                client.close()
            except Exception:
                pass
        elapsed = time.perf_counter() - start
        print(f"Planning LLM call elapsed: {elapsed:.2f}s")
    else:
        print("Running OFFLINE planning demo with calibrated parameters...")
        params = PlanningParameters(
            goal_topic="Full Screen Gesture Function",
            goal_title="Full screen gestures",
            action_name="Disable Full Screen Gestures",
            action_description="It will disable full screen gestures.",
            link_query="navigation bar settings",
        )

    print()
    print("--- Planning Parameters ---")
    print(params.model_dump_json(indent=2))

    response, link_decision, violations = build_planned_pipeline(
        complaint=args.query,
        extraction=extraction,
        params=params,
        deeplink_catalogue=catalogue,
    )

    print()
    print("--- Link Decision ---")
    print(link_decision.model_dump_json(indent=2))

    print()
    print("--- Rule Violations ---")
    fatal = [v for v in violations if is_fatal_code(v.code)]
    non_fatal = [v for v in violations if not is_fatal_code(v.code)]
    print(f"Total violations: {len(violations)} (Fatal: {len(fatal)}, Non-fatal: {len(non_fatal)})")
    for v in violations:
        tag = "[FATAL]" if is_fatal_code(v.code) else "[NON-FATAL]"
        print(f"  {tag} {v.code}: {v.message}")

    print()
    print("--- Assembled ContextDeeplinkResponse JSON ---")
    print(response.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())