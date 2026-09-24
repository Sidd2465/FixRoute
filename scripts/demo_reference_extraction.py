"""Manual source-grounded reference extraction demo for FixRoute.

This script intentionally does not connect to the response builder, and it only
performs a live API call when the user explicitly passes --run-live.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_loader import load_references
from app.extraction import extract_from_reference
from scripts.demo_reference_assembly import _extract_markdown_section

REFERENCES_PATH = PROJECT_ROOT / "student_kit" / "siis_responses.json"
TARGET_SECTION_HEADING = "## 4. Full Screen Gesture Function"


def _load_api_key() -> str:
    """Load the API key without overriding an already-set environment value."""
    from dotenv import load_dotenv

    load_dotenv(
        PROJECT_ROOT / ".env",
        override=False,
    )

    key = os.environ.get("GEMINI_API_KEY")
    if key is None or not key.strip():
        raise RuntimeError("Missing GEMINI_API_KEY in the environment or project-root .env file.")

    return key.strip()


def _open_google_client(api_key: str) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def get_full_screen_gesture_section(reference_id: str = "row_21") -> str:
    references = load_references(REFERENCES_PATH)
    for entry in references:
        if entry.get("id") == reference_id:
            content = entry["siis_response"]["content"]
            return _extract_markdown_section(content, TARGET_SECTION_HEADING)
    raise ValueError(f"Reference {reference_id!r} was not found in the SIIS catalogue.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manual source-grounded extraction demo for row_21."
    )
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="Require explicit confirmation before contacting the external Google API.",
    )
    args = parser.parse_args(argv)

    total_start = time.perf_counter()

    if not args.run_live:
        print("Live extraction is disabled.")
        print("This demo will not run without --run-live.")
        print("Run with --run-live to send the reference text to the Google model API.")
        print("Command: ~/.venvs/fixroute-check/bin/python -m scripts.demo_reference_extraction --run-live")
        return 0

    try:
        api_key = _load_api_key()
    except Exception as exc:
        print("Status: KEY_ERROR")
        print(f"Explanation: {exc}")
        return 1

    try:
        reference_entry = next(
            entry for entry in load_references(REFERENCES_PATH) if entry.get("id") == "row_21"
        )
        reference_id = reference_entry["id"]
        complaint = reference_entry["original_query"]
        source_section = get_full_screen_gesture_section(reference_id=reference_id)
    except Exception as exc:
        print("Status: REFERENCE_ERROR")
        print(f"Explanation: {exc}")
        return 1

    extraction_start = time.perf_counter()
    try:
        client = _open_google_client(api_key)
        try:
            result = extract_from_reference(
                source_section,
                client=client,
                model_name="gemini-3.6-flash",
            )
        finally:
            try:
                client.close()
            except Exception:
                pass
    except Exception as exc:
        underlying = exc.__cause__ if exc.__cause__ is not None else exc
        print("Status: EXTRACTION_FAILED")
        print(f"Error type: {type(underlying).__name__}")

        code = getattr(underlying, "code", None)
        if isinstance(code, int):
            print(f"API status code: {code}")

        message = getattr(underlying, "message", None)
        if isinstance(message, str):
            safe_message = message.replace(api_key, "[REDACTED]")
            print(f"Server explanation: {safe_message[:1000]}")
        else:
            fallback = str(underlying)
            if api_key:
                fallback = fallback.replace(api_key, "[REDACTED]")
            print(f"Server explanation: {fallback[:1000]}")
        return 1

    extraction_duration = time.perf_counter() - extraction_start
    total_duration = time.perf_counter() - total_start

    print(f"Reference ID: {reference_id}")
    print(f"Original complaint: {complaint}")
    print("\nExact source subsection:")
    print(source_section)
    print("\nActual extracted JSON:")
    print(result.model_dump_json(indent=2))
    print(f"\nExtraction-call duration: {extraction_duration:.2f}s")
    print(f"Total main-function duration: {total_duration:.2f}s")
    print("\nSource extraction only. The original complaint does not establish that the user uses full-screen gestures. This output is not yet an approved troubleshooting plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
