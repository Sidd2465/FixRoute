"""Data loading helpers for the Smart Guided Troubleshooting Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List


def _read_text_file(path: str | Path) -> str:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: {resolved_path}")

    try:
        return resolved_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8: {resolved_path}") from exc


def _validate_count(payload: dict[str, Any], label: str, path: str | Path, actual_count: int) -> int:
    if "count" not in payload:
        raise ValueError(f"{label} must include an integer 'count' field.")

    count = payload["count"]
    if type(count) is not int:
        if count is None:
            raise ValueError(f"{label} has an invalid 'count' field: expected int, got None.")
        raise ValueError(
            f"{label} has an invalid 'count' field: expected int, got {type(count).__name__}."
        )
    if count < 0:
        raise ValueError(f"{label} has a negative 'count' value: {count}.")
    if count != actual_count:
        raise ValueError(
            f"Count mismatch in {path}: declared count is {count}, but found {actual_count} entries."
        )
    return count


def _load_json_document(path: str | Path, label: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        raw_bytes = file_path.read_bytes()
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File contains invalid UTF-8: {file_path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read {label}: {file_path}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {file_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object at the top level.")

    return payload


def load_complaints(path: str | Path) -> list[str]:
    """Load complaint lines from an input file while preserving complaint text."""
    text = _read_text_file(path)
    complaints = [line.strip() for line in text.splitlines()]
    return [line for line in complaints if line]


def load_references(path: str | Path) -> list[dict]:
    """Load validated SIIS reference entries from a JSON catalog."""
    payload = _load_json_document(path, "siis_responses.json")
    responses = payload.get("responses")
    if not isinstance(responses, list):
        raise ValueError("siis_responses.json must contain a top-level 'responses' list.")

    _validate_count(payload, "siis_responses.json", path, len(responses))

    seen_ids: set[str] = set()
    validated: list[dict] = []

    for index, entry in enumerate(responses):
        if not isinstance(entry, dict):
            raise ValueError(f"Reference entry at index {index} must be an object.")

        reference_id = entry.get("id")
        if not isinstance(reference_id, str):
            raise ValueError(f"Reference entry at index {index} is missing a string 'id'.")
        if reference_id in seen_ids:
            raise ValueError(f"Duplicate reference id found: {reference_id}")
        seen_ids.add(reference_id)

        original_query = entry.get("original_query")
        if not isinstance(original_query, str):
            raise ValueError(
                f"Reference '{reference_id}' is missing a string 'original_query'."
            )

        siis_response = entry.get("siis_response")
        if not isinstance(siis_response, dict):
            raise ValueError(
                f"Reference '{reference_id}' is missing a valid 'siis_response' object."
            )

        title = siis_response.get("title")
        content = siis_response.get("content")
        if not isinstance(title, str) or not isinstance(content, str):
            raise ValueError(
                f"Reference '{reference_id}' must contain a 'siis_response' object with "
                "string 'title' and 'content' fields."
            )

        validated.append(entry)

    return validated


def load_deeplinks(path: str | Path) -> list[dict]:
    """Load validated deeplink entries from a JSON catalog."""
    payload = _load_json_document(path, "deeplinks.json")
    deeplinks = payload.get("deeplinks")
    if not isinstance(deeplinks, list):
        raise ValueError("deeplinks.json must contain a top-level 'deeplinks' list.")

    _validate_count(payload, "deeplinks.json", path, len(deeplinks))

    seen_ids: set[str] = set()
    validated: list[dict] = []

    for index, entry in enumerate(deeplinks):
        if not isinstance(entry, dict):
            raise ValueError(f"Deeplink entry at index {index} must be an object.")

        deeplink_id = entry.get("id")
        if not isinstance(deeplink_id, str):
            raise ValueError(f"Deeplink entry at index {index} is missing a string 'id'.")
        if deeplink_id in seen_ids:
            raise ValueError(f"Duplicate deeplink id found: {deeplink_id}")
        seen_ids.add(deeplink_id)

        deeplink = entry.get("deeplink")
        description = entry.get("description")
        if not isinstance(deeplink, str):
            raise ValueError(f"Deeplink '{deeplink_id}' is missing a string 'deeplink'.")
        if not isinstance(description, str):
            raise ValueError(f"Deeplink '{deeplink_id}' is missing a string 'description'.")

        validated.append(entry)

    return validated
