"""Manual extraction demo for FixRoute.

This script is deliberately isolated and only uses a mocked or real model when
run manually. It does not connect to the response builder or any production flow.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_api_key() -> str:
    try:
        from dotenv import dotenv_values
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("python-dotenv is required to load the project .env file") from exc

    values = dotenv_values(PROJECT_ROOT / ".env")
    key = values.get("GEMINI_API_KEY")
    if key is None or not str(key).strip():
        raise RuntimeError("Missing GEMINI_API_KEY in the project-root .env file.")
    os.environ.setdefault("GEMINI_API_KEY", str(key).strip())
    return str(key).strip()


def main() -> int:
    start = time.perf_counter()
    model_name = "gemini-3.6-flash"

    try:
        api_key = _load_api_key()
    except Exception as exc:
        print("Status: KEY_ERROR")
        print(f"Explanation: {exc}")
        return 1

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover
        print("Status: IMPORT_ERROR")
        print(f"Explanation: Google Gen AI SDK is unavailable. {exc.__class__.__name__}: {exc}")
        return 1

    try:
        from app.extraction import extract_from_reference
    except Exception as exc:  # pragma: no cover
        print("Status: IMPORT_ERROR")
        print(f"Explanation: Extraction module could not be imported. {exc.__class__.__name__}: {exc}")
        return 1

    reference_text = (
        "If Example Mail is freezing, clear its cache. Open Settings.\n"
        "Tap Apps. Select Example Mail. Tap Storage. Tap Clear cache.\n"
        "Do not tap Clear data; doing so removes the app's stored data."
    )

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        try:
            result = extract_from_reference(
                reference_text,
                client=client,
                model_name=model_name,
            )
        finally:
            try:
                client.close()
            except Exception:
                pass
    except Exception as exc:
        # Recover the original exception preserved by "raise ... from exc".
        underlying = exc.__cause__ if exc.__cause__ is not None else exc

        print("Status: EXTRACTION_FAILED")
        print("Error type:", type(underlying).__name__)

        code = getattr(underlying, "code", None)
        if isinstance(code, int):
            print("API status code:", code)

        message = getattr(underlying, "message", None)
        if isinstance(message, str):
            safe_message = message.replace(api_key, "[REDACTED]")
            print("Server explanation:", safe_message[:1000])
        else:
            print("No provider explanation available.")
            print("Wrapper error type:", type(exc).__name__)

        return 1

    elapsed = time.perf_counter() - start
    print(f"Model: {model_name}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(result.model_dump(mode="json", by_alias=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
