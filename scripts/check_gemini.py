"""Standalone Gemini connectivity check for FixRoute.

This module is intentionally isolated. Importing it does not load secrets or make
API calls. The check is performed only when `main()` is executed directly.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_api_key() -> str:
    """Load the project-root .env without overriding existing environment values."""
    try:
        from dotenv import dotenv_values
    except Exception as exc:  # pragma: no cover - import path guard
        raise RuntimeError("python-dotenv is required to load the project .env file.") from exc

    env_path = _project_root() / ".env"
    env_values = dotenv_values(env_path)
    key = env_values.get("GEMINI_API_KEY")
    if key is None or not str(key).strip():
        raise RuntimeError("Missing GEMINI_API_KEY in the project-root .env file.")

    os.environ.setdefault("GEMINI_API_KEY", str(key).strip())
    return str(key).strip()


def main() -> int:
    start = time.perf_counter()
    model_name = "gemini-3.6-flash"

    try:
        api_key = _load_api_key()
    except Exception as exc:  # pragma: no cover - intentionally safe failure path
        print(f"Status: KEY_ERROR\nExplanation: {exc}")
        return 1

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # pragma: no cover - import guard
        print("Status: IMPORT_ERROR")
        print(f"Explanation: Google Gen AI SDK is unavailable. {exc.__class__.__name__}: {exc}")
        return 1

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=30_000,  # 30 seconds, expressed in milliseconds,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    except Exception as exc:  # pragma: no cover - safe initialization failure path
        print("Status: CLIENT_INIT_ERROR")
        print(f"Explanation: Failed to initialize Gemini client. {exc.__class__.__name__}")
        return 1

    try:
                response = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: FixRoute connection OK",
            config=types.GenerateContentConfig(
                max_output_tokens=1024,
                automatic_function_calling=(
                    types.AutomaticFunctionCallingConfig(disable=True)
                ),
            ),
        )
        
    except Exception as exc:
        print("Status: REQUEST_FAILED")
        print(f"Error type: {type(exc).__name__}")

        status_code = getattr(exc, "code", None)
        if isinstance(status_code, int):
            print(f"API status code: {status_code}")

        # Read only the SDK's error message.
        message = getattr(exc, "message", None)

        if isinstance(message, str):
            # Hide our API key if it appears in the message.
            safe_message = message.replace(api_key, "[REDACTED]")
            print("Server explanation:", safe_message[:1000])
        else:
            print("No server explanation was provided.")

        return 1
    finally:
        try:
            client.close()
        except Exception:
            pass

    text = ""
    try:
        if hasattr(response, "text") and response.text:
            text = str(response.text).strip()
        elif hasattr(response, "candidates") and response.candidates:
            parts = getattr(response.candidates[0], "content", None)
            if parts is not None:
                text = "".join(
                    part.text for part in getattr(parts, "parts", []) if getattr(part, "text", None)
                ).strip()
    except Exception:
        text = ""

    elapsed = time.perf_counter() - start
    if not text:
        print("Status: EMPTY_RESPONSE")
        print("Explanation: Gemini returned no usable content.")
        return 1

    print(f"Model: {model_name}")
    print(f"Response: {text}")
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
