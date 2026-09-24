import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import ContextDeeplinkResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_OUTPUT_PATH = PROJECT_ROOT / "student_kit" / "sample_output.json"


def test_sample_response_matches_schema():
    payload = json.loads(SAMPLE_OUTPUT_PATH.read_text(encoding="utf-8"))
    validated = ContextDeeplinkResponse.model_validate(payload["response"])

    assert len(validated.contexts) > 0
    assert validated.contexts[0].title == "Screen display damage"


def test_empty_contexts_is_valid():
    validated = ContextDeeplinkResponse.model_validate({"contexts": []})

    assert validated.contexts == []


def test_non_list_contexts_is_rejected():
    with pytest.raises(ValidationError):
        ContextDeeplinkResponse.model_validate({"contexts": "not-a-list"})
