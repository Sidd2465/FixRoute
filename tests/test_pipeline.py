"""Tests for the FixRoute unified troubleshooting pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from app.cache import SemanticCache
from app.data_loader import load_deeplinks, load_references
from app.pipeline import TroubleshootPipeline
from app.schemas import ContextDeeplinkResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPLINKS_PATH = PROJECT_ROOT / "student_kit" / "deeplinks.json"
SIIS_PATH = PROJECT_ROOT / "student_kit" / "siis_responses.json"


@pytest.fixture
def catalogue() -> list[dict[str, Any]]:
    return load_deeplinks(DEEPLINKS_PATH)


@pytest.fixture
def references() -> list[dict[str, Any]]:
    return load_references(SIIS_PATH)


class MockModels:
    def __init__(self, extraction_json: str, planning_json: str) -> None:
        self.extraction_json = extraction_json
        self.planning_json = planning_json
        self.call_count = 0

    def generate_content(self, **kwargs: Any) -> Any:
        self.call_count += 1
        class DummyResponse:
            def __init__(self, text: str) -> None:
                self.text = text

        # First call is extraction, second call is planning
        text = self.extraction_json if self.call_count == 1 else self.planning_json
        return DummyResponse(text)


class MockGenAIClient:
    def __init__(self, extraction_json: str, planning_json: str) -> None:
        self.models = MockModels(extraction_json, planning_json)


def test_pipeline_unmatched_query_returns_empty_envelope(
    catalogue: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> None:
    cache = SemanticCache()
    pipeline = TroubleshootPipeline(
        deeplink_catalogue=catalogue,
        reference_catalogue=references,
        cache=cache,
        client=None,
    )

    envelope = pipeline.process("quantum entanglement failure on warp drive")
    assert envelope.response.contexts == []
    assert envelope.meta.cache_hit is False
    assert envelope.query_variations == []
    assert envelope.meta.cost_usd == 0.0


def test_pipeline_cache_hit_returns_sub_millisecond(
    catalogue: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> None:
    cache = SemanticCache()
    query = "My Galaxy S22 screen inputs are delayed"
    cached_response = ContextDeeplinkResponse(contexts=[])
    cache.set(query, cached_response, query_variations=["screen touches lagging"])

    pipeline = TroubleshootPipeline(
        deeplink_catalogue=catalogue,
        reference_catalogue=references,
        cache=cache,
        client=None,
    )

    envelope = pipeline.process(query)
    assert envelope.meta.cache_hit is True
    assert envelope.meta.model == "semantic_cache"
    assert envelope.meta.latency_ms < 50.0  # sub-50ms cache return


def test_pipeline_cold_execution_and_cache_population(
    catalogue: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> None:
    extraction_payload = json.dumps(
        {
            "steps": [
                {"text": "Go to Settings", "source_quote": "Go to Settings"},
                {"text": "Tap Display", "source_quote": "Tap Display"},
                {"text": "Tap Navigation bar", "source_quote": "Tap Navigation bar"},
                {"text": "Select Buttons", "source_quote": "Select Buttons"},
            ],
            "warnings": [],
            "conditions": [],
            "missing_details": [],
        }
    )
    planning_payload = json.dumps(
        {
            "goal_topic": "Full Screen Gesture Function",
            "goal_title": "Full screen gestures",
            "action_name": "Disable Full Screen Gestures",
            "action_description": "It will disable full screen gestures.",
            "link_query": "navigation bar settings",
        }
    )

    mock_client = MockGenAIClient(extraction_payload, planning_payload)
    cache = SemanticCache()

    pipeline = TroubleshootPipeline(
        deeplink_catalogue=catalogue,
        reference_catalogue=references,
        cache=cache,
        client=mock_client,
    )

    complaint = (
        "My Galaxy S22 screen inputs are delayed and the touch responsiveness is laggy, "
        "causing a noticeable delay when I try to interact with the phone."
    )

    # 1. Cold execution
    assert len(cache) == 0
    cold_envelope = pipeline.process(complaint)

    assert cold_envelope.meta.cache_hit is False
    assert cold_envelope.meta.model == "gemini-3.6-flash"
    assert len(cold_envelope.response.contexts) == 1

    goal = cold_envelope.response.contexts[0]
    assert goal.title == "Full screen gestures"
    assert goal.actions[0].stepGroups[0].actionableDeeplink.deeplink == "bixby://masked/act/2f3dd95259"

    # Verify entry was written to cache
    assert len(cache) == 1

    # 2. Warm execution on the same query
    warm_envelope = pipeline.process(complaint)
    assert warm_envelope.meta.cache_hit is True
    assert warm_envelope.meta.model == "semantic_cache"
    assert warm_envelope.meta.latency_ms < 50.0
    assert len(warm_envelope.response.contexts) == 1