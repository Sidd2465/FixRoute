"""Tests for the FixRoute FastAPI REST API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.cache import SemanticCache
from app.data_loader import load_deeplinks, load_references
from app.pipeline import TroubleshootPipeline
from app.schemas import ContextDeeplinkResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPLINKS_PATH = PROJECT_ROOT / "student_kit" / "deeplinks.json"
SIIS_PATH = PROJECT_ROOT / "student_kit" / "siis_responses.json"


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

        text = self.extraction_json if self.call_count == 1 else self.planning_json
        return DummyResponse(text)


class MockGenAIClient:
    def __init__(self, extraction_json: str, planning_json: str) -> None:
        self.models = MockModels(extraction_json, planning_json)


@pytest.fixture
def mock_pipeline() -> TroubleshootPipeline:
    catalogue = load_deeplinks(DEEPLINKS_PATH)
    references = load_references(SIIS_PATH)
    cache = SemanticCache()

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
    client = MockGenAIClient(extraction_payload, planning_payload)

    return TroubleshootPipeline(
        deeplink_catalogue=catalogue,
        reference_catalogue=references,
        cache=cache,
        client=client,
    )


@pytest.fixture
def client(mock_pipeline: TroubleshootPipeline) -> TestClient:
    app = create_app(pipeline=mock_pipeline)
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["deeplinks_count"] == 578
    assert data["references_count"] == 20
    assert data["cache_size"] == 0


def test_troubleshoot_empty_query_rejected(client: TestClient) -> None:
    response = client.post("/v1/troubleshoot", json={"query": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_troubleshoot_unmatched_query_returns_empty_contexts(client: TestClient) -> None:
    response = client.post(
        "/v1/troubleshoot",
        json={"query": "quantum drive failure on spacecraft"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "quantum drive failure on spacecraft"
    assert data["response"]["contexts"] == []
    assert data["meta"]["cache_hit"] is False
    assert data["meta"]["cost_usd"] == 0.0


def test_troubleshoot_cold_and_warm_flow(client: TestClient) -> None:
    complaint = (
        "My Galaxy S22 screen inputs are delayed and the touch responsiveness is laggy, "
        "causing a noticeable delay when I try to interact with the phone."
    )

    # 1. Cold Request
    r_cold = client.post("/v1/troubleshoot", json={"query": complaint})
    assert r_cold.status_code == 200
    cold_data = r_cold.json()

    assert cold_data["meta"]["cache_hit"] is False
    assert cold_data["meta"]["model"] == "gemini-3.6-flash"
    assert len(cold_data["response"]["contexts"]) == 1

    goal = cold_data["response"]["contexts"][0]
    assert goal["title"] == "Full screen gestures"
    assert goal["actions"][0]["stepGroups"][0]["actionableDeeplink"]["deeplink"] == "bixby://masked/act/2f3dd95259"

    # 2. Warm Request (Cache Hit)
    r_warm = client.post("/v1/troubleshoot", json={"query": complaint})
    assert r_warm.status_code == 200
    warm_data = r_warm.json()

    assert warm_data["meta"]["cache_hit"] is True
    assert warm_data["meta"]["model"] == "semantic_cache"
    assert warm_data["meta"]["latency_ms"] < 50.0  # sub-50ms cache return!
    assert len(warm_data["response"]["contexts"]) == 1