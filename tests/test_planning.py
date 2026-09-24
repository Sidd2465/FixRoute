"""Tests for the FixRoute planning module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from app.data_loader import load_deeplinks
from app.extraction_schemas import ExtractionResult
from app.planning import build_planned_pipeline, generate_plan_parameters, normalize_plan_parameters
from app.planning_schemas import PlanningParameters
from app.rule_validation import is_fatal_code

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "row21_extraction.json"
DEEPLINKS_PATH = PROJECT_ROOT / "student_kit" / "deeplinks.json"


@pytest.fixture
def row21_extraction() -> ExtractionResult:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return ExtractionResult.model_validate(data)


@pytest.fixture
def deeplink_catalogue() -> list[dict[str, Any]]:
    return load_deeplinks(DEEPLINKS_PATH)


class DummyModels:
    def __init__(self, response_text: str) -> None:
        self._text = response_text

    def generate_content(self, **kwargs: Any) -> Any:
        class DummyResponse:
            def __init__(self, text: str) -> None:
                self.text = text

        return DummyResponse(self._text)


class DummyClient:
    def __init__(self, response_text: str) -> None:
        self.models = DummyModels(response_text)


def test_normalize_plan_parameters_pads_short_description() -> None:
    raw = PlanningParameters(
        goal_topic="Gestures",
        goal_title="Full screen gestures",
        action_name="Disable Gestures",
        action_description="It will work.",
        link_query="navigation bar",
    )
    norm = normalize_plan_parameters(raw)
    assert 5 <= len(norm.action_description.split()) <= 7
    assert norm.action_description.startswith("It will ")


def test_normalize_plan_parameters_truncates_long_description() -> None:
    raw = PlanningParameters(
        goal_topic="Gestures",
        goal_title="Full screen gestures",
        action_name="Disable Gestures",
        action_description="It will disable full screen gestures and quickly fix navigation problems on device.",
        link_query="navigation bar",
    )
    norm = normalize_plan_parameters(raw)
    assert 5 <= len(norm.action_description.split()) <= 7
    assert norm.action_description.startswith("It will ")


def test_normalize_plan_parameters_prepends_it_will() -> None:
    raw = PlanningParameters(
        goal_topic="Gestures",
        goal_title="Full screen gestures",
        action_name="Disable Gestures",
        action_description="Disables full screen gestures on device.",
        link_query="navigation bar",
    )
    norm = normalize_plan_parameters(raw)
    assert norm.action_description.startswith("It will ")
    assert 5 <= len(norm.action_description.split()) <= 7


def test_normalize_plan_parameters_normalizes_goal_title() -> None:
    raw = PlanningParameters(
        goal_topic="Gestures",
        goal_title="Disabling full screen gestures settings",
        action_name="Disable Gestures",
        action_description="It will disable full screen gestures.",
        link_query="navigation bar",
    )
    norm = normalize_plan_parameters(raw)
    assert len(norm.goal_title.split()) in (2, 3)


def test_generate_plan_parameters_mock_success(row21_extraction: ExtractionResult) -> None:
    mock_payload = json.dumps(
        {
            "goal_topic": "Full Screen Gesture Function",
            "goal_title": "Full screen gestures",
            "action_name": "Disable Full Screen Gestures",
            "action_description": "It will disable full screen gestures.",
            "link_query": "navigation bar settings",
        }
    )
    client = DummyClient(mock_payload)
    params = generate_plan_parameters("Screen lag complaint", row21_extraction, client=client)

    assert params.goal_topic == "Full Screen Gesture Function"
    assert params.goal_title == "Full screen gestures"
    assert params.action_name == "Disable Full Screen Gestures"
    assert params.action_description == "It will disable full screen gestures."
    assert params.link_query == "navigation bar settings"


def test_build_planned_pipeline_offline_with_real_catalogue(
    row21_extraction: ExtractionResult,
    deeplink_catalogue: list[dict[str, Any]],
) -> None:
    params = PlanningParameters(
        goal_topic="Full Screen Gesture Function",
        goal_title="Full screen gestures",
        action_name="Disable Full Screen Gestures",
        action_description="It will disable full screen gestures.",
        link_query="navigation bar settings",
    )

    response, link_decision, violations = build_planned_pipeline(
        complaint="My screen is lagging",
        extraction=row21_extraction,
        params=params,
        deeplink_catalogue=deeplink_catalogue,
    )

    # 1. Deeplink attached DL-0169
    assert link_decision.reason == "attached"
    assert link_decision.catalogue_id == "DL-0169"

    # 2. Response structure
    assert len(response.contexts) == 1
    goal = response.contexts[0]
    assert goal.title == "Full screen gestures"
    assert goal.goal == "Follow these steps to perform this Full Screen Gesture Function Troubleshooting"
    assert len(goal.actions) == 1

    action = goal.actions[0]
    assert action.actionName == "Disable Full Screen Gestures"
    assert action.category.value == "auto" if hasattr(action.category, "value") else action.category == "auto"

    group = action.stepGroups[0]
    assert group.actionableDeeplink is not None
    assert group.actionableDeeplink.deeplink == "bixby://masked/act/2f3dd95259"
    assert group.validationDeeplink is not None
    assert group.validationDeeplink.deeplink == "bixby://masked/val/d8310c1b7d"
    assert len(group.steps) == 4

    # 3. Rule validation: zero fatal violations
    fatal_violations = [v for v in violations if is_fatal_code(v.code)]
    assert fatal_violations == []


def test_build_planned_pipeline_unattached_link(
    row21_extraction: ExtractionResult,
    deeplink_catalogue: list[dict[str, Any]],
) -> None:
    params = PlanningParameters(
        goal_topic="Full Screen Gesture Function",
        goal_title="Full screen gestures",
        action_name="Disable Full Screen Gestures",
        action_description="It will disable full screen gestures.",
        link_query="quantum flux capacitor settings",
    )

    response, link_decision, violations = build_planned_pipeline(
        complaint="My screen is lagging",
        extraction=row21_extraction,
        params=params,
        deeplink_catalogue=deeplink_catalogue,
    )

    assert link_decision.reason in ("no_candidates", "score_below_threshold")
    action = response.contexts[0].actions[0]
    assert action.category.value == "manual" if hasattr(action.category, "value") else action.category == "manual"

    group = action.stepGroups[0]
    assert group.actionableDeeplink is None
    assert group.validationDeeplink is None

    fatal_violations = [v for v in violations if is_fatal_code(v.code)]
    assert fatal_violations == []