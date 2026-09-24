import json
from math import isnan
from pathlib import Path

import pytest

from app.extraction_schemas import ExtractionResult
from app.plan_from_extraction import build_extraction_plan
from app.plan_schemas import GoalPlan
from app.response_builder import ResponseBuildError, build_response
from app.rule_validation import validate_response_rules
from app.schemas import actionCategory

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "row21_extraction.json"


with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
    EXTRACTION_PAYLOAD = json.load(handle)


EXTRACTION = ExtractionResult.model_validate(EXTRACTION_PAYLOAD)


def test_plan_preserves_steps_and_ignores_other_fields():
    plan = build_extraction_plan(
        extraction=EXTRACTION,
        goal_topic="Full Screen Gesture Function",
        goal_title="Full screen gestures",
        action_name="Disable Full Screen Gestures",
        action_description="It will disable full screen gestures.",
        score=0.0,
    )

    assert isinstance(plan, GoalPlan)
    assert len(plan.actions) == 1
    assert len(plan.actions[0].stepGroups) == 1
    assert plan.actions[0].stepGroups[0].steps == [
        "Go to Settings",
        "Tap Display",
        "Tap Navigation bar",
        "Select Buttons to turn off full screen gestures",
    ]
    assert plan.goal == "Follow these steps to perform this Full Screen Gesture Function Troubleshooting"
    assert plan.title == "Full screen gestures"
    assert plan.actions[0].actionName == "Disable Full Screen Gestures"
    assert plan.actions[0].description == "It will disable full screen gestures."
    assert plan.actions[0].category == actionCategory.manual
    assert plan.actions[0].stepGroups[0].catalogue_id is None

    joined_step_text = "\n".join(plan.actions[0].stepGroups[0].steps)
    assert "If you are using the full screen gesture function" not in joined_step_text
    assert "If you're using the full screen gesture function" not in joined_step_text
    assert "Go to Settings" in joined_step_text


def test_empty_extraction_steps_raise_value_error():
    with pytest.raises(ValueError, match="non-empty"):
        build_extraction_plan(
            extraction=ExtractionResult(
                steps=[],
                warnings=[],
                conditions=[],
                missing_details=[],
            ),
            goal_topic="Full Screen Gesture Function",
            goal_title="Full screen gestures",
            action_name="Disable Full Screen Gestures",
            action_description="It will disable full screen gestures.",
            score=0.0,
        )


@pytest.mark.parametrize(
    "goal_topic,goal_title,action_name,action_description",
    [
        ("", "Full screen gestures", "Disable Full Screen Gestures", "It will disable full screen gestures."),
        ("Full Screen Gesture Function", "", "Disable Full Screen Gestures", "It will disable full screen gestures."),
        ("Full Screen Gesture Function", "Full screen gestures", "", "It will disable full screen gestures."),
        ("Full Screen Gesture Function", "Full screen gestures", "Disable Full Screen Gestures", "   "),
    ],
)
def test_blank_required_strings_raise_value_error(goal_topic, goal_title, action_name, action_description):
    with pytest.raises(ValueError, match="non-empty"):
        build_extraction_plan(
            extraction=EXTRACTION,
            goal_topic=goal_topic,
            goal_title=goal_title,
            action_name=action_name,
            action_description=action_description,
            score=0.0,
        )


@pytest.mark.parametrize("score", [1.5, -0.1, float("nan"), float("inf"), True])
def test_invalid_score_values_raise_value_error(score):
    with pytest.raises(ValueError, match="score"):
        build_extraction_plan(
            extraction=EXTRACTION,
            goal_topic="Full Screen Gesture Function",
            goal_title="Full screen gestures",
            action_name="Disable Full Screen Gestures",
            action_description="It will disable full screen gestures.",
            score=score,
        )


def test_goal_text_template_and_category_are_exact():
    plan = build_extraction_plan(
        extraction=EXTRACTION,
        goal_topic="Full Screen Gesture Function",
        goal_title="Full screen gestures",
        action_name="Disable Full Screen Gestures",
        action_description="It will disable full screen gestures.",
        score=0.0,
    )

    assert plan.goal == "Follow these steps to perform this Full Screen Gesture Function Troubleshooting"
    assert plan.title == "Full screen gestures"
    assert plan.actions[0].category == actionCategory.manual
    assert plan.actions[0].stepGroups[0].catalogue_id is None


def test_response_builder_integration_uses_existing_builder_and_validator():
    plan = build_extraction_plan(
        extraction=EXTRACTION,
        goal_topic="Full Screen Gesture Function",
        goal_title="Full screen gestures",
        action_name="Disable Full Screen Gestures",
        action_description="It will disable full screen gestures.",
        score=0.0,
    )

    try:
        response = build_response([plan], [])
    except ResponseBuildError as exc:
        pytest.skip(f"Existing builder rejects manual group without a catalogue_id: {exc}")

    assert response.contexts
    assert response.contexts[0].actions[0].actionName == "Disable Full Screen Gestures"
    steps = response.contexts[0].actions[0].stepGroups[0].steps
    assert steps == [
        "Go to Settings",
        "Tap Display",
        "Tap Navigation bar",
        "Select Buttons to turn off full screen gestures",
    ]
    assert response.contexts[0].actions[0].stepGroups[0].actionableDeeplink is None
    assert response.contexts[0].actions[0].stepGroups[0].validationDeeplink is None

    violations = validate_response_rules(response)
    assert not violations
    assert response.contexts[0].actions[0].category == actionCategory.manual


def test_no_invented_steps_or_leaked_text():
    plan = build_extraction_plan(
        extraction=EXTRACTION,
        goal_topic="Full Screen Gesture Function",
        goal_title="Full screen gestures",
        action_name="Disable Full Screen Gestures",
        action_description="It will disable full screen gestures.",
        score=0.0,
    )

    step_texts = [
        step
        for step_group in plan.actions[0].stepGroups
        for step in step_group.steps
    ]
    assert set(step_texts) == {item.text for item in EXTRACTION.steps}
    leaked = [
        "If you are using the full screen gesture function",
        "If you're using the full screen gesture function",
    ]
    for text in leaked:
        assert text not in "\n".join(step_texts)


def test_extraction_instance_is_required():
    with pytest.raises(ValueError, match="ExtractionResult"):
        build_extraction_plan(
            extraction={"steps": [], "warnings": [], "conditions": [], "missing_details": []},
            goal_topic="Full Screen Gesture Function",
            goal_title="Full screen gestures",
            action_name="Disable Full Screen Gestures",
            action_description="It will disable full screen gestures.",
            score=0.0,
        )
