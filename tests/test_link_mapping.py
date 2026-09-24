import json
from pathlib import Path

import pytest

from app.data_loader import load_deeplinks
from app.extraction_schemas import ExtractionResult
from app.link_mapping import LinkDecision, apply_link_decision, map_links_for_group
from app.plan_from_extraction import build_extraction_plan
from app.plan_schemas import GoalPlan
from app.response_builder import build_response
from app.rule_validation import is_fatal_code, validate_response_rules


@pytest.fixture
def synthetic_catalogue():
    return [
        {
            "id": "DL-1001",
            "deeplink": "bixby://masked/act/settings_open",
            "description": "Opens the settings page.",
            "message": "Open Settings",
            "qna_description": "Navigate to device settings.",
            "originalType": "onURL",
        },
        {
            "id": "DL-1002",
            "deeplink": "bixby://masked/act/open_nav_bar",
            "description": "Opens the navigation bar settings page.",
            "message": "Open navigation bar settings",
            "qna_description": "Configure navigation bar settings and gestures.",
            "originalType": "onURL",
        },
        {
            "id": "DL-1003",
            "deeplink": "bixby://masked/act/backup_enable",
            "description": "Enables backup settings.",
            "message": "Enable backup",
            "qna_description": "Turn on backup for device data.",
            "originalType": "onURL",
        },
    ]


def test_gate_logic_for_small_catalogue(synthetic_catalogue):
    no_overlap = map_links_for_group(query="quantum gravity reactor", catalogue=synthetic_catalogue)
    assert no_overlap.reason == "no_candidates"
    assert no_overlap.attached is False
    assert no_overlap.catalogue_id is None

    low_relevance = map_links_for_group(
        query="Open backup toggles",
        catalogue=synthetic_catalogue,
        min_score=0.95,
    )
    assert low_relevance.reason == "score_below_threshold"
    assert low_relevance.attached is False

    close = map_links_for_group(query="Open Settings page", catalogue=synthetic_catalogue)
    assert close.reason == "attached"
    assert close.catalogue_id == "DL-1001"

    pair_catalogue = [
        {
            "id": "DL-2001",
            "deeplink": "bixby://masked/act/left",
            "description": "Open the settings menu.",
            "message": "Open settings",
            "originalType": "onURL",
        },
        {
            "id": "DL-2002",
            "deeplink": "bixby://masked/act/right",
            "description": "Open the settings menu.",
            "message": "Open settings",
            "originalType": "onURL",
        },
    ]
    tied = map_links_for_group(query="Open Settings page", catalogue=pair_catalogue, min_margin=0.15)
    assert tied.reason == "margin_too_small"
    assert tied.attached is False

    distinct = map_links_for_group(query="Open navigation bar settings", catalogue=synthetic_catalogue)
    assert distinct.reason == "attached"
    assert distinct.catalogue_id == "DL-1002"


def test_parameter_validation_rejects_bad_inputs(synthetic_catalogue):
    with pytest.raises(ValueError, match="non-empty string"):
        map_links_for_group(query="   ", catalogue=synthetic_catalogue)

    with pytest.raises(ValueError, match="min_score"):
        map_links_for_group(query="settings", catalogue=synthetic_catalogue, min_score=1.5)

    with pytest.raises(ValueError, match="min_margin"):
        map_links_for_group(query="settings", catalogue=synthetic_catalogue, min_margin=-0.1)

    with pytest.raises(ValueError, match="top_k"):
        map_links_for_group(query="settings", catalogue=synthetic_catalogue, top_k=0)

    with pytest.raises(ValueError, match="top_k"):
        map_links_for_group(query="settings", catalogue=synthetic_catalogue, top_k=True)


def test_single_candidate_never_attaches():
    one_entry = {
        "id": "DL-1001",
        "deeplink": "bixby://masked/act/settings_open",
        "description": "Opens the settings page.",
        "message": "Open Settings",
        "qna_description": "Navigate to device settings.",
        "originalType": "onURL",
    }

    decision = map_links_for_group(query="Open Settings page", catalogue=[one_entry])

    assert decision.attached is False
    assert decision.reason == "margin_too_small"
    assert decision.margin == 0.0


def test_default_margin_gate_rejects_tie():
    pair_catalogue = [
        {
            "id": "DL-2001",
            "deeplink": "bixby://masked/act/left",
            "description": "Open the settings menu.",
            "message": "Open settings",
            "originalType": "onURL",
        },
        {
            "id": "DL-2002",
            "deeplink": "bixby://masked/act/right",
            "description": "Open the settings menu.",
            "message": "Open settings",
            "originalType": "onURL",
        },
    ]

    decision = map_links_for_group(query="Open Settings page", catalogue=pair_catalogue)

    assert decision.reason == "margin_too_small"
    assert decision.attached is False


def test_decision_is_deterministic_and_non_mutating(synthetic_catalogue):
    first = map_links_for_group(query="navigation bar settings", catalogue=synthetic_catalogue)
    second = map_links_for_group(query="navigation bar settings", catalogue=synthetic_catalogue)
    assert first == second

    plan = GoalPlan(
        goal="Follow these steps to perform this Navigation Configuration",
        title="Navigation settings",
        score=0.0,
        actions=[
            {
                "actionName": "Open navigation settings",
                "description": "It will open navigation settings.",
                "category": "manual",
                "stepGroups": [{"steps": ["Open Settings."], "catalogue_id": None}],
            }
        ],
    )
    before = json.dumps(plan.model_dump(mode="python"), sort_keys=True)
    result = apply_link_decision(plan=plan, decision=first)
    after = json.dumps(plan.model_dump(mode="python"), sort_keys=True)
    assert before == after
    assert result.actions[0].category.value == "auto"
    assert result.actions[0].stepGroups[0].catalogue_id == "DL-1002"


def test_unattached_and_attached_decisions_behave_correctly(synthetic_catalogue):
    unattached = map_links_for_group(query="quantum gravity reactor", catalogue=synthetic_catalogue)
    plan = GoalPlan(
        goal="Follow these steps to perform this Navigation Configuration",
        title="Navigation settings",
        score=0.0,
        actions=[
            {
                "actionName": "Open navigation settings",
                "description": "It will open navigation settings.",
                "category": "manual",
                "stepGroups": [{"steps": ["Open Settings."], "catalogue_id": None}],
            }
        ],
    )
    unchanged = apply_link_decision(plan=plan, decision=unattached)
    assert unchanged.actions[0].category.value == "manual"
    assert unchanged.actions[0].stepGroups[0].catalogue_id is None

    attached = map_links_for_group(query="navigation bar settings", catalogue=synthetic_catalogue)
    changed = apply_link_decision(plan=plan, decision=attached)
    assert changed.actions[0].category.value == "auto"
    assert changed.actions[0].stepGroups[0].catalogue_id == "DL-1002"


def test_real_catalogue_requires_attached_navigation_bar_match():
    catalogue = load_deeplinks(Path(__file__).resolve().parent.parent / "student_kit" / "deeplinks.json")
    decision = map_links_for_group(query="navigation bar settings", catalogue=catalogue)
    assert decision.reason == "attached", (
        "navigation bar settings did not attach. Top 3 candidates were: "
        f"{[(item['id'], item['score']) for item in decision.candidates]} "
        "and step-joined query candidates were not checked because the real catalogue match was expected."
    )
    assert decision.catalogue_id == "DL-0169", (
        "Expected DL-0169 for 'navigation bar settings', got "
        f"{[(item['id'], item['score']) for item in decision.candidates]}"
    )


def test_builder_integration_with_row_21_plan_and_real_catalogue():
    catalogue = load_deeplinks(Path(__file__).resolve().parent.parent / "student_kit" / "deeplinks.json")
    fixture_text = Path(__file__).resolve().parent.joinpath("fixtures", "row21_extraction.json").read_text(encoding="utf-8")
    extraction = ExtractionResult.model_validate(json.loads(fixture_text))
    plan = build_extraction_plan(
        extraction=extraction,
        goal_topic="Full Screen Gesture Function",
        goal_title="Full screen gestures",
        action_name="Disable Full Screen Gestures",
        action_description="It will disable full screen gestures.",
        score=0.0,
    )

    decision = map_links_for_group(query="navigation bar settings", catalogue=catalogue)
    assert decision.reason == "attached"
    target_entry = next(entry for entry in catalogue if entry["id"] == decision.catalogue_id)
    plan_after = apply_link_decision(plan=plan, decision=decision)
    response = build_response([plan_after], catalogue)
    violations = validate_response_rules(response)
    fatal = [item for item in violations if is_fatal_code(item.code)]
    assert not fatal, f"Fatal validation issues: {fatal}"

    step_group = response.contexts[0].actions[0].stepGroups[0]
    assert step_group.actionableDeeplink is not None
    assert step_group.actionableDeeplink.deeplink == target_entry["deeplink"]
    assert step_group.validationDeeplink is not None
    assert step_group.validationDeeplink.deeplink == target_entry["validation"]["deeplink"]


def test_apply_link_decision_raises_when_plan_shape_does_not_match():
    decision = LinkDecision(
        attached=True,
        catalogue_id="DL-999",
        chosen_score=0.9,
        margin=0.2,
        reason="attached",
        candidates=[],
    )
    plan = GoalPlan(
        goal="Follow these steps to perform this Demo Troubleshooting",
        title="Demo",
        score=0.0,
        actions=[
            {
                "actionName": "Example",
                "description": "It will do something.",
                "category": "manual",
                "stepGroups": [
                    {"steps": ["Step one."], "catalogue_id": None},
                    {"steps": ["Step two."], "catalogue_id": None},
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="exactly one action"):
        apply_link_decision(plan=plan, decision=decision)
