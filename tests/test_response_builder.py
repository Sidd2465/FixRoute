import copy
import json

import pytest

from app.plan_schemas import ActionPlan, GoalPlan, StepGroupPlan
from app.response_builder import ResponseBuildError, build_response
from app.schemas import ContextDeeplinkResponse, actionCategory


@pytest.fixture
def sample_catalogue() -> list[dict]:
    # Synthetic assembly-test data only; not an official reference or recommendation.
    return [
        {
            "id": "DL-1001",
            "deeplink": "bixby://masked/act/open_settings",
            "description": "Opens the device Settings page.",
            "message": "Open Settings",
            "originalType": "onURL",
            "classes": {"theme": "settings"},
            "validation": {
                "deeplink": "bixby://masked/val/settings_active",
                "key": "Settings",
                "resultType": "boolean",
                "condition": "equal",
                "value": "True",
            },
        },
        {
            "id": "DL-1002",
            "deeplink": "bixby://masked/act/hold_power",
            "description": "Holds the power button for a short refresh.",
            "message": "Hold power", 
            "originalType": "onURL",
        },
        {
            "id": "DL-1003",
            "deeplink": "bixby://masked/act/reset_device",
            "description": "Resets the device to factory defaults.",
            "message": "Reset device",
            "originalType": "onURL",
        },
    ]


@pytest.fixture
def assembly_fixture() -> list[GoalPlan]:
    # Synthetic assembly-test data only; not an official reference or recommendation.
    return [
        GoalPlan(
            goal="Follow these steps to perform this Wi-Fi Troubleshooting",
            title="Wi Fi",
            score=0.87,
            actions=[
                ActionPlan(
                    actionName="Open settings",
                    description="It will open the settings page.",
                    category=actionCategory.auto,
                    stepGroups=[
                        StepGroupPlan(
                            steps=["Open Settings."],
                            catalogue_id="DL-1001",
                        )
                    ],
                ),
                ActionPlan(
                    actionName="Hold power button",
                    description="It will hold the power button.",
                    category=actionCategory.manual,
                    stepGroups=[
                        StepGroupPlan(
                            steps=["Press and hold the power button."],
                            catalogue_id=None,
                        )
                    ],
                ),
                ActionPlan(
                    actionName="Reset device",
                    description="It will reset the device now.",
                    category=actionCategory.critical,
                    stepGroups=[
                        StepGroupPlan(
                            steps=["Confirm the reset prompt."],
                            catalogue_id="DL-1003",
                        )
                    ],
                ),
            ],
        )
    ]


def test_valid_auto_manual_and_critical_actions(sample_catalogue, assembly_fixture):
    response = build_response(assembly_fixture, sample_catalogue)

    assert response.contexts[0].actions[0].category == actionCategory.auto
    assert response.contexts[0].actions[1].category == actionCategory.manual
    assert response.contexts[0].actions[2].category == actionCategory.critical

    auto_link = response.contexts[0].actions[0].stepGroups[0].actionableDeeplink
    assert auto_link is not None
    assert auto_link.deeplink == "bixby://masked/act/open_settings"
    assert auto_link.originalType == "onURL"
    assert response.contexts[0].actions[0].stepGroups[0].validationDeeplink is not None


def test_exact_uri_and_supported_metadata_copied(sample_catalogue, assembly_fixture):
    response = build_response(assembly_fixture, sample_catalogue)

    action = response.contexts[0].actions[0]
    deeplink = action.stepGroups[0].actionableDeeplink
    assert deeplink is not None
    assert deeplink.deeplink == "bixby://masked/act/open_settings"
    assert deeplink.description == "Opens the device Settings page."
    assert deeplink.message == "Open Settings"
    assert deeplink.originalType == "onURL"
    assert deeplink.classes == {"theme": "settings"}
    assert "id" not in deeplink.model_dump()
    assert "qna_description" not in deeplink.model_dump()


def test_optional_validation_metadata_is_copied(sample_catalogue, assembly_fixture):
    response = build_response(assembly_fixture, sample_catalogue)

    validation = response.contexts[0].actions[0].stepGroups[0].validationDeeplink
    assert validation is not None
    assert validation.deeplink == "bixby://masked/val/settings_active"
    assert validation.key == "Settings"
    assert validation.resultType.value == "boolean"
    assert validation.condition.value == "equal"
    assert validation.value == "True"


def test_message_metadata_semantics(sample_catalogue):
    missing_message = copy.deepcopy(sample_catalogue[0])
    missing_message.pop("message")

    response = build_response(
        [
            GoalPlan(
                goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                title="Wi Fi",
                score=0.8,
                actions=[
                    ActionPlan(
                        actionName="Open settings",
                        description="It will open the settings page.",
                        category=actionCategory.auto,
                        stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                    )
                ],
            )
        ],
        [missing_message],
    )
    assert response.contexts[0].actions[0].stepGroups[0].actionableDeeplink is not None
    assert response.contexts[0].actions[0].stepGroups[0].actionableDeeplink.message == ""

    explicit_none = copy.deepcopy(sample_catalogue[0])
    explicit_none["message"] = None
    response = build_response(
        [
            GoalPlan(
                goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                title="Wi Fi",
                score=0.8,
                actions=[
                    ActionPlan(
                        actionName="Open settings",
                        description="It will open the settings page.",
                        category=actionCategory.auto,
                        stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                    )
                ],
            )
        ],
        [explicit_none],
    )
    assert response.contexts[0].actions[0].stepGroups[0].actionableDeeplink is not None
    assert response.contexts[0].actions[0].stepGroups[0].actionableDeeplink.message is None

    explicit_string = copy.deepcopy(sample_catalogue[0])
    explicit_string["message"] = "Custom message"
    response = build_response(
        [
            GoalPlan(
                goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                title="Wi Fi",
                score=0.8,
                actions=[
                    ActionPlan(
                        actionName="Open settings",
                        description="It will open the settings page.",
                        category=actionCategory.auto,
                        stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                    )
                ],
            )
        ],
        [explicit_string],
    )
    assert response.contexts[0].actions[0].stepGroups[0].actionableDeeplink is not None
    assert response.contexts[0].actions[0].stepGroups[0].actionableDeeplink.message == "Custom message"

    bad_message = copy.deepcopy(sample_catalogue[0])
    bad_message["message"] = ["not", "allowed"]
    with pytest.raises(ResponseBuildError, match="invalid actionable message metadata"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            [bad_message],
        )


def test_catalogue_and_plan_inputs_are_list_validated():
    with pytest.raises(ResponseBuildError, match="plans must be a list"):
        build_response(None, [])

    with pytest.raises(ResponseBuildError, match="plans must be a list"):
        build_response({"goal": "x"}, [])

    with pytest.raises(ResponseBuildError, match="catalogue_entries must be a list"):
        build_response([], None)

    with pytest.raises(ResponseBuildError, match="catalogue_entries must be a list"):
        build_response([], {"id": "DL-1"})


def test_catalogue_ids_must_be_non_empty_strings(sample_catalogue):
    bad_catalogue = copy.deepcopy(sample_catalogue)
    bad_catalogue[0]["id"] = 123
    with pytest.raises(ResponseBuildError, match="non-empty string 'id'"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            bad_catalogue,
        )

    invalid_plan_dict = {
        "goal": "Follow these steps to perform this Wi-Fi Troubleshooting",
        "title": "Wi Fi",
        "score": 0.8,
        "actions": [
            {
                "actionName": "Open settings",
                "description": "It will open the settings page.",
                "category": "auto",
                "stepGroups": [
                    {"steps": ["Open Settings."], "catalogue_id": 123}
                ],
            }
        ],
    }
    with pytest.raises(ResponseBuildError, match="Invalid plan at index"):
        build_response([invalid_plan_dict], sample_catalogue)


def test_malformed_nested_validation_metadata_rejected(sample_catalogue):
    missing_key = copy.deepcopy(sample_catalogue[0])
    missing_key["validation"] = {"deeplink": "bixby://masked/val/settings_active"}
    with pytest.raises(ResponseBuildError, match="missing a valid key value"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            [missing_key],
        )

    missing_uri = copy.deepcopy(sample_catalogue[0])
    missing_uri["validation"] = {"key": "Settings"}
    with pytest.raises(ResponseBuildError, match="missing a valid deeplink field"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            [missing_uri],
        )

    bad_result_type = copy.deepcopy(sample_catalogue[0])
    bad_result_type["validation"]["resultType"] = "made_up"
    with pytest.raises(ResponseBuildError, match="unsupported validation resultType"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            [bad_result_type],
        )

    bad_condition = copy.deepcopy(sample_catalogue[0])
    bad_condition["validation"]["condition"] = "unknown"
    with pytest.raises(ResponseBuildError, match="unsupported validation condition"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            [bad_condition],
        )

    bad_value = copy.deepcopy(sample_catalogue[0])
    bad_value["validation"]["value"] = 123
    with pytest.raises(ResponseBuildError, match="validation value must be a string"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            [bad_value],
        )

    non_object_validation = copy.deepcopy(sample_catalogue[0])
    non_object_validation["validation"] = ["not", "an", "object"]
    with pytest.raises(ResponseBuildError, match="must be an object"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            [non_object_validation],
        )

    bad_classes = copy.deepcopy(sample_catalogue[0])
    bad_classes["classes"] = ["wrong", "shape"]
    with pytest.raises(ResponseBuildError, match="invalid 'classes' metadata"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            [bad_classes],
        )

    with pytest.raises(ResponseBuildError, match="Invalid plan at index"):
        build_response([
            {"goal": "bad dict"}
        ], sample_catalogue)


def test_input_output_independence_with_nested_classes_and_steps(sample_catalogue):
    plan = [
        GoalPlan(
            goal="Follow these steps to perform this Wi-Fi Troubleshooting",
            title="Wi Fi",
            score=0.8,
            actions=[
                ActionPlan(
                    actionName="Open settings",
                    description="It will open the settings page.",
                    category=actionCategory.auto,
                    stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                )
            ],
        )
    ]
    original_plan = copy.deepcopy(plan)
    original_catalogue = copy.deepcopy(sample_catalogue)

    response = build_response(plan, sample_catalogue)

    response.contexts[0].actions[0].stepGroups[0].actionableDeeplink.classes["theme"] = "mutated-output"
    response.contexts[0].actions[0].stepGroups[0].steps.append("Injected output step")

    assert plan == original_plan
    assert sample_catalogue == original_catalogue

    plan[0].actions[0].stepGroups[0].steps.append("Injected input step")
    sample_catalogue[0]["classes"]["theme"] = "mutated-input"

    assert response.contexts[0].actions[0].stepGroups[0].steps == ["Open Settings.", "Injected output step"]
    assert response.contexts[0].actions[0].stepGroups[0].actionableDeeplink.classes == {"theme": "mutated-output"}
    assert plan[0].actions[0].stepGroups[0].steps == ["Open Settings.", "Injected input step"]
    assert sample_catalogue[0]["classes"] == {"theme": "mutated-input"}


def test_unknown_and_duplicate_catalogue_ids_rejected(sample_catalogue):
    with pytest.raises(ResponseBuildError, match="Unknown catalogue_id"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Missing link",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-9999")],
                        )
                    ],
                )
            ],
            sample_catalogue,
        )

    duplicate_entries = copy.deepcopy(sample_catalogue)
    duplicate_entries.append({
        "id": "DL-1001",
        "deeplink": "bixby://masked/act/duplicate_entry",
        "description": "Duplicate entry.",
        "message": "Duplicate",
        "originalType": "onURL",
    })
    with pytest.raises(ResponseBuildError, match="Duplicate catalogue_id"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            duplicate_entries,
        )


def test_missing_auto_link_and_forbidden_manual_link(sample_catalogue):
    with pytest.raises(ResponseBuildError, match="Auto actions require a catalogue_id"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id=None)],
                        )
                    ],
                )
            ],
            sample_catalogue,
        )

    with pytest.raises(ResponseBuildError, match="Manual action step group must not specify a catalogue_id"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Manual step",
                            description="It will hold the power button.",
                            category=actionCategory.manual,
                            stepGroups=[StepGroupPlan(steps=["Press and hold the power button."], catalogue_id="DL-1002")],
                        )
                    ],
                )
            ],
            sample_catalogue,
        )


def test_placeholder_rejection(sample_catalogue):
    bad_catalogue = copy.deepcopy(sample_catalogue)
    bad_catalogue[0]["deeplink"] = "bixby://dummy_positive"

    with pytest.raises(ResponseBuildError, match="bixby://dummy_positive"):
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="It will open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            bad_catalogue,
        )


def test_invalid_plans_rejected_by_existing_rule_validation(sample_catalogue):
    with pytest.raises(ResponseBuildError) as exc_info:
        build_response(
            [
                GoalPlan(
                    goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                    title="Wi Fi",
                    score=0.8,
                    actions=[
                        ActionPlan(
                            actionName="Open settings",
                            description="Please open the settings page.",
                            category=actionCategory.auto,
                            stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                        )
                    ],
                )
            ],
            sample_catalogue,
        )

    assert exc_info.value.violations
    assert any(v.code == "action_description_prefix" for v in exc_info.value.violations)


def test_empty_plans_produce_empty_contexts():
    response = build_response([], [])
    assert response == ContextDeeplinkResponse(contexts=[])
    assert response.contexts == []


def test_serialized_fixture_matches_official_schema(sample_catalogue):
    response = build_response(
        [
            GoalPlan(
                goal="Follow these steps to perform this Wi-Fi Troubleshooting",
                title="Wi Fi",
                score=0.8,
                actions=[
                    ActionPlan(
                        actionName="Open settings",
                        description="It will open the settings page.",
                        category=actionCategory.auto,
                        stepGroups=[StepGroupPlan(steps=["Open Settings."], catalogue_id="DL-1001")],
                    )
                ],
            )
        ],
        sample_catalogue,
    )
    payload = response.model_dump(mode="json")
    json_text = json.dumps(payload)
    reparsed = ContextDeeplinkResponse.model_validate_json(json_text)

    assert reparsed.contexts[0].actions[0].stepGroups[0].actionableDeeplink is not None
    assert reparsed.contexts[0].actions[0].stepGroups[0].actionableDeeplink.deeplink == "bixby://masked/act/open_settings"
    assert reparsed == response
