import copy

import pytest

from app.rule_validation import RuleViolation, validate_response_rules
from app.schemas import Action, ContextDeeplinkResponse, Goal, StepGroup, actionCategory


def _goal(goal_text: str, title: str, actions: list[Action], score: float = 0.8) -> Goal:
    return Goal(goal=goal_text, title=title, actions=actions, score=score)


def _manual_action(
    description: str = "It will open settings page for backup.",
    step_groups: list[StepGroup] | None = None,
) -> Action:
    return Action(
        actionName="Example action",
        description=description,
        stepGroups=[StepGroup(steps=["Open Settings."], actionableDeeplink=None)]
        if step_groups is None
        else step_groups,
        category=actionCategory.manual,
    )


def _auto_action(
    description: str = "It will open settings page for backup.",
    step_groups: list[StepGroup] | None = None,
) -> Action:
    return Action(
        actionName="Example action",
        description=description,
        stepGroups=[
            StepGroup(
                steps=["Open Settings."],
                actionableDeeplink={
                    "deeplink": "bixby://masked/act/example",
                    "description": "Example deeplink",
                    "message": "Open settings",
                    "originalType": "onURL",
                },
            )
        ]
        if step_groups is None
        else step_groups,
        category=actionCategory.auto,
    )


@pytest.mark.parametrize(
    ("score", "should_pass"),
    [
        (0, True),
        (1, True),
        (-0.1, False),
        (1.1, False),
        (float("nan"), False),
        (float("inf"), False),
        (-float("inf"), False),
    ],
)
def test_goal_score_boundaries(score, should_pass):
    response = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [_manual_action("It will open settings for Wi-Fi.")],
                score=score,
            )
        ]
    )
    violations = validate_response_rules(response)
    has_violation = any(v.code == "goal_score_out_of_range" for v in violations)
    assert has_violation is (not should_pass)


@pytest.mark.parametrize(
    ("title", "should_pass"),
    [
        ("Wi Fi", True),
        ("Wi Fi display", True),
        ("Wi", False),
        ("Wi Fi display now", False),
    ],
)
def test_goal_title_word_count_boundary(title, should_pass):
    response = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                title,
                [_manual_action("It will open settings for Wi-Fi.")],
                score=0.8,
            )
        ]
    )
    violations = validate_response_rules(response)
    has_violation = any(v.code == "goal_title_word_count" for v in violations)
    assert has_violation is (not should_pass)


def test_goal_text_format_requires_non_empty_topic():
    valid = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [_manual_action("It will open settings for Wi-Fi.")],
            )
        ]
    )
    assert not any(v.code == "goal_text_format" for v in validate_response_rules(valid))

    invalid = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Troubleshooting",
                "Wi Fi",
                [_manual_action("It will open settings for Wi-Fi.")],
            )
        ]
    )
    assert any(v.code == "goal_text_format" for v in validate_response_rules(invalid))

    whitespace_topic = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this    Troubleshooting",
                "Wi Fi",
                [_manual_action("It will open settings for Wi-Fi.")],
            )
        ]
    )
    assert any(v.code == "goal_text_format" for v in validate_response_rules(whitespace_topic))


@pytest.mark.parametrize(
    ("description", "should_pass"),
    [
        ("It will open the settings page.", True),
        ("It will open the settings page now.", True),
        ("It will open.", False),
        ("It will open the settings page today for a check.", False),
    ],
)
def test_action_description_length_and_prefix_boundaries(description, should_pass):
    response = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [_manual_action(description)],
            )
        ]
    )
    violations = validate_response_rules(response)
    has_prefix_violation = any(v.code == "action_description_prefix" for v in violations)
    has_length_violation = any(v.code == "action_description_word_count" for v in violations)
    assert (has_prefix_violation or has_length_violation) is (not should_pass)


def test_action_description_prefix_regression_rejects_it_willow():
    response = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [_manual_action("It willow open the settings page.")],
            )
        ]
    )
    assert any(v.code == "action_description_prefix" for v in validate_response_rules(response))


@pytest.mark.parametrize(
    ("path_value", "should_pass"),
    [
        ("https://example.com", False),
        ("www.example.com", False),
        ("[guide](https://example.com)", False),
        ("[guide](/help)", False),
        ("bixby://masked/act/open_settings", True),
        ("Open Settings.", True),
    ],
)
def test_human_readable_fields_reject_web_urls_and_markdown_links(path_value, should_pass):
    response = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [
                    Action(
                        actionName="Open support page",
                        description="It will open the settings page.",
                        stepGroups=[
                            StepGroup(
                                steps=[path_value],
                                actionableDeeplink={
                                    "deeplink": "bixby://masked/act/open_settings",
                                    "description": "Open the settings page.",
                                    "message": "Open settings",
                                    "originalType": "onURL",
                                },
                            )
                        ],
                    )
                ],
            )
        ]
    )
    violations = validate_response_rules(response)
    has_violation = any(v.code == "human_readable_url_detected" for v in violations)
    assert has_violation is (not should_pass)

    if path_value.startswith("bixby://"):
        assert not any(v.code == "human_readable_url_detected" for v in validate_response_rules(response))

    nested = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [
                    Action(
                        actionName="Open support page",
                        description="It will open the settings page.",
                        stepGroups=[
                            StepGroup(
                                steps=["Open Settings."],
                                actionableDeeplink={
                                    "deeplink": "bixby://masked/act/open_settings",
                                    "description": f"Open {path_value}",
                                    "message": f"Open {path_value}",
                                    "originalType": "onURL",
                                },
                            )
                        ],
                    )
                ],
            )
        ]
    )
    if not should_pass:
        assert any(v.code == "human_readable_url_detected" for v in validate_response_rules(nested))


def test_manual_and_auto_action_deeplink_rules():
    manual = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [
                    Action(
                        actionName="Open settings",
                        description="It will open the settings page.",
                        stepGroups=[
                            StepGroup(
                                steps=["Open Settings."],
                                actionableDeeplink={
                                    "deeplink": "bixby://masked/act/open_settings",
                                    "description": "Open the settings page.",
                                    "message": "Open settings",
                                    "originalType": "onURL",
                                },
                            )
                        ],
                        category=actionCategory.manual,
                    )
                ],
            )
        ]
    )
    assert any(v.code == "manual_action_has_actionable_deeplink" for v in validate_response_rules(manual))

    auto = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [
                    Action(
                        actionName="Open settings",
                        description="It will open the settings page.",
                        stepGroups=[StepGroup(steps=["Open Settings."], actionableDeeplink=None)],
                        category=actionCategory.auto,
                    )
                ],
            )
        ]
    )
    assert any(v.code == "auto_action_missing_actionable_deeplink" for v in validate_response_rules(auto))


@pytest.mark.parametrize(
    ("goal_actions", "expected_code"),
    [
        ([], "empty_goal_actions"),
        ([
            Action(
                actionName="Critical step",
                description="It will open the settings page.",
                stepGroups=[StepGroup(steps=["Open Settings."], actionableDeeplink=None)],
                category=actionCategory.critical,
            ),
            Action(
                actionName="Follow up step",
                description="It will continue the process.",
                stepGroups=[StepGroup(steps=["Continue."], actionableDeeplink=None)],
                category=actionCategory.manual,
            ),
        ], "critical_action_order"),
        ([
            Action(
                actionName="Open settings",
                description="It will open the settings page.",
                stepGroups=[],
                category=actionCategory.manual,
            )
        ], "empty_step_groups"),
        ([
            Action(
                actionName="Open settings",
                description="It will open the settings page.",
                stepGroups=[StepGroup(steps=["   "], actionableDeeplink=None)],
                category=actionCategory.manual,
            )
        ], "blank_step_string"),
        ([
            Action(
                actionName="Open settings",
                description="It will open the settings page.",
                stepGroups=[StepGroup(steps=[], actionableDeeplink=None)],
                category=actionCategory.manual,
            )
        ], "empty_steps_list"),
    ],
)
def test_boundary_and_empty_structures(goal_actions, expected_code):
    response = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                goal_actions,
            )
        ]
    )
    assert any(v.code == expected_code for v in validate_response_rules(response))

    empty_contexts = ContextDeeplinkResponse(contexts=[])
    assert validate_response_rules(empty_contexts) == []


def test_multiple_simultaneous_violations_and_valid_examples():
    response = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [
                    Action(
                        actionName="Open https://example.com",
                        description="Please open the settings page.",
                        stepGroups=[StepGroup(steps=["Open Settings."], actionableDeeplink=None)],
                    )
                ],
                score=float("nan"),
            )
        ]
    )
    violations = validate_response_rules(response)
    assert {v.code for v in violations} >= {
        "goal_score_out_of_range",
        "action_description_prefix",
        "human_readable_url_detected",
    }

    valid = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Configuration",
                "Wi Fi",
                [
                    Action(
                        actionName="Open settings",
                        description="It will open the settings page.",
                        stepGroups=[
                            StepGroup(
                                steps=["Open Settings."],
                                actionableDeeplink={
                                    "deeplink": "bixby://masked/act/open_settings",
                                    "description": "Open the settings page.",
                                    "message": "Open settings",
                                    "originalType": "onURL",
                                },
                            )
                        ],
                        category=actionCategory.auto,
                    )
                ],
            )
        ]
    )
    assert validate_response_rules(valid) == []


def test_validation_does_not_mutate_input():
    original = ContextDeeplinkResponse(
        contexts=[
            _goal(
                "Follow these steps to perform this Wi-Fi Troubleshooting",
                "Wi Fi",
                [
                    Action(
                        actionName="Open settings",
                        description="It will open the settings page.",
                        stepGroups=[StepGroup(steps=["Open Settings."], actionableDeeplink=None)],
                    )
                ],
            )
        ]
    )
    snapshot = copy.deepcopy(original)

    validate_response_rules(original)

    assert original == snapshot


def test_sample_response_has_known_rule_violations():
    from pathlib import Path

    import json

    sample_path = Path(__file__).resolve().parents[1] / "student_kit" / "sample_output.json"
    with sample_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    response = ContextDeeplinkResponse.model_validate(payload["response"])
    violations = validate_response_rules(response)

    assert any(v.code == "action_description_word_count" for v in violations)
    assert any(v.code == "action_description_prefix" for v in violations) is False
    assert any(v.code == "goal_text_format" for v in violations) is False
    assert violations
