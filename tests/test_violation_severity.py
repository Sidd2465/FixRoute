import pytest

from app.rule_validation import (
    FATAL_VIOLATION_CODES,
    NON_FATAL_VIOLATION_CODES,
    is_fatal_code,
)


@pytest.mark.parametrize("code", [
    "human_readable_url_detected",
    "goal_score_out_of_range",
    "goal_text_format",
    "empty_goal_actions",
    "critical_action_order",
    "empty_step_groups",
    "empty_steps_list",
    "blank_step_string",
    "manual_action_has_actionable_deeplink",
    "auto_action_missing_actionable_deeplink",
])
def test_fatal_codes_are_fatal(code):
    assert is_fatal_code(code) is True


@pytest.mark.parametrize("code", [
    "goal_title_word_count",
    "action_description_prefix",
    "action_description_word_count",
])
def test_non_fatal_codes_are_not_fatal(code):
    assert is_fatal_code(code) is False


def test_unknown_code_is_fatal_by_default():
    assert is_fatal_code("future_rule") is True


def test_severity_sets_are_disjoint_and_cover_all_known_codes():
    combined = FATAL_VIOLATION_CODES | NON_FATAL_VIOLATION_CODES
    assert FATAL_VIOLATION_CODES.isdisjoint(NON_FATAL_VIOLATION_CODES)
    assert len(combined) == 13
    assert combined == {
        "human_readable_url_detected",
        "goal_score_out_of_range",
        "goal_text_format",
        "empty_goal_actions",
        "critical_action_order",
        "empty_step_groups",
        "empty_steps_list",
        "blank_step_string",
        "manual_action_has_actionable_deeplink",
        "auto_action_missing_actionable_deeplink",
        "goal_title_word_count",
        "action_description_prefix",
        "action_description_word_count",
    }
