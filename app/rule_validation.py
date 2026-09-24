"""Deterministic format validation for troubleshooting responses.

These checks intentionally validate response structure and wording constraints only.
They do not establish source grounding, atomic UI-step correctness, matching of a
real device screen, catalogue membership, operation direction, or real-world safety.
An empty violation list means only that these implemented checks passed; it does not
mean that the response is fully correct in a production or safety-critical sense.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.schemas import Action, ContextDeeplinkResponse, Goal, actionCategory


_URL_PATTERN = re.compile(r"(?i)(?:https?://|www\.)\S+|\[[^\]]+\]\((?:[^\)]*)\)")
_GOAL_TEXT_PATTERN = re.compile(
    r"^Follow these steps to perform this\s+(.+?)\s+(Troubleshooting|Configuration)$"
)

FATAL_VIOLATION_CODES: frozenset[str] = frozenset({
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
})

NON_FATAL_VIOLATION_CODES: frozenset[str] = frozenset({
    "goal_title_word_count",
    "action_description_prefix",
    "action_description_word_count",
})


def is_fatal_code(code: str) -> bool:
    """Authoritative severity split. Unknown codes are treated as
    fatal (fail-safe). Rationale: URL leaks, structural integrity,
    disruptive-last ordering, and category/deeplink taxonomy
    consistency are contract violations; word-count and 'It will'
    prefix checks are soft style boundaries (the official sample
    itself violates description word count)."""
    return code not in NON_FATAL_VIOLATION_CODES


@dataclass(frozen=True)
class RuleViolation:
    code: str
    path: str
    message: str


def validate_response_rules(response: ContextDeeplinkResponse | dict[str, Any]) -> list[RuleViolation]:
    """Validate a response object and return all discovered rule violations."""
    if isinstance(response, dict):
        response = ContextDeeplinkResponse.model_validate(response)

    violations: list[RuleViolation] = []
    for goal_index, goal in enumerate(response.contexts):
        _validate_goal(goal_index, goal, violations)
    _scan_output_strings(response.model_dump(mode="python", exclude_none=False), violations)
    return violations


def _goal_category_value(action: Action | None) -> str:
    if action is None or action.category is None:
        return "manual"
    category = action.category
    if hasattr(category, "value"):
        return str(category.value).lower()
    return str(category).lower()


def _iter_string_values(node: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            next_path = f"{path}.{key}" if path else str(key)
            if isinstance(value, str):
                yield next_path, value
            else:
                yield from _iter_string_values(value, next_path)
        return

    if isinstance(node, list):
        for index, value in enumerate(node):
            next_path = f"{path}[{index}]" if path else f"[{index}]"
            if isinstance(value, str):
                yield next_path, value
            else:
                yield from _iter_string_values(value, next_path)
        return

    if isinstance(node, tuple):
        for index, value in enumerate(node):
            next_path = f"{path}[{index}]" if path else f"[{index}]"
            if isinstance(value, str):
                yield next_path, value
            else:
                yield from _iter_string_values(value, next_path)
        return

    if hasattr(node, "model_dump"):
        yield from _iter_string_values(node.model_dump(mode="python", exclude_none=False), path)


def _contains_web_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text.startswith("bixby://"):
        return False
    return bool(_URL_PATTERN.search(text))


def _scan_output_strings(node: Any, violations: list[RuleViolation]) -> None:
    seen_paths: set[str] = set()
    for path, value in _iter_string_values(node):
        if path in seen_paths:
            continue
        if _contains_web_url(value):
            seen_paths.add(path)
            _add_violation(
                violations,
                "human_readable_url_detected",
                path,
                "Human-readable output fields must not contain HTTP/HTTPS URLs, www-style addresses, or Markdown links.",
            )


def _validate_goal(goal_index: int, goal: Goal, violations: list[RuleViolation]) -> None:
    goal_path = f"contexts[{goal_index}]"
    score = goal.score
    if isinstance(score, bool) or score is None:
        _add_violation(
            violations,
            "goal_score_out_of_range",
            f"{goal_path}.score",
            "Goal score must be a finite number in the range [0, 1].",
        )
    else:
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            _add_violation(
                violations,
                "goal_score_out_of_range",
                f"{goal_path}.score",
                "Goal score must be a finite number in the range [0, 1].",
            )
        else:
            if not math.isfinite(score_value) or score_value < 0 or score_value > 1:
                _add_violation(
                    violations,
                    "goal_score_out_of_range",
                    f"{goal_path}.score",
                    "Goal score must be a finite number in the range [0, 1].",
                )

    title = goal.title or ""
    title_words = title.split()
    if len(title_words) not in (2, 3):
        _add_violation(
            violations,
            "goal_title_word_count",
            f"{goal_path}.title",
            "Goal title must contain 2–3 whitespace-separated words.",
        )

    goal_text = goal.goal or ""
    match = _GOAL_TEXT_PATTERN.fullmatch(goal_text.strip())
    if not match:
        _add_violation(
            violations,
            "goal_text_format",
            f"{goal_path}.goal",
            "Goal text must match 'Follow these steps to perform this <Topic> Troubleshooting' or '... Configuration'.",
        )
    else:
        topic = match.group(1).strip()
        if not topic:
            _add_violation(
                violations,
                "goal_text_format",
                f"{goal_path}.goal",
                "Goal text topic must contain non-whitespace text.",
            )

    if not goal.actions:
        _add_violation(
            violations,
            "empty_goal_actions",
            f"{goal_path}.actions",
            "Goal actions must include at least one action.",
        )

    critical_seen = False
    for action_index, action in enumerate(goal.actions):
        action_path = f"{goal_path}.actions[{action_index}]"
        category = _goal_category_value(action)

        if category == "critical":
            critical_seen = True
        elif critical_seen:
            _add_violation(
                violations,
                "critical_action_order",
                action_path,
                "Once a critical action appears, no non-critical action may follow it.",
            )

        description = action.description or ""
        words = description.strip().split()
        first_two_words = words[:2]
        if first_two_words != ["It", "will"]:
            _add_violation(
                violations,
                "action_description_prefix",
                f"{action_path}.description",
                "Action description must begin with the exact words 'It will'.",
            )
        if not (5 <= len(words) <= 7):
            _add_violation(
                violations,
                "action_description_word_count",
                f"{action_path}.description",
                "Action description must contain 5–7 whitespace-separated words.",
            )

        if not action.stepGroups:
            _add_violation(
                violations,
                "empty_step_groups",
                f"{action_path}.stepGroups",
                "Actions must include at least one step group.",
            )

        for step_group_index, step_group in enumerate(action.stepGroups):
            step_group_path = f"{action_path}.stepGroups[{step_group_index}]"
            if not step_group.steps:
                _add_violation(
                    violations,
                    "empty_steps_list",
                    f"{step_group_path}.steps",
                    "Step groups must contain at least one step.",
                )

            for step_index, step in enumerate(step_group.steps):
                step_path = f"{step_group_path}.steps[{step_index}]"
                if step is None or not isinstance(step, str) or not step.strip():
                    _add_violation(
                        violations,
                        "blank_step_string",
                        step_path,
                        "Step strings must be non-empty after trimming whitespace.",
                    )

            if category == "manual" and step_group.actionableDeeplink is not None:
                _add_violation(
                    violations,
                    "manual_action_has_actionable_deeplink",
                    f"{step_group_path}.actionableDeeplink",
                    "Manual actions must not carry an actionableDeeplink.",
                )
            if category == "auto" and step_group.actionableDeeplink is None:
                _add_violation(
                    violations,
                    "auto_action_missing_actionable_deeplink",
                    f"{step_group_path}.actionableDeeplink",
                    "Auto actions must have an actionableDeeplink for each step group.",
                )


def _add_violation(violations: list[RuleViolation], code: str, path: str, message: str) -> None:
    violations.append(RuleViolation(code=code, path=path, message=message))
