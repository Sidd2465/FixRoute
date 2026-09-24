"""Deterministic extraction-to-plan conversion for FixRoute.

This helper intentionally ignores extraction.warnings, extraction.conditions, and
extraction.missing_details when building a plan. Those fields describe source
context and safety cues, not actionable step text. This planner copies only the
ordered extraction steps verbatim into a single manual action group.
"""

from __future__ import annotations

import math
from typing import Any

from app.extraction_schemas import ExtractionResult
from app.plan_schemas import ActionPlan, GoalPlan, StepGroupPlan
from app.schemas import actionCategory


def build_extraction_plan(
    *,
    extraction: ExtractionResult,
    goal_topic: str,
    goal_title: str,
    action_name: str,
    action_description: str,
    score: float = 0.0,
) -> GoalPlan:
    """Build a single manual GoalPlan from a grounded extraction result."""
    if not isinstance(extraction, ExtractionResult):
        raise ValueError("extraction must be an ExtractionResult instance.")
    if not extraction.steps:
        raise ValueError("extraction.steps must be non-empty.")

    for label, value in (
        ("goal_topic", goal_topic),
        ("goal_title", goal_title),
        ("action_name", action_name),
        ("action_description", action_description),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string.")

    if isinstance(score, bool):
        raise ValueError("score must be a finite number within [0, 1].")
    if not isinstance(score, (int, float)):
        raise ValueError("score must be a finite number within [0, 1].")
    if not math.isfinite(float(score)) or float(score) < 0.0 or float(score) > 1.0:
        raise ValueError("score must be a finite number within [0, 1].")

    step_texts = [item.text for item in extraction.steps]

    action_plan = ActionPlan(
        actionName=action_name.strip(),
        description=action_description.strip(),
        category=actionCategory.manual,
        stepGroups=[
            StepGroupPlan(
                steps=list(step_texts),
                catalogue_id=None,
            )
        ],
    )

    goal_text = f"Follow these steps to perform this {goal_topic.strip()} Troubleshooting"

    return GoalPlan(
        goal=goal_text,
        title=goal_title.strip(),
        score=float(score),
        actions=[action_plan],
    )
