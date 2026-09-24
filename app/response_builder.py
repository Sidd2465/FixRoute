"""Deterministic response assembly for FixRoute plans.

This layer proves only that a response can be assembled in the official schema,
that each requested catalogue ID resolves exactly, and that the result passes the
implemented deterministic rule checks. It does not prove source grounding,
correct screen selection, operation direction, or user-safety outcomes.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from app.plan_schemas import ActionPlan, GoalPlan, StepGroupPlan
from app.rule_validation import RuleViolation, validate_response_rules
from app.schemas import (
    Action,
    ContextDeeplinkResponse,
    Deeplink,
    Goal,
    StepGroup,
    ValidationDeepLink,
    actionCategory,
    Condition,
    ResultTypes,
)


class ResponseBuildError(ValueError):
    """Raised when an assembled response cannot be built or validated."""

    def __init__(self, message: str, violations: list[RuleViolation] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.violations = list(violations) if violations else []


def build_response(
    plans: list[GoalPlan | dict[str, Any]],
    catalogue_entries: list[dict[str, Any]],
) -> ContextDeeplinkResponse:
    """Construct an official response from internal plan models."""
    if not isinstance(plans, list):
        raise ResponseBuildError("plans must be a list of GoalPlan objects or dictionaries.")
    if not isinstance(catalogue_entries, list):
        raise ResponseBuildError("catalogue_entries must be a list of dictionaries.")
    if not plans:
        return ContextDeeplinkResponse(contexts=[])

    resolved_plans: list[GoalPlan] = []
    for index, plan in enumerate(plans):
        try:
            if isinstance(plan, GoalPlan):
                resolved_plan = copy.deepcopy(plan)
            else:
                resolved_plan = GoalPlan.model_validate(plan)
        except ValidationError as exc:
            raise ResponseBuildError(f"Invalid plan at index {index}: {plan!r}", []) from exc

        resolved_plans.append(resolved_plan)

    resolved_catalogue = _index_catalogue(catalogue_entries)

    official_contexts: list[Goal] = []
    for goal_index, goal_plan in enumerate(resolved_plans):
        official_actions: list[Action] = []
        for action_index, action_plan in enumerate(goal_plan.actions):
            official_step_groups: list[StepGroup] = []
            for step_group_index, step_group_plan in enumerate(action_plan.stepGroups):
                _validate_action_category_constraints(action_plan.category, step_group_plan)

                actionable_deeplink = None
                validation_deeplink = None
                catalogue_id = step_group_plan.catalogue_id
                if catalogue_id is not None:
                    entry = _resolve_catalogue_entry(catalogue_id, resolved_catalogue)
                    actionable_deeplink = _build_actionable_deeplink(entry)
                    validation_deeplink = _build_validation_deeplink(entry)

                official_step_groups.append(
                    StepGroup(
                        steps=list(copy.deepcopy(step_group_plan.steps)),
                        validationDeeplink=validation_deeplink,
                        actionableDeeplink=actionable_deeplink,
                    )
                )

            try:
                official_actions.append(
                    Action(
                        actionName=action_plan.actionName,
                        description=action_plan.description,
                        stepGroups=official_step_groups,
                        category=action_plan.category,
                    )
                )
            except ValidationError as exc:
                raise ResponseBuildError(
                    f"Invalid action payload for goal {goal_index}, action {action_index}: {action_plan!r}"
                ) from exc

        try:
            official_contexts.append(
                Goal(
                    goal=goal_plan.goal,
                    title=goal_plan.title,
                    actions=official_actions,
                    score=float(goal_plan.score),
                )
            )
        except ValidationError as exc:
            raise ResponseBuildError(
                f"Invalid goal payload for goal {goal_index}: {goal_plan!r}"
            ) from exc

    try:
        response = ContextDeeplinkResponse(contexts=official_contexts)
    except ValidationError as exc:
        raise ResponseBuildError("Invalid official response assembly for the supplied plans.") from exc

    violations = validate_response_rules(response)
    if violations:
        raise ResponseBuildError(
            "The assembled response failed the deterministic rule validation checks.",
            violations,
        )
    return response


def _index_catalogue(catalogue_entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(catalogue_entries, list):
        raise ResponseBuildError("catalogue_entries must be supplied as a list of dictionaries.")

    catalogue: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(catalogue_entries):
        if not isinstance(entry, dict):
            raise ResponseBuildError(f"Catalogue entry at index {index} must be a dictionary.")

        catalogue_id = entry.get("id")
        if not isinstance(catalogue_id, str) or not catalogue_id.strip():
            raise ResponseBuildError(f"Catalogue entry at index {index} is missing a valid non-empty string 'id'.")

        if catalogue_id in catalogue:
            raise ResponseBuildError(f"Duplicate catalogue_id '{catalogue_id}' was supplied.")

        catalogue[catalogue_id] = copy.deepcopy(entry)

    return catalogue


def _resolve_catalogue_entry(catalogue_id: str, catalogue: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(catalogue_id, str) or not catalogue_id.strip():
        raise ResponseBuildError("Catalogue IDs must be non-empty strings.")

    entry = catalogue.get(catalogue_id)
    if entry is None:
        raise ResponseBuildError(f"Unknown catalogue_id '{catalogue_id}' was requested.")

    return copy.deepcopy(entry)


def _validate_action_category_constraints(category: actionCategory, step_group_plan: StepGroupPlan) -> None:
    if category == actionCategory.manual and step_group_plan.catalogue_id is not None:
        raise ResponseBuildError(
            f"Manual action step group must not specify a catalogue_id; got '{step_group_plan.catalogue_id}'."
        )

    if category == actionCategory.auto and (step_group_plan.catalogue_id is None or not step_group_plan.catalogue_id.strip()):
        raise ResponseBuildError("Auto actions require a catalogue_id for each step group.")


def _build_actionable_deeplink(entry: dict[str, Any]) -> Deeplink:
    entry_id = entry.get("id")
    deeplink = entry.get("deeplink")
    description = entry.get("description")

    if not isinstance(deeplink, str) or not deeplink.strip():
        raise ResponseBuildError(f"Catalogue entry '{entry_id}' is missing a valid deeplink string.")
    if deeplink == "bixby://dummy_positive":
        raise ResponseBuildError(
            "The placeholder bixby://dummy_positive is not permitted in the initial response builder."
        )
    if not isinstance(description, str) or not description.strip():
        raise ResponseBuildError(f"Catalogue entry '{entry_id}' is missing a valid description string.")

    payload: dict[str, Any] = {
        "deeplink": deeplink,
        "description": description,
    }
    if "message" in entry:
        message = entry["message"]
        if message is None or isinstance(message, str):
            payload["message"] = message
        else:
            raise ResponseBuildError(
                f"Catalogue entry '{entry_id}' has invalid actionable message metadata; expected None or str, got {type(message).__name__}."
            )

    if "classes" in entry and entry["classes"] is not None:
        classes = entry["classes"]
        if not isinstance(classes, dict):
            raise ResponseBuildError(f"Catalogue entry '{entry_id}' has invalid 'classes' metadata.")
        payload["classes"] = copy.deepcopy(classes)

    if "originalType" in entry and entry["originalType"] is not None:
        original_type = entry["originalType"]
        if not isinstance(original_type, str):
            raise ResponseBuildError(f"Catalogue entry '{entry_id}' has an invalid 'originalType' value.")
        payload["originalType"] = original_type

    try:
        return Deeplink.model_validate(payload)
    except ValidationError as exc:
        raise ResponseBuildError(
            f"Invalid actionable deeplink metadata for catalogue entry '{entry_id}': {payload!r}"
        ) from exc


def _build_validation_deeplink(entry: dict[str, Any]) -> ValidationDeepLink | None:
    entry_id = entry.get("id")
    validation = entry.get("validation")
    if validation is None:
        return None
    if not isinstance(validation, dict):
        raise ResponseBuildError(
            f"Catalogue entry '{entry_id}' has invalid validation metadata; it must be an object/dictionary."
        )

    allowed_keys = {"deeplink", "key", "resultType", "condition", "value"}
    unexpected = sorted(set(validation) - allowed_keys)
    if unexpected:
        raise ResponseBuildError(
            f"Catalogue entry '{entry_id}' has unsupported validation keys: {unexpected}."
        )

    deeplink = validation.get("deeplink")
    key = validation.get("key")
    if not isinstance(deeplink, str) or not deeplink.strip():
        raise ResponseBuildError(
            f"Catalogue entry '{entry_id}' validation metadata is missing a valid deeplink field."
        )
    if deeplink == "bixby://dummy_positive":
        raise ResponseBuildError(
            "The placeholder bixby://dummy_positive is not permitted in the initial response builder."
        )
    if not isinstance(key, str) or not key.strip():
        raise ResponseBuildError(
            f"Catalogue entry '{entry_id}' validation metadata is missing a valid key value."
        )

    payload: dict[str, Any] = {"deeplink": deeplink, "key": key}

    result_type = validation.get("resultType")
    if "resultType" in validation:
        if result_type is None:
            raise ResponseBuildError(
                f"Catalogue entry '{entry_id}' validation metadata has a null resultType; the value must be a supported enum string."
            )
        try:
            payload["resultType"] = ResultTypes(str(result_type))
        except ValueError as exc:
            raise ResponseBuildError(
                f"Catalogue entry '{entry_id}' has an unsupported validation resultType '{result_type}'."
            ) from exc

    condition = validation.get("condition")
    if "condition" in validation:
        if condition is None:
            raise ResponseBuildError(
                f"Catalogue entry '{entry_id}' validation metadata has a null condition; the value must be a supported enum string."
            )
        try:
            payload["condition"] = Condition(str(condition))
        except ValueError as exc:
            raise ResponseBuildError(
                f"Catalogue entry '{entry_id}' has an unsupported validation condition '{condition}'."
            ) from exc

    value = validation.get("value")
    if "value" in validation:
        if not isinstance(value, str):
            raise ResponseBuildError(
                f"Catalogue entry '{entry_id}' validation value must be a string when supplied."
            )
        payload["value"] = value

    try:
        return ValidationDeepLink.model_validate(payload)
    except ValidationError as exc:
        raise ResponseBuildError(
            f"Invalid validation metadata for catalogue entry '{entry_id}': {payload!r}"
        ) from exc
