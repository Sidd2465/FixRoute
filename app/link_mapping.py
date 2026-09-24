"""Explicit deeplink gating for extracted plans.

This layer makes a deterministic attachment decision based on the existing lexical
retriever output. It does not invent retrieval logic or mutate the supplied plan.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

from app.plan_schemas import ActionPlan, GoalPlan, StepGroupPlan
from app.retrieval import DeeplinkRetriever
from app.schemas import actionCategory


@dataclass(frozen=True)
class LinkDecision:
    attached: bool
    catalogue_id: str | None
    chosen_score: float | None
    margin: float | None
    reason: str
    candidates: list[dict[str, Any]] = field(default_factory=list)


def map_links_for_group(
    *,
    query: str,
    catalogue: list[dict[str, Any]],
    min_score: float = 0.5,
    min_margin: float = 0.10,
    top_k: int = 3,
) -> LinkDecision:
    """Attach a catalogue entry only when the gating thresholds are satisfied."""
    if isinstance(query, bool) or not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    if not isinstance(catalogue, list):
        raise ValueError("catalogue must be a list of dictionaries.")

    for label, value, lower, upper in (
        ("min_score", min_score, 0.0, 1.0),
        ("min_margin", min_margin, 0.0, 1.0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite number within [{lower}, {upper}].")
        value_float = float(value)
        if not math.isfinite(value_float) or value_float < lower or value_float > upper:
            raise ValueError(f"{label} must be a finite number within [{lower}, {upper}].")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    retriever = DeeplinkRetriever(catalogue)
    ranked = retriever.search(query, top_k=top_k)
    if not ranked:
        return LinkDecision(
            attached=False,
            catalogue_id=None,
            chosen_score=None,
            margin=None,
            reason="no_candidates",
            candidates=[],
        )

    candidates = [
        {
            "id": item["entry"]["id"],
            "score": float(item["similarity_score"]),
            "text": item["entry"].get("message") or item["entry"].get("description") or "",
        }
        for item in ranked
    ]
    top1_score = float(candidates[0]["score"])
    min_score_value = float(min_score)
    if top1_score < min_score_value:
        return LinkDecision(
            attached=False,
            catalogue_id=None,
            chosen_score=top1_score,
            margin=None,
            reason="score_below_threshold",
            candidates=candidates,
        )

    # Deliberate conservative rule: with only one candidate there
    # is no comparative evidence, so the margin is treated as 0
    # and the margin gate fails. A lone candidate can never
    # attach. Do not change this to margin = top1_score.
    second_score = float(candidates[1]["score"]) if len(candidates) > 1 else top1_score
    margin = top1_score - second_score
    if margin < float(min_margin):
        return LinkDecision(
            attached=False,
            catalogue_id=None,
            chosen_score=top1_score,
            margin=float(margin),
            reason="margin_too_small",
            candidates=candidates,
        )

    chosen = candidates[0]
    return LinkDecision(
        attached=True,
        catalogue_id=str(chosen["id"]),
        chosen_score=float(chosen["score"]),
        margin=float(margin),
        reason="attached",
        candidates=candidates,
    )


def apply_link_decision(*, plan: GoalPlan, decision: LinkDecision) -> GoalPlan:
    """Apply a link decision to a plan without mutating the input plan."""
    if not isinstance(plan, GoalPlan):
        raise ValueError("plan must be a GoalPlan instance.")
    if not isinstance(decision, LinkDecision):
        raise ValueError("decision must be a LinkDecision instance.")

    if not decision.attached:
        return copy.deepcopy(plan)

    if len(plan.actions) != 1 or len(plan.actions[0].stepGroups) != 1:
        raise ValueError("Attached decisions require exactly one action and one step group.")

    updated = copy.deepcopy(plan)
    updated.actions[0].category = actionCategory.auto
    updated.actions[0].stepGroups[0].catalogue_id = decision.catalogue_id
    return updated
