from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.schemas import actionCategory


class StepGroupPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[str]
    catalogue_id: str | None = None


class ActionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actionName: str
    description: str
    category: actionCategory
    stepGroups: list[StepGroupPlan]


class GoalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    title: str
    score: float
    actions: list[ActionPlan]
