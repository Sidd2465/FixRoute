"""Schemas for the FixRoute planning layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlanningParameters(BaseModel):
    """Metadata parameters required to build a compliant GoalPlan and route deep links."""

    model_config = ConfigDict(extra="forbid")

    goal_topic: str = Field(
        description="Feature or issue noun phrase for the goal template, e.g. 'Full Screen Gesture Function'."
    )
    goal_title: str = Field(
        description="EXACTLY 2 or 3 words in sentence case with no trailing punctuation, e.g. 'Full screen gestures'."
    )
    action_name: str = Field(
        description="Title Case action label, e.g. 'Disable Full Screen Gestures'."
    )
    action_description: str = Field(
        description="EXACTLY 5 to 7 words, starting with 'It will' and ending with a period, e.g. 'It will disable full screen gestures.'."
    )
    link_query: str = Field(
        description="A 2 to 5 word search query for the device Settings catalogue, e.g. 'navigation bar settings'."
    )