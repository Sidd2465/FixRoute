"""Planning module connecting extracted steps to schema-compliant response assembly."""

from __future__ import annotations

import json
from typing import Any, Sequence

from pydantic import ValidationError

from app.extraction_schemas import ExtractionResult
from app.link_mapping import LinkDecision, apply_link_decision, map_links_for_group
from app.plan_from_extraction import build_extraction_plan
from app.planning_schemas import PlanningParameters
from app.response_builder import build_response
from app.rule_validation import RuleViolation, validate_response_rules
from app.schemas import ContextDeeplinkResponse


def normalize_plan_parameters(p: PlanningParameters) -> PlanningParameters:
    """Safety normalizer guaranteeing supplemental business-rule compliance."""
    # 1. Normalize action_description: must start with "It will " and be 5 to 7 words.
    desc = p.action_description.strip().rstrip(".")
    words = desc.split()
    if not (len(words) >= 2 and words[0].lower() == "it" and words[1].lower() == "will"):
        words = ["It", "will"] + words
    else:
        words[0] = "It"
        words[1] = "will"

    if len(words) < 5:
        padding = ["safely", "apply", "changes", "now"]
        while len(words) < 5:
            words.append(padding.pop(0))
    elif len(words) > 7:
        words = words[:6]

    norm_desc = " ".join(words) + "."

    # 2. Normalize goal_title: must be 2 or 3 words in sentence case.
    title = p.goal_title.strip().rstrip(".")
    t_words = title.split()
    if len(t_words) < 2:
        t_words.append("troubleshooting")
    elif len(t_words) > 3:
        t_words = t_words[:3]

    s = " ".join(t_words)
    norm_title = s[0].upper() + s[1:]

    return PlanningParameters(
        goal_topic=p.goal_topic.strip(),
        goal_title=norm_title,
        action_name=p.action_name.strip(),
        action_description=norm_desc,
        link_query=p.link_query.strip(),
    )


def generate_plan_parameters(
    complaint: str,
    extraction: ExtractionResult,
    *,
    client: Any,
    model_name: str = "gemini-3.6-flash",
) -> PlanningParameters:
    """Generate plan metadata from complaint and extracted steps via Gemini structured output."""
    try:
        from google.genai import types
    except Exception as exc:
        raise RuntimeError("Google Gen AI SDK is required for planning generation.") from exc

    system_instruction = (
        "You are an expert Samsung troubleshooting planner. Given a user complaint "
        "and extracted troubleshooting steps, generate plan metadata adhering strictly "
        "to these requirements:\n"
        "1. goal_topic: A concise noun phrase naming the feature or issue (e.g. 'Full Screen Gesture Function' or 'Email Storage').\n"
        "2. goal_title: EXACTLY 2 or 3 words in sentence case with no trailing punctuation (e.g. 'Full screen gestures' or 'Clear email cache').\n"
        "3. action_name: A short Title Case label for the action (e.g. 'Disable Full Screen Gestures' or 'Clear Email Cache').\n"
        "4. action_description: EXACTLY 5 to 7 words, starting with 'It will' and ending with a period (e.g. 'It will disable full screen gestures.').\n"
        "5. link_query: A 2 to 5 word search query for the device Settings catalogue (e.g. 'navigation bar settings' or 'apps storage settings')."
    )

    steps_text = "\n".join(f"{i+1}. {s.text}" for i, s in enumerate(extraction.steps))
    user_prompt = (
        f"User Complaint:\n{complaint}\n\n"
        f"Extracted Steps:\n{steps_text}\n\n"
        "Generate the plan parameters."
    )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=PlanningParameters.model_json_schema(),
        system_instruction=system_instruction,
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=config,
        )
    except Exception as exc:
        raise RuntimeError(f"Planning LLM call failed: {exc}") from exc

    raw_text = response.text or "{}"
    try:
        data = json.loads(raw_text)
        params = PlanningParameters.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Planning response failed schema validation: {raw_text[:200]}") from exc

    return normalize_plan_parameters(params)


def build_planned_pipeline(
    complaint: str,
    extraction: ExtractionResult,
    params: PlanningParameters,
    deeplink_catalogue: Sequence[dict[str, Any]],
    score: float = 0.0,
) -> tuple[ContextDeeplinkResponse, LinkDecision, list[RuleViolation]]:
    """Assemble the planned pipeline from extraction and planning parameters.

    Returns (response, link_decision, rule_violations).
    """
    plan = build_extraction_plan(
        extraction=extraction,
        goal_topic=params.goal_topic,
        goal_title=params.goal_title,
        action_name=params.action_name,
        action_description=params.action_description,
        score=score,
    )

    link_decision = map_links_for_group(query=params.link_query, catalogue=deeplink_catalogue)
    linked_plan = apply_link_decision(plan=plan, decision=link_decision)
    response = build_response([linked_plan], deeplink_catalogue)
    violations = validate_response_rules(response)

    return response, link_decision, violations