"""Unified end-to-end troubleshooting pipeline service for FixRoute."""

from __future__ import annotations

import time
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.cache import SemanticCache
from app.extraction import extract_from_reference
from app.planning import build_planned_pipeline, generate_plan_parameters
from app.reference_routing import route_complaint, select_subsection, _extract_markdown_section
from app.rule_validation import is_fatal_code
from app.schemas import ContextDeeplinkResponse


class MetaInfo(BaseModel):
    """Metadata for the troubleshooting response envelope."""

    model_config = ConfigDict(extra="forbid")

    latency_ms: float
    cache_hit: bool
    model: str
    cost_usd: float


class TroubleshootEnvelope(BaseModel):
    """Official Theme 2 response envelope."""

    model_config = ConfigDict(extra="forbid")

    query: str
    query_variations: list[str] = Field(default_factory=list)
    response: ContextDeeplinkResponse
    meta: MetaInfo


class TroubleshootPipeline:
    """Orchestrates caching, routing, extraction, planning, linking, and response assembly."""

    def __init__(
        self,
        deeplink_catalogue: Sequence[dict[str, Any]],
        reference_catalogue: Sequence[dict[str, Any]],
        cache: SemanticCache | None = None,
        client: Any | None = None,
        model_name: str = "gemini-3.6-flash",
    ) -> None:
        self.deeplink_catalogue = deeplink_catalogue
        self.reference_catalogue = reference_catalogue
        self.cache = cache if cache is not None else SemanticCache()
        self.client = client
        self.model_name = model_name

    def process(
        self,
        query: str,
        siis_response: dict[str, Any] | None = None,
    ) -> TroubleshootEnvelope:
        """Process a troubleshooting query end-to-end."""
        start_time = time.perf_counter()
        clean_query = query.strip()

        # 1. Semantic Cache check
        cached_hit = self.cache.get(clean_query)
        if cached_hit is not None:
            latency = (time.perf_counter() - start_time) * 1000.0
            return TroubleshootEnvelope(
                query=clean_query,
                query_variations=cached_hit.query_variations,
                response=cached_hit.response,
                meta=MetaInfo(
                    latency_ms=round(latency, 2),
                    cache_hit=True,
                    model="semantic_cache",
                    cost_usd=0.0,
                ),
            )

        # 2. Reference & Subsection determination
        content: str = ""
        if siis_response is not None:
            raw_content = str(siis_response.get("content", "")).strip()
            # Attempt subsection routing if content has sections
            sub_decision, _ = select_subsection(complaint=clean_query, content=raw_content)
            if sub_decision.selected_heading is not None:
                try:
                    content = _extract_markdown_section(raw_content, sub_decision.selected_heading)
                except Exception:
                    content = raw_content
            else:
                content = raw_content
        else:
            # Route complaint across reference catalogue
            routing = route_complaint(complaint=clean_query, references=self.reference_catalogue)
            if routing.reference_id is None or not (routing.content or "").strip():
                # No grounded reference match: return empty response as required
                latency = (time.perf_counter() - start_time) * 1000.0
                return TroubleshootEnvelope(
                    query=clean_query,
                    query_variations=[],
                    response=ContextDeeplinkResponse(contexts=[]),
                    meta=MetaInfo(
                        latency_ms=round(latency, 2),
                        cache_hit=False,
                        model=self.model_name,
                        cost_usd=0.0,
                    ),
                )
            content = routing.content or ""

        # 3. Grounded Step Extraction
        if self.client is None:
            raise RuntimeError("Pipeline requires an active GenAI client for cold extraction.")

        extraction = extract_from_reference(
            content,
            client=self.client,
            model_name=self.model_name,
        )

        if not extraction.steps:
            latency = (time.perf_counter() - start_time) * 1000.0
            return TroubleshootEnvelope(
                query=clean_query,
                query_variations=[],
                response=ContextDeeplinkResponse(contexts=[]),
                meta=MetaInfo(
                    latency_ms=round(latency, 2),
                    cache_hit=False,
                    model=self.model_name,
                    cost_usd=0.0,
                ),
            )

        # 4. Planning Layer
        params = generate_plan_parameters(
            clean_query,
            extraction,
            client=self.client,
            model_name=self.model_name,
        )

        # 5. Link Mapping & Response Assembly
        response, link_decision, violations = build_planned_pipeline(
            complaint=clean_query,
            extraction=extraction,
            params=params,
            deeplink_catalogue=self.deeplink_catalogue,
        )

        # 6. Cache population if valid and zero fatal violations
        fatal_violations = [v for v in violations if is_fatal_code(v.code)]
        variations = [v for v in [params.link_query, params.goal_title] if v]
        if response.contexts and not fatal_violations:
            self.cache.set(clean_query, response, query_variations=variations)

        latency = (time.perf_counter() - start_time) * 1000.0
        return TroubleshootEnvelope(
            query=clean_query,
            query_variations=variations,
            response=response,
            meta=MetaInfo(
                latency_ms=round(latency, 2),
                cache_hit=False,
                model=self.model_name,
                cost_usd=0.0,
            ),
        )