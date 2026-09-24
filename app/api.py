"""FastAPI REST API for FixRoute Guided Troubleshooting Engine."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.cache import SemanticCache
from app.data_loader import load_deeplinks, load_references
from app.pipeline import TroubleshootEnvelope, TroubleshootPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEEPLINKS_PATH = PROJECT_ROOT / "student_kit" / "deeplinks.json"
DEFAULT_SIIS_PATH = PROJECT_ROOT / "student_kit" / "siis_responses.json"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "cache.json"


class TroubleshootRequest(BaseModel):
    """Payload accepted by POST /v1/troubleshoot."""

    query: str = Field(..., description="User complaint or troubleshooting query")
    siis_response: dict[str, Any] | None = Field(
        default=None,
        description="Optional SIIS response payload with title and content",
    )


class HealthResponse(BaseModel):
    """Payload returned by GET /health."""

    status: str
    ready: bool
    cache_size: int
    deeplinks_count: int
    references_count: int


def _init_default_pipeline() -> TroubleshootPipeline:
    """Initialize the default production pipeline from disk assets and .env."""
    deeplinks = load_deeplinks(DEFAULT_DEEPLINKS_PATH)
    references = load_references(DEFAULT_SIIS_PATH)
    cache = SemanticCache(persistence_path=DEFAULT_CACHE_PATH)

    client = None
    try:
        from dotenv import dotenv_values

        values = dotenv_values(PROJECT_ROOT / ".env")
        key = values.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if key and str(key).strip():
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=str(key).strip(),
                http_options=types.HttpOptions(
                    timeout=30_000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
    except Exception:
        client = None

    return TroubleshootPipeline(
        deeplink_catalogue=deeplinks,
        reference_catalogue=references,
        cache=cache,
        client=client,
    )


def create_app(pipeline: TroubleshootPipeline | None = None) -> FastAPI:
    """Application factory for FixRoute API."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if not hasattr(app.state, "pipeline") or app.state.pipeline is None:
            app.state.pipeline = _init_default_pipeline()
        yield

    app = FastAPI(
        title="FixRoute - Smart Guided Troubleshooting Engine",
        version="1.0.0",
        description="Theme 2 GenAI Troubleshooting Engine converting user complaints to grounded One UI actions.",
        lifespan=lifespan,
    )

    if pipeline is not None:
        app.state.pipeline = pipeline

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current_pipeline: TroubleshootPipeline = app.state.pipeline
        return HealthResponse(
            status="ok",
            ready=True,
            cache_size=len(current_pipeline.cache),
            deeplinks_count=len(current_pipeline.deeplink_catalogue),
            references_count=len(current_pipeline.reference_catalogue),
        )

    @app.post(
        "/v1/troubleshoot",
        response_model=TroubleshootEnvelope,
        status_code=status.HTTP_200_OK,
    )
    def troubleshoot(payload: TroubleshootRequest) -> TroubleshootEnvelope:
        clean_query = payload.query.strip()
        if not clean_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query string cannot be empty or whitespace.",
            )

        current_pipeline: TroubleshootPipeline = app.state.pipeline
        try:
            return current_pipeline.process(
                query=clean_query,
                siis_response=payload.siis_response,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline processing failed: {exc}",
            ) from exc

    return app


# Default app instance for uvicorn
app = create_app()