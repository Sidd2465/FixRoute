"""Tests for the FixRoute semantic cache."""

from __future__ import annotations

import pytest

from app.cache import SemanticCache, normalize_query_text
from app.schemas import Action, ContextDeeplinkResponse, Goal, StepGroup, actionCategory


@pytest.fixture
def dummy_response() -> ContextDeeplinkResponse:
    group = StepGroup(
        groupId="group-1",
        stepGroupTitle="Gestures",
        steps=["Step 1", "Step 2"],
    )
    action = Action(
        actionName="Disable Gestures",
        description="It will disable full screen gestures.",
        stepGroups=[group],
        category=actionCategory.auto,
    )
    goal = Goal(
        goal="Follow these steps to perform this Gestures Troubleshooting",
        title="Full screen gestures",
        actions=[action],
        score=0.0,
    )
    return ContextDeeplinkResponse(contexts=[goal])


def test_normalize_query_text() -> None:
    assert normalize_query_text("  My Screen is Lagging!  ") == "my screen is lagging"
    assert normalize_query_text("Touch-screen, delay???") == "touch screen delay"


def test_cache_miss_on_empty(dummy_response: ContextDeeplinkResponse) -> None:
    cache = SemanticCache()
    assert len(cache) == 0
    assert cache.get("My screen inputs are delayed") is None


def test_cache_exact_hit(dummy_response: ContextDeeplinkResponse) -> None:
    cache = SemanticCache()
    query = "My Galaxy S22 screen inputs are delayed"
    cache.set(query, dummy_response, query_variations=["screen touches lagging"])

    assert len(cache) == 1

    # Exact string hit
    hit = cache.get(query)
    assert hit is not None
    assert hit.similarity == 1.0
    assert len(hit.response.contexts) == 1

    # Normalized casing/punctuation hit
    hit_norm = cache.get("  my galaxy s22 screen inputs are delayed!  ")
    assert hit_norm is not None
    assert hit_norm.similarity == 1.0


def test_cache_variation_hit(dummy_response: ContextDeeplinkResponse) -> None:
    cache = SemanticCache()
    query = "My Galaxy S22 screen inputs are delayed"
    cache.set(query, dummy_response, query_variations=["screen touches lagging"])

    hit_var = cache.get("screen touches lagging")
    assert hit_var is not None
    assert hit_var.similarity == 1.0


def test_cache_semantic_paraphrase_hit(dummy_response: ContextDeeplinkResponse) -> None:
    cache = SemanticCache(default_threshold=0.60)
    query = "My Galaxy S22 screen inputs are delayed and touch responsiveness is laggy"
    cache.set(query, dummy_response)

    # Paraphrased version
    paraphrase = "touch responsiveness is delayed and laggy on my galaxy screen"
    hit = cache.get(paraphrase)
    assert hit is not None
    assert hit.similarity >= 0.60
    assert hit.query == query


def test_cache_semantic_miss_on_unrelated(dummy_response: ContextDeeplinkResponse) -> None:
    cache = SemanticCache(default_threshold=0.60)
    query = "My Galaxy S22 screen inputs are delayed"
    cache.set(query, dummy_response)

    # Completely unrelated query
    hit = cache.get("Email server not responding on phone")
    assert hit is None


def test_cache_clear(dummy_response: ContextDeeplinkResponse) -> None:
    cache = SemanticCache()
    cache.set("query 1", dummy_response)
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert cache.get("query 1") is None


def test_cache_disk_persistence(tmp_path: pytest.TempPathFactory, dummy_response: ContextDeeplinkResponse) -> None:
    cache_file = tmp_path / "test_cache.json"
    cache1 = SemanticCache(persistence_path=cache_file)
    cache1.set("Persistent query", dummy_response, query_variations=["var1"])
    assert len(cache1) == 1

    # Open fresh instance from same file
    cache2 = SemanticCache(persistence_path=cache_file)
    assert len(cache2) == 1
    hit = cache2.get("Persistent query")
    assert hit is not None
    assert hit.query == "Persistent query"