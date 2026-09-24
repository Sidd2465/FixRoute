"""Semantic cache for FixRoute troubleshooting queries.

Provides sub-millisecond query lookups using exact normalization and
TF-IDF semantic similarity against stored queries and query variations.
"""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import ContextDeeplinkResponse


def normalize_query_text(text: str) -> str:
    """Normalize query text for exact and semantic comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


class CacheEntry(BaseModel):
    """An individual entry stored in the semantic cache."""

    model_config = ConfigDict(extra="forbid")

    query: str
    normalized_query: str
    query_variations: list[str] = Field(default_factory=list)
    response: ContextDeeplinkResponse
    created_at: float = Field(default_factory=time.time)


class CacheMatch(BaseModel):
    """Result returned when a query hits the cache."""

    model_config = ConfigDict(extra="forbid")

    query: str
    query_variations: list[str]
    response: ContextDeeplinkResponse
    similarity: float
    matched_query: str


class SemanticCache:
    """In-memory and file-backed semantic cache."""

    def __init__(
        self,
        persistence_path: Path | str | None = None,
        default_threshold: float = 0.65,
    ) -> None:
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.default_threshold = default_threshold
        self._entries: list[CacheEntry] = []
        self._corpus: list[str] = []
        self._entry_indices: list[int] = []

        if self.persistence_path and self.persistence_path.is_file():
            self._load_from_disk()

    def __len__(self) -> int:
        return len(self._entries)

    def _rebuild_index(self) -> None:
        """Rebuild flattened corpus mapping variations to entry indices."""
        self._corpus = []
        self._entry_indices = []
        for idx, entry in enumerate(self._entries):
            # Include main query
            self._corpus.append(entry.normalized_query)
            self._entry_indices.append(idx)
            # Include variations
            for var in entry.query_variations:
                norm_var = normalize_query_text(var)
                if norm_var and norm_var != entry.normalized_query:
                    self._corpus.append(norm_var)
                    self._entry_indices.append(idx)

    def get(self, query: str, threshold: float | None = None) -> CacheMatch | None:
        """Look up a query. Returns CacheMatch on hit, None on miss."""
        if not query or not query.strip() or not self._entries:
            return None

        min_sim = threshold if threshold is not None else self.default_threshold
        norm_query = normalize_query_text(query)

        # 1. Exact match check (ultra fast)
        for entry in self._entries:
            if entry.normalized_query == norm_query:
                return CacheMatch(
                    query=entry.query,
                    query_variations=copy.deepcopy(entry.query_variations),
                    response=copy.deepcopy(entry.response),
                    similarity=1.0,
                    matched_query=entry.query,
                )
            for var in entry.query_variations:
                if normalize_query_text(var) == norm_query:
                    return CacheMatch(
                        query=entry.query,
                        query_variations=copy.deepcopy(entry.query_variations),
                        response=copy.deepcopy(entry.response),
                        similarity=1.0,
                        matched_query=var,
                    )

        # 2. Semantic TF-IDF similarity check
        if not self._corpus:
            return None

        try:
            vec_word = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
            vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))

            all_texts = self._corpus + [norm_query]
            m_w = vec_word.fit_transform(all_texts)
            m_c = vec_char.fit_transform(all_texts)

            corpus_len = len(self._corpus)
            corpus_w = m_w[:corpus_len]
            q_w = m_w[corpus_len:]

            corpus_c = m_c[:corpus_len]
            q_c = m_c[corpus_len:]

            sim_w = cosine_similarity(q_w, corpus_w).flatten()
            sim_c = cosine_similarity(q_c, corpus_c).flatten()
            combined = 0.5 * sim_w + 0.5 * sim_c
        except Exception:
            return None

        best_idx = int(combined.argmax())
        best_score = float(combined[best_idx])

        if best_score >= min_sim:
            matched_entry_idx = self._entry_indices[best_idx]
            entry = self._entries[matched_entry_idx]
            matched_text = self._corpus[best_idx]
            return CacheMatch(
                query=entry.query,
                query_variations=copy.deepcopy(entry.query_variations),
                response=copy.deepcopy(entry.response),
                similarity=best_score,
                matched_query=matched_text,
            )

        return None

    def set(
        self,
        query: str,
        response: ContextDeeplinkResponse,
        query_variations: Sequence[str] | None = None,
    ) -> None:
        """Add or update an entry in the cache."""
        norm_query = normalize_query_text(query)
        variations = list(query_variations or [])

        # Check if already present, update response
        for entry in self._entries:
            if entry.normalized_query == norm_query:
                entry.response = copy.deepcopy(response)
                for v in variations:
                    if v not in entry.query_variations:
                        entry.query_variations.append(v)
                self._rebuild_index()
                self._save_to_disk()
                return

        new_entry = CacheEntry(
            query=query.strip(),
            normalized_query=norm_query,
            query_variations=variations,
            response=copy.deepcopy(response),
        )
        self._entries.append(new_entry)
        self._rebuild_index()
        self._save_to_disk()

    def clear(self) -> None:
        """Clear all entries."""
        self._entries = []
        self._corpus = []
        self._entry_indices = []
        if self.persistence_path and self.persistence_path.is_file():
            self.persistence_path.unlink(missing_ok=True)

    def _save_to_disk(self) -> None:
        if not self.persistence_path:
            return
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.model_dump(mode="json") for e in self._entries]
        self.persistence_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_from_disk(self) -> None:
        try:
            content = self.persistence_path.read_text(encoding="utf-8")
            data = json.loads(content)
            self._entries = [CacheEntry.model_validate(item) for item in data]
            self._rebuild_index()
        except Exception:
            self._entries = []
            self._rebuild_index()