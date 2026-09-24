"""Baseline lexical deeplink retriever.

This module intentionally performs lexical retrieval only. It uses TF-IDF and cosine
similarity over descriptive text; it does not attempt semantic understanding,
probabilistic scoring, or retrieval-augmented reasoning.

Scores are similarity values, not probabilities or confidence scores. A positive
score simply indicates lexical overlap in the indexed text; it does not prove that a
candidate is correct or appropriate for the user's request.
"""

from __future__ import annotations

import copy
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DeeplinkRetriever:
    """Retrieve deeplink catalogue entries by lexical similarity."""

    def __init__(self, entries: list[dict[str, Any]]):
        if not isinstance(entries, list):
            raise ValueError("Catalogue entries must be provided as a list.")
        if not entries:
            raise ValueError("Catalogue is empty.")

        self.entries = copy.deepcopy(entries)
        self.description_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
        )
        self.combined_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
        )

        self.description_documents = [self._description_text(entry) for entry in self.entries]
        self.combined_documents = [self._combined_text(entry) for entry in self.entries]

        self.description_matrix = self.description_vectorizer.fit_transform(
            self.description_documents
        )
        self.combined_matrix = self.combined_vectorizer.fit_transform(
            self.combined_documents
        )

    @staticmethod
    def _pick_text(entry: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @classmethod
    def _description_text(cls, entry: dict[str, Any]) -> str:
        parts = [cls._pick_text(entry, "description")]
        return " ".join(part for part in parts if part)

    @classmethod
    def _combined_text(cls, entry: dict[str, Any]) -> str:
        parts = [
            cls._pick_text(entry, "description"),
            cls._pick_text(entry, "message"),
            cls._pick_text(entry, "qna_description"),
        ]
        return " ".join(part for part in parts if part)

    def _score_candidates(self, query: str) -> list[tuple[float, str, dict[str, Any]]]:
        if not query or not query.strip():
            return []

        query_desc = self.description_vectorizer.transform([query])
        query_combined = self.combined_vectorizer.transform([query])

        description_scores = cosine_similarity(query_desc, self.description_matrix).ravel()
        combined_scores = cosine_similarity(query_combined, self.combined_matrix).ravel()

        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for index, entry in enumerate(self.entries):
            description_score = float(description_scores[index])
            combined_score = float(combined_scores[index])
            score = 0.8 * description_score + 0.2 * combined_score

            if score <= 0:
                continue

            entry_id = str(entry.get("id", ""))
            ranked.append((score, entry_id, copy.deepcopy(entry)))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked

    def search(self, action_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if not isinstance(action_text, str):
            return []

        query = action_text.strip()
        if not query:
            return []

        ranked = self._score_candidates(query)
        if not ranked:
            return []

        limited = ranked[:top_k]
        return [
            {"entry": copy.deepcopy(entry), "similarity_score": float(score)}
            for score, _, entry in limited
        ]
