"""Semantic retrieval over agent fact memory.

Uses ChromaDB when available (proper vector database with persistent
embeddings). Falls back to sentence-transformers in-process, then to
lightweight token-overlap scoring.

Chroma is the right tool for this: embeddings are computed once when
facts are stored, and queries are fast similarity searches — not
re-encoding the entire fact list on every question.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)


class ChromaFactStore:
    """Per-agent fact store backed by ChromaDB.

    Each agent gets a Chroma collection. Facts are embedded once on add.
    Queries are fast vector similarity searches.

    Falls back gracefully when Chroma is not available.
    """

    def __init__(self, agent_id: str, chroma_host: str = "localhost", chroma_port: int = 8001):
        self._agent_id = agent_id
        self._collection = None
        self._client = None

        try:
            import chromadb
            self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            # One collection per agent — agent's own knowledge
            collection_name = f"agent_{agent_id.replace('-', '_')[:50]}"
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"agent_id": agent_id},
            )
            logger.debug(f"Chroma collection ready for agent {agent_id}")
        except Exception as e:
            logger.debug(f"Chroma not available for agent {agent_id}: {e}")
            self._collection = None

    @property
    def available(self) -> bool:
        return self._collection is not None

    def add_fact(self, fact_id: str, text: str, metadata: dict | None = None) -> None:
        """Add a fact to the agent's vector store. Embedding computed once."""
        if not self._collection or not text.strip():
            return
        try:
            self._collection.upsert(
                ids=[fact_id],
                documents=[text.strip()],
                metadatas=[metadata or {}],
            )
        except Exception as e:
            logger.debug(f"Chroma add_fact failed: {e}")

    def query(self, question: str, top_k: int = 5) -> list[tuple[float, str, dict]]:
        """Query agent's knowledge by semantic similarity.

        Returns list of (score, text, metadata) tuples, highest relevance first.
        """
        if not self._collection:
            return []
        try:
            results = self._collection.query(
                query_texts=[question],
                n_results=top_k,
            )
            output: list[tuple[float, str, dict]] = []
            if results and results.get("documents"):
                docs = results["documents"][0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                for i, doc in enumerate(docs):
                    # Chroma returns distances (lower = more similar)
                    # Convert to similarity score (higher = better)
                    distance = distances[i] if i < len(distances) else 1.0
                    score = max(0.0, 1.0 - distance)
                    meta = metadatas[i] if i < len(metadatas) else {}
                    output.append((score, doc, meta))
            return output
        except Exception as e:
            logger.debug(f"Chroma query failed: {e}")
            return []

    @property
    def count(self) -> int:
        if not self._collection:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0


class SemanticFactRetriever:
    """Ranks fact strings by relevance to a query.

    Uses sentence-transformers embeddings when available, otherwise falls back
    to lightweight token-overlap scoring. This is the in-process fallback
    when Chroma is not available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.backend = "token-overlap"
        self._model = None
        self._np = None

        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self._np = np
            self._model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
        except Exception:
            self.backend = "token-overlap"

    def rank(self, query: str, facts: Iterable[str], top_k: int = 5) -> list[tuple[float, int, str]]:
        fact_list = [fact for fact in facts if fact and fact.strip()]
        if not fact_list:
            return []

        if self.backend == "sentence-transformers" and self._model is not None and self._np is not None:
            query_embedding = self._model.encode([query], normalize_embeddings=True)
            fact_embeddings = self._model.encode(fact_list, normalize_embeddings=True)
            scores = (fact_embeddings @ query_embedding[0]).tolist()
            indexed = [(score, idx, text) for idx, (score, text) in enumerate(zip(scores, fact_list))]
            ranked = sorted(indexed, key=lambda item: item[0], reverse=True)
            return ranked[:top_k]

        q_tokens = self._tokenize(query)
        ranked: list[tuple[float, int, str]] = []
        for idx, fact in enumerate(fact_list):
            f_tokens = self._tokenize(fact)
            if not q_tokens:
                score = 0.0
            else:
                score = len(q_tokens & f_tokens) / max(len(q_tokens), 1)
            if score > 0:
                ranked.append((score, idx, fact))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:top_k]

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9']+", text.lower())
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "is", "are", "was", "were",
            "i", "me", "my", "you", "your", "it", "that", "this", "what", "where", "when", "how", "have", "has",
            "had", "do", "did", "so", "can", "could", "would", "should", "be", "been",
        }
        return {word for word in words if word and word not in stop}
