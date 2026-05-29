"""Semantic retrieval over agent fact memory.

Uses ChromaDB when available (proper vector database with persistent
embeddings). Falls back to the TEI embedding service (S72), then to
lightweight token-overlap scoring when TEI is also unreachable.

Chroma is the right tool for this: embeddings are computed once when
facts are stored, and queries are fast similarity searches — not
re-encoding the entire fact list on every question.

All chroma calls are wrapped in a hard per-call timeout via a shared
ThreadPoolExecutor. Without this, a hung chroma server (sync HttpClient
has no built-in request timeout) parks the asyncio.to_thread worker
calling it forever, which in turn freezes the agent tick loop on the
first iteration after startup. See server-design-plan S98.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, Iterable, TypeVar

from .embeddings import TEIEmbedder, get_embedder

logger = logging.getLogger(__name__)

# Per-call timeout for any chroma operation. Generous enough for normal
# operation, tight enough that a dead chroma can't freeze the sim tick.
_CHROMA_CALL_TIMEOUT_SECONDS = 5.0

# Shared executor for chroma calls. Single pool serves all agent fact
# stores; size scales with expected concurrent in-flight chroma calls
# (one per agent tick under sequential mode is fine, more under Ray).
_CHROMA_EXECUTOR = ThreadPoolExecutor(max_workers=32, thread_name_prefix="chroma")

_T = TypeVar("_T")


def _chroma_call(fn: Callable[[], _T], default: _T, op: str) -> _T:
    """Run a chroma call under a hard timeout. Return default on failure."""
    future = _CHROMA_EXECUTOR.submit(fn)
    try:
        return future.result(timeout=_CHROMA_CALL_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        future.cancel()
        logger.warning(
            f"Chroma {op} timed out after {_CHROMA_CALL_TIMEOUT_SECONDS}s — "
            f"chroma server may be unresponsive"
        )
        return default
    except Exception as e:
        logger.debug(f"Chroma {op} failed: {e}")
        return default


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
            # Wrap connect + collection setup in the same hard timeout —
            # chroma can hang at startup too (e.g. unhealthy container).
            client = _chroma_call(
                lambda: chromadb.HttpClient(host=chroma_host, port=chroma_port),
                default=None, op="connect",
            )
            if client is None:
                self._collection = None
                return
            self._client = client
            collection_name = f"agent_{agent_id.replace('-', '_')[:50]}"
            self._collection = _chroma_call(
                lambda: client.get_or_create_collection(
                    name=collection_name, metadata={"agent_id": agent_id},
                ),
                default=None, op="get_or_create_collection",
            )
            if self._collection is not None:
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
        collection = self._collection
        clean_text = text.strip()
        meta = metadata or {}
        _chroma_call(
            lambda: collection.upsert(
                ids=[fact_id], documents=[clean_text], metadatas=[meta],
            ),
            default=None,
            op="add_fact",
        )

    def query(self, question: str, top_k: int = 5) -> list[tuple[float, str, dict]]:
        """Query agent's knowledge by semantic similarity.

        Returns list of (score, text, metadata) tuples, highest relevance first.
        """
        if not self._collection:
            return []
        collection = self._collection
        results = _chroma_call(
            lambda: collection.query(query_texts=[question], n_results=top_k),
            default=None,
            op="query",
        )
        if not results or not results.get("documents"):
            return []
        output: list[tuple[float, str, dict]] = []
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

    @property
    def count(self) -> int:
        if not self._collection:
            return 0
        collection = self._collection
        return _chroma_call(lambda: collection.count(), default=0, op="count")


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. TEI returns L2-normalized vectors (normalize=True),
    so the dot product already is the cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


class SemanticFactRetriever:
    """Ranks fact strings by relevance to a query.

    Embeddings come from the TEI service (S72: BAAI/bge-m3). When TEI is
    unreachable, falls back to lightweight token-overlap scoring. No
    in-process embedding model is loaded — S72 retired the previous
    sentence-transformers path, which cost ~150-250 MB of PyTorch per
    process and duplicated the network embedding service.

    This is the fallback retriever used when Chroma is not available.
    """

    def __init__(self, embedder: TEIEmbedder | None = None):
        self._embedder = embedder if embedder is not None else get_embedder()
        # Reflects the path the most recent rank() call actually took.
        self.backend = "tei"

    def rank(self, query: str, facts: Iterable[str], top_k: int = 5) -> list[tuple[float, int, str]]:
        fact_list = [fact for fact in facts if fact and fact.strip()]
        if not fact_list:
            return []

        ranked = self._rank_with_tei(query, fact_list, top_k)
        if ranked is not None:
            self.backend = "tei"
            return ranked

        self.backend = "token-overlap"
        return self._rank_with_overlap(query, fact_list, top_k)

    def _rank_with_tei(
        self, query: str, fact_list: list[str], top_k: int
    ) -> list[tuple[float, int, str]] | None:
        """Embed query + facts via TEI and rank by cosine. Returns None if
        TEI is unavailable so the caller can fall back."""
        vectors = self._embedder.embed([query, *fact_list])
        if not vectors or len(vectors) != len(fact_list) + 1:
            return None
        q_vec = vectors[0]
        ranked = [
            (_cosine(q_vec, vec), idx, text)
            for idx, (vec, text) in enumerate(zip(vectors[1:], fact_list))
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[:top_k]

    def _rank_with_overlap(
        self, query: str, fact_list: list[str], top_k: int
    ) -> list[tuple[float, int, str]]:
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
