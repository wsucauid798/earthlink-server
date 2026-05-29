"""World semantic layer — Chroma's real job (S79–S82).

Chroma is no longer per-agent memory (that moved to pgvector, S75–S78). Its
role now is a SINGLE shared collection over civilisation content the world has
resolved from Earth via EarthProxy adapters. Agents semantic-search this live
layer by location / topic / goal over the recent window.

- Embeddings: TEI (S72, BAAI/bge-m3) — same model as pgvector, passed to
  Chroma explicitly so Chroma never runs its own embedding function.
- Async throughout (`chromadb.AsyncHttpClient`) — satisfies S91's intent. The
  legacy sync `ChromaFactStore` is being removed (S99/S78), so there is nothing
  left to refactor there; the world store is async from the start and lives off
  the tick thread.
- TTL: Chroma has no native expiry, so each document carries an `expires_at`
  epoch in metadata, aligned to the Redis TTL (S80). Queries filter on it and a
  periodic `prune()` deletes lapsed rows.

Degrades gracefully: if Chroma or TEI is unreachable, writes are skipped and
queries return [] — never raises, never blocks the resolution path.
"""

from __future__ import annotations

import hashlib
import logging

from agents.embeddings import TEIEmbedder, get_embedder

logger = logging.getLogger(__name__)

COLLECTION_NAME = "world_semantic"


def _doc_id(domain: str, source: str, location_id: int | None, text: str) -> str:
    """Stable id so re-resolving the same fact upserts (refreshes) rather than
    duplicating. Content-addressed within (domain, source, location)."""
    h = hashlib.sha1(f"{domain}|{source}|{location_id}|{text}".encode()).hexdigest()
    return h[:32]


class WorldSemanticStore:
    """Async Chroma-backed semantic index over EarthProxy civilisation content."""

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        embedder: TEIEmbedder | None = None,
    ):
        self._host = chroma_host
        self._port = chroma_port
        self._embedder = embedder if embedder is not None else get_embedder()
        self._client = None
        self._collection = None

    @property
    def available(self) -> bool:
        return self._collection is not None

    async def connect(self) -> bool:
        """Connect and get/create the shared collection. Returns availability."""
        try:
            import chromadb

            self._client = await chromadb.AsyncHttpClient(host=self._host, port=self._port)
            self._collection = await self._client.get_or_create_collection(
                name=COLLECTION_NAME, metadata={"role": "world_semantic_layer"}
            )
            logger.info("World semantic layer connected (collection=%s)", COLLECTION_NAME)
            return True
        except Exception as e:
            logger.warning("World semantic layer unavailable: %s", e)
            self._collection = None
            return False

    async def add_facts(self, facts: list, expires_at: float, tick_count: int | None = None) -> int:
        """Embed (TEI) and upsert resolved EarthFacts. Returns rows written.

        `facts` are EarthFact-like objects (have .domain/.topic/.text/.source/
        .location_id/.confidence). `expires_at` is the epoch the docs lapse at
        (aligned to the Redis TTL by the caller)."""
        if not self.available:
            return 0
        items = [f for f in facts if getattr(f, "text", "").strip()]
        if not items:
            return 0

        texts = [f.text.strip() for f in items]
        vectors = await self._embedder.embed_async(texts)
        if not vectors or len(vectors) != len(texts):
            logger.debug("world semantic add skipped: TEI unavailable (%d facts)", len(texts))
            return 0

        ids, metadatas = [], []
        for f in items:
            ids.append(_doc_id(f.domain, f.source, f.location_id, f.text))
            metadatas.append(
                {
                    "domain": f.domain or "",
                    "topic": f.topic or "",
                    "source": f.source or "",
                    "location_id": int(f.location_id) if f.location_id is not None else -1,
                    "confidence": float(getattr(f, "confidence", 0.8)),
                    "expires_at": float(expires_at),
                    "tick_count": int(tick_count) if tick_count is not None else -1,
                }
            )
        try:
            await self._collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
            return len(ids)
        except Exception as e:
            logger.debug("world semantic upsert failed: %s", e)
            return 0

    async def query(
        self,
        question: str,
        top_k: int = 5,
        *,
        location_id: int | None = None,
        domain: str | None = None,
        now: float | None = None,
    ) -> list[tuple[float, str, dict]]:
        """Semantic-search the live world layer. Returns (score, text, metadata),
        most relevant first, excluding lapsed (expired) documents."""
        if not self.available or not question.strip():
            return []
        vectors = await self._embedder.embed_async([question])
        if not vectors:
            return []

        clauses = []
        if location_id is not None:
            clauses.append({"location_id": {"$eq": int(location_id)}})
        if domain is not None:
            clauses.append({"domain": {"$eq": domain}})
        if now is not None:
            clauses.append({"expires_at": {"$gt": float(now)}})
        where = None if not clauses else (clauses[0] if len(clauses) == 1 else {"$and": clauses})

        try:
            res = await self._collection.query(
                query_embeddings=[vectors[0]], n_results=top_k, where=where
            )
        except Exception as e:
            logger.debug("world semantic query failed: %s", e)
            return []

        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        out = []
        for i, doc in enumerate(docs):
            dist = dists[i] if i < len(dists) else 1.0
            out.append((max(0.0, 1.0 - float(dist)), doc, metas[i] if i < len(metas) else {}))
        return out

    async def prune(self, now: float) -> int:
        """Delete lapsed documents (expires_at <= now). Returns best-effort."""
        if not self.available:
            return 0
        try:
            await self._collection.delete(where={"expires_at": {"$lte": float(now)}})
            return 1
        except Exception as e:
            logger.debug("world semantic prune failed: %s", e)
            return 0

    async def count(self) -> int:
        if not self.available:
            return 0
        try:
            return await self._collection.count()
        except Exception:
            return 0


_STORE: WorldSemanticStore | None = None


def get_world_semantic_store(chroma_host: str = "localhost", chroma_port: int = 8000) -> WorldSemanticStore:
    global _STORE
    if _STORE is None:
        _STORE = WorldSemanticStore(chroma_host=chroma_host, chroma_port=chroma_port)
    return _STORE
