"""pgvector-backed agent semantic memory (S75).

Replaces the per-agent ``ChromaFactStore`` for committed agent memory. Where
Chroma kept one collection per agent and embedded server-side, this store is
a single ``agent_facts`` table scoped by ``agent_id``, with embeddings
computed client-side through the TEI service (S72: BAAI/bge-m3, 1024 dims).

Ranking uses cosine distance on the HNSW index built in migration
``c5e7a09f8b21`` (``vector_cosine_ops``). TEI returns L2-normalized vectors,
so cosine distance is the canonical operator and ``score = 1 - distance``.

Async by construction (the DB layer is async). Every method accepts an
optional ``session`` so the S76 write path can persist facts inside the same
transaction as the rest of an agent's knowledge state. When no session is
passed the store opens and commits its own.

Graceful degradation mirrors S72/S98: if TEI cannot embed, ``query`` returns
``[]`` and writes are skipped (logged) rather than raising — a dead embedding
service must never wedge the tick loop. Facts still live in the in-memory
knowledge state and can be backfilled later (S78).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db.engine import async_session as _default_session_factory
from db.models import AgentFact

from .embeddings import TEIEmbedder, get_embedder

logger = logging.getLogger(__name__)


class PgVectorFactStore:
    """Agent fact memory backed by the pgvector ``agent_facts`` table."""

    def __init__(
        self,
        session_factory: async_sessionmaker | None = None,
        embedder: TEIEmbedder | None = None,
    ):
        self._session_factory = session_factory or _default_session_factory
        self._embedder = embedder if embedder is not None else get_embedder()

    # -- writes ----------------------------------------------------------

    async def add_fact(
        self,
        agent_id: str,
        content: str,
        *,
        location_id: int | None = None,
        tick_count: int | None = None,
        metadata: dict | None = None,
        session: AsyncSession | None = None,
    ) -> int:
        """Embed and persist one fact. Returns the number of rows written."""
        return await self.add_facts(
            agent_id,
            [
                {
                    "content": content,
                    "location_id": location_id,
                    "tick_count": tick_count,
                    "metadata": metadata,
                }
            ],
            session=session,
        )

    async def add_facts(
        self,
        agent_id: str,
        facts: Iterable[dict],
        *,
        session: AsyncSession | None = None,
    ) -> int:
        """Embed a batch of facts in one TEI call and persist them.

        Each fact is a dict with ``content`` (required) and optional
        ``location_id``, ``tick_count``, ``metadata``. Returns the number of
        rows written (0 if TEI is unavailable — the batch is skipped, not
        partially written, so the caller's transaction stays consistent).
        """
        items = [f for f in facts if str(f.get("content", "")).strip()]
        if not items:
            return 0

        contents = [str(f["content"]).strip() for f in items]
        vectors = await self._embed_chunked(contents)
        if vectors is None:
            logger.warning(
                "pgvector add_facts skipped for agent %s: TEI unavailable "
                "(%d facts not embedded)",
                agent_id,
                len(contents),
            )
            return 0

        rows = [
            AgentFact(
                agent_id=agent_id,
                content=content,
                embedding=vector,
                location_id=item.get("location_id"),
                tick_count=item.get("tick_count"),
                fact_metadata=item.get("metadata") or {},
            )
            for content, vector, item in zip(contents, vectors, items)
        ]
        await self._persist(rows, session)
        return len(rows)

    async def add_facts_for_agents(
        self,
        grouped: Iterable[tuple[str, list[dict]]],
        *,
        session: AsyncSession | None = None,
        batch_size: int = 256,
    ) -> dict[str, int]:
        """Embed and persist facts for many agents in as few TEI calls as
        possible — used by the periodic save (S76), which can otherwise fire
        one embedding call per agent (up to 1000) every flush.

        ``grouped`` is ``(agent_id, facts)`` pairs. All contents are embedded
        across shared chunked TEI calls, then written as `agent_facts` rows.
        Returns ``{agent_id: rows_written}``. On TEI failure nothing is
        written and an empty dict is returned (the whole batch retries next
        save), preserving the atomic-with-knowledge-state guarantee.
        """
        flat: list[tuple[str, dict, str]] = []
        for agent_id, facts in grouped:
            for fact in facts:
                content = str(fact.get("content", "")).strip()
                if content:
                    flat.append((agent_id, fact, content))
        if not flat:
            return {}

        vectors = await self._embed_chunked([c for _, _, c in flat], batch_size)
        if vectors is None:
            logger.warning(
                "pgvector batch flush skipped: TEI unavailable (%d facts across %d agents)",
                len(flat),
                len({a for a, _, _ in flat}),
            )
            return {}

        rows: list[AgentFact] = []
        counts: dict[str, int] = {}
        for (agent_id, fact, content), vector in zip(flat, vectors):
            rows.append(
                AgentFact(
                    agent_id=agent_id,
                    content=content,
                    embedding=vector,
                    location_id=fact.get("location_id"),
                    tick_count=fact.get("tick_count"),
                    fact_metadata=fact.get("metadata") or {},
                )
            )
            counts[agent_id] = counts.get(agent_id, 0) + 1
        await self._persist(rows, session)
        return counts

    async def _embed_chunked(
        self, contents: list[str], batch_size: int = 256
    ) -> Optional[list[list[float]]]:
        """Embed ``contents`` across chunked TEI calls. Returns all vectors in
        order, or ``None`` if any chunk fails (caller treats as TEI-down)."""
        out: list[list[float]] = []
        for start in range(0, len(contents), batch_size):
            chunk = contents[start : start + batch_size]
            vectors = await self._embedder.embed_async(chunk)
            if not vectors or len(vectors) != len(chunk):
                return None
            out.extend(vectors)
        return out

    # -- reads (the retriever) ------------------------------------------

    async def query(
        self,
        agent_id: str,
        question: str,
        top_k: int = 5,
        *,
        location_id: int | None = None,
        session: AsyncSession | None = None,
    ) -> list[tuple[float, str, dict]]:
        """Return the agent's most relevant facts as (score, content, metadata),
        highest relevance first. Empty list if TEI is unavailable."""
        if not question.strip():
            return []
        vectors = await self._embedder.embed_async([question])
        if not vectors:
            return []
        q_vec = vectors[0]

        distance = AgentFact.embedding.cosine_distance(q_vec).label("distance")
        stmt = select(AgentFact.content, AgentFact.fact_metadata, distance).where(
            AgentFact.agent_id == agent_id
        )
        if location_id is not None:
            stmt = stmt.where(AgentFact.location_id == location_id)
        stmt = stmt.order_by(distance).limit(top_k)

        rows = await self._execute(stmt, session, fetch="all")
        return [
            (max(0.0, 1.0 - float(dist)), content, dict(meta or {}))
            for content, meta, dist in rows
        ]

    async def count(self, agent_id: str, *, session: AsyncSession | None = None) -> int:
        stmt = (
            select(func.count())
            .select_from(AgentFact)
            .where(AgentFact.agent_id == agent_id)
        )
        return await self._execute(stmt, session, fetch="scalar") or 0

    # -- internals -------------------------------------------------------

    async def _persist(self, rows: Sequence[AgentFact], session: AsyncSession | None) -> None:
        if session is not None:
            # Caller owns the transaction (S76 atomic write) — add, don't commit.
            session.add_all(list(rows))
            return
        async with self._session_factory() as own:
            own.add_all(list(rows))
            await own.commit()

    async def _execute(self, stmt, session: AsyncSession | None, *, fetch: str):
        if session is not None:
            result = await session.execute(stmt)
            return result.scalar_one() if fetch == "scalar" else result.all()
        async with self._session_factory() as own:
            result = await own.execute(stmt)
            return result.scalar_one() if fetch == "scalar" else result.all()


_STORE: PgVectorFactStore | None = None


def get_fact_store() -> PgVectorFactStore:
    """Return the process-wide shared pgvector fact store."""
    global _STORE
    if _STORE is None:
        _STORE = PgVectorFactStore()
    return _STORE
