"""S78: Backfill per-agent ChromaDB facts into the pgvector ``agent_facts`` table.

Preserve the research record — each agent's learned semantic memory — before
the legacy per-agent Chroma collections are dropped. From S76/S77 pgvector is
the durable home for agent memory; this one-shot migration moves the facts
Chroma accumulated *before* the cutover so nothing learned is lost.

Chroma's embeddings are NOT reused: they came from Chroma's default model
(384-dim all-MiniLM), incompatible with the 1024-dim bge-m3 vectors pgvector
expects (S71). Every fact is re-embedded through TEI on the way in.

Idempotent by ``(agent_id, content)``: a fact whose text already exists for
that agent in ``agent_facts`` is skipped, so the backfill is safe to re-run and
will not duplicate facts the live write path (S76) has since written. Identical
sentences collapse to one row — a semantic store gains nothing from duplicate
vectors, and the dedup is what makes re-runs and live-path overlap safe.

Dropping the source collections is destructive and OFF by default. Run the
backfill, check the row counts it reports, then re-run with ``--drop-collections``
to remove the per-agent collections. A collection is dropped only once its facts
are confirmed present in pgvector; if TEI was unavailable for any agent the drop
is refused wholesale so a partial backfill can never lose data.

The shared world-semantic collection (S79) is never touched.

Usage::

    python -m agents.backfill                    # backfill only (no drop)
    python -m agents.backfill --dry-run          # report only, write nothing
    python -m agents.backfill --drop-collections # backfill, then drop collections
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field

from sqlalchemy import select

from config import settings
from db.engine import async_session, dispose_engine
from db.models import AgentFact

from .fact_store import PgVectorFactStore

logger = logging.getLogger(__name__)

# The S79 world-semantic layer shares the same Chroma server but is NOT
# per-agent memory — it must survive this migration.
WORLD_SEMANTIC_COLLECTION = "world_semantic"


@dataclass
class BackfillSummary:
    """What the backfill did, for logging and tests."""

    agents: int = 0              # per-agent collections seen
    facts_written: int = 0       # rows newly inserted into agent_facts
    facts_skipped: int = 0       # facts already present (dedup)
    collections_dropped: int = 0
    tei_unavailable: bool = False  # any agent had fresh facts TEI could not embed
    droppable: list[str] = field(default_factory=list)  # collections safe to drop

    @property
    def complete(self) -> bool:
        """True when every fact is durably in pgvector (drop is then safe)."""
        return not self.tei_unavailable


def _connect_chroma():
    import chromadb

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)


def _collection_name(item) -> str:
    """list_collections() yields Collection objects on most versions and plain
    names on others — normalise to the name string."""
    return item if isinstance(item, str) else item.name


def _per_agent_collections(client) -> list[tuple[object, str]]:
    """Return ``(collection, agent_id)`` for every per-agent collection.

    The original agent id is read from the collection's ``agent_id`` metadata
    (set when the store was created) rather than the collection name, which is
    a lossy munge (``agent_{id.replace('-','_')[:50]}``). The world-semantic
    collection and anything else is excluded.
    """
    out: list[tuple[object, str]] = []
    for item in client.list_collections():
        name = _collection_name(item)
        if name == WORLD_SEMANTIC_COLLECTION:
            continue
        collection = item if not isinstance(item, str) else client.get_collection(name)
        meta = getattr(collection, "metadata", None) or {}
        agent_id = meta.get("agent_id")
        if not agent_id and not name.startswith("agent_"):
            continue  # not a per-agent collection
        out.append((collection, agent_id or name))
    return out


def _facts_from_collection(collection) -> list[dict]:
    """Read every stored fact out of a Chroma collection as pgvector fact dicts.

    Chroma stored each fact's metadata as ``{"predicate", "location_id"}`` with
    ``location_id or 0`` — so 0 means "no location" and maps back to ``None`` to
    honour the nullable FK. Tick count was never stored in Chroma, so it stays
    ``None``; the value field likewise wasn't persisted there.
    """
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    facts: list[dict] = []
    for i, document in enumerate(documents):
        if not document or not document.strip():
            continue
        meta = metadatas[i] if i < len(metadatas) else {}
        meta = meta or {}
        location_id = meta.get("location_id") or None  # 0/None → no location
        facts.append(
            {
                "content": document.strip(),
                "location_id": location_id,
                "tick_count": None,
                "metadata": {"predicate": meta.get("predicate", "")},
            }
        )
    return facts


async def _existing_contents(session, agent_id: str) -> set[str]:
    result = await session.execute(
        select(AgentFact.content).where(AgentFact.agent_id == agent_id)
    )
    return {row[0] for row in result.all()}


def _dedup(facts: list[dict], existing: set[str]) -> list[dict]:
    """Drop facts already in pgvector for this agent and collapse duplicate
    contents within the batch (identical text → identical vector → one row)."""
    fresh: list[dict] = []
    seen: set[str] = set()
    for fact in facts:
        content = fact["content"]
        if content in existing or content in seen:
            continue
        seen.add(content)
        fresh.append(fact)
    return fresh


async def run(
    *,
    client=None,
    store: PgVectorFactStore | None = None,
    session_factory=None,
    dry_run: bool = False,
    drop_collections: bool = False,
) -> BackfillSummary:
    """Backfill all per-agent Chroma collections into pgvector.

    Dependencies are injectable so the flow is unit-testable without live
    Chroma / Postgres / TEI. Each agent is committed in its own transaction, so
    an interrupted run leaves completed agents durably persisted (mirrors the
    per-country streaming in ``fetch_geography``).
    """
    client = client or _connect_chroma()
    session_factory = session_factory or async_session
    store = store or PgVectorFactStore(session_factory=session_factory)
    summary = BackfillSummary()

    for collection, agent_id in _per_agent_collections(client):
        summary.agents += 1
        name = _collection_name(collection)
        facts = _facts_from_collection(collection)

        if not facts:
            # Empty collection — nothing to preserve, safe to drop.
            summary.droppable.append(name)
            continue

        async with session_factory() as session:
            existing = await _existing_contents(session, agent_id)
            fresh = _dedup(facts, existing)
            already = len(facts) - len(fresh)
            summary.facts_skipped += already

            if not fresh:
                # Everything already in pgvector (prior run or live write path).
                summary.droppable.append(name)
                logger.info("agent %s: all %d facts already in pgvector", agent_id, already)
                continue

            if dry_run:
                summary.facts_written += len(fresh)
                logger.info(
                    "agent %s: would write %d new facts (%d already present)",
                    agent_id, len(fresh), already,
                )
                continue

            written = await store.add_facts(agent_id, fresh, session=session)
            if written == 0:
                # add_facts writes nothing only when TEI can't embed (fresh is
                # non-empty and pre-stripped). Don't commit, don't mark
                # droppable — the batch retries on the next run.
                summary.tei_unavailable = True
                logger.warning(
                    "agent %s: TEI unavailable — %d facts not embedded, will retry; "
                    "collection NOT dropped", agent_id, len(fresh),
                )
                continue

            await session.commit()
            summary.facts_written += written
            summary.droppable.append(name)
            logger.info(
                "agent %s: wrote %d new facts (%d already present)",
                agent_id, written, already,
            )

    if drop_collections and not dry_run:
        if not summary.complete:
            logger.warning(
                "Refusing to drop collections: TEI was unavailable for at least one "
                "agent, so the backfill is incomplete. Fix TEI and re-run."
            )
        else:
            for name in summary.droppable:
                client.delete_collection(name)
                summary.collections_dropped += 1
            logger.info("Dropped %d per-agent Chroma collections", summary.collections_dropped)

    logger.info(
        "Backfill summary: %d agents, %d facts written, %d already present, %d collections dropped%s",
        summary.agents, summary.facts_written, summary.facts_skipped,
        summary.collections_dropped,
        "" if summary.complete else " (INCOMPLETE — TEI unavailable, re-run)",
    )
    return summary


async def _main(dry_run: bool, drop_collections: bool) -> None:
    try:
        await run(dry_run=dry_run, drop_collections=drop_collections)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backfill per-agent Chroma facts into pgvector (S78).")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument(
        "--drop-collections",
        action="store_true",
        help="drop per-agent Chroma collections after a complete backfill (destructive)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.dry_run, args.drop_collections))
