"""Tests for the S78 Chroma→pgvector backfill (agents/backfill.py).

No live services: Chroma client, TEI embedder, and the async session are all
faked, so we exercise the migration's logic — collection filtering, fact
mapping, idempotent dedup, TEI-down safety, and drop gating.
"""

import pytest

from agents import backfill
from agents.fact_store import PgVectorFactStore


# -- fakes ---------------------------------------------------------------


class _FakeCollection:
    def __init__(self, name, metadata=None, documents=None, metadatas=None):
        self.name = name
        self.metadata = metadata or {}
        self._documents = documents or []
        self._metadatas = metadatas or [{} for _ in (documents or [])]

    def get(self, include=None):
        return {"documents": list(self._documents), "metadatas": list(self._metadatas)}


class _FakeChroma:
    def __init__(self, collections):
        self._collections = collections
        self.deleted = []

    def list_collections(self):
        return list(self._collections)

    def get_collection(self, name):
        return next(c for c in self._collections if c.name == name)

    def delete_collection(self, name):
        self.deleted.append(name)
        self._collections = [c for c in self._collections if c.name != name]


class _FakeEmbedder:
    def __init__(self, fail=False):
        self._fail = fail
        self.calls = []

    async def embed_async(self, texts):
        self.calls.append(list(texts))
        if self._fail:
            return None
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    """One shared session standing in for every per-agent transaction.

    ``existing`` seeds the rows an _existing_contents query returns; rows added
    via add_all accumulate so later agents see prior writes within the run.
    """

    def __init__(self, existing=None):
        self.added = []
        self.commits = 0
        self._existing = list(existing or [])

    async def execute(self, stmt):  # only used for the SELECT content query
        return _FakeResult([(c,) for c in self._existing])

    def add_all(self, objs):
        self.added.extend(objs)

    async def commit(self):
        self.commits += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _store(session, embedder):
    return PgVectorFactStore(session_factory=lambda: session, embedder=embedder)


# -- tests ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfills_facts_and_maps_metadata():
    col = _FakeCollection(
        name="agent_abc",
        metadata={"agent_id": "abc-123"},
        documents=["Population of London is 9000000.", "I visited York."],
        metadatas=[
            {"predicate": "population", "location_id": 5},
            {"predicate": "visit", "location_id": 0},  # 0 == no location
        ],
    )
    chroma = _FakeChroma([col])
    session = _FakeSession()
    embedder = _FakeEmbedder()

    summary = await backfill.run(
        client=chroma, store=_store(session, embedder), session_factory=lambda: session
    )

    assert summary.agents == 1
    assert summary.facts_written == 2
    assert summary.facts_skipped == 0
    assert session.commits == 1
    # Re-embedded through TEI (not Chroma's vectors).
    assert embedder.calls == [["Population of London is 9000000.", "I visited York."]]
    rows = session.added
    assert {r.agent_id for r in rows} == {"abc-123"}  # un-munged id from metadata
    by_content = {r.content: r for r in rows}
    assert by_content["Population of London is 9000000."].location_id == 5
    assert by_content["I visited York."].location_id is None  # 0 → None
    assert by_content["I visited York."].fact_metadata == {"predicate": "visit"}


@pytest.mark.asyncio
async def test_skips_world_semantic_and_non_agent_collections():
    chroma = _FakeChroma(
        [
            _FakeCollection(name="world_semantic", documents=["civilisation fact"]),
            _FakeCollection(name="random_other", documents=["nope"]),  # no agent_ prefix/meta
            _FakeCollection(
                name="agent_x", metadata={"agent_id": "x"}, documents=["keep me"]
            ),
        ]
    )
    session = _FakeSession()

    summary = await backfill.run(
        client=chroma, store=_store(session, _FakeEmbedder()), session_factory=lambda: session
    )

    assert summary.agents == 1  # only agent_x
    assert summary.facts_written == 1
    assert session.added[0].content == "keep me"


@pytest.mark.asyncio
async def test_idempotent_dedup_against_existing_and_within_batch():
    col = _FakeCollection(
        name="agent_a",
        metadata={"agent_id": "a"},
        documents=["already there", "new fact", "new fact"],  # dup within batch
        metadatas=[{"predicate": "p"}, {"predicate": "p"}, {"predicate": "p"}],
    )
    chroma = _FakeChroma([col])
    session = _FakeSession(existing=["already there"])
    embedder = _FakeEmbedder()

    summary = await backfill.run(
        client=chroma, store=_store(session, embedder), session_factory=lambda: session
    )

    assert summary.facts_written == 1  # "new fact" once
    assert summary.facts_skipped == 2  # existing + the in-batch dup
    assert embedder.calls == [["new fact"]]
    assert [r.content for r in session.added] == ["new fact"]


@pytest.mark.asyncio
async def test_dry_run_writes_nothing():
    col = _FakeCollection(
        name="agent_a", metadata={"agent_id": "a"}, documents=["f1", "f2"],
        metadatas=[{"predicate": "p"}, {"predicate": "p"}],
    )
    chroma = _FakeChroma([col])
    session = _FakeSession()
    embedder = _FakeEmbedder()

    summary = await backfill.run(
        client=chroma, store=_store(session, embedder), session_factory=lambda: session,
        dry_run=True, drop_collections=True,
    )

    assert summary.facts_written == 2  # reported as "would write"
    assert session.added == []
    assert session.commits == 0
    assert embedder.calls == []      # never embedded
    assert chroma.deleted == []      # drop suppressed under dry-run


@pytest.mark.asyncio
async def test_tei_down_writes_nothing_and_blocks_drop():
    col = _FakeCollection(
        name="agent_a", metadata={"agent_id": "a"}, documents=["f1"],
        metadatas=[{"predicate": "p"}],
    )
    chroma = _FakeChroma([col])
    session = _FakeSession()

    summary = await backfill.run(
        client=chroma, store=_store(session, _FakeEmbedder(fail=True)),
        session_factory=lambda: session, drop_collections=True,
    )

    assert summary.facts_written == 0
    assert summary.tei_unavailable is True
    assert summary.complete is False
    assert session.commits == 0
    assert chroma.deleted == []  # incomplete backfill must never drop


@pytest.mark.asyncio
async def test_drop_collections_after_complete_backfill():
    chroma = _FakeChroma(
        [
            _FakeCollection(name="agent_a", metadata={"agent_id": "a"}, documents=["f1"],
                            metadatas=[{"predicate": "p"}]),
            _FakeCollection(name="agent_empty", metadata={"agent_id": "e"}),  # no docs
        ]
    )
    session = _FakeSession()

    summary = await backfill.run(
        client=chroma, store=_store(session, _FakeEmbedder()),
        session_factory=lambda: session, drop_collections=True,
    )

    assert summary.complete is True
    assert summary.collections_dropped == 2  # written one + empty one
    assert set(chroma.deleted) == {"agent_a", "agent_empty"}


@pytest.mark.asyncio
async def test_all_facts_already_present_marks_droppable_without_writing():
    col = _FakeCollection(
        name="agent_a", metadata={"agent_id": "a"}, documents=["seen"],
        metadatas=[{"predicate": "p"}],
    )
    chroma = _FakeChroma([col])
    session = _FakeSession(existing=["seen"])
    embedder = _FakeEmbedder()

    summary = await backfill.run(
        client=chroma, store=_store(session, embedder), session_factory=lambda: session,
        drop_collections=True,
    )

    assert summary.facts_written == 0
    assert summary.facts_skipped == 1
    assert embedder.calls == []          # nothing to embed
    assert chroma.deleted == ["agent_a"]  # already preserved → safe to drop
