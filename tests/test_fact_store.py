"""Tests for the pgvector-backed PgVectorFactStore (S75).

No live Postgres: the async session is faked so we exercise the store's
Python logic — TEI embedding integration, distance→score mapping, graceful
degradation when TEI is down, and the external-session (no-commit) path used
by the S76 atomic write. Similarity ranking against the real HNSW index is
verified at deploy time (S77).
"""

import pytest

from agents.fact_store import PgVectorFactStore


class _FakeEmbedder:
    def __init__(self, table=None, fail=False):
        self._table = table or {}
        self._fail = fail
        self.calls = []

    async def embed_async(self, texts):
        self.calls.append(list(texts))
        if self._fail:
            return None
        return [self._table.get(t, [0.0, 0.0, 0.0, 0.0]) for t in texts]


class _FakeResult:
    def __init__(self, rows, scalar):
        self._rows = rows
        self._scalar = scalar

    def all(self):
        return self._rows

    def scalar_one(self):
        return self._scalar


class _FakeSession:
    def __init__(self, rows=None, scalar=0):
        self.added = []
        self.committed = False
        self.executed = []
        self._rows = rows or []
        self._scalar = scalar

    async def execute(self, stmt):
        self.executed.append(stmt)
        return _FakeResult(self._rows, self._scalar)

    def add_all(self, objs):
        self.added.extend(objs)

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _factory(session):
    return lambda: session


@pytest.mark.asyncio
async def test_add_facts_embeds_and_commits_with_own_session():
    session = _FakeSession()
    embedder = _FakeEmbedder(table={"pop 1000": [1.0, 0.0, 0.0, 0.0]})
    store = PgVectorFactStore(session_factory=_factory(session), embedder=embedder)

    n = await store.add_facts(
        "A1",
        [
            {"content": "pop 1000", "location_id": 2, "tick_count": 5,
             "metadata": {"predicate": "population"}},
            {"content": "   "},  # blank → dropped before embedding
        ],
    )

    assert n == 1
    assert embedder.calls == [["pop 1000"]]  # blank never embedded
    assert session.committed is True
    assert len(session.added) == 1
    row = session.added[0]
    assert row.agent_id == "A1"
    assert row.content == "pop 1000"
    assert row.embedding == [1.0, 0.0, 0.0, 0.0]
    assert row.location_id == 2
    assert row.tick_count == 5
    assert row.fact_metadata == {"predicate": "population"}


@pytest.mark.asyncio
async def test_add_facts_skipped_when_tei_down():
    session = _FakeSession()
    store = PgVectorFactStore(session_factory=_factory(session), embedder=_FakeEmbedder(fail=True))

    n = await store.add_facts("A1", [{"content": "anything"}])

    assert n == 0
    assert session.added == []
    assert session.committed is False


@pytest.mark.asyncio
async def test_external_session_is_added_but_not_committed():
    """S76 atomic write: the caller owns the transaction, so the store must
    add rows to the provided session without committing or opening its own."""
    external = _FakeSession()

    def _boom():
        raise AssertionError("store opened its own session instead of reusing the caller's")

    store = PgVectorFactStore(
        session_factory=_boom,
        embedder=_FakeEmbedder(table={"x": [0.0, 1.0, 0.0, 0.0]}),
    )

    n = await store.add_fact("A1", "x", session=external)

    assert n == 1
    assert len(external.added) == 1
    assert external.committed is False


@pytest.mark.asyncio
async def test_query_maps_cosine_distance_to_score():
    rows = [
        ("fact A", {"predicate": "x"}, 0.1),
        ("fact B", {}, 0.4),
    ]
    session = _FakeSession(rows=rows)
    embedder = _FakeEmbedder(table={"where is A": [1.0, 0.0, 0.0, 0.0]})
    store = PgVectorFactStore(session_factory=_factory(session), embedder=embedder)

    result = await store.query("A1", "where is A", top_k=5)

    assert embedder.calls == [["where is A"]]
    assert result[0] == (pytest.approx(0.9), "fact A", {"predicate": "x"})
    assert result[1] == (pytest.approx(0.6), "fact B", {})


@pytest.mark.asyncio
async def test_query_empty_when_tei_down_or_blank():
    session = _FakeSession(rows=[("f", {}, 0.0)])
    down = PgVectorFactStore(session_factory=_factory(session), embedder=_FakeEmbedder(fail=True))
    assert await down.query("A1", "anything") == []

    up = PgVectorFactStore(session_factory=_factory(session), embedder=_FakeEmbedder())
    assert await up.query("A1", "   ") == []  # blank question never hits the DB
    assert session.executed == []


@pytest.mark.asyncio
async def test_count_returns_scalar():
    session = _FakeSession(scalar=7)
    store = PgVectorFactStore(session_factory=_factory(session), embedder=_FakeEmbedder())

    assert await store.count("A1") == 7


@pytest.mark.asyncio
async def test_add_facts_for_agents_batches_across_agents():
    session = _FakeSession()
    embedder = _FakeEmbedder(
        table={
            "a1 fact one": [1.0, 0, 0, 0],
            "a1 fact two": [0, 1.0, 0, 0],
            "a2 fact": [0, 0, 1.0, 0],
        }
    )
    store = PgVectorFactStore(session_factory=_factory(session), embedder=embedder)

    counts = await store.add_facts_for_agents(
        [
            ("A1", [{"content": "a1 fact one"}, {"content": "  "}, {"content": "a1 fact two"}]),
            ("A2", [{"content": "a2 fact", "location_id": 5}]),
        ]
    )

    assert counts == {"A1": 2, "A2": 1}
    # All non-blank contents embedded in a single chunked call (< batch_size).
    assert embedder.calls == [["a1 fact one", "a1 fact two", "a2 fact"]]
    assert len(session.added) == 3
    assert {r.agent_id for r in session.added} == {"A1", "A2"}
    assert session.committed is True


@pytest.mark.asyncio
async def test_add_facts_for_agents_chunks_large_batches():
    session = _FakeSession()
    embedder = _FakeEmbedder()  # default vector for any content
    store = PgVectorFactStore(session_factory=_factory(session), embedder=embedder)

    grouped = [("A1", [{"content": f"fact {i}"} for i in range(5)])]
    counts = await store.add_facts_for_agents(grouped, batch_size=2)

    assert counts == {"A1": 5}
    # 5 facts at batch_size=2 → 3 TEI calls (2 + 2 + 1).
    assert [len(c) for c in embedder.calls] == [2, 2, 1]


@pytest.mark.asyncio
async def test_add_facts_for_agents_tei_down_writes_nothing():
    session = _FakeSession()
    store = PgVectorFactStore(session_factory=_factory(session), embedder=_FakeEmbedder(fail=True))

    counts = await store.add_facts_for_agents([("A1", [{"content": "x"}])])

    assert counts == {}
    assert session.added == []
    assert session.committed is False
