"""Tests for the world semantic layer (S79-S82)."""

from types import SimpleNamespace

import pytest

from world.semantic import WorldSemanticStore, _doc_id


def _fact(text, domain="history", topic="founding", source="wikipedia", location_id=1, confidence=0.8):
    return SimpleNamespace(
        text=text, domain=domain, topic=topic, source=source,
        location_id=location_id, confidence=confidence,
    )


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


class _FakeCollection:
    def __init__(self, query_result=None):
        self.upserts = []
        self.deletes = []
        self.queries = []
        self._query_result = query_result or {"documents": [[]], "distances": [[]], "metadatas": [[]]}

    async def upsert(self, ids, documents, embeddings, metadatas):
        self.upserts.append({"ids": ids, "documents": documents, "embeddings": embeddings, "metadatas": metadatas})

    async def query(self, query_embeddings, n_results, where=None):
        self.queries.append({"n_results": n_results, "where": where})
        return self._query_result

    async def delete(self, where=None):
        self.deletes.append(where)

    async def count(self):
        return 42


def _store(embedder, collection):
    s = WorldSemanticStore(embedder=embedder)
    s._collection = collection
    return s


@pytest.mark.asyncio
async def test_add_facts_embeds_and_upserts_with_ttl_metadata():
    emb = _FakeEmbedder(table={"London was founded by the Romans.": [1.0, 0, 0, 0]})
    col = _FakeCollection()
    store = _store(emb, col)

    n = await store.add_facts(
        [_fact("London was founded by the Romans.", location_id=5), _fact("   ")],
        expires_at=1000.0,
        tick_count=12,
    )

    assert n == 1
    assert emb.calls == [["London was founded by the Romans."]]
    up = col.upserts[0]
    assert up["documents"] == ["London was founded by the Romans."]
    assert up["embeddings"] == [[1.0, 0, 0, 0]]
    meta = up["metadatas"][0]
    assert meta["location_id"] == 5
    assert meta["domain"] == "history"
    assert meta["expires_at"] == 1000.0
    assert meta["tick_count"] == 12
    # id is stable/content-addressed
    assert up["ids"][0] == _doc_id("history", "wikipedia", 5, "London was founded by the Romans.")


@pytest.mark.asyncio
async def test_add_facts_skips_when_tei_down():
    col = _FakeCollection()
    store = _store(_FakeEmbedder(fail=True), col)
    assert await store.add_facts([_fact("anything")], expires_at=1.0) == 0
    assert col.upserts == []


@pytest.mark.asyncio
async def test_query_filters_by_location_and_ttl_and_maps_scores():
    col = _FakeCollection(query_result={
        "documents": [["fact A", "fact B"]],
        "distances": [[0.1, 0.4]],
        "metadatas": [[{"domain": "news"}, {"domain": "history"}]],
    })
    store = _store(_FakeEmbedder(table={"what's happening": [1.0, 0, 0, 0]}), col)

    res = await store.query("what's happening", top_k=5, location_id=7, now=500.0)

    assert res[0] == (pytest.approx(0.9), "fact A", {"domain": "news"})
    assert res[1] == (pytest.approx(0.6), "fact B", {"domain": "history"})
    where = col.queries[0]["where"]
    assert where == {"$and": [{"location_id": {"$eq": 7}}, {"expires_at": {"$gt": 500.0}}]}


@pytest.mark.asyncio
async def test_query_empty_when_unavailable_or_blank():
    store = WorldSemanticStore(embedder=_FakeEmbedder())  # no collection
    assert await store.query("x", now=1.0) == []
    store2 = _store(_FakeEmbedder(), _FakeCollection())
    assert await store2.query("   ") == []


@pytest.mark.asyncio
async def test_prune_deletes_lapsed():
    col = _FakeCollection()
    store = _store(_FakeEmbedder(), col)
    await store.prune(now=999.0)
    assert col.deletes == [{"expires_at": {"$lte": 999.0}}]


def test_doc_id_stable():
    a = _doc_id("history", "wikipedia", 5, "text")
    b = _doc_id("history", "wikipedia", 5, "text")
    c = _doc_id("history", "wikipedia", 6, "text")
    assert a == b and a != c and len(a) == 32
