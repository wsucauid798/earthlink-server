"""Tests for the S77 pgvector read path in AutonomousAgent.

Covers the pgvector retrieval seam (hit→record mapping, empty→None) and the
answer_async branching (prefer pgvector, fall back to the sync Chroma/
in-process path). The full compose tail is exercised by test_agents_learning;
here the seams are stubbed so the branching is tested without a Geography.
"""

import pytest

import agents.fact_store as fact_store_mod
import agents.system as sysmod
from agents.system import AgentTraits, AutonomousAgent


def _agent(monkeypatch) -> AutonomousAgent:
    monkeypatch.setattr(sysmod, "_get_chroma_store", lambda agent_id: None)
    return AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )


class _FakeStore:
    def __init__(self, hits):
        self._hits = hits
        self.last = None

    async def query(self, agent_id, question, top_k=5):
        self.last = (agent_id, question, top_k)
        return self._hits


@pytest.mark.asyncio
async def test_retrieve_ranked_pgvector_maps_hits(monkeypatch):
    agent = _agent(monkeypatch)
    store = _FakeStore(
        [(0.9, "fact A", {}), (0.5, "new fact", {"predicate": "x", "location_id": 3})]
    )
    monkeypatch.setattr(fact_store_mod, "get_fact_store", lambda: store)
    memory_records = [{"text": "fact A", "predicate": "pop", "location_id": 2}]

    ranked = await agent._retrieve_ranked_pgvector("where", memory_records)

    assert store.last == ("A1", "where", 10)
    assert ranked[0]["score"] == 0.9
    assert ranked[0]["record"] is memory_records[0]  # reuse the richer record
    assert ranked[1]["record"] == {"text": "new fact", "predicate": "x", "location_id": 3}


@pytest.mark.asyncio
async def test_retrieve_ranked_pgvector_none_when_empty(monkeypatch):
    agent = _agent(monkeypatch)
    monkeypatch.setattr(fact_store_mod, "get_fact_store", lambda: _FakeStore([]))

    assert await agent._retrieve_ranked_pgvector("q", []) is None


def _stub_common_seams(agent):
    agent.get_visited_places = lambda geography: []
    agent._episodic_answer = lambda *args, **kwargs: None
    agent._memory_facts = lambda geography: []
    agent._finalize_answer = (
        lambda q, question, ranked, backend, vp, geography: {"retrieval_backend": backend}
    )


@pytest.mark.asyncio
async def test_answer_async_prefers_pgvector(monkeypatch):
    agent = _agent(monkeypatch)
    _stub_common_seams(agent)

    async def _pg(q, mr):
        return [{"score": 1.0, "record": {"text": "f"}}]

    def _no_sync(q, mr):
        raise AssertionError("sync retrieval must not run when pgvector has hits")

    agent._retrieve_ranked_pgvector = _pg
    agent._retrieve_ranked_sync = _no_sync

    result = await agent.answer_async("q", geography=None)

    assert result["retrieval_backend"] == "pgvector"


@pytest.mark.asyncio
async def test_answer_async_falls_back_when_pgvector_empty(monkeypatch):
    agent = _agent(monkeypatch)
    _stub_common_seams(agent)

    async def _pg(q, mr):
        return None

    agent._retrieve_ranked_pgvector = _pg
    agent._retrieve_ranked_sync = lambda q, mr: ([{"score": 1.0, "record": {}}], "chroma")

    result = await agent.answer_async("q", geography=None)

    assert result["retrieval_backend"] == "chroma"
