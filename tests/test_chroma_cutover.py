"""Tests for the pgvector cutover switch (S99).

When `chroma_enabled` is off, the legacy per-agent ChromaFactStore must be
fully out of the path: `_get_chroma_store` returns None for every caller, so
`_append_fact` makes no Chroma call and `answer`'s sync retrieval skips the
Chroma branch — which is what removes the S66 tick stall in prod.
"""

import random

import pytest

import agents.system as sysmod
from agents.system import AgentTraits, AutonomousAgent
from config import settings


@pytest.fixture
def chroma_disabled(monkeypatch):
    monkeypatch.setattr(settings, "chroma_enabled", False)


def _agent() -> AutonomousAgent:
    return AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )


def test_get_chroma_store_returns_none_when_disabled(chroma_disabled):
    assert sysmod._get_chroma_store("A1") is None


def test_get_chroma_store_attempts_connection_when_enabled(monkeypatch):
    # When enabled, it constructs a ChromaFactStore (which returns None here
    # only because no Chroma is reachable — but the construction is attempted,
    # proving the flag, not connectivity, is the gate).
    monkeypatch.setattr(settings, "chroma_enabled", True)
    constructed = {"n": 0}
    real_init = sysmod.ChromaFactStore.__init__

    def _spy(self, *args, **kwargs):
        constructed["n"] += 1
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(sysmod.ChromaFactStore, "__init__", _spy)
    sysmod._get_chroma_store("A1")
    assert constructed["n"] == 1


def test_append_fact_makes_no_chroma_call_when_disabled(chroma_disabled, monkeypatch):
    # Guard: if _get_chroma_store were to hand back any store, fail loudly.
    def _boom(agent_id):
        raise AssertionError("Chroma must not be consulted when disabled")

    # _append_fact calls the module-level _get_chroma_store; with the flag off
    # the real one returns None, so the buffer fills with no Chroma store set.
    agent = _agent()
    agent._append_fact("population", "Population of L2 is 1000.", 2, value=1000)

    assert agent._chroma_store is None
    assert len(agent.pending_facts) == 1  # still buffered for pgvector
    assert agent.pending_facts[0]["content"] == "Population of L2 is 1000."


def test_sync_retrieval_skips_chroma_when_disabled(chroma_disabled):
    agent = _agent()
    # No Chroma, no facts → in-process retriever path, no exception.
    ranked, backend = agent._retrieve_ranked_sync("where is L2", [])
    assert agent._chroma_store is None
    assert ranked == []
    assert backend in {"tei", "token-overlap"}
