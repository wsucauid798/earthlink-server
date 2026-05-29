"""Tests for the S76 pgvector write-path wiring in AutonomousAgent / AgentSystem.

Covers the in-memory buffering and drain bookkeeping that feeds
World._save_state. The actual transactional flush (PgVectorFactStore +
session) is covered by test_fact_store.py; here we verify facts are buffered
with the right shape, consecutive duplicates are suppressed, and the
front-drop preserves facts appended during an async save.
"""

import random

import pytest

import agents.system as sysmod
from agents.system import AgentSystem, AgentTraits, AutonomousAgent


def _agent(monkeypatch) -> AutonomousAgent:
    # Keep Chroma out of the picture so _append_fact never touches the network.
    monkeypatch.setattr(sysmod, "_get_chroma_store", lambda agent_id: None)
    return AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )


def test_append_fact_buffers_with_tick_and_metadata(monkeypatch):
    agent = _agent(monkeypatch)
    agent._current_tick_count = 7

    agent._append_fact("population", "Population of L2 is 1000.", 2, value=1000)

    assert len(agent.pending_facts) == 1
    fact = agent.pending_facts[0]
    assert fact["content"] == "Population of L2 is 1000."
    assert fact["location_id"] == 2
    assert fact["tick_count"] == 7
    assert fact["metadata"] == {"predicate": "population", "value": 1000}


def test_append_fact_without_value_omits_value_key(monkeypatch):
    agent = _agent(monkeypatch)
    agent._append_fact("visit", "I visited L1.", 1)

    assert agent.pending_facts[0]["metadata"] == {"predicate": "visit"}
    assert agent.pending_facts[0]["tick_count"] is None


def test_consecutive_duplicate_not_buffered(monkeypatch):
    agent = _agent(monkeypatch)
    agent._append_fact("visit", "I visited L1.", 1)
    agent._append_fact("visit", "I visited L1.", 1)  # exact consecutive duplicate

    assert len(agent.pending_facts) == 1


def test_drop_pending_facts_from_front_preserves_tail(monkeypatch):
    agent = _agent(monkeypatch)
    agent._append_fact("a", "fact 1", 1)
    agent._append_fact("b", "fact 2", 1)
    agent._append_fact("c", "fact 3", 1)

    # Simulate: first two were flushed+committed; a third arrived mid-save.
    agent.drop_pending_facts(2)
    assert [f["content"] for f in agent.pending_facts] == ["fact 3"]

    # Over-count clears everything; non-positive is a no-op.
    agent.drop_pending_facts(0)
    assert len(agent.pending_facts) == 1
    agent.drop_pending_facts(5)
    assert agent.pending_facts == []


def test_iter_local_agents_yields_in_sequential_mode(monkeypatch):
    agent = _agent(monkeypatch)
    system = AgentSystem(agents=[agent], _rng=random.Random(0))

    assert system._use_ray is False
    assert list(system.iter_local_agents()) == [agent]


def test_consider_world_facts_commits_relevant_novel(monkeypatch):
    agent = _agent(monkeypatch)
    candidates = [
        {"score": 0.9, "text": "London was founded by the Romans.", "domain": "history", "location_id": 5},
        {"score": 0.1, "text": "Irrelevant low-score fact.", "domain": "news", "location_id": 5},
    ]
    committed = agent.consider_world_facts(candidates, min_score=0.4)
    assert committed == 1
    texts = [f["content"] for f in agent.pending_facts]
    assert "London was founded by the Romans." in texts
    assert "Irrelevant low-score fact." not in texts


def test_consider_world_facts_skips_already_known(monkeypatch):
    agent = _agent(monkeypatch)
    agent._append_fact("history", "Known fact.", 5)
    before = len(agent.pending_facts)
    committed = agent.consider_world_facts([{"score": 0.95, "text": "Known fact.", "location_id": 5}])
    assert committed == 0
    assert len(agent.pending_facts) == before


def test_consider_world_facts_goal_location_clears_lower_bar(monkeypatch):
    from agents.system import AgentGoal
    agent = _agent(monkeypatch)
    agent.current_goal = AgentGoal(kind="investigate_gap", target_location_id=7, priority=0.9)
    # score 0.32 is below 0.4 default but above 0.4-0.1 for the goal location.
    committed = agent.consider_world_facts([{"score": 0.32, "text": "Goal-place fact.", "location_id": 7}])
    assert committed == 1
