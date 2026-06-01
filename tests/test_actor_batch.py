"""Tests for the S100 batched Ray actor (agents/actor.py).

`_AgentActorImpl` is a plain class until Ray decorates it, so we exercise the
batching logic directly — no Ray runtime. Covers: one event per agent from a
batch tick, list-shaped summaries/snapshots/persisted, per-agent addressing of
detail/answer, and social updates applied only to the targeted agents.
"""

from datetime import datetime, timezone

import pytest

from agents.actor import _AgentActorImpl
from agents.system import AgentTraits, AutonomousAgent
from world.geography import ConnectionData, Geography, LocationData
from world.weather import WeatherState


class _StubWeather:
    def get_weather(self, location_id: int):
        return WeatherState(
            location_id=location_id, temperature_c=12.0, humidity_pct=70.0,
            wind_speed_kmh=15.0, conditions="cloudy",
        )

    def get_all_weather(self):
        return {}


def _geo() -> Geography:
    locations = {
        1: LocationData(1, "L1", "town", 51.5, -0.1, None, None, None, None, None, None, 1000, None),
        2: LocationData(2, "L2", "town", 51.5, 0.0, None, None, None, None, None, None, 1000, None),
    }
    return Geography.from_memory(locations, [ConnectionData(1, 2, 1.0, "road", None, None)])


def _agent(agent_id: str, location_id: int = 1) -> AutonomousAgent:
    return AutonomousAgent(
        agent_id=agent_id, name=agent_id, location_id=location_id,
        learning_rate=0.2, exploration_bias=0.1,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.7, comfort_temperature_c=20.0),
    )


def _actor(ids):
    return _AgentActorImpl([_agent(i).to_persisted() for i in ids], seed=42)


def _tick(actor):
    return actor.tick(_geo(), _StubWeather(), datetime.now(timezone.utc), 1, {}, None, 1.0)


def test_batch_tick_returns_one_event_per_agent():
    actor = _actor(["a1", "a2", "a3"])
    events = _tick(actor)
    assert [e["agent_id"] for e in events] == ["a1", "a2", "a3"]
    assert actor.agent_ids == ["a1", "a2", "a3"]


def test_list_shaped_summaries_snapshots_persisted():
    actor = _actor(["a1", "a2"])
    geo = _geo()
    assert {s["id"] for s in actor.get_summary(geo)} == {"a1", "a2"}
    assert {s["agent_id"] for s in actor.get_social_snapshot()} == {"a1", "a2"}
    assert len(actor.to_persisted()) == 2


def test_detail_and_answer_address_by_id():
    actor = _actor(["a1", "a2"])
    geo = _geo()
    assert actor.get_detail("a2", geo)["id"] == "a2"
    assert actor.get_detail("ghost", geo) is None
    assert actor.answer("a1", "where am I?", geo) is not None
    assert actor.answer("ghost", "where am I?", geo) is None


def test_social_updates_apply_only_to_targeted_agents():
    actor = _actor(["a1", "a2"])
    actor.apply_social_updates({"a1": {"new_facts": [{"text": "shared fact"}]}})

    snaps = {s["agent_id"]: s for s in actor.get_social_snapshot()}
    assert any(f.get("text") == "shared fact" for f in snaps["a1"]["recent_facts"])
    assert not any(f.get("text") == "shared fact" for f in snaps["a2"]["recent_facts"])


def test_social_updates_ignore_unknown_agent():
    actor = _actor(["a1"])
    # Must not raise when an update targets an agent this actor doesn't hold.
    actor.apply_social_updates({"not-here": {"new_facts": [{"text": "x"}]}})
    assert actor.agent_ids == ["a1"]
