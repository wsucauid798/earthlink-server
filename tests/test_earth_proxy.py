"""Tests for the Earth proxy and adapter pipeline.

Tests the full chain: adapter -> EarthProxy (Redis or fallback) -> agent ingestion.
All tests use a mock adapter (no internet required) and the in-memory fallback
(no Redis required).
"""

import pytest
import asyncio
from datetime import datetime, timezone

from adapters.base import EarthAdapter, EarthFact
from world.earth_proxy import EarthProxy


# ---------------------------------------------------------------------------
# Mock adapter for testing (no internet required)
# ---------------------------------------------------------------------------

class MockAdapter(EarthAdapter):
    """A fake adapter that returns canned facts."""

    def __init__(self, canned_facts: list[EarthFact] | None = None):
        self._canned = canned_facts or []
        self._resolve_count = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def domains(self) -> list[str]:
        return ["history", "culture"]

    async def resolve(self, query, location_name=None, location_id=None, max_results=5):
        self._resolve_count += 1
        facts = []
        for f in self._canned[:max_results]:
            facts.append(
                EarthFact(
                    domain=f.domain,
                    topic=f.topic,
                    text=f.text,
                    source=self.name,
                    location_id=location_id or f.location_id,
                    location_name=location_name or f.location_name,
                    confidence=f.confidence,
                    timestamp=datetime.now(timezone.utc),
                )
            )
        return facts

    async def health_check(self):
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canned_facts(location_id: int = 1, location_name: str = "London") -> list[EarthFact]:
    return [
        EarthFact(
            domain="history",
            topic="founding",
            text="London was founded by the Romans as Londinium in AD 43.",
            source="mock",
            location_id=location_id,
            location_name=location_name,
        ),
        EarthFact(
            domain="culture",
            topic="theatre",
            text="London's West End is one of the world's major theatre districts.",
            source="mock",
            location_id=location_id,
            location_name=location_name,
        ),
        EarthFact(
            domain="institution",
            topic="parliament",
            text="The Palace of Westminster is the seat of the UK Parliament.",
            source="mock",
            location_id=location_id,
            location_name=location_name,
        ),
    ]


# ---------------------------------------------------------------------------
# EarthFact tests
# ---------------------------------------------------------------------------

def test_earth_fact_to_dict():
    """EarthFact should serialise to a clean dictionary."""
    fact = EarthFact(
        domain="history",
        topic="founding",
        text="London was founded by the Romans.",
        source="wikipedia",
        location_id=1,
        location_name="London",
    )
    d = fact.to_dict()
    assert d["domain"] == "history"
    assert d["topic"] == "founding"
    assert d["text"] == "London was founded by the Romans."
    assert d["source"] == "wikipedia"
    assert d["location_id"] == 1


# ---------------------------------------------------------------------------
# EarthProxy tests (uses in-memory fallback — no Redis needed for tests)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_proxy_register_adapter():
    """Registering an adapter should increase adapter count."""
    proxy = EarthProxy()
    assert proxy.adapter_count == 0
    proxy.register_adapter(MockAdapter())
    assert proxy.adapter_count == 1


@pytest.mark.asyncio
async def test_proxy_resolve_for_location():
    """Proxy should resolve facts for a location via registered adapter."""
    canned = _make_canned_facts()
    adapter = MockAdapter(canned_facts=canned)
    proxy = EarthProxy()
    proxy.register_adapter(adapter)

    facts = await proxy.resolve_for_location(location_id=1, location_name="London")
    assert len(facts) == 3
    assert all(isinstance(f, EarthFact) for f in facts)
    assert facts[0].text == "London was founded by the Romans as Londinium in AD 43."
    assert adapter._resolve_count == 1


@pytest.mark.asyncio
async def test_proxy_cache_prevents_duplicate_resolve():
    """Once resolved, the proxy should return cached facts without re-querying."""
    canned = _make_canned_facts()
    adapter = MockAdapter(canned_facts=canned)
    proxy = EarthProxy()
    proxy.register_adapter(adapter)

    await proxy.resolve_for_location(location_id=1, location_name="London")
    await proxy.resolve_for_location(location_id=1, location_name="London")
    assert adapter._resolve_count == 1  # only called once


@pytest.mark.asyncio
async def test_proxy_force_re_resolve():
    """force=True should re-query the adapter even if cached."""
    canned = _make_canned_facts()
    adapter = MockAdapter(canned_facts=canned)
    proxy = EarthProxy()
    proxy.register_adapter(adapter)

    await proxy.resolve_for_location(location_id=1, location_name="London")
    await proxy.resolve_for_location(location_id=1, location_name="London", force=True)
    assert adapter._resolve_count == 2


@pytest.mark.asyncio
async def test_proxy_is_location_resolved():
    """is_location_resolved should track which locations have live content."""
    proxy = EarthProxy()
    proxy.register_adapter(MockAdapter(canned_facts=_make_canned_facts()))

    assert not await proxy.is_location_resolved(1)
    await proxy.resolve_for_location(location_id=1, location_name="London")
    assert await proxy.is_location_resolved(1)
    assert not await proxy.is_location_resolved(999)


@pytest.mark.asyncio
async def test_proxy_resolve_query():
    """Free-form query should resolve and cache by query string."""
    canned = _make_canned_facts()
    adapter = MockAdapter(canned_facts=canned)
    proxy = EarthProxy()
    proxy.register_adapter(adapter)

    facts = await proxy.resolve_query("history of London")
    assert len(facts) == 3

    # Should be cached
    facts2 = await proxy.resolve_query("history of London")
    assert adapter._resolve_count == 1  # only one actual resolve


@pytest.mark.asyncio
async def test_proxy_no_adapters_returns_empty():
    """Proxy with no adapters should return empty results gracefully."""
    proxy = EarthProxy()
    facts = await proxy.resolve_for_location(location_id=1, location_name="London")
    assert facts == []


@pytest.mark.asyncio
async def test_proxy_health():
    """Health check should report adapter status."""
    proxy = EarthProxy()
    proxy.register_adapter(MockAdapter())
    health = await proxy.health()
    assert health == {"mock": True}


@pytest.mark.asyncio
async def test_proxy_total_resolves_counter():
    """total_resolves should count how many times the proxy queried Earth."""
    canned = _make_canned_facts()
    proxy = EarthProxy()
    proxy.register_adapter(MockAdapter(canned_facts=canned))

    assert proxy.total_resolves == 0
    await proxy.resolve_for_location(location_id=1, location_name="London")
    assert proxy.total_resolves == 1
    await proxy.resolve_for_location(location_id=2, location_name="Manchester")
    assert proxy.total_resolves == 2


@pytest.mark.asyncio
async def test_proxy_uses_fallback_when_no_redis():
    """Without Redis, proxy should use in-memory fallback transparently."""
    proxy = EarthProxy()
    assert not proxy.using_redis

    proxy.register_adapter(MockAdapter(canned_facts=_make_canned_facts()))
    facts = await proxy.resolve_for_location(location_id=1, location_name="London")
    assert len(facts) == 3

    # Should be cached in fallback
    facts2 = await proxy.get_resolved_facts(1)
    assert len(facts2) == 3


@pytest.mark.asyncio
async def test_proxy_get_resolved_facts_empty_for_unknown():
    """get_resolved_facts for an unresolved location returns empty."""
    proxy = EarthProxy()
    facts = await proxy.get_resolved_facts(999)
    assert facts == []


# ---------------------------------------------------------------------------
# Agent perception tests — civilisation flows through observation,
# not through a backdoor injection. The world presents facts as part
# of the location, the agent perceives them the same as weather.
# ---------------------------------------------------------------------------

def _make_agent_and_geography():
    """Create a minimal agent + geography for perception tests."""
    from agents.system import AutonomousAgent, AgentTraits, AgentObservation
    from world.geography import LocationData

    location = LocationData(
        id=1, name="London", type="capital", lat=51.5, lng=-0.13,
        elevation=11.0, terrain="urban",
        admin_level_1="United Kingdom", admin_level_2="England",
        admin_level_3="Greater London", admin_level_4=None,
        population=8900000, metadata=None,
    )
    agent = AutonomousAgent(
        agent_id="test-01",
        name="Tester",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.3,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.8, comfort_temperature_c=18.0),
    )
    return agent, location


def test_agent_learns_from_perceived_earth_facts():
    """Agent learns from civilisation content presented by the world.
    This is the agent's own knowledge — not a copy of civilisation."""
    from agents.system import AgentObservation
    agent, location = _make_agent_and_geography()

    earth_facts = _make_canned_facts()
    observation = AgentObservation(
        current_location=location,
        current_weather=None,
        current_wind=None,
        neighbour_ids=[],
        simulation_time=datetime.now(timezone.utc),
        earth_facts=earth_facts,
    )

    agent._ingest_observation(observation)

    earth_predicates = [f for f in agent.knowledge.facts if f["predicate"].startswith("earth_")]
    assert len(earth_predicates) == 3

    texts = [f["text"] for f in earth_predicates]
    assert "London was founded by the Romans as Londinium in AD 43." in texts
    assert "London's West End is one of the world's major theatre districts." in texts
    assert "The Palace of Westminster is the seat of the UK Parliament." in texts


def test_earth_facts_have_correct_predicates():
    """Earth facts should be stored with earth_{domain} predicates —
    same naming pattern as weather/geography facts."""
    from agents.system import AgentObservation
    agent, location = _make_agent_and_geography()

    observation = AgentObservation(
        current_location=location,
        current_weather=None,
        current_wind=None,
        neighbour_ids=[],
        simulation_time=datetime.now(timezone.utc),
        earth_facts=_make_canned_facts(),
    )
    agent._ingest_observation(observation)

    predicates = [f["predicate"] for f in agent.knowledge.facts if f["predicate"].startswith("earth_")]
    assert "earth_history" in predicates
    assert "earth_culture" in predicates
    assert "earth_institution" in predicates


def test_agent_skips_empty_earth_facts():
    """Agent should skip earth facts with empty text during perception."""
    from agents.system import AgentObservation
    agent, location = _make_agent_and_geography()

    facts = [
        EarthFact(domain="history", topic="empty", text="", source="mock", location_id=1),
        EarthFact(domain="history", topic="blank", text="   ", source="mock", location_id=1),
        EarthFact(domain="culture", topic="real", text="A real fact.", source="mock", location_id=1),
    ]
    observation = AgentObservation(
        current_location=location,
        current_weather=None,
        current_wind=None,
        neighbour_ids=[],
        simulation_time=datetime.now(timezone.utc),
        earth_facts=facts,
    )
    agent._ingest_observation(observation)

    earth_predicates = [f for f in agent.knowledge.facts if f["predicate"].startswith("earth_")]
    assert len(earth_predicates) == 1
    assert earth_predicates[0]["text"] == "A real fact."


def test_earth_facts_not_re_ingested_on_repeated_observation():
    """Once an agent has perceived civilisation at a location, re-perceiving
    the same location should not duplicate facts. You read a plaque once."""
    from agents.system import AgentObservation
    agent, location = _make_agent_and_geography()

    earth_facts = _make_canned_facts()
    observation = AgentObservation(
        current_location=location,
        current_weather=None,
        current_wind=None,
        neighbour_ids=[],
        simulation_time=datetime.now(timezone.utc),
        earth_facts=earth_facts,
    )

    # Perceive twice at the same location
    agent._ingest_observation(observation)
    agent._ingest_observation(observation)

    earth_predicates = [f for f in agent.knowledge.facts if f["predicate"].startswith("earth_")]
    assert len(earth_predicates) == 3  # not 6


@pytest.mark.asyncio
async def test_full_pipeline_proxy_to_perception():
    """Full pipeline: adapter resolves -> proxy caches in Redis/fallback ->
    world presents through observation -> agent perceives. No backdoor."""
    from agents.system import AgentObservation
    agent, location = _make_agent_and_geography()

    # World resolves civilisation for this location (LOD)
    proxy = EarthProxy()
    proxy.register_adapter(MockAdapter(canned_facts=_make_canned_facts()))
    resolved = await proxy.resolve_for_location(location_id=1, location_name="London")
    assert len(resolved) == 3

    # World presents resolved facts through observation
    observation = AgentObservation(
        current_location=location,
        current_weather=None,
        current_wind=None,
        neighbour_ids=[],
        simulation_time=datetime.now(timezone.utc),
        earth_facts=await proxy.get_resolved_facts(1),
    )

    # Agent perceives — same pipeline as weather and geography
    agent._ingest_observation(observation)

    texts = [f["text"] for f in agent.knowledge.facts]
    assert any("Romans" in t for t in texts)
    assert any("Parliament" in t for t in texts)
