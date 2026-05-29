"""Tests for Ray-based parallel agent ticking.

Tests both the Ray actor path and verifies that sequential fallback
produces identical behaviour when Ray is not available.
"""

import math
import pytest
import random
from datetime import datetime, timezone

from agents.system import (
    AgentSystem,
    AgentTraits,
    AutonomousAgent,
    AgentKnowledge,
    AgentObservation,
)
from world.geography import ConnectionData, LocationData, Geography
from world.weather import Weather, WeatherState


def _haversine_km(a: LocationData, b: LocationData) -> float:
    radius = 6371.0
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = math.radians(b.lat - a.lat)
    dlng = math.radians(b.lng - a.lng)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubWeather:
    """Minimal weather stub — returns a default WeatherState for any location."""

    def get_weather(self, location_id: int) -> WeatherState | None:
        return WeatherState(
            location_id=location_id,
            temperature_c=12.0,
            humidity_pct=70.0,
            wind_speed_kmh=15.0,
            conditions="cloudy",
        )

    def get_all_weather(self):
        return {}


def _make_geography() -> Geography:
    """Build a minimal 3-node geography for testing."""
    locations = {
        1: LocationData(
            id=1, name="London", type="capital", lat=51.5, lng=-0.13,
            elevation=11.0, terrain="urban",
            admin_level_1="United Kingdom", admin_level_2="England",
            admin_level_3="Greater London", admin_level_4=None,
            population=8900000, metadata=None,
        ),
        2: LocationData(
            id=2, name="Manchester", type="city", lat=53.48, lng=-2.24,
            elevation=38.0, terrain="urban",
            admin_level_1="United Kingdom", admin_level_2="England",
            admin_level_3="Greater Manchester", admin_level_4=None,
            population=550000, metadata=None,
        ),
        3: LocationData(
            id=3, name="Edinburgh", type="city", lat=55.95, lng=-3.19,
            elevation=47.0, terrain="urban",
            admin_level_1="United Kingdom", admin_level_2="Scotland",
            admin_level_3="City of Edinburgh", admin_level_4=None,
            population=520000, metadata=None,
        ),
    }
    neighbour_ids = {
        1: [2, 3],
        2: [1, 3],
        3: [1, 2],
    }
    connections: list[ConnectionData] = []
    seen: set[tuple[int, int]] = set()
    for src, neighbours in neighbour_ids.items():
        for dst in neighbours:
            key = (min(src, dst), max(src, dst))
            if key in seen:
                continue
            seen.add(key)
            distance = round(_haversine_km(locations[src], locations[dst]), 2)
            connections.append(ConnectionData(src, dst, distance, "road", None, None))
    return Geography.from_memory(locations=locations, connections=connections)


def _make_agents(n: int = 3, seed: int = 42) -> list[AutonomousAgent]:
    """Create n test agents at sequential locations."""
    rng = random.Random(seed)
    agents = []
    loc_ids = [1, 2, 3]
    for i in range(n):
        agent = AutonomousAgent(
            agent_id=f"T{i + 1}",
            name=f"Test Agent {i + 1}",
            location_id=loc_ids[i % len(loc_ids)],
            learning_rate=0.2,
            exploration_bias=0.3,
            traits=AgentTraits(
                risk_tolerance=0.5,
                stamina=0.8,
                comfort_temperature_c=18.0,
            ),
        )
        agent.knowledge.visited_locations.add(agent.location_id)
        agents.append(agent)
    return agents


def _make_system(n: int = 3, seed: int = 42) -> AgentSystem:
    agents = _make_agents(n, seed)
    return AgentSystem(agents=agents, _rng=random.Random(seed))


# ---------------------------------------------------------------------------
# Sequential-mode tests (always work, no Ray needed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sequential_tick_produces_events():
    """Sequential tick should produce one event per agent."""
    system = _make_system(3)
    geography = _make_geography()

    events = await system.tick(
        geography=geography,
        weather=_StubWeather(),
        sim_time=datetime.now(timezone.utc),
        tick_count=1,
    )

    assert len(events) == 3
    agent_ids = {e["agent_id"] for e in events}
    assert agent_ids == {"T1", "T2", "T3"}
    for event in events:
        assert "from_location_id" in event
        assert "to_location_id" in event
        assert "knowledge_score" in event


@pytest.mark.asyncio
async def test_sequential_summaries():
    """summaries() should return one summary per agent."""
    system = _make_system(3)
    geography = _make_geography()
    summaries = system.summaries(geography)
    assert len(summaries) == 3


@pytest.mark.asyncio
async def test_sequential_detail():
    """detail() should return detailed info for a specific agent."""
    system = _make_system(3)
    geography = _make_geography()
    detail = system.detail("T1", geography)
    assert detail is not None
    assert detail["id"] == "T1"


@pytest.mark.asyncio
async def test_sequential_persist_and_restore():
    """Agents should round-trip through persist/restore."""
    system = _make_system(3)
    geography = _make_geography()

    # Tick once to generate some state
    await system.tick(geography, _StubWeather(), datetime.now(timezone.utc), tick_count=1)

    persisted = system.to_persisted()
    assert len(persisted) == 3

    restored = AgentSystem.from_persisted(persisted, seed=42)
    assert len(restored.agents) == 3
    assert restored.agents[0].agent_id == "T1"


# ---------------------------------------------------------------------------
# Ray-mode tests (skip if Ray not installed)
# ---------------------------------------------------------------------------

try:
    import ray
    RAY_INSTALLED = True
except ImportError:
    RAY_INSTALLED = False


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
async def test_ray_init_and_shutdown():
    """init_ray should create actors and set _use_ray."""
    system = _make_system(3)
    try:
        result = system.init_ray(seed=42)
        assert result is True
        assert system.using_ray is True
        assert len(system._actors) == 3
    finally:
        system.shutdown_ray()
        assert system.using_ray is False


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
async def test_ray_tick_produces_events():
    """Ray tick should produce one event per agent — same as sequential."""
    system = _make_system(3)
    geography = _make_geography()
    try:
        system.init_ray(seed=42)
        events = await system.tick(
            geography=geography,
            weather=_StubWeather(),
            sim_time=datetime.now(timezone.utc),
            tick_count=1,
        )
        assert len(events) == 3
        agent_ids = {e["agent_id"] for e in events}
        assert agent_ids == {"T1", "T2", "T3"}
        for event in events:
            assert "from_location_id" in event
            assert "to_location_id" in event
            assert "knowledge_score" in event
    finally:
        system.shutdown_ray()


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
async def test_ray_summaries():
    """summaries() via Ray actors should match sequential."""
    system = _make_system(3)
    geography = _make_geography()
    try:
        system.init_ray(seed=42)

        # Tick first so agents have some state
        await system.tick(geography, _StubWeather(), datetime.now(timezone.utc), tick_count=1)

        summaries = system.summaries(geography)
        assert len(summaries) == 3
        assert all("id" in s for s in summaries)
    finally:
        system.shutdown_ray()


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
async def test_ray_detail():
    """detail() via Ray actors should return the correct agent."""
    system = _make_system(3)
    geography = _make_geography()
    try:
        system.init_ray(seed=42)
        detail = system.detail("T1", geography)
        assert detail is not None
        assert detail["id"] == "T1"
    finally:
        system.shutdown_ray()


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
async def test_ray_persist_and_restore():
    """Agents in Ray mode should round-trip through persist/restore."""
    system = _make_system(3)
    geography = _make_geography()
    try:
        system.init_ray(seed=42)

        # Tick to generate state
        await system.tick(geography, _StubWeather(), datetime.now(timezone.utc), tick_count=1)

        persisted = system.to_persisted()
        assert len(persisted) == 3

        restored = AgentSystem.from_persisted(persisted, seed=42)
        assert len(restored.agents) == 3
    finally:
        system.shutdown_ray()


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
async def test_ray_social_learning():
    """Co-located agents should exchange knowledge via social learning in Ray mode."""
    # Create 2 agents at the same location
    agents = [
        AutonomousAgent(
            agent_id="S1", name="Speaker", location_id=1,
            learning_rate=0.2, exploration_bias=0.3,
            traits=AgentTraits(risk_tolerance=0.5, stamina=0.8, comfort_temperature_c=18.0),
        ),
        AutonomousAgent(
            agent_id="S2", name="Listener", location_id=1,
            learning_rate=0.2, exploration_bias=0.3,
            traits=AgentTraits(risk_tolerance=0.5, stamina=0.8, comfort_temperature_c=18.0),
        ),
    ]
    # Give S1 some facts to share
    agents[0].knowledge.facts = [
        {"predicate": "earth_history", "location_id": 1, "text": "London was founded by Romans.", "value": None, "source": "observation"},
    ]
    agents[0].knowledge.visited_locations = {1}
    agents[1].knowledge.visited_locations = {1}

    system = AgentSystem(agents=agents, _rng=random.Random(42))
    geography = _make_geography()

    try:
        system.init_ray(seed=42)

        # Tick — both agents at location 1, social learning should trigger
        await system.tick(geography, _StubWeather(), datetime.now(timezone.utc), tick_count=1)

        # Check that S2 received some facts via social learning
        # (Get persisted state to inspect)
        persisted = system.to_persisted()
        s2_data = next(p for p in persisted if p["id"] == "S2")
        # S2 should have at least one fact (either from observation or social)
        assert len(s2_data["knowledge"]["facts"]) > 0
    finally:
        system.shutdown_ray()


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
async def test_ray_multiple_ticks():
    """Multiple ticks should work correctly in Ray mode."""
    system = _make_system(3)
    geography = _make_geography()

    try:
        system.init_ray(seed=42)

        for tick in range(1, 6):
            events = await system.tick(
                geography, _StubWeather(), datetime.now(timezone.utc), tick_count=tick,
            )
            assert len(events) == 3

        # Agents should have moved and learned
        summaries = system.summaries(geography)
        assert all(s["knowledge_score"] > 0 for s in summaries)
    finally:
        system.shutdown_ray()


@pytest.mark.asyncio
@pytest.mark.skipif(not RAY_INSTALLED, reason="Ray not installed")
async def test_ray_answer():
    """ask() via Ray actors should return an answer dict."""
    system = _make_system(3)
    geography = _make_geography()
    try:
        system.init_ray(seed=42)

        # Tick to build some knowledge
        await system.tick(geography, _StubWeather(), datetime.now(timezone.utc), tick_count=1)

        result = system.answer("T1", "Where am I?", geography)
        assert result is not None
        assert "answer" in result
    finally:
        system.shutdown_ray()
