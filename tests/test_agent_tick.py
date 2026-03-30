"""Tests for the agent tick loop.

Every tick, every agent must:
1. Perceive — observe current location, features, conditions, connections
2. Learn — ingest facts, build knowledge, revise beliefs
3. Decide — choose where to go based on curiosity, knowledge gaps, goals
4. Move — go there, pay energy proportional to distance
5. Communicate — exchange knowledge with co-located agents
6. Produce traceable evidence — tick event records all of the above

No exceptions. No resting. No dead ticks.
"""

from datetime import datetime, timezone
import random
import asyncio

from agents.system import AgentSystem, AgentTraits, AutonomousAgent
from world.geography import Geography, LocationData, ConnectionData


class DummyWeather:
    def get_weather(self, location_id):
        class W:
            temperature_c = 10.0
            precipitation_mm = 0.0
            wind_speed_kmh = 5.0
            conditions = "clear"
        return W()


def build_geography():
    locations = {
        1: LocationData(1, "A", "city", 51.5, -0.1, 25.0, "urban", "UK", "England", "London", None, 100000, None),
        2: LocationData(2, "B", "town", 51.6, -0.1, 30.0, "rural", "UK", "England", "Essex", None, 5000, None),
        3: LocationData(3, "C", "town", 51.5, 0.0, 10.0, "urban", "UK", "England", "Kent", None, 8000, None),
        4: LocationData(4, "D", "village", 51.7, 0.0, 50.0, "rural", "UK", "England", "Herts", None, 1000, None),
    }
    c12 = ConnectionData(1, 2, 2.0, "road", None, None)
    c13 = ConnectionData(1, 3, 3.0, "road", None, None)
    c24 = ConnectionData(2, 4, 1.5, "path", None, None)
    c34 = ConnectionData(3, 4, 2.0, "road", None, None)
    return Geography(
        locations=locations,
        connections=[c12, c13, c24, c34],
        _adjacency={
            1: [c12, c13],
            2: [c12, c24],
            3: [c13, c34],
            4: [c24, c34],
        },
    )


# === 1. PERCEIVE ===

def test_agent_perceives_every_tick():
    """Agent must observe its location, weather, connections each tick."""
    geo = build_geography()
    agent = AutonomousAgent(
        agent_id="A1", name="A1", location_id=1,
        learning_rate=0.2, exploration_bias=0.3,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.7, comfort_temperature_c=20.0),
    )
    obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
    assert obs.current_location is not None
    assert obs.current_location.id == 1
    assert len(obs.neighbour_ids) > 0
    assert obs.current_weather is not None


# === 2. LEARN ===

def test_agent_learns_every_tick():
    """Agent must gain knowledge over ticks."""
    geo = build_geography()
    agent = AutonomousAgent(
        agent_id="A1", name="A1", location_id=1,
        learning_rate=0.2, exploration_bias=0.5,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.7, comfort_temperature_c=20.0),
    )
    rng = random.Random(42)
    initial = agent.knowledge.knowledge_score

    for _ in range(5):
        obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
        next_loc = agent.choose_next_location(obs, geo, DummyWeather(), rng)
        agent.apply_action(next_loc, geo)
        next_obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
        agent.learn(next_obs)

    assert agent.knowledge.knowledge_score > initial


# === 3. DECIDE ===

def test_agent_decides_every_tick():
    """Agent must choose a next location — never returns None."""
    geo = build_geography()
    agent = AutonomousAgent(
        agent_id="A1", name="A1", location_id=1,
        learning_rate=0.2, exploration_bias=0.3,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.7, comfort_temperature_c=20.0),
    )
    obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
    result = agent.choose_next_location(obs, geo, DummyWeather(), random.Random(42))
    assert result is not None
    assert result in obs.neighbour_ids or result == agent.location_id


def test_zero_energy_agent_still_decides():
    """Energy must never prevent decision-making."""
    geo = build_geography()
    agent = AutonomousAgent(
        agent_id="A1", name="A1", location_id=1,
        learning_rate=0.2, exploration_bias=0.5,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=20.0),
        energy=0.0,
    )
    obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
    result = agent.choose_next_location(obs, geo, DummyWeather(), random.Random(42))
    assert result is not None


# === 4. MOVE ===

def test_agent_moves_to_new_locations():
    """Agent must visit multiple locations over time."""
    geo = build_geography()
    agent = AutonomousAgent(
        agent_id="A1", name="A1", location_id=1,
        learning_rate=0.2, exploration_bias=0.8,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.7, comfort_temperature_c=20.0),
    )
    rng = random.Random(42)
    for _ in range(20):
        obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
        next_loc = agent.choose_next_location(obs, geo, DummyWeather(), rng)
        agent.apply_action(next_loc, geo)
        next_obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
        agent.learn(next_obs)

    assert agent.knowledge.discovered_location_count > 1


def test_movement_costs_energy():
    """Moving must cost energy proportional to distance."""
    geo = build_geography()
    agent = AutonomousAgent(
        agent_id="A1", name="A1", location_id=1,
        learning_rate=0.2, exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.7, comfort_temperature_c=20.0),
        energy=100.0,
    )
    agent.apply_action(2, geo)  # 2km road
    obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
    agent.learn(obs)
    assert agent.energy < 100.0


# === 5. COMMUNICATE ===

def test_colocated_agents_communicate():
    """Agents at the same location must exchange knowledge."""
    geo = build_geography()
    system = AgentSystem.bootstrap(geo, count=3, learning_rate=0.2, exploration_bias=0.1, seed=42)

    # Put all agents at location 1
    for agent in system.agents:
        agent.location_id = 1
        agent.knowledge.facts.append({
            "predicate": "earth_history",
            "subject": "A",
            "value": f"Fact from {agent.agent_id}",
            "confidence": 0.9,
            "source": "test",
        })

    # Run social learning
    system._social_learn(geography=geo, tick_count=1, sim_time=datetime.now(timezone.utc))

    # Each agent should know about facts from others
    for agent in system.agents:
        other_facts = [f for f in agent.knowledge.facts if f.get("source") == "social"]
        # At least some social learning should have occurred
        assert len(agent.knowledge.social_memory) > 0 or len(other_facts) > 0


# === 6. TRACEABLE EVIDENCE ===

def test_tick_event_has_all_dimensions():
    """Tick event must capture perceive, learn, decide, move data."""
    geo = build_geography()
    system = AgentSystem.bootstrap(geo, count=2, learning_rate=0.2, exploration_bias=0.3, seed=42)

    events = asyncio.run(
        system.tick(geo, DummyWeather(), datetime.now(timezone.utc), tick_count=1)
    )

    assert len(events) == 2
    for event in events:
        # Perceive + Move
        assert "from_location_id" in event
        assert "to_location_id" in event
        assert "moved" in event
        assert "distance_km" in event
        # Learn
        assert "knowledge_score" in event
        assert "facts_learned" in event
        assert "visited_count" in event
        # Decide
        assert "goal" in event
        assert "q_value" in event
        assert "reward" in event
        # Energy
        assert "energy" in event


# === NO DEAD STATES ===

def test_no_rest_action_exists():
    """There must be no 'rest' action anywhere."""
    geo = build_geography()
    agent = AutonomousAgent(
        agent_id="A1", name="A1", location_id=1,
        learning_rate=0.2, exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=20.0),
        energy=1.0,
    )
    for _ in range(10):
        obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
        next_loc = agent.choose_next_location(obs, geo, DummyWeather(), random.Random(42))
        agent.apply_action(next_loc, geo)
        next_obs = agent.perceive(geo, DummyWeather(), datetime.now(timezone.utc))
        agent.learn(next_obs)
        assert agent.last_action != "rest"
        assert agent.last_action != "observe"
        assert agent.last_action != "traveling"
        assert agent.last_action != "departing"
        assert agent.last_action != "arrived"
