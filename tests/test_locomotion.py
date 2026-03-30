"""Tests for the physical locomotion system (S58).

Agents move between locations each tick. Movement costs energy
proportional to distance and terrain. No teleportation paralysis —
full cognitive loop runs every tick.
"""

from datetime import datetime, timezone

from agents.system import AgentTraits, AutonomousAgent, AgentObservation
from world.geography import Geography, LocationData, ConnectionData


def build_geography():
    """3-node graph: L1 --1km road-- L2 --5km path-- L3"""
    locations = {
        1: LocationData(1, "L1", "town", 51.5, -0.1, None, None, None, None, None, None, 1000, None),
        2: LocationData(2, "L2", "town", 51.5, 0.0, None, None, None, None, None, None, 1000, None),
        3: LocationData(3, "L3", "town", 51.6, 0.0, None, None, None, None, None, None, 1000, None),
    }
    c12 = ConnectionData(1, 2, 1.0, "road", None, None)
    c23 = ConnectionData(2, 3, 5.0, "path", None, None)
    return Geography(
        locations=locations,
        connections=[c12, c23],
        _adjacency={1: [c12], 2: [c12, c23], 3: [c23]},
    )


def make_agent(location_id=1):
    return AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=location_id,
        learning_rate=0.2,
        exploration_bias=0.1,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.7, comfort_temperature_c=20.0),
    )


# --- Movement ---

def test_move_changes_location():
    agent = make_agent(location_id=1)
    geo = build_geography()
    agent.apply_action(2, geo)
    assert agent.location_id == 2
    assert agent.last_action == "explore"


def test_move_records_distance():
    agent = make_agent(location_id=1)
    geo = build_geography()
    agent.apply_action(2, geo)
    assert agent.last_move_distance_km == 1.0


def test_stay_is_explore():
    agent = make_agent(location_id=1)
    geo = build_geography()
    agent.apply_action(1, geo)
    assert agent.location_id == 1
    assert agent.last_action == "explore"
    assert agent.last_move_distance_km == 0.0


def test_action_is_always_explore():
    """Whether moving or staying, the agent is always exploring."""
    agent = make_agent(location_id=1)
    geo = build_geography()

    agent.apply_action(2, geo)
    assert agent.last_action == "explore"

    agent.apply_action(2, geo)  # stay
    assert agent.last_action == "explore"


# --- Energy cost ---

def test_terrain_energy_modifiers_exist():
    agent = make_agent()
    for terrain in ["road", "rail", "path", "waterway", "proximity"]:
        mod = agent._terrain_energy_modifier(terrain)
        assert mod > 0, f"{terrain} has no modifier"


def test_road_is_cheapest():
    agent = make_agent()
    assert agent._terrain_energy_modifier("road") < agent._terrain_energy_modifier("path")
    assert agent._terrain_energy_modifier("road") < agent._terrain_energy_modifier("waterway")


def test_movement_costs_energy():
    agent = make_agent()
    cost_short_road = agent._movement_energy_cost(1.0, "road")
    cost_long_road = agent._movement_energy_cost(5.0, "road")
    assert cost_long_road > cost_short_road


def test_harder_terrain_costs_more():
    agent = make_agent()
    cost_road = agent._movement_energy_cost(3.0, "road")
    cost_path = agent._movement_energy_cost(3.0, "path")
    cost_water = agent._movement_energy_cost(3.0, "waterway")
    assert cost_path > cost_road
    assert cost_water > cost_path


def test_move_drains_energy():
    """Longer moves should have a net energy cost."""
    agent = make_agent(location_id=2)
    agent.energy = 100.0
    geo = build_geography()
    agent.apply_action(3, geo)  # 5km path — expensive
    obs = AgentObservation(
        current_location=geo.get_location(3),
        current_weather=None,
        current_wind=None,
        neighbour_ids=[2],
        simulation_time=datetime.now(timezone.utc),
    )
    agent.learn(obs)
    assert agent.energy < 100.0


def test_exploring_in_place_has_net_recovery():
    """Staying in place costs 0.3 but recovers 0.5 — net positive."""
    agent = make_agent(location_id=1)
    agent.energy = 50.0
    geo = build_geography()
    agent.apply_action(1, geo)  # stay = explore in place
    obs = AgentObservation(
        current_location=geo.get_location(1),
        current_weather=None,
        current_wind=None,
        neighbour_ids=[2],
        simulation_time=datetime.now(timezone.utc),
    )
    agent.learn(obs)
    assert agent.energy > 50.0  # net recovery when stationary


def test_energy_never_below_zero():
    agent = make_agent(location_id=2)
    agent.energy = 0.5
    geo = build_geography()
    agent.apply_action(3, geo)  # 5km path — expensive
    obs = AgentObservation(
        current_location=geo.get_location(3),
        current_weather=None,
        current_wind=None,
        neighbour_ids=[2],
        simulation_time=datetime.now(timezone.utc),
    )
    agent.learn(obs)
    assert agent.energy >= 0.0
