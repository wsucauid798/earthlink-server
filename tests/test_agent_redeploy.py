from agents.system import AgentKnowledge, AgentSystem, AgentTraits, AutonomousAgent
from world.geography import LocationData


class _FakeGeography:
    def __init__(self, locations: dict[int, LocationData]):
        self._locations = locations

    def get_location(self, location_id: int):
        return self._locations.get(location_id)


def _agent(agent_id: str, location_id: int) -> AutonomousAgent:
    return AutonomousAgent(
        agent_id=agent_id,
        name=f"Agent {agent_id}",
        location_id=location_id,
        learning_rate=0.2,
        exploration_bias=0.3,
        traits=AgentTraits(
            risk_tolerance=0.5,
            stamina=0.5,
            comfort_temperature_c=15.0,
        ),
        knowledge=AgentKnowledge(),
    )


def test_redeploy_to_spawn_pool_expands_country_coverage():
    system = AgentSystem(
        agents=[
            _agent("A1", 1),
            _agent("A2", 1),
            _agent("A3", 1),
            _agent("A4", 1),
        ]
    )

    locations = {
        1: LocationData(1, "Legacy A", "city", 0.0, 0.0, None, None, "CountryA", None, None, None, 1000, None),
        10: LocationData(10, "A-1", "city", 1.0, 1.0, None, None, "CountryA", None, None, None, 1000, None),
        20: LocationData(20, "B-1", "city", 2.0, 2.0, None, None, "CountryB", None, None, None, 1000, None),
        30: LocationData(30, "C-1", "city", 3.0, 3.0, None, None, "CountryC", None, None, None, 1000, None),
    }
    geography = _FakeGeography(locations)

    before = system.country_coverage(geography)
    assert before == {"CountryA"}

    spawn_pool = [locations[10], locations[20], locations[30]]
    moved = system.redeploy_to_spawn_pool(spawn_pool)

    assert moved >= 2
    after = system.country_coverage(geography)
    assert after == {"CountryA", "CountryB", "CountryC"}
