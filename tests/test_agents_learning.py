from datetime import datetime, timezone
import random

from agents.system import AgentGoal, AgentKnowledge, AgentSystem, AgentTraits, AutonomousAgent, AgentObservation
from world.geography import Geography, LocationData, ConnectionData


class DummyWeather:
    def __init__(self, by_location: dict[int, float]):
        self.by_location = by_location

    def get_weather(self, location_id: int):
        temp = self.by_location.get(location_id)
        if temp is None:
            return None

        class WeatherStateLike:
            def __init__(self, t: float):
                self.temperature_c = t
                self.precipitation_mm = 0.0
                self.wind_speed_kmh = 5.0
                self.conditions = "clear"

        return WeatherStateLike(temp)


def build_geography() -> Geography:
    locations = {
        1: LocationData(1, "L1", "town", 0.0, 0.0, None, None, None, None, None, None, 1000, None),
        2: LocationData(2, "L2", "town", 0.0, 1.0, None, None, None, None, None, None, 1000, None),
        3: LocationData(3, "L3", "town", 1.0, 0.0, None, None, None, None, None, None, 1000, None),
    }
    c12 = ConnectionData(1, 2, 1.0, "road", None, None)
    c13 = ConnectionData(1, 3, 1.0, "road", None, None)
    return Geography(
        locations=locations,
        connections=[c12, c13],
        _adjacency={1: [c12, c13], 2: [c12], 3: [c13]},
    )


def test_q_policy_prefers_highest_learned_action_value():
    geography = build_geography()
    weather = DummyWeather({1: 10.0, 2: 12.0, 3: 30.0})

    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )
    agent.knowledge.q_values = {1: {2: 1.5, 3: 0.2}}

    observation = agent.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))
    selected = agent.choose_next_location(observation, geography, weather, random.Random(1))

    assert selected == 2


def test_agent_knowledge_persists_q_values_roundtrip():
    knowledge = AgentKnowledge(
        visited_locations={1, 2},
        visit_counts={1: 2, 2: 1},
        location_scores={2: 0.45},
        condition_counts={"clear": 3},
        q_values={1: {2: 0.9, 3: -0.1}},
    )

    restored = AgentKnowledge.from_dict(knowledge.to_dict())

    assert restored.q_values[1][2] == 0.9
    assert restored.q_values[1][3] == -0.1
    assert restored.visit_counts[1] == 2


def test_bootstrap_agents_have_distinct_learning_profiles():
    geography = build_geography()
    system = AgentSystem.bootstrap(
        geography=geography,
        count=3,
        learning_rate=0.2,
        exploration_bias=0.2,
        seed=42,
    )

    signatures = {
        (
            round(a.learning_rate, 3),
            round(a.exploration_bias, 3),
            round(a.discount_factor, 3),
            round(a.traits.risk_tolerance, 3),
            round(a.traits.stamina, 3),
            round(a.traits.comfort_temperature_c, 1),
        )
        for a in system.agents
    }

    assert len(signatures) > 1


def test_learning_dynamics_are_trait_independent():
    geography = build_geography()
    weather = DummyWeather({1: 10.0, 2: 12.0, 3: 30.0})

    low_stamina = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.1, stamina=0.1, comfort_temperature_c=5.0),
        energy=80.0,
    )
    high_stamina = AutonomousAgent(
        agent_id="A2",
        name="Agent A2",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.9, stamina=0.9, comfort_temperature_c=25.0),
        energy=80.0,
    )

    # Force the same state/action/transition for both agents.
    low_stamina._last_state_id = 1
    low_stamina._last_action_id = 2
    high_stamina._last_state_id = 1
    high_stamina._last_action_id = 2

    low_stamina.apply_action(2, geography)
    high_stamina.apply_action(2, geography)

    next_obs_low = low_stamina.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))
    next_obs_high = high_stamina.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))

    low_stamina.learn(next_obs_low)
    high_stamina.learn(next_obs_high)

    assert low_stamina.last_reward == high_stamina.last_reward
    assert low_stamina.energy == high_stamina.energy


def test_low_energy_agent_creates_recovery_goal_but_still_explores():
    """Low energy triggers a recover goal, but the agent still moves — explorers don't stop."""
    geography = build_geography()
    weather = DummyWeather({1: 10.0, 2: 12.0, 3: 30.0})
    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
        energy=10.0,
    )

    observation = agent.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))
    agent.refresh_goal(observation, geography, random.Random(7))
    selected = agent.choose_next_location(observation, geography, weather, random.Random(7))

    assert agent.current_goal is not None
    assert agent.current_goal.kind == "recover"
    # Agent still picks a location — it doesn't stop
    assert selected in observation.neighbour_ids or selected == agent.location_id


def test_social_learning_shares_high_value_action_when_agents_meet():
    geography = build_geography()
    system = AgentSystem.bootstrap(
        geography=geography,
        count=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        seed=42,
    )

    a1, a2 = system.agents[0], system.agents[1]
    a1.location_id = 1
    a2.location_id = 1

    # Agent A1 knows that action 2 is good at state 1; A2 doesn't.
    a1.knowledge.q_values = {1: {2: 2.0}}
    a2.knowledge.q_values = {1: {2: 0.0}}

    system._social_learn(geography=geography)

    assert a2.knowledge.q_values[1][2] > 0.0


def test_social_learning_exchanges_facts_between_colocated_agents():
    geography = build_geography()
    system = AgentSystem.bootstrap(
        geography=geography,
        count=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        seed=42,
    )

    a1, a2 = system.agents[0], system.agents[1]
    a1.location_id = 1
    a2.location_id = 1

    a1._append_fact("population", "Population of L2 is 1200.", 2, value=1200)
    assert not any("Population of L2 is 1200." in str(f.get("text", "")) for f in a2.knowledge.facts)

    system._social_learn(geography=geography)

    assert any("Population of L2 is 1200." in str(f.get("text", "")) for f in a2.knowledge.facts)
    assert a2.knowledge.beliefs["population:2"]["value"] == 1200


def test_social_belief_merge_uses_confidence_weighting():
    geography = build_geography()
    system = AgentSystem.bootstrap(
        geography=geography,
        count=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        seed=42,
    )

    a1, a2 = system.agents[0], system.agents[1]
    a1.location_id = 1
    a2.location_id = 1

    # Learner already has strong confidence in existing value.
    a2.knowledge.beliefs["population:2"] = {
        "predicate": "population",
        "location_id": 2,
        "value": 1000,
        "confidence": 0.9,
        "evidence_count": 4,
    }

    # Sender has conflicting but lower-confidence belief.
    a1._append_fact("population", "Population of L2 is 1200.", 2, value=1200)
    a1.knowledge.beliefs["population:2"]["confidence"] = 0.45

    system._social_learn(geography=geography)

    assert a2.knowledge.beliefs["population:2"]["value"] == 1000


def test_social_dialogue_records_messages_between_colocated_agents():
    geography = build_geography()
    system = AgentSystem.bootstrap(
        geography=geography,
        count=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        seed=42,
    )

    a1, a2 = system.agents[0], system.agents[1]
    a1.location_id = 1
    a2.location_id = 1
    a1._append_fact("population", "Population of L2 is 1200.", 2, value=1200)

    system._social_learn(geography=geography, tick_count=7, sim_time=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc))

    assert len(a1.knowledge.dialogue_events) > 0
    assert len(a2.knowledge.dialogue_events) > 0
    inbound = [e for e in a2.knowledge.dialogue_events if e.get("direction") == "in"]
    assert inbound
    assert inbound[0].get("from_agent") == a1.agent_id
    assert "population" in str(inbound[0].get("text", "")).lower()


def test_listener_learns_from_dialogue_messages():
    geography = build_geography()
    system = AgentSystem.bootstrap(
        geography=geography,
        count=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        seed=42,
    )

    a1, a2 = system.agents[0], system.agents[1]
    a1.location_id = 1
    a2.location_id = 1

    a1._append_fact("population", "Population of L2 is 1200.", 2, value=1200)
    assert not any("Population of L2 is 1200." in str(f.get("text", "")) for f in a2.knowledge.facts)

    system._social_learn(geography=geography, tick_count=8, sim_time=datetime(2025, 1, 1, 12, 5, tzinfo=timezone.utc))

    learned = [f for f in a2.knowledge.facts if str(f.get("source")) == "dialogue"]
    assert learned
    assert any("Population of L2 is 1200." in str(f.get("text", "")) for f in learned)


def test_agent_can_answer_unseen_memory_question():
    geography = build_geography()

    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )
    agent.knowledge.visited_locations = {1, 2}
    agent.knowledge.visit_counts = {1: 1, 2: 2}

    result = agent.answer("Tell me locations you explored so far", geography)

    assert result["agent_id"] == "A1"
    assert "visited" in result["answer"].lower() or "locations" in result["answer"].lower()
    assert any(place["location_id"] == 2 for place in result["visited_places"])


def test_agent_learns_location_facts_from_observation():
    geography = build_geography()
    weather = DummyWeather({1: 10.0, 2: 12.0, 3: 30.0})

    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )
    obs = agent.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))
    agent._ingest_observation(obs)

    texts = [fact.get("text", "") for fact in agent.knowledge.facts]
    assert any("Population of L2" in text for text in texts)
    assert any("Coordinates of L2" in text for text in texts)
    assert any("Weather in L2" in text for text in texts)


def test_agent_answers_unseen_population_question_from_learned_facts():
    geography = build_geography()
    weather = DummyWeather({2: 12.0})

    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )
    obs = agent.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))
    agent._ingest_observation(obs)

    # Unseen phrasing (not pre-programmed template):
    response = agent.answer("Could you tell me how big L2 is by population?", geography)

    assert "population" in response["answer"].lower()
    assert "1000" in response["answer"]


def test_answer_contains_confidence_and_provenance():
    geography = build_geography()
    weather = DummyWeather({2: 12.0})

    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )

    obs = agent.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))
    agent._ingest_observation(obs)
    response = agent.answer("what can you tell me about L2", geography)

    assert 0.0 <= response["answer_confidence"] <= 1.0
    assert response["answer_certainty"] in {"known", "likely", "uncertain"}
    assert isinstance(response["supporting_facts"], list)


def test_multi_hop_location_answer_composes_related_facts():
    geography = build_geography()
    weather = DummyWeather({2: 12.0})

    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )

    obs = agent.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))
    agent._ingest_observation(obs)
    response = agent.answer("What do you know about L2 including population and coordinates?", geography)

    assert "about l2" in response["answer"].lower()
    assert "population" in response["answer"].lower()
    assert "coordinates" in response["answer"].lower()


def test_contradiction_detection_and_belief_revision():
    geography = build_geography()
    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=2,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )

    agent._append_fact("population", "Population of L2 is 1000.", 2, value=1000)
    agent._append_fact("population", "Population of L2 is 1200.", 2, value=1200)

    belief = agent.knowledge.beliefs["population:2"]
    assert belief["value"] == 1200
    assert len(agent.knowledge.belief_conflicts) >= 1

    response = agent.answer("What is the population of L2?", geography)
    assert "1200" in response["answer"]


def test_agent_can_answer_recently_from_timeline_memory():
    geography = build_geography()
    weather = DummyWeather({1: 10.0, 2: 12.0, 3: 14.0})
    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )

    obs1 = agent.perceive(geography, weather, datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc))
    agent.learn(obs1, tick_count=1)
    agent.apply_action(2, geography)
    obs2 = agent.perceive(geography, weather, datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc))
    agent.learn(obs2, tick_count=2)
    agent.apply_action(3, geography)
    obs3 = agent.perceive(geography, weather, datetime(2025, 1, 1, 10, 2, tzinfo=timezone.utc))
    agent.learn(obs3, tick_count=3)

    response = agent.answer("Where have you been recently?", geography)
    assert "recently" in response["answer"].lower()
    assert "l2" in response["answer"].lower() or "l3" in response["answer"].lower()
    assert response["retrieval_backend"] == "episodic_timeline"


def test_agent_can_answer_before_place_from_timeline_memory():
    geography = build_geography()
    weather = DummyWeather({1: 10.0, 2: 12.0, 3: 14.0})
    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
    )

    obs1 = agent.perceive(geography, weather, datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc))
    agent.learn(obs1, tick_count=1)
    agent.apply_action(2, geography)
    obs2 = agent.perceive(geography, weather, datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc))
    agent.learn(obs2, tick_count=2)
    agent.apply_action(3, geography)
    obs3 = agent.perceive(geography, weather, datetime(2025, 1, 1, 10, 2, tzinfo=timezone.utc))
    agent.learn(obs3, tick_count=3)

    response = agent.answer("What happened before L3?", geography)
    assert "before l3" in response["answer"].lower()
    assert "l2" in response["answer"].lower() or "l1" in response["answer"].lower()
    assert response["retrieval_backend"] == "episodic_timeline"


def test_refresh_goal_prioritizes_detected_knowledge_gap_target():
    geography = build_geography()
    weather = DummyWeather({1: 10.0, 2: 12.0, 3: 14.0})
    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
        energy=90.0,
    )

    agent.knowledge.visited_locations = {1, 2, 3}
    agent.knowledge.visit_counts = {1: 2, 2: 1, 3: 2}

    # L1 and L3 are well-covered; L2 intentionally missing key predicates.
    for location_id, location_name in [(1, "L1"), (3, "L3")]:
        agent._append_fact("location_type", f"{location_name} is a town.", location_id, value="town")
        agent._append_fact("coordinates", f"Coordinates of {location_name} are known.", location_id, value="0,0")
        agent._append_fact("connections", f"{location_name} has connections.", location_id)
        agent._append_fact("population", f"Population of {location_name} is 1000.", location_id, value=1000)

    agent._append_fact("location_type", "L2 is a town.", 2, value="town")

    observation = agent.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))
    agent.refresh_goal(observation, geography, random.Random(1))

    assert agent.current_goal is not None
    assert agent.current_goal.kind == "investigate_gap"
    assert agent.current_goal.target_location_id == 2


def test_investigate_gap_goal_biases_action_toward_gap_target():
    geography = build_geography()
    weather = DummyWeather({1: 10.0, 2: 12.0, 3: 14.0})
    agent = AutonomousAgent(
        agent_id="A1",
        name="Agent A1",
        location_id=1,
        learning_rate=0.2,
        exploration_bias=0.0,
        traits=AgentTraits(risk_tolerance=0.5, stamina=0.5, comfort_temperature_c=12.0),
        energy=90.0,
    )

    # Raw Q-values prefer 3, but gap investigation target should redirect to 2.
    agent.knowledge.q_values = {1: {2: 0.2, 3: 0.5}}
    agent.current_goal = AgentGoal(kind="investigate_gap", target_location_id=2, priority=0.9)

    observation = agent.perceive(geography, weather, datetime(2025, 1, 1, tzinfo=timezone.utc))
    chosen = agent.choose_next_location(observation, geography, weather, random.Random(2))

    assert chosen == 2


def test_agent_knowledge_serialization_with_new_fields():
    """Test that AgentKnowledge serializes and deserializes correctly with A55-A59 fields."""
    # Create knowledge with new fields populated
    knowledge = AgentKnowledge(
        visited_locations={1, 2, 3},
        facts=[{"text": "test fact", "predicate": "test"}],
        # A55/A56 - Conversations
        active_conversations={
            "conv_A1_A2_100": {
                "participants": ["A1", "A2"],
                "topic": "location:London",
                "initiator": "A1",
                "started_tick": 100,
                "turn_count": 2,
                "messages": [
                    {"from": "A1", "text": "Tell me about London", "tick": 100},
                    {"from": "A2", "text": "London is a capital city", "tick": 101},
                ],
                "status": "active",
            }
        },
        conversation_history=[
            {
                "participants": ["A1", "A3"],
                "topic": "weather",
                "concluded_tick": 95,
                "turn_count": 3,
            }
        ],
        # A57/A58 - Teaching/Learning
        teaching_events=[
            {
                "taught_agent": "A2",
                "topic": "location:1",
                "facts_taught": 3,
                "tick": 50,
                "effectiveness": 0.8,
            }
        ],
        learning_requests=[
            {
                "from_agent": "A1",
                "to_agent": "A3",
                "topic": "predicate:weather",
                "tick": 60,
                "fulfilled": True,
            }
        ],
        # A59 - Social network
        interaction_network={
            "A2": {
                "interaction_count": 15,
                "last_interaction_tick": 100,
                "topics_discussed": {"weather": 5, "geography": 10},
                "knowledge_quality": 0.85,
                "centrality_score": 0.6,
                "is_hub": False,
            }
        },
    )

    # Serialize
    serialized = knowledge.to_dict()

    # Verify new fields are in serialized dict
    assert "active_conversations" in serialized
    assert "conversation_history" in serialized
    assert "teaching_events" in serialized
    assert "learning_requests" in serialized
    assert "interaction_network" in serialized

    # Verify content
    assert "conv_A1_A2_100" in serialized["active_conversations"]
    assert len(serialized["conversation_history"]) == 1
    assert len(serialized["teaching_events"]) == 1
    assert len(serialized["learning_requests"]) == 1
    assert "A2" in serialized["interaction_network"]

    # Deserialize
    restored = AgentKnowledge.from_dict(serialized)

    # Verify new fields are restored
    assert len(restored.active_conversations) == 1
    assert "conv_A1_A2_100" in restored.active_conversations
    assert restored.active_conversations["conv_A1_A2_100"]["topic"] == "location:London"
    assert restored.active_conversations["conv_A1_A2_100"]["turn_count"] == 2

    assert len(restored.conversation_history) == 1
    assert restored.conversation_history[0]["topic"] == "weather"

    assert len(restored.teaching_events) == 1
    assert restored.teaching_events[0]["taught_agent"] == "A2"
    assert restored.teaching_events[0]["effectiveness"] == 0.8

    assert len(restored.learning_requests) == 1
    assert restored.learning_requests[0]["from_agent"] == "A1"
    assert restored.learning_requests[0]["fulfilled"] is True

    assert len(restored.interaction_network) == 1
    assert "A2" in restored.interaction_network
    assert restored.interaction_network["A2"]["interaction_count"] == 15
    assert restored.interaction_network["A2"]["is_hub"] is False


def test_agent_knowledge_backward_compatibility():
    """Test that old AgentKnowledge data (without new fields) deserializes correctly."""
    # Simulate old data without new fields
    old_data = {
        "visited_locations": [1, 2],
        "visit_counts": {"1": 3, "2": 1},
        "location_scores": {"1": 1.5, "2": 0.8},
        "condition_counts": {"clear": 5},
        "q_values": {"1": {"2": 0.5}},
        "facts": [{"text": "old fact", "predicate": "test"}],
        "beliefs": {},
        "belief_conflicts": [],
        "episodic_events": [],
        "dialogue_events": [],
        "social_memory": {
            "A2": {"name": "Agent 2", "trust": 0.7, "familiarity": 0.5}
        },
        "encounters": [],
        # NOTE: No active_conversations, conversation_history, teaching_events,
        # learning_requests, or interaction_network
    }

    # Deserialize old data
    knowledge = AgentKnowledge.from_dict(old_data)

    # Verify new fields exist with empty defaults
    assert knowledge.active_conversations == {}
    assert knowledge.conversation_history == []
    assert knowledge.teaching_events == []
    assert knowledge.learning_requests == []
    assert knowledge.interaction_network == {}

    # Verify old fields are intact
    assert len(knowledge.visited_locations) == 2
    assert 1 in knowledge.visited_locations
    assert knowledge.visit_counts[1] == 3
    assert "A2" in knowledge.social_memory
    assert knowledge.social_memory["A2"]["trust"] == 0.7
