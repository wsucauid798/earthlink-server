"""Autonomous world agents.

Agents are embodied entities that perceive the world, choose actions,
move through real geography, and continuously update their own knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import re
import random

logger = logging.getLogger(__name__)

from world.geography import Geography, LocationData
from world.weather import Weather, WeatherState

from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from world.earth_proxy import EarthProxy
    from world.wind import Wind, WindState

from .retrieval import ChromaFactStore, SemanticFactRetriever


_SEMANTIC_RETRIEVER: SemanticFactRetriever | None = None


def _get_retriever() -> SemanticFactRetriever:
    global _SEMANTIC_RETRIEVER
    if _SEMANTIC_RETRIEVER is None:
        _SEMANTIC_RETRIEVER = SemanticFactRetriever()
    return _SEMANTIC_RETRIEVER


def _get_chroma_store(agent_id: str) -> ChromaFactStore | None:
    """Get or create a Chroma fact store for an agent. Returns None if unavailable."""
    try:
        from config import settings
        store = ChromaFactStore(agent_id, chroma_host=settings.chroma_host, chroma_port=settings.chroma_port)
        return store if store.available else None
    except Exception:
        return None


@dataclass
class AgentKnowledge:
    """Learned state for a single autonomous agent."""

    visited_locations: set[int] = field(default_factory=set)
    visit_counts: dict[int, int] = field(default_factory=dict)
    location_scores: dict[int, float] = field(default_factory=dict)
    condition_counts: dict[str, int] = field(default_factory=dict)
    q_values: dict[int, dict[int, float]] = field(default_factory=dict)
    facts: list[dict] = field(default_factory=list)
    beliefs: dict[str, dict] = field(default_factory=dict)
    belief_conflicts: list[dict] = field(default_factory=list)
    episodic_events: list[dict] = field(default_factory=list)
    dialogue_events: list[dict] = field(default_factory=list)

    # --- Social memory (A53/A54/A60) ---
    # Persistent memory of other agents this agent has encountered.
    # Key: agent_id.  Value: relationship record.
    social_memory: dict[str, dict] = field(default_factory=dict)
    # Encounter event log (capped).
    encounters: list[dict] = field(default_factory=list)

    # --- Conversation state (A55/A56) ---
    # Active conversations this agent is currently engaged in.
    # Key: conversation_id.  Value: conversation record with participants, topic, messages, etc.
    active_conversations: dict[str, dict] = field(default_factory=dict)
    # Archived conversation history (capped at 1000).
    conversation_history: list[dict] = field(default_factory=list)

    # --- Teaching/learning tracking (A57/A58) ---
    # Teaching events this agent has conducted (capped at 1000).
    teaching_events: list[dict] = field(default_factory=list)
    # Learning requests this agent has made or received (capped at 1000).
    learning_requests: list[dict] = field(default_factory=list)

    # --- Social network analysis (A59) ---
    # Interaction network tracking communication patterns with peers.
    # Key: agent_id.  Value: interaction statistics (count, topics, quality, etc.).
    interaction_network: dict[str, dict] = field(default_factory=dict)

    @property
    def discovered_location_count(self) -> int:
        return len(self.visited_locations)

    @property
    def knowledge_score(self) -> float:
        # Compact scalar confidence/progress metric for quick API display.
        score = 0.0
        score += self.discovered_location_count * 1.0
        score += len(self.condition_counts) * 0.5
        score += sum(max(v, 0.0) for v in self.location_scores.values())
        return round(score, 2)

    def to_dict(self) -> dict:
        return {
            "visited_locations": sorted(self.visited_locations),
            "visit_counts": {str(k): v for k, v in self.visit_counts.items()},
            "location_scores": {str(k): v for k, v in self.location_scores.items()},
            "condition_counts": dict(self.condition_counts),
            "q_values": {
                str(state_id): {str(action_id): value for action_id, value in action_map.items()}
                for state_id, action_map in self.q_values.items()
            },
            "facts": list(self.facts),
            "beliefs": dict(self.beliefs),
            "belief_conflicts": list(self.belief_conflicts),
            "episodic_events": list(self.episodic_events),
            "dialogue_events": list(self.dialogue_events),
            "social_memory": {k: dict(v) for k, v in self.social_memory.items()},
            "encounters": list(self.encounters),
            # A55/A56 - Conversations
            "active_conversations": {k: dict(v) for k, v in self.active_conversations.items()},
            "conversation_history": list(self.conversation_history),
            # A57/A58 - Teaching/Learning
            "teaching_events": list(self.teaching_events),
            "learning_requests": list(self.learning_requests),
            # A59 - Social network
            "interaction_network": {k: dict(v) for k, v in self.interaction_network.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "AgentKnowledge":
        payload = payload or {}
        visited_raw = payload.get("visited_locations", [])
        visit_counts_raw = payload.get("visit_counts", {})
        location_scores_raw = payload.get("location_scores", {})
        condition_counts = payload.get("condition_counts", {})
        q_values_raw = payload.get("q_values", {})
        facts_raw = payload.get("facts", [])
        beliefs_raw = payload.get("beliefs", {})
        conflicts_raw = payload.get("belief_conflicts", [])
        events_raw = payload.get("episodic_events", [])
        dialogue_raw = payload.get("dialogue_events", [])
        social_memory_raw = payload.get("social_memory", {})
        encounters_raw = payload.get("encounters", [])
        # A55/A56 - Conversations
        active_conversations_raw = payload.get("active_conversations", {})
        conversation_history_raw = payload.get("conversation_history", [])
        # A57/A58 - Teaching/Learning
        teaching_events_raw = payload.get("teaching_events", [])
        learning_requests_raw = payload.get("learning_requests", [])
        # A59 - Social network
        interaction_network_raw = payload.get("interaction_network", {})

        return cls(
            visited_locations={int(v) for v in visited_raw},
            visit_counts={int(k): int(v) for k, v in visit_counts_raw.items()},
            location_scores={int(k): float(v) for k, v in location_scores_raw.items()},
            condition_counts={str(k): int(v) for k, v in condition_counts.items()},
            q_values={
                int(state_id): {int(action_id): float(value) for action_id, value in actions.items()}
                for state_id, actions in q_values_raw.items()
            },
            facts=[fact for fact in facts_raw if isinstance(fact, dict)],
            beliefs={str(k): v for k, v in beliefs_raw.items() if isinstance(v, dict)},
            belief_conflicts=[c for c in conflicts_raw if isinstance(c, dict)],
            episodic_events=[e for e in events_raw if isinstance(e, dict)],
            dialogue_events=[d for d in dialogue_raw if isinstance(d, dict)],
            social_memory={str(k): v for k, v in social_memory_raw.items() if isinstance(v, dict)},
            encounters=[e for e in encounters_raw if isinstance(e, dict)],
            # A55/A56 - Conversations
            active_conversations={str(k): v for k, v in active_conversations_raw.items() if isinstance(v, dict)},
            conversation_history=[c for c in conversation_history_raw if isinstance(c, dict)],
            # A57/A58 - Teaching/Learning
            teaching_events=[t for t in teaching_events_raw if isinstance(t, dict)],
            learning_requests=[r for r in learning_requests_raw if isinstance(r, dict)],
            # A59 - Social network
            interaction_network={str(k): v for k, v in interaction_network_raw.items() if isinstance(v, dict)},
        )


@dataclass
class AgentTraits:
    """Persisted trait placeholders (not used in runtime policy logic)."""

    risk_tolerance: float
    stamina: float
    comfort_temperature_c: float


@dataclass
class AgentObservation:
    """Perception snapshot consumed by decision policy.

    This is what the world presents to the agent each tick — geography,
    weather, connections, time, and civilisation content at this location.
    Everything the agent needs to see a complete Earth.
    """

    current_location: LocationData
    current_weather: WeatherState | None
    current_wind: WindState | None
    neighbour_ids: list[int]
    simulation_time: datetime
    earth_facts: list = field(default_factory=list)  # civilisation content at this location


@dataclass
class AgentGoal:
    """Agent-generated objective.

    Goal types:
    - "explore": Visit new or specific locations
    - "investigate_gap": Go to location with knowledge gaps
    - "recover": Stay put to regain energy
    - "seek_agent": Find a specific agent at their last known location (A55/A58/A59)
    """

    kind: str
    target_location_id: int | None = None
    target_agent_id: str | None = None  # For "seek_agent" goals
    priority: float = 1.0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "target_location_id": self.target_location_id,
            "target_agent_id": self.target_agent_id,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "AgentGoal | None":
        if not payload:
            return None
        return cls(
            kind=str(payload.get("kind", "explore")),
            target_location_id=(None if payload.get("target_location_id") is None else int(payload["target_location_id"])),
            target_agent_id=(None if payload.get("target_agent_id") is None else str(payload["target_agent_id"])),
            priority=float(payload.get("priority", 1.0)),
        )


@dataclass
class AutonomousAgent:
    """A rational learning entity acting inside the world."""

    agent_id: str
    name: str
    location_id: int
    learning_rate: float
    exploration_bias: float
    traits: AgentTraits
    discount_factor: float = 0.92
    energy: float = 100.0
    last_move_distance_km: float = 0.0
    last_move_connection_type: str = "road"
    last_action: str = "spawned"
    # Locomotion
    speed_kmh: float = 5.0  # base walking speed (km/h), affects energy cost
    last_reward: float = 0.0
    current_goal: AgentGoal | None = None
    goal_age_ticks: int = 0
    knowledge: AgentKnowledge = field(default_factory=AgentKnowledge)
    _last_state_id: int | None = field(default=None, init=False, repr=False)
    _last_action_id: int | None = field(default=None, init=False, repr=False)
    _earth_observed_locations: set = field(default_factory=set, init=False, repr=False)
    _chroma_store: ChromaFactStore | None = field(default=None, init=False, repr=False)
    _fact_counter: int = field(default=0, init=False, repr=False)

    _BELIEF_PREDICATES = {
        "location_type",
        "population",
        "elevation",
        "terrain",
        "nation",
        "region",
        "coordinates",
    }

    def perceive(self, geography: Geography, weather: Weather, sim_time: datetime, earth_facts: list | None = None, wind: Wind | None = None) -> AgentObservation:
        """Perceive the world. The world presents everything — geography, weather,
        wind, civilisation — as one seamless environment. The agent just sees Earth."""
        current_location = geography.get_location(self.location_id)
        if current_location is None:
            # Agent's location not in cache — create a minimal placeholder
            current_location = LocationData(
                id=self.location_id, name="Unknown", type="town",
                lat=0.0, lng=0.0, elevation=None, terrain=None,
                admin_level_1=None, admin_level_2=None,
                admin_level_3=None, admin_level_4=None,
                population=None, metadata=None,
            )

        wind_state = None
        if wind:
            wind_state = wind.get_wind(
                self.location_id,
                lat=current_location.lat,
                lng=current_location.lng,
                terrain=current_location.terrain,
            )

        return AgentObservation(
            current_location=current_location,
            current_weather=weather.get_weather(self.location_id),
            current_wind=wind_state,
            neighbour_ids=[
                conn.to_id if conn.from_id == self.location_id else conn.from_id
                for conn in geography.get_neighbours(self.location_id)
            ],
            simulation_time=sim_time,
            earth_facts=earth_facts or [],
        )

    def choose_next_location(self, observation: AgentObservation, geography: Geography, weather: Weather, rng: random.Random) -> int:
        if not observation.neighbour_ids:
            return self.location_id

        state_id = observation.current_location.id
        self._last_state_id = state_id

        # epsilon-greedy policy on learned Q-values, conditioned by current autonomous goal.
        if rng.random() < self.exploration_bias:
            selected = rng.choice(observation.neighbour_ids)
            self._last_action_id = selected
            return selected

        action_values = self.knowledge.q_values.get(state_id, {})
        if not action_values:
            selected = rng.choice(observation.neighbour_ids)
            self._last_action_id = selected
            return selected

        scored_actions = []
        for action_id in observation.neighbour_ids:
            q = action_values.get(action_id, 0.0)
            q += self._goal_bonus(action_id, observation, geography)
            scored_actions.append((q, action_id))

        max_q = max(value for value, _ in scored_actions)
        best_actions = [
            action_id
            for value, action_id in scored_actions
            if value == max_q
        ]
        selected = rng.choice(best_actions)
        self._last_action_id = selected
        return selected

    def _terrain_energy_modifier(self, connection_type: str) -> float:
        """Terrain affects energy cost. Roads cheapest, waterways most expensive."""
        modifiers = {
            "road": 1.0,
            "rail": 1.2,
            "path": 1.5,
            "waterway": 2.5,
            "proximity": 2.0,
        }
        return modifiers.get(connection_type, 1.5)

    def _movement_energy_cost(self, distance_km: float, connection_type: str) -> float:
        """Energy cost for moving a given distance over a given terrain.

        Proportional to distance and terrain difficulty. An agent that
        moves further or over harder terrain pays more energy.
        """
        terrain_mod = self._terrain_energy_modifier(connection_type)
        return distance_km * terrain_mod

    def apply_action(self, new_location_id: int, geography: Geography, tick_interval_seconds: float = 1.0) -> None:
        if new_location_id == self.location_id:
            # Staying at current location — still exploring it (deeper observation)
            self.last_action = "explore"
            self.last_move_distance_km = 0.0
            self.last_move_connection_type = "road"
            return

        # Move to new location — pay energy proportional to distance and terrain
        distance_km = 0.0
        connection_type = "proximity"
        for location, connection in geography.get_nearby_locations(self.location_id):
            if location.id == new_location_id:
                distance_km = connection.distance_km
                connection_type = connection.connection_type
                break

        self.location_id = new_location_id
        self.last_move_distance_km = distance_km
        self.last_move_connection_type = connection_type
        self.last_action = "explore"

    def learn(self, next_observation: AgentObservation, tick_count: int | None = None) -> None:
        self._ingest_observation(next_observation, tick_count=tick_count)
        reward = self._reward_signal()
        self.last_reward = round(reward, 4)

        self.knowledge.visited_locations.add(self.location_id)
        self.knowledge.visit_counts[self.location_id] = self.knowledge.visit_counts.get(self.location_id, 0) + 1

        if self._last_state_id is not None and self._last_action_id is not None:
            state_actions = self.knowledge.q_values.setdefault(self._last_state_id, {})
            old_q = state_actions.get(self._last_action_id, 0.0)

            next_actions = self.knowledge.q_values.get(next_observation.current_location.id, {})
            if next_observation.neighbour_ids:
                next_max_q = max(next_actions.get(action_id, 0.0) for action_id in next_observation.neighbour_ids)
            else:
                next_max_q = 0.0

            td_target = reward + (self.discount_factor * next_max_q)
            new_q = old_q + (self.learning_rate * (td_target - old_q))
            state_actions[self._last_action_id] = round(new_q, 6)

        current_value = self.knowledge.location_scores.get(self.location_id, 0.0)
        updated = current_value + self.learning_rate * (reward - current_value)
        self.knowledge.location_scores[self.location_id] = round(updated, 4)

        if next_observation.current_weather and next_observation.current_weather.conditions:
            key = next_observation.current_weather.conditions.lower()
            self.knowledge.condition_counts[key] = self.knowledge.condition_counts.get(key, 0) + 1

        # Energy: light movement cost, strong recovery. Agents should spend
        # the vast majority of their time exploring, not recovering.
        # A 20km road move costs 2 energy. Recovery is 5/tick. Agents
        # stay above the recover threshold under sustained exploration.
        if self.last_move_distance_km > 0:
            cost = self._movement_energy_cost(self.last_move_distance_km, self.last_move_connection_type) * 0.1
            cost = min(cost, 3.0)  # hard cap: no single move costs more than 3
            self.energy = max(0.0, self.energy - cost)
        self.energy = min(100.0, self.energy + 5.0)

    def to_summary(self, geography: Geography) -> dict:
        loc = geography.get_location(self.location_id)
        return {
            "id": self.agent_id,
            "name": self.name,
            "location_id": self.location_id,
            "location_name": loc.name if loc else None,
            "lat": loc.lat if loc else None,
            "lng": loc.lng if loc else None,
            "last_action": self.last_action,
            "energy": round(self.energy, 1),
            "knowledge_score": self.knowledge.knowledge_score,
            "visited_locations": self.knowledge.discovered_location_count,
            "known_agents": len(self.knowledge.social_memory),
            "policy": "q_learning",
            "last_reward": self.last_reward,
            "goal": self.current_goal.to_dict() if self.current_goal else None,
        }

    def to_detail(self, geography: Geography) -> dict:
        summary = self.to_summary(geography)
        top_learned = sorted(
            self.knowledge.location_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        visited_places = self.get_visited_places(geography)
        summary["top_locations"] = [
            {
                "location_id": location_id,
                "location_name": geography.get_location(location_id).name if geography.get_location(location_id) else None,
                "score": round(score, 3),
                "visits": self.knowledge.visit_counts.get(location_id, 0),
            }
            for location_id, score in top_learned
        ]
        summary["known_conditions"] = dict(sorted(self.knowledge.condition_counts.items(), key=lambda item: item[1], reverse=True))
        summary["visited_places"] = visited_places

        # Social connections (A53/A54/A60)
        social_connections = []
        for agent_id, mem in sorted(
            self.knowledge.social_memory.items(),
            key=lambda item: item[1].get("familiarity", 0),
            reverse=True,
        ):
            social_connections.append({
                "agent_id": agent_id,
                "name": mem.get("name"),
                "encounter_count": mem.get("encounter_count", 0),
                "familiarity": mem.get("familiarity", 0.0),
                "trust": mem.get("trust", 0.5),
                "facts_received": mem.get("facts_received", 0),
                "facts_shared": mem.get("facts_shared", 0),
                "first_met_location": mem.get("first_met_location"),
                "last_met_location": mem.get("last_met_location"),
                "last_met_tick": mem.get("last_met_tick"),
            })
        summary["social_connections"] = social_connections
        summary["recent_encounters"] = list(self.knowledge.encounters[-20:])
        return summary

    def get_visited_places(self, geography: Geography, limit: int = 500) -> list[dict]:
        ordered = sorted(
            self.knowledge.visited_locations,
            key=lambda location_id: self.knowledge.visit_counts.get(location_id, 0),
            reverse=True,
        )[:limit]

        places: list[dict] = []
        for location_id in ordered:
            location = geography.get_location(location_id)
            places.append(
                {
                    "location_id": location_id,
                    "location_name": location.name if location else None,
                    "visits": self.knowledge.visit_counts.get(location_id, 0),
                }
            )
        return places

    def answer(self, question: str, geography: Geography) -> dict:
        q = question.strip()
        q_lower = q.lower()
        visited_places = self.get_visited_places(geography)

        if "recent" in q_lower or "lately" in q_lower:
            text = self._answer_recently(geography)
            return {
                "agent_id": self.agent_id,
                "question": question,
                "answer": text,
                "visited_places": visited_places,
                "retrieval_backend": "episodic_timeline",
                "answer_confidence": 0.9,
                "answer_certainty": "known",
                "supporting_facts": self._timeline_supporting_facts(limit=8),
            }

        if "before" in q_lower:
            text, support = self._answer_before(question, geography)
            return {
                "agent_id": self.agent_id,
                "question": question,
                "answer": text,
                "visited_places": visited_places,
                "retrieval_backend": "episodic_timeline",
                "answer_confidence": 0.85 if support else 0.25,
                "answer_certainty": "known" if support else "uncertain",
                "supporting_facts": support,
            }

        memory_records = self._memory_facts(geography)

        # Try Chroma first (fast vector search, embeddings pre-computed).
        # Fall back to in-process retriever if Chroma is not available.
        ranked_records = []
        if self._chroma_store is None:
            self._chroma_store = _get_chroma_store(self.agent_id)
        if self._chroma_store and self._chroma_store.count > 0:
            chroma_results = self._chroma_store.query(q, top_k=10)
            for score, text, meta in chroma_results:
                matching = [r for r in memory_records if r.get("text") == text]
                record = matching[0] if matching else {"text": text, "predicate": meta.get("predicate", ""), "location_id": meta.get("location_id")}
                ranked_records.append({"score": score, "record": record})
        else:
            facts = [record["text"] for record in memory_records]
            retriever = _get_retriever()
            ranked = retriever.rank(q, facts, top_k=10)
            ranked_records = [
                {
                    "score": float(score),
                    "record": memory_records[idx],
                }
                for score, idx, _ in ranked
            ]

        # Re-rank: boost earth facts and diversify so basic "I visited X"
        # facts don't drown out richer civilisation knowledge.
        ranked_records = self._rerank_with_diversity(q, ranked_records)

        if ranked_records:
            text = self._compose_answer(q, ranked_records, geography)
        else:
            names = [place["location_name"] for place in visited_places[:8] if place["location_name"]]
            if names:
                text = (
                    f"I know {len(visited_places)} visited places. "
                    f"Some are {', '.join(names)}. "
                    "Ask me about places, visits, weather conditions, goals, or rewards."
                )
            else:
                text = "I do not have enough memory yet to answer that."

        backend = "sentence-transformers"
        if self._chroma_store and self._chroma_store.count > 0:
            backend = "chroma"

        return {
            "agent_id": self.agent_id,
            "question": question,
            "answer": text,
            "visited_places": visited_places,
            "retrieval_backend": backend,
            "answer_confidence": self._answer_confidence(ranked_records),
            "answer_certainty": self._certainty_label(self._answer_confidence(ranked_records)),
            "supporting_facts": self._supporting_facts(ranked_records, geography),
        }

    def _memory_facts(self, geography: Geography) -> list[dict]:
        memory_facts: list[dict] = []
        for fact in self.knowledge.facts[-500:]:
            text = str(fact.get("text", "")).strip()
            if not text:
                continue
            memory_facts.append(
                {
                    "text": text,
                    "predicate": fact.get("predicate"),
                    "location_id": fact.get("location_id"),
                    "source": "memory",
                }
            )

        for belief_key, belief in self.knowledge.beliefs.items():
            location_id = belief.get("location_id")
            location_name = None
            if location_id is not None:
                location = geography.get_location(int(location_id))
                location_name = location.name if location else f"location {location_id}"
            predicate = belief.get("predicate", "fact")
            value = belief.get("value")
            confidence = round(float(belief.get("confidence", 0.0)), 2)
            text = (
                f"My current belief for {predicate} at {location_name} is {value} "
                f"with confidence {confidence}."
            )
            memory_facts.append(
                {
                    "text": text,
                    "predicate": "belief",
                    "location_id": location_id,
                    "source": "belief_state",
                }
            )

        for conflict in self.knowledge.belief_conflicts[-20:]:
            location_id = conflict.get("location_id")
            location_name = None
            if location_id is not None:
                location = geography.get_location(int(location_id))
                location_name = location.name if location else f"location {location_id}"
            text = (
                f"I revised {conflict.get('predicate')} for {location_name} from "
                f"{conflict.get('old_value')} to {conflict.get('new_value')} due to conflicting evidence."
            )
            memory_facts.append(
                {
                    "text": text,
                    "predicate": "belief_revision",
                    "location_id": location_id,
                    "source": "belief_conflict",
                }
            )

        for event in self._episodes_sorted()[-120:]:
            text = str(event.get("text", "")).strip()
            if not text:
                continue
            memory_facts.append(
                {
                    "text": text,
                    "predicate": "episode",
                    "location_id": event.get("location_id"),
                    "source": "episodic_memory",
                }
            )

        summary_text = (
            f"My energy is {round(self.energy, 1)}. "
            f"My last action was {self.last_action}. "
            f"My last reward was {self.last_reward}. "
            f"I have visited {self.knowledge.discovered_location_count} places."
        )
        memory_facts.append(
            {
                "text": summary_text,
                "predicate": "agent_state",
                "location_id": self.location_id,
                "source": "derived",
            }
        )
        return memory_facts

    def _compose_answer(self, question: str, ranked_records: list[dict], geography: Geography) -> str:
        by_location: dict[int, list[dict]] = {}
        for item in ranked_records:
            location_id = item["record"].get("location_id")
            if location_id is None:
                continue
            by_location.setdefault(int(location_id), []).append(item)

        question_lower = question.lower()
        preferred_location: int | None = None
        for location_id in by_location:
            location = geography.get_location(location_id)
            if location and location.name.lower() in question_lower:
                preferred_location = location_id
                break

        if preferred_location is None and by_location:
            preferred_location = max(by_location, key=lambda lid: max(i["score"] for i in by_location[lid]))

        if preferred_location is not None:
            location = geography.get_location(preferred_location)
            location_name = location.name if location else str(preferred_location)
            selected = sorted(by_location[preferred_location], key=lambda item: item["score"], reverse=True)[:8]
            fragments = [item["record"]["text"] for item in selected]
            return f"About {location_name}: " + " ".join(fragments)

        fragments = [item["record"]["text"] for item in ranked_records[:8]]
        return " ".join(fragments)

    # --- Retrieval re-ranking ------------------------------------------------

    # Predicates that are trivially generic ("I visited X", "X is a Y")
    _TRIVIAL_PREDICATES: ClassVar[set[str]] = {"visit", "visits", "location_type", "coordinates", "connections", "agent_state"}

    # Question keywords → earth_* predicate domains they should boost
    _DOMAIN_KEYWORDS: ClassVar[dict[str, list[str]]] = {
        "history": ["earth_history", "earth_heritage", "earth_archive", "earth_government"],
        "historic": ["earth_history", "earth_heritage", "earth_archive"],
        "culture": ["earth_culture", "earth_literature", "earth_archaeology"],
        "cultural": ["earth_culture", "earth_literature", "earth_archaeology"],
        "news": ["earth_news", "earth_social", "earth_discourse"],
        "politic": ["earth_politics", "earth_institution", "earth_law"],
        "parliament": ["earth_politics", "earth_institution"],
        "crime": ["earth_crime", "earth_safety"],
        "population": ["earth_demographics", "earth_geography"],
        "economy": ["earth_economy", "earth_demographics"],
        "science": ["earth_science", "earth_knowledge", "earth_technology"],
        "literature": ["earth_literature", "earth_culture"],
        "heritage": ["earth_heritage", "earth_history", "earth_architecture"],
        "geography": ["earth_geography", "earth_search"],
        "transport": ["earth_transport", "earth_geography"],
        "weather": [],  # basic facts are fine for weather
        "social": ["earth_social", "earth_discourse"],
    }

    def _rerank_with_diversity(self, question: str, ranked_records: list[dict]) -> list[dict]:
        """Re-rank retrieval results to surface earth facts alongside basic facts.

        Strategy:
        1. Boost all earth_* facts by a base multiplier (richer content deserves
           more weight than "I visited X").
        2. If the question mentions a domain keyword (history, culture, etc.),
           give matching earth predicates an extra boost.
        3. Diversify: cap trivial basic facts at 3 so earth facts get slots.
        """
        if not ranked_records:
            return ranked_records

        q_lower = question.lower()

        # Detect domain keywords in the question
        boosted_predicates: set[str] = set()
        for keyword, domains in self._DOMAIN_KEYWORDS.items():
            if keyword in q_lower:
                boosted_predicates.update(domains)

        # Apply score adjustments
        for item in ranked_records:
            predicate = str(item["record"].get("predicate", ""))
            base_score = float(item["score"])

            if predicate.startswith("earth_"):
                # Base boost for all earth facts (richer content)
                item["score"] = base_score * 1.15
                # Extra boost if the question targets this domain
                if predicate in boosted_predicates:
                    item["score"] = base_score * 1.35

        # Re-sort by adjusted score
        ranked_records.sort(key=lambda r: r["score"], reverse=True)

        # Diversify: interleave so trivial basic facts don't monopolise the top
        trivial: list[dict] = []
        substantive: list[dict] = []
        for item in ranked_records:
            predicate = str(item["record"].get("predicate", ""))
            if predicate in self._TRIVIAL_PREDICATES:
                trivial.append(item)
            else:
                substantive.append(item)

        # Take up to 3 trivial facts (for context), fill the rest with substantive
        max_trivial = 3
        result: list[dict] = []
        t_idx, s_idx = 0, 0
        trivial_count = 0
        # Merge by score, respecting the trivial cap
        all_items = sorted(trivial + substantive, key=lambda r: r["score"], reverse=True)
        for item in all_items:
            predicate = str(item["record"].get("predicate", ""))
            if predicate in self._TRIVIAL_PREDICATES:
                if trivial_count >= max_trivial:
                    continue
                trivial_count += 1
            result.append(item)

        return result

    def _answer_confidence(self, ranked_records: list[dict]) -> float:
        if not ranked_records:
            return 0.0

        normalized_scores = [max(0.0, min(1.0, float(item["score"]))) for item in ranked_records]
        top = normalized_scores[0]
        avg = sum(normalized_scores) / len(normalized_scores)
        coverage = min(len(normalized_scores) / 5.0, 1.0)
        confidence = 0.5 * top + 0.3 * avg + 0.2 * coverage
        return round(max(0.0, min(1.0, confidence)), 3)

    def _certainty_label(self, confidence: float) -> str:
        if confidence >= 0.66:
            return "known"
        if confidence >= 0.40:
            return "likely"
        return "uncertain"

    def _supporting_facts(self, ranked_records: list[dict], geography: Geography) -> list[dict]:
        result: list[dict] = []
        for item in ranked_records[:5]:
            record = item["record"]
            location_id = record.get("location_id")
            location_name = None
            if location_id is not None:
                location = geography.get_location(int(location_id))
                location_name = location.name if location else None
            result.append(
                {
                    "text": record.get("text", ""),
                    "score": round(float(item["score"]), 4),
                    "predicate": record.get("predicate"),
                    "location_id": location_id,
                    "location_name": location_name,
                    "source": record.get("source", "memory"),
                }
            )
        return result

    def _ingest_observation(self, observation: AgentObservation, tick_count: int | None = None) -> None:
        loc = observation.current_location
        visit_count = self.knowledge.visit_counts.get(loc.id, 0) + 1

        self._append_episode(
            event_type="observation",
            text=f"At {loc.name}, I observed conditions and state while action was {self.last_action}.",
            location_id=loc.id,
            location_name=loc.name,
            simulation_time=observation.simulation_time,
            tick_count=tick_count,
            action=self.last_action,
        )

        self._append_fact("visit", f"I visited {loc.name}.", loc.id)
        self._append_fact("location_type", f"{loc.name} is a {loc.type}.", loc.id, value=loc.type)

        if loc.population is not None:
            self._append_fact("population", f"Population of {loc.name} is {loc.population}.", loc.id, value=loc.population)
        if loc.elevation is not None:
            self._append_fact("elevation", f"Elevation of {loc.name} is {round(loc.elevation, 1)} metres.", loc.id, value=round(loc.elevation, 1))
        if loc.terrain:
            self._append_fact("terrain", f"Terrain of {loc.name} is {loc.terrain}.", loc.id, value=loc.terrain)
        if loc.admin_level_2:
            self._append_fact("nation", f"{loc.name} is in {loc.admin_level_2}.", loc.id, value=loc.admin_level_2)
        if loc.admin_level_3:
            self._append_fact("region", f"{loc.name} is in region {loc.admin_level_3}.", loc.id, value=loc.admin_level_3)

        self._append_fact("coordinates", f"Coordinates of {loc.name} are lat {loc.lat} lng {loc.lng}.", loc.id, value=f"{loc.lat},{loc.lng}")
        self._append_fact("visits", f"I have visited {loc.name} {visit_count} times.", loc.id)

        if observation.current_weather:
            ws = observation.current_weather
            if ws.temperature_c is not None:
                self._append_fact("temperature", f"Temperature in {loc.name} is {round(ws.temperature_c, 1)} C.", loc.id)
            if ws.conditions:
                self._append_fact("weather_condition", f"Weather in {loc.name} is {ws.conditions}.", loc.id)
            if ws.wind_speed_kmh is not None:
                self._append_fact("wind", f"Wind speed in {loc.name} is {round(ws.wind_speed_kmh, 1)} kmh.", loc.id)

        self._append_fact("connections", f"{loc.name} has {len(observation.neighbour_ids)} connected nearby places.", loc.id)

        # Earth facts are ingested via _ingest_earth_facts (called here
        # and also pre-move in the tick loop to catch resolved facts).
        self._ingest_earth_facts(observation)

    def _ingest_earth_facts(self, observation: AgentObservation) -> None:
        """Ingest civilisation/earth facts from an observation.

        Called both during _ingest_observation (post-move learn) and
        pre-move in the sequential tick loop. The latter is critical because
        the world resolves facts for the agent's CURRENT location at tick
        start, but the agent moves before learn() is called — so the new
        location won't be resolved yet. Pre-move ingestion ensures the
        agent actually learns from resolved earth content.

        _earth_observed_locations tracks locations where facts have
        ACTUALLY been ingested. If the agent arrives before the proxy has
        resolved facts, the location stays eligible for re-ingestion on
        subsequent ticks.
        """
        loc = observation.current_location
        if not observation.earth_facts or loc.id in self._earth_observed_locations:
            return

        ingested = 0
        for ef in observation.earth_facts:
            text = getattr(ef, "text", None)
            if isinstance(ef, dict):
                text = ef.get("text", "")
            if not text or not text.strip():
                continue
            domain = getattr(ef, "domain", "culture") if not isinstance(ef, dict) else ef.get("domain", "culture")
            topic = getattr(ef, "topic", "earth") if not isinstance(ef, dict) else ef.get("topic", "earth")
            self._append_fact(f"earth_{domain}", text.strip(), loc.id, value=topic)
            ingested += 1

        if ingested > 0:
            self._earth_observed_locations.add(loc.id)
            logger.info(f"Agent {self.agent_id} ingested {ingested} earth facts at {loc.name}")

    def _append_episode(
        self,
        event_type: str,
        text: str,
        location_id: int | None,
        location_name: str | None,
        simulation_time: datetime,
        tick_count: int | None,
        action: str | None = None,
    ) -> None:
        entry = {
            "event_type": event_type,
            "text": text.strip(),
            "location_id": location_id,
            "location_name": location_name,
            "time": simulation_time.isoformat() if simulation_time else None,
            "tick": tick_count,
            "action": action,
        }
        self.knowledge.episodic_events.append(entry)
        if len(self.knowledge.episodic_events) > 5000:
            self.knowledge.episodic_events = self.knowledge.episodic_events[-5000:]

    def _episodes_sorted(self) -> list[dict]:
        def key(event: dict) -> tuple[int, str]:
            tick = event.get("tick")
            tick_value = int(tick) if isinstance(tick, int) else -1
            time_value = str(event.get("time", ""))
            return tick_value, time_value

        events = list(self.knowledge.episodic_events)
        events.sort(key=key)
        return events

    def _answer_recently(self, geography: Geography) -> str:
        recent = self._episodes_sorted()[-8:]
        if not recent:
            return "I do not have recent episodic memories yet."

        names = [event.get("location_name") for event in recent if event.get("location_name")]
        if not names:
            return "I have recent memories, but no location names in those events yet."

        compressed: list[str] = []
        for name in names:
            if not compressed or compressed[-1] != name:
                compressed.append(name)
        return "Recently I have been at " + ", ".join(compressed) + "."

    def _answer_before(self, question: str, geography: Geography) -> tuple[str, list[dict]]:
        question_lower = question.lower()
        target_id = None
        for location_id in self.knowledge.visited_locations:
            location = geography.get_location(location_id)
            if location and location.name.lower() in question_lower:
                target_id = location_id
                break

        timeline = self._episodes_sorted()
        if target_id is None:
            return "I could not identify the reference place in that question.", []

        target_indices = [i for i, event in enumerate(timeline) if event.get("location_id") == target_id]
        if not target_indices:
            target_loc = geography.get_location(target_id)
            target_name = target_loc.name if target_loc else str(target_id)
            return f"I do not have episodic memory for being at {target_name} yet.", []

        last_target_idx = target_indices[-1]
        prior = timeline[max(0, last_target_idx - 6):last_target_idx]
        if not prior:
            target_loc = geography.get_location(target_id)
            target_name = target_loc.name if target_loc else str(target_id)
            return f"I have no recorded events before my latest memory of {target_name}.", []

        names = [event.get("location_name") for event in prior if event.get("location_name")]
        target_loc = geography.get_location(target_id)
        target_name = target_loc.name if target_loc else str(target_id)
        summary = f"Before {target_name}, I remember being at " + ", ".join(names) + "."

        support = [
            {
                "text": event.get("text", ""),
                "score": 1.0,
                "predicate": "episode",
                "location_id": event.get("location_id"),
                "location_name": event.get("location_name"),
                "source": "episodic_memory",
            }
            for event in prior[-5:]
        ]
        return summary, support

    def _timeline_supporting_facts(self, limit: int = 8) -> list[dict]:
        recent = self._episodes_sorted()[-limit:]
        return [
            {
                "text": event.get("text", ""),
                "score": 1.0,
                "predicate": "episode",
                "location_id": event.get("location_id"),
                "location_name": event.get("location_name"),
                "source": "episodic_memory",
            }
            for event in recent
        ]

    def _append_fact(self, predicate: str, text: str, location_id: int | None = None, value: str | int | float | None = None) -> None:
        normalized = text.strip()
        if not normalized:
            return

        entry = {
            "predicate": predicate,
            "location_id": location_id,
            "text": normalized,
            "value": value,
        }

        # Prevent immediate duplicate spam and cap memory.
        if self.knowledge.facts and self.knowledge.facts[-1].get("text") == normalized:
            return

        self.knowledge.facts.append(entry)
        if len(self.knowledge.facts) > 5000:
            self.knowledge.facts = self.knowledge.facts[-5000:]

        # Also store in Chroma for fast semantic retrieval (if available).
        # Embedding is computed once here, queries are fast similarity search.
        if self._chroma_store is None:
            self._chroma_store = _get_chroma_store(self.agent_id)
        if self._chroma_store:
            self._fact_counter += 1
            self._chroma_store.add_fact(
                fact_id=f"{self.agent_id}_{self._fact_counter}",
                text=normalized,
                metadata={"predicate": predicate, "location_id": location_id or 0},
            )

        if predicate in self._BELIEF_PREDICATES and location_id is not None and value is not None:
            self._revise_belief(predicate=predicate, location_id=location_id, value=value)

    def _revise_belief(self, predicate: str, location_id: int, value: str | int | float) -> None:
        belief_key = f"{predicate}:{location_id}"
        existing = self.knowledge.beliefs.get(belief_key)

        if existing is None:
            self.knowledge.beliefs[belief_key] = {
                "predicate": predicate,
                "location_id": location_id,
                "value": value,
                "confidence": 0.7,
                "evidence_count": 1,
            }
            return

        old_value = existing.get("value")
        if old_value == value:
            existing["evidence_count"] = int(existing.get("evidence_count", 1)) + 1
            existing["confidence"] = min(1.0, float(existing.get("confidence", 0.7)) + 0.05)
            return

        conflict = {
            "predicate": predicate,
            "location_id": location_id,
            "old_value": old_value,
            "new_value": value,
        }
        self.knowledge.belief_conflicts.append(conflict)
        if len(self.knowledge.belief_conflicts) > 1000:
            self.knowledge.belief_conflicts = self.knowledge.belief_conflicts[-1000:]

        existing["value"] = value
        existing["evidence_count"] = int(existing.get("evidence_count", 1)) + 1
        existing["confidence"] = max(0.35, float(existing.get("confidence", 0.7)) * 0.75)

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9']+", text.lower())
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "is", "are", "was", "were",
            "i", "me", "my", "you", "your", "it", "that", "this", "what", "where", "when", "how", "have", "has",
            "had", "do", "did", "so", "can", "could", "would", "should", "be", "been",
        }
        return {word for word in words if word and word not in stop}

    def to_persisted(self) -> dict:
        return {
            "id": self.agent_id,
            "name": self.name,
            "location_id": self.location_id,
            "energy": self.energy,
            "last_move_distance_km": self.last_move_distance_km,
            "last_move_connection_type": self.last_move_connection_type,
            "last_action": self.last_action,
            "learning_rate": self.learning_rate,
            "exploration_bias": self.exploration_bias,
            "risk_tolerance": self.traits.risk_tolerance,
            "stamina": self.traits.stamina,
            "comfort_temperature_c": self.traits.comfort_temperature_c,
            "discount_factor": self.discount_factor,
            "goal": self.current_goal.to_dict() if self.current_goal else None,
            "goal_age_ticks": self.goal_age_ticks,
            "knowledge": self.knowledge.to_dict(),
        }

    @classmethod
    def from_persisted(cls, payload: dict) -> "AutonomousAgent":
        return cls(
            agent_id=str(payload["id"]),
            name=str(payload["name"]),
            location_id=int(payload["location_id"]),
            learning_rate=float(payload["learning_rate"]),
            exploration_bias=float(payload["exploration_bias"]),
            traits=AgentTraits(
                risk_tolerance=float(payload["risk_tolerance"]),
                stamina=float(payload["stamina"]),
                comfort_temperature_c=float(payload["comfort_temperature_c"]),
            ),
            discount_factor=float(payload.get("discount_factor", 0.92)),
            energy=float(payload.get("energy", 100.0)),
            last_move_distance_km=float(payload.get("last_move_distance_km", 0.0)),
            last_move_connection_type=str(payload.get("last_move_connection_type", "road")),
            last_action=str(payload.get("last_action", "spawned")),
            current_goal=AgentGoal.from_dict(payload.get("goal")),
            goal_age_ticks=int(payload.get("goal_age_ticks", 0)),
            knowledge=AgentKnowledge.from_dict(payload.get("knowledge")),
        )

    def _find_specialist_for_location(self, location_id: int) -> str | None:
        """Find agent in social memory who is a specialist about this location (A59)."""
        # Look for agents who have high interaction count and know about geography
        best_specialist = None
        best_score = 0.0

        for peer_id, network_data in self.knowledge.interaction_network.items():
            mem = self.knowledge.social_memory.get(peer_id)
            if not mem:
                continue

            # Score based on:
            # - Topic expertise in geography (from A59)
            # - Trust level
            # - Teaching effectiveness
            expertise = network_data.get("topic_expertise", {}).get("geography", 0.0)
            if expertise < 0.3:  # Not enough expertise
                continue

            trust = mem.get("trust", 0.5)
            teaching_eff = mem.get("teaching_effectiveness", 0.5)

            score = expertise * 0.5 + trust * 0.3 + teaching_eff * 0.2

            if score > best_score:
                best_score = score
                best_specialist = peer_id

        return best_specialist if best_score > 0.4 else None

    def refresh_goal(self, observation: AgentObservation, geography: Geography, rng: random.Random) -> None:
        self.goal_age_ticks += 1

        if self.energy < 15:
            self.current_goal = AgentGoal(kind="recover", target_location_id=self.location_id, priority=1.0)
            self.goal_age_ticks = 0
            return

        if self.current_goal and self.current_goal.kind == "recover" and self.energy >= 30:
            self.current_goal = None

        if self.current_goal and self.current_goal.target_location_id == self.location_id:
            self.current_goal = None

        if self.current_goal and self.goal_age_ticks <= 120:
            return

        gap_target = self._select_knowledge_gap_target(observation, geography, rng)
        if gap_target is not None and gap_target != self.location_id:
            self.current_goal = AgentGoal(kind="investigate_gap", target_location_id=gap_target, priority=0.9)
            self.goal_age_ticks = 0
            return

        # --- Seek specialist agent for knowledge gap (A58/A59 hybrid seeking) ---
        # If agent has knowledge gap and knows a specialist, seek them out
        if gap_target is not None:
            # Find specialist for this location's topic
            specialist = self._find_specialist_for_location(gap_target)
            if specialist:
                last_location = self.knowledge.social_memory.get(specialist, {}).get("last_met_location")
                if last_location and last_location != self.location_id:
                    self.current_goal = AgentGoal(
                        kind="seek_agent",
                        target_location_id=last_location,
                        target_agent_id=specialist,
                        priority=0.85
                    )
                    self.goal_age_ticks = 0
                    return

        unseen_neighbours = [nid for nid in observation.neighbour_ids if nid not in self.knowledge.visited_locations]
        if unseen_neighbours:
            self.current_goal = AgentGoal(kind="explore", target_location_id=rng.choice(unseen_neighbours), priority=0.8)
            self.goal_age_ticks = 0
            return

        # Fall back to broad exploration — pick a random neighbour's neighbour
        all_known = list(self.knowledge.visited_locations)
        if all_known:
            self.current_goal = AgentGoal(kind="explore", target_location_id=rng.choice(all_known), priority=0.6)
            self.goal_age_ticks = 0

    def _goal_bonus(self, action_id: int, observation: AgentObservation, geography: Geography) -> float:
        if not self.current_goal:
            return 0.0

        if self.current_goal.kind == "explore":
            bonus = 0.0
            if action_id not in self.knowledge.visited_locations:
                bonus += 0.35

            if self.current_goal.target_location_id is not None:
                if action_id == self.current_goal.target_location_id:
                    bonus += 0.7
                next_hop = self._next_hop_toward(observation.current_location.id, self.current_goal.target_location_id, geography)
                if next_hop == action_id:
                    bonus += 0.25
            return bonus

        if self.current_goal.kind == "investigate_gap":
            bonus = 0.0
            if self.current_goal.target_location_id is not None:
                if action_id == self.current_goal.target_location_id:
                    bonus += 0.9
                next_hop = self._next_hop_toward(observation.current_location.id, self.current_goal.target_location_id, geography)
                if next_hop == action_id:
                    bonus += 0.35

            bonus += min(0.4, self._knowledge_gap_score(action_id, geography) * 0.5)
            return bonus

        if self.current_goal.kind == "recover" and action_id == self.location_id:
            return 0.25

        if self.current_goal.kind == "seek_agent":
            # Navigate toward last known location of target agent
            bonus = 0.0
            if self.current_goal.target_location_id is not None:
                if action_id == self.current_goal.target_location_id:
                    bonus += 0.85  # High bonus for reaching target location
                next_hop = self._next_hop_toward(observation.current_location.id, self.current_goal.target_location_id, geography)
                if next_hop == action_id:
                    bonus += 0.35  # Bonus for moving toward target
            return bonus

        return 0.0

    def _next_hop_toward(self, source_id: int, target_id: int, geography: Geography) -> int | None:
        if source_id == target_id:
            return source_id

        visited: set[int] = {source_id}
        queue = deque([(source_id, None)])
        parents: dict[int, int | None] = {source_id: None}

        while queue:
            node, _ = queue.popleft()
            for connection in geography.get_neighbours(node):
                neighbour_id = connection.to_id if connection.from_id == node else connection.from_id
                if neighbour_id in visited:
                    continue
                visited.add(neighbour_id)
                parents[neighbour_id] = node
                if neighbour_id == target_id:
                    cursor = target_id
                    while parents.get(cursor) != source_id and parents.get(cursor) is not None:
                        cursor = parents[cursor]
                    return cursor
                queue.append((neighbour_id, node))

        return None

    def _select_knowledge_gap_target(self, observation: AgentObservation, geography: Geography, rng: random.Random) -> int | None:
        # Score locations the agent knows about (visited + current neighbours)
        candidates = list(self.knowledge.visited_locations | set(observation.neighbour_ids))
        if not candidates:
            return None

        scored: list[tuple[float, int]] = []
        for location_id in candidates:
            score = self._knowledge_gap_score(location_id, geography)
            if score <= 0.0:
                continue
            if location_id == self.location_id:
                score *= 0.5
            scored.append((score, location_id))

        if not scored:
            return None

        max_score = max(score for score, _ in scored)
        best = [location_id for score, location_id in scored if score >= (max_score - 1e-9)]
        nearby_best = [location_id for location_id in best if location_id in observation.neighbour_ids]
        if nearby_best:
            return rng.choice(nearby_best)
        return rng.choice(best)

    def _knowledge_gap_score(self, location_id: int, geography: Geography) -> float:
        score = 0.0
        if location_id not in self.knowledge.visited_locations:
            score += 1.0

        location = geography.get_location(location_id)
        if location is None:
            return round(score, 4)

        expected = self._expected_predicates(location)
        if expected:
            known = self._known_predicates_for_location(location_id)
            missing_count = sum(1 for predicate in expected if predicate not in known)
            score += missing_count / float(len(expected))

        visits = self.knowledge.visit_counts.get(location_id, 0)
        if visits > 0:
            score *= max(0.25, 1.0 - min(visits, 10) * 0.08)

        return round(max(0.0, score), 4)

    def _expected_predicates(self, location: LocationData) -> set[str]:
        expected = {"location_type", "coordinates", "connections"}
        if location.population is not None:
            expected.add("population")
        if location.elevation is not None:
            expected.add("elevation")
        if location.terrain:
            expected.add("terrain")
        if location.admin_level_2:
            expected.add("nation")
        if location.admin_level_3:
            expected.add("region")
        return expected

    def _known_predicates_for_location(self, location_id: int) -> set[str]:
        known: set[str] = set()

        for fact in self.knowledge.facts:
            if not isinstance(fact, dict):
                continue
            if fact.get("location_id") != location_id:
                continue
            predicate = fact.get("predicate")
            if isinstance(predicate, str) and predicate:
                known.add(predicate)

        for belief in self.knowledge.beliefs.values():
            if not isinstance(belief, dict):
                continue
            if belief.get("location_id") != location_id:
                continue
            predicate = belief.get("predicate")
            if isinstance(predicate, str) and predicate:
                known.add(predicate)

        return known

    def _reward_signal(self) -> float:
        """
        Transition-outcome reward for learned policy updates.
        This intentionally avoids hand-crafted weather routing heuristics.
        """
        first_visit = self.location_id not in self.knowledge.visited_locations
        novelty_reward = 1.0 if first_visit else -0.05

        movement_cost = self.last_move_distance_km * 0.01
        energy_signal = (self.energy - 50.0) / 50.0

        reward = novelty_reward - movement_cost + (0.25 * energy_signal)
        return max(-2.0, min(2.0, reward))

@dataclass
class AgentSystem:
    """Manages all autonomous agents in the world.

    Supports two modes:
    - **Sequential** (default): agents tick one-by-one in the coordinator.
    - **Ray parallel**: each agent runs as a Ray actor; tick() fans out
      in parallel and social learning is coordinated via remote calls.

    The mode is selected automatically: if Ray is installed and
    ``init_ray()`` succeeds, agents run as actors.  Otherwise everything
    falls back to the sequential path — same behaviour, same results.
    """

    agents: list[AutonomousAgent] = field(default_factory=list)
    _rng: random.Random = field(default_factory=lambda: random.Random(42))
    social_learning_rate: float = 0.2

    # --- Co-location duration tracking (A60) ---
    # Maps frozenset({agent_id_a, agent_id_b}) → consecutive ticks co-located.
    _colocation_ticks: dict[frozenset, int] = field(default_factory=dict, init=False, repr=False)

    # --- Ray fields (populated by init_ray) ---
    _use_ray: bool = field(default=False, init=False, repr=False)
    _actors: list = field(default_factory=list, init=False, repr=False)
    _agent_id_to_actor: dict = field(default_factory=dict, init=False, repr=False)
    _agent_locations: dict = field(default_factory=dict, init=False, repr=False)

    # --- Ray lifecycle -----------------------------------------------------

    # Minimum agent count to justify Ray overhead.
    # Each Ray actor is a separate Python process that loads PyTorch / sentence-transformers
    # — ~150–250 MB resident per actor. At 1000 actors that's ~200 GB RAM, so the threshold
    # is also a guard against accidentally exhausting host memory.
    # Sequential mode with `asyncio.to_thread` per-agent dispatch (see `_tick_sequential`)
    # keeps the main event loop free; up to ~1000 agents on a single host is well-behaved.
    # Below this threshold, sequential mode is faster and far lighter on memory.
    # Effectively keeps Ray off on a single VPS. Each Ray actor is its own
    # Python process (~150–250 MB resident with PyTorch + sentence-transformers),
    # so 1000 actors would need ~200 GB RAM — far past any single host.
    # Sequential + asyncio.to_thread (see `_tick_sequential`) scales fine into
    # the thousands on one host because all agents share one process and the
    # blocking work is I/O-bound (Chroma HTTP). Ray is only viable when wired
    # to a remote worker pool — re-evaluate this threshold then.
    RAY_MIN_AGENTS = 10000

    def init_ray(self, seed: int = 42) -> bool:
        """Create a Ray actor for each agent. Returns True if Ray mode activated.

        Skipped when:
        - Ray is not installed
        - EARTHLINK_DISABLE_RAY=1 is set (e.g. in Docker where memory is tight)
        - Agent count is below RAY_MIN_AGENTS (overhead > benefit)
        """
        import os

        if os.environ.get("EARTHLINK_DISABLE_RAY", "").strip() in ("1", "true", "yes"):
            logger.info("Ray disabled via EARTHLINK_DISABLE_RAY — sequential mode")
            return False

        from .actor import RAY_AVAILABLE, AgentActor

        if not RAY_AVAILABLE or AgentActor is None:
            logger.info("Ray not available — staying in sequential mode")
            return False

        if len(self.agents) < self.RAY_MIN_AGENTS:
            logger.info(
                f"Only {len(self.agents)} agents (< {self.RAY_MIN_AGENTS}) — "
                f"sequential mode is more efficient than Ray"
            )
            return False

        try:
            import ray as _ray
            import multiprocessing

            if not _ray.is_initialized():
                # Limit Ray to half the available CPUs (min 2) so it doesn't
                # saturate the machine.  Override with RAY_NUM_CPUS env var.
                default_cpus = max(2, multiprocessing.cpu_count() // 2)
                num_cpus = int(os.environ.get("RAY_NUM_CPUS", str(default_cpus)))
                _ray.init(
                    num_cpus=num_cpus,
                    ignore_reinit_error=True,
                    logging_level=logging.WARNING,
                )
                logger.info(
                    f"Ray initialised: {num_cpus} CPUs allocated "
                    f"(of {multiprocessing.cpu_count()} available), "
                    f"{_ray.cluster_resources().get('GPU', 0)} GPUs"
                )

            self._actors = []
            self._agent_id_to_actor = {}
            for i, agent in enumerate(self.agents):
                actor_seed = seed + i
                handle = AgentActor.remote(agent.to_persisted(), actor_seed)
                self._actors.append(handle)
                self._agent_id_to_actor[agent.agent_id] = handle

            self._agent_locations = {a.agent_id: a.location_id for a in self.agents}
            self._use_ray = True
            logger.info(f"Ray mode active: {len(self._actors)} agent actors created")
            return True

        except Exception as exc:
            logger.warning(f"Failed to initialise Ray actors: {exc}")
            self._use_ray = False
            self._actors = []
            self._agent_id_to_actor = {}
            return False

    def shutdown_ray(self) -> None:
        """Kill all Ray actors and optionally shut down Ray."""
        if not self._use_ray:
            return

        try:
            import ray as _ray

            for actor in self._actors:
                _ray.kill(actor)
            self._actors = []
            self._agent_id_to_actor = {}
            self._use_ray = False
            logger.info("Ray actors shut down")
        except Exception as exc:
            logger.warning(f"Error shutting down Ray actors: {exc}")

    @property
    def using_ray(self) -> bool:
        return self._use_ray

    # --- Factory ----------------------------------------------------------

    @classmethod
    async def bootstrap(
        cls,
        geography: Geography,
        count: int,
        learning_rate: float,
        exploration_bias: float,
        seed: int,
    ) -> "AgentSystem":
        rng = random.Random(seed)
        spawn_pool = await geography.get_spawn_locations(
            types=["capital", "city", "town"], limit=max(count, 200)
        )
        if not spawn_pool:
            return cls(agents=[], _rng=rng)

        agents: list[AutonomousAgent] = []
        for i in range(count):
            base = spawn_pool[i % len(spawn_pool)]
            agent_id = f"A{i + 1}"
            personal_learning_rate = max(0.05, learning_rate + rng.uniform(-0.08, 0.08))
            personal_exploration = min(0.95, max(0.05, exploration_bias + rng.uniform(-0.25, 0.25)))
            agent = AutonomousAgent(
                agent_id=agent_id,
                name=f"Agent {agent_id}",
                location_id=base.id,
                learning_rate=personal_learning_rate,
                exploration_bias=personal_exploration,
                traits=AgentTraits(
                    risk_tolerance=min(1.0, max(0.0, rng.uniform(0.2, 0.9))),
                    stamina=min(1.0, max(0.0, rng.uniform(0.2, 0.9))),
                    comfort_temperature_c=rng.uniform(2.0, 24.0),
                ),
                discount_factor=min(0.99, max(0.75, rng.uniform(0.85, 0.98))),
                energy=rng.uniform(75.0, 100.0),
            )
            agent.knowledge.visited_locations.add(base.id)
            agent.knowledge.visit_counts[base.id] = 1
            agents.append(agent)

        return cls(agents=agents, _rng=rng)

    async def tick(self, geography: Geography, weather: Weather, sim_time: datetime, tick_count: int | None = None, earth_proxy: EarthProxy | None = None, wind: Wind | None = None, tick_interval: float = 1.0) -> list[dict]:
        """Tick all agents. Dispatches to Ray (parallel) or sequential path."""
        if self._use_ray:
            return await self._tick_ray(geography, weather, sim_time, tick_count, earth_proxy, wind, tick_interval)
        return await self._tick_sequential(geography, weather, sim_time, tick_count, earth_proxy, wind, tick_interval)

    # --- Sequential tick (original logic) ---------------------------------

    async def _tick_sequential(self, geography: Geography, weather: Weather, sim_time: datetime, tick_count: int | None = None, earth_proxy: EarthProxy | None = None, wind: Wind | None = None, tick_interval: float = 1.0) -> list[dict]:
        """Sequential agent tick.

        The async outer shell awaits earth_proxy facts (true async HTTP). The
        per-agent sync body — which calls into ChromaDB's sync HTTP client and
        does CPU-bound RL/learn work — is dispatched to a worker thread via
        asyncio.to_thread() so it never blocks the main event loop. Without
        this, with hundreds of agents the main loop is pinned and uvicorn
        cannot service HTTP requests.
        """
        import asyncio

        events: list[dict] = []
        for agent in self.agents:
            earth_facts = []
            if earth_proxy and await earth_proxy.is_location_resolved(agent.location_id):
                earth_facts = await earth_proxy.get_resolved_facts(agent.location_id)

            # Run all sync agent work in a worker thread. This includes
            # ChromaDB HttpClient calls (sync HTTP) and CPU-bound Q-learning.
            previous_location = agent.location_id
            await asyncio.to_thread(
                self._tick_one_agent_sync_premove,
                agent, geography, weather, sim_time, earth_facts, wind, tick_interval,
            )

            next_earth_facts = []
            if earth_proxy and await earth_proxy.is_location_resolved(agent.location_id):
                next_earth_facts = await earth_proxy.get_resolved_facts(agent.location_id)

            await asyncio.to_thread(
                self._tick_one_agent_sync_postmove,
                agent, geography, weather, sim_time, next_earth_facts, wind, tick_count,
            )

            q_value = 0.0
            if agent._last_state_id is not None and agent._last_action_id is not None:
                q_value = agent.knowledge.q_values.get(agent._last_state_id, {}).get(agent._last_action_id, 0.0)

            loc = geography.get_location(agent.location_id)
            events.append(
                {
                    "agent_id": agent.agent_id,
                    # Perceive: what the agent observed
                    "location_id": agent.location_id,
                    "from_location_id": previous_location,
                    "to_location_id": agent.location_id,
                    "lat": loc.lat if loc else None,
                    "lng": loc.lng if loc else None,
                    "location_name": loc.name if loc else None,
                    # Learn: what knowledge changed
                    "knowledge_score": agent.knowledge.knowledge_score,
                    "facts_learned": len(next_earth_facts) if next_earth_facts else 0,
                    "visited_count": agent.knowledge.discovered_location_count,
                    # Decide: what goal drove the decision
                    "goal": agent.current_goal.to_dict() if agent.current_goal else None,
                    "q_value": round(q_value, 6),
                    "reward": agent.last_reward,
                    # Move: where and how far
                    "moved": previous_location != agent.location_id,
                    "distance_km": round(agent.last_move_distance_km, 2),
                    # Communicate: handled in social_learn, logged separately
                    # Energy
                    "energy": round(agent.energy, 1),
                }
            )

        # Social learning is also sync (data shuffling + sync chroma writes)
        # — run it off the main loop too.
        await asyncio.to_thread(
            self._social_learn,
            geography=geography, tick_count=tick_count, sim_time=sim_time,
        )
        return events

    # --- Per-agent sync helpers (called via asyncio.to_thread) -------------

    def _tick_one_agent_sync_premove(self, agent, geography, weather, sim_time, earth_facts, wind, tick_interval) -> None:
        """Sync portion of one agent's tick BEFORE moving.

        perceive -> ingest earth facts -> refresh goal -> choose next -> apply move.
        Runs in a worker thread; safe to do sync HTTP (Chroma) here.
        """
        observation = agent.perceive(geography, weather, sim_time, earth_facts=earth_facts, wind=wind)
        # Ingest earth facts for current location BEFORE moving.
        if earth_facts:
            agent._ingest_earth_facts(observation)
        agent.refresh_goal(observation, geography, self._rng)
        next_location = agent.choose_next_location(observation, geography, weather, self._rng)
        agent.apply_action(next_location, geography, tick_interval_seconds=tick_interval)

    def _tick_one_agent_sync_postmove(self, agent, geography, weather, sim_time, next_earth_facts, wind, tick_count) -> None:
        """Sync portion of one agent's tick AFTER moving.

        perceive new location -> learn. Sync (Chroma writes happen inside learn).
        """
        next_observation = agent.perceive(geography, weather, sim_time, earth_facts=next_earth_facts, wind=wind)
        agent.learn(next_observation, tick_count=tick_count)

    # --- Ray parallel tick ------------------------------------------------

    async def _tick_ray(self, geography: Geography, weather: Weather, sim_time: datetime, tick_count: int | None = None, earth_proxy: EarthProxy | None = None, wind: Wind | None = None, tick_interval: float = 1.0) -> list[dict]:
        """Fan out tick to all Ray actors in parallel, then coordinate social learning."""
        import asyncio
        import ray as _ray

        # 1. Build earth-facts map for all known agent locations
        earth_facts_map: dict = {}
        if earth_proxy:
            all_locations = set(self._agent_locations.values())
            for loc_id in all_locations:
                if await earth_proxy.is_location_resolved(loc_id):
                    facts = await earth_proxy.get_resolved_facts(loc_id)
                    if facts:
                        earth_facts_map[loc_id] = facts

        # 2. Put shared state in object store (zero-copy on same node)
        geo_ref = _ray.put(geography)
        weather_ref = _ray.put(weather)
        wind_ref = _ray.put(wind)
        facts_ref = _ray.put(earth_facts_map)

        # 3. Fan out tick to all actors — true parallelism
        futures = [
            actor.tick.remote(geo_ref, weather_ref, sim_time, tick_count, facts_ref, wind_ref, tick_interval)
            for actor in self._actors
        ]

        # 4. Await results without blocking the event loop
        events = await asyncio.get_event_loop().run_in_executor(
            None, _ray.get, futures,
        )

        # 5. Update location tracking from results
        for event in events:
            self._agent_locations[event["agent_id"]] = event["to_location_id"]

        # 6. Social learning — coordinated via remote calls
        await self._social_learn_ray(geography=geography, tick_count=tick_count, sim_time=sim_time)

        return events

    async def _social_learn_ray(self, geography: Geography, tick_count: int | None = None, sim_time: datetime | None = None) -> None:
        """Social learning across Ray actors.

        Groups actors by location, pulls social snapshots from co-located
        actors, computes exchanges (same algorithm as sequential mode),
        and pushes updates back.
        """
        import asyncio
        import ray as _ray

        # Group actors by location
        by_location: dict[int, list[tuple[str, object]]] = {}
        for agent_id, loc_id in self._agent_locations.items():
            actor = self._agent_id_to_actor.get(agent_id)
            if actor is not None:
                by_location.setdefault(loc_id, []).append((agent_id, actor))

        for location_id, group in by_location.items():
            if len(group) < 2:
                continue

            # Pull social snapshots from co-located actors
            snapshot_futures = [actor.get_social_snapshot.remote() for _, actor in group]
            snapshots = await asyncio.get_event_loop().run_in_executor(
                None, _ray.get, snapshot_futures,
            )

            # Compute social updates for each agent
            updates = self._compute_social_updates(
                snapshots, location_id, tick_count, sim_time,
            )

            # Push updates back to actors
            update_futures = [
                actor.apply_social_update.remote(updates[i])
                for i, (_, actor) in enumerate(group)
            ]
            await asyncio.get_event_loop().run_in_executor(
                None, _ray.get, update_futures,
            )

    def _compute_social_updates(
        self,
        snapshots: list[dict],
        location_id: int,
        tick_count: int | None,
        sim_time: datetime | None,
    ) -> list[dict]:
        """Compute social learning updates from agent snapshots.

        Mirrors the logic of ``_social_dialogue``, ``_social_exchange_facts``,
        and Q-value blending from the sequential path, but produces
        serialisable update dicts instead of mutating agents directly.
        """
        n = len(snapshots)
        updates: list[dict] = [{
            "new_facts": [],
            "belief_updates": {},
            "belief_conflicts": [],
            "q_updates": {},
            "dialogue_events": [],
            "social_memory": {},
            "encounters": [],
        } for _ in range(n)]

        time_iso = sim_time.isoformat() if sim_time else None

        # --- Record encounters & update social memory (A53/A54/A60) --------
        for i, snap_a in enumerate(snapshots):
            for j, snap_b in enumerate(snapshots):
                if j <= i:
                    continue
                pair = frozenset({snap_a["agent_id"], snap_b["agent_id"]})
                duration = self._colocation_ticks.get(pair, 1)

                # Build/update social memory records
                for idx, me, other in [(i, snap_a, snap_b), (j, snap_b, snap_a)]:
                    existing = me.get("social_memory", {}).get(other["agent_id"])
                    if existing is None:
                        record = {
                            "name": other.get("name", ""),
                            "first_met_tick": tick_count,
                            "first_met_location": location_id,
                            "last_met_tick": tick_count,
                            "last_met_location": location_id,
                            "encounter_count": 1,
                            "total_colocation_ticks": duration,
                            "familiarity": 0.0,
                            "trust": 0.5,
                            "facts_received": 0,
                            "facts_shared": 0,
                        }
                    else:
                        record = dict(existing)
                        if tick_count is not None and tick_count != record.get("last_met_tick"):
                            record["encounter_count"] = record.get("encounter_count", 1) + 1
                        record["last_met_tick"] = tick_count
                        record["last_met_location"] = location_id
                        record["total_colocation_ticks"] = record.get("total_colocation_ticks", 0) + 1
                    encounters_count = record.get("encounter_count", 1)
                    record["familiarity"] = round(1.0 - (1.0 / (1.0 + 0.15 * encounters_count)), 3)
                    updates[idx]["social_memory"][other["agent_id"]] = record

                    # Log encounter event on first tick of co-location
                    if duration == 1:
                        updates[idx]["encounters"].append({
                            "agent_id": other["agent_id"],
                            "agent_name": other.get("name", ""),
                            "location_id": location_id,
                            "tick": tick_count,
                            "time": time_iso,
                        })

        # --- Dialogue: each speaker shares their best recent fact ----------
        for si, speaker in enumerate(snapshots):
            fact = self._pick_dialogue_fact(speaker)
            if fact is None:
                continue
            text = str(fact.get("text", "")).strip()
            if not text:
                continue

            predicate = fact.get("predicate")
            value = fact.get("value")
            confidence = self._snapshot_fact_confidence(speaker, fact)

            for li, listener in enumerate(snapshots):
                if li == si:
                    continue

                message = {
                    "from_agent": speaker["agent_id"],
                    "to_agent": listener["agent_id"],
                    "location_id": location_id,
                    "predicate": predicate,
                    "text": text,
                    "confidence": round(confidence, 3),
                    "tick": tick_count,
                    "time": sim_time.isoformat() if sim_time else None,
                }

                # Speaker outgoing
                updates[si]["dialogue_events"].append({**message, "direction": "out"})
                # Listener incoming
                updates[li]["dialogue_events"].append({**message, "direction": "in"})

                # Listener gains the fact
                updates[li]["new_facts"].append({
                    "predicate": predicate,
                    "location_id": fact.get("location_id"),
                    "text": text,
                    "value": value,
                    "source": "dialogue",
                    "source_agent": speaker["agent_id"],
                    "source_confidence": round(confidence, 3),
                })

                # Update social memory fact counts + trust (A53/A54)
                sp_mem = updates[si]["social_memory"].get(listener["agent_id"])
                if sp_mem:
                    sp_mem["facts_shared"] = sp_mem.get("facts_shared", 0) + 1
                li_mem = updates[li]["social_memory"].get(speaker["agent_id"])
                if li_mem:
                    li_mem["facts_received"] = li_mem.get("facts_received", 0) + 1
                    old_trust = li_mem.get("trust", 0.5)
                    li_mem["trust"] = round(min(1.0, old_trust + 0.01 * confidence), 3)

                # Belief merge for listener
                if (
                    isinstance(predicate, str)
                    and predicate in AutonomousAgent._BELIEF_PREDICATES
                    and isinstance(fact.get("location_id"), int)
                    and value is not None
                ):
                    self._merge_belief_update(
                        updates[li], listener,
                        predicate, int(fact["location_id"]), value, confidence,
                    )

        # --- Fact exchange -------------------------------------------------
        for si, sender in enumerate(snapshots):
            shared_facts = sender["recent_facts"][-40:]
            for li, learner in enumerate(snapshots):
                if li == si:
                    continue

                learner_texts = learner.get("recent_facts_texts_500", set())
                for fact in shared_facts:
                    if not isinstance(fact, dict):
                        continue
                    text = str(fact.get("text", "")).strip()
                    if not text or text in learner_texts:
                        continue

                    sender_conf = self._snapshot_fact_confidence(sender, fact)
                    transfer_strength = self.social_learning_rate * sender_conf
                    if transfer_strength < 0.08:
                        continue

                    copied = {
                        "predicate": fact.get("predicate"),
                        "location_id": fact.get("location_id"),
                        "text": text,
                        "value": fact.get("value"),
                        "source": "social",
                        "source_agent": sender["agent_id"],
                        "source_confidence": round(sender_conf, 3),
                    }
                    updates[li]["new_facts"].append(copied)
                    learner_texts.add(text)

                    # Update social memory fact counts + trust (A53/A54)
                    s_mem = updates[si]["social_memory"].get(learner["agent_id"])
                    if s_mem:
                        s_mem["facts_shared"] = s_mem.get("facts_shared", 0) + 1
                    l_mem = updates[li]["social_memory"].get(sender["agent_id"])
                    if l_mem:
                        l_mem["facts_received"] = l_mem.get("facts_received", 0) + 1
                        old_t = l_mem.get("trust", 0.5)
                        l_mem["trust"] = round(min(1.0, old_t + 0.005 * sender_conf), 3)

                    predicate = copied.get("predicate")
                    loc_id = copied.get("location_id")
                    value = copied.get("value")
                    if (
                        isinstance(predicate, str)
                        and predicate in AutonomousAgent._BELIEF_PREDICATES
                        and isinstance(loc_id, int)
                        and value is not None
                    ):
                        self._merge_belief_update(
                            updates[li], learner,
                            predicate, loc_id, value, sender_conf,
                        )

        # --- Q-value blending ----------------------------------------------
        peer_best_q = 0.0
        peer_best_action: int | None = None
        for snap in snapshots:
            action_map = snap["q_values"].get(location_id, {})
            if not action_map:
                continue
            action_id, q_val = max(action_map.items(), key=lambda item: item[1])
            if peer_best_action is None or q_val > peer_best_q:
                peer_best_action = action_id
                peer_best_q = q_val

        if peer_best_action is not None:
            for idx, snap in enumerate(snapshots):
                learner_map = snap["q_values"].get(location_id, {})
                old_q = learner_map.get(peer_best_action, 0.0)
                new_q = round(
                    old_q + self.social_learning_rate * (peer_best_q - old_q),
                    6,
                )
                updates[idx].setdefault("q_updates", {})[str(location_id)] = {
                    str(peer_best_action): new_q,
                }

        return updates

    # --- Social learning helpers (snapshot-based) -------------------------

    @staticmethod
    def _pick_dialogue_fact(snapshot: dict) -> dict | None:
        """Pick the best fact for an agent to share in dialogue (from snapshot)."""
        candidates = [
            f for f in snapshot["recent_facts"]
            if isinstance(f, dict) and str(f.get("text", "")).strip()
        ]
        if not candidates:
            return None

        scored: list[tuple[float, dict]] = []
        for fact in candidates:
            confidence = AgentSystem._snapshot_fact_confidence(snapshot, fact)
            scored.append((confidence, fact))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_conf = scored[0][0]
        top_facts = [fact for conf, fact in scored if conf >= top_conf - 1e-9]
        rng = random.Random(hash(snapshot["agent_id"]))
        return rng.choice(top_facts)

    @staticmethod
    def _snapshot_fact_confidence(snapshot: dict, fact: dict) -> float:
        """Compute a speaker's confidence about a fact (from snapshot)."""
        predicate = fact.get("predicate")
        location_id = fact.get("location_id")
        if isinstance(predicate, str) and isinstance(location_id, int):
            belief_key = f"{predicate}:{location_id}"
            belief = snapshot["beliefs"].get(belief_key)
            if isinstance(belief, dict):
                return max(0.0, min(1.0, float(belief.get("confidence", 0.6))))
        return 0.55

    @staticmethod
    def _merge_belief_update(
        update: dict,
        learner_snapshot: dict,
        predicate: str,
        location_id: int,
        value,
        sender_confidence: float,
    ) -> None:
        """Compute a belief merge and add it to the update dict."""
        belief_key = f"{predicate}:{location_id}"
        existing = learner_snapshot["beliefs"].get(belief_key)

        # Also check if we already wrote an update for this key
        pending = update["belief_updates"].get(belief_key)
        if pending is not None:
            existing = pending

        if existing is None:
            update["belief_updates"][belief_key] = {
                "predicate": predicate,
                "location_id": location_id,
                "value": value,
                "confidence": round(max(0.35, min(0.95, 0.35 + sender_confidence * 0.45)), 3),
                "evidence_count": 1,
            }
            return

        existing_value = existing.get("value")
        existing_confidence = float(existing.get("confidence", 0.6))
        if existing_value == value:
            update["belief_updates"][belief_key] = {
                **existing,
                "evidence_count": int(existing.get("evidence_count", 1)) + 1,
                "confidence": round(min(1.0, existing_confidence + 0.1 * sender_confidence), 3),
            }
            return

        if sender_confidence > existing_confidence + 0.12:
            update["belief_conflicts"].append({
                "predicate": predicate,
                "location_id": location_id,
                "old_value": existing_value,
                "new_value": value,
                "source": "social",
            })
            update["belief_updates"][belief_key] = {
                **existing,
                "value": value,
                "evidence_count": int(existing.get("evidence_count", 1)) + 1,
                "confidence": round(max(0.35, min(0.95, (existing_confidence + sender_confidence) / 2.0)), 3),
            }
        else:
            update["belief_updates"][belief_key] = {
                **existing,
                "confidence": round(max(0.2, existing_confidence - 0.03 * sender_confidence), 3),
            }

    def summaries(self, geography: Geography) -> list[dict]:
        if self._use_ray:
            import ray as _ray
            geo_ref = _ray.put(geography)
            futures = [a.get_summary.remote(geo_ref) for a in self._actors]
            return _ray.get(futures)
        return [agent.to_summary(geography) for agent in self.agents]

    def detail(self, agent_id: str, geography: Geography) -> dict | None:
        if self._use_ray:
            import ray as _ray
            actor = self._agent_id_to_actor.get(agent_id)
            if actor is None:
                # Case-insensitive lookup
                for aid, act in self._agent_id_to_actor.items():
                    if aid.lower() == agent_id.lower():
                        actor = act
                        break
            if actor is None:
                return None
            geo_ref = _ray.put(geography)
            return _ray.get(actor.get_detail.remote(geo_ref))
        for agent in self.agents:
            if agent.agent_id.lower() == agent_id.lower():
                return agent.to_detail(geography)
        return None

    def answer(self, agent_id: str, question: str, geography: Geography) -> dict | None:
        if self._use_ray:
            import ray as _ray
            actor = self._agent_id_to_actor.get(agent_id)
            if actor is None:
                for aid, act in self._agent_id_to_actor.items():
                    if aid.lower() == agent_id.lower():
                        actor = act
                        break
            if actor is None:
                return None
            geo_ref = _ray.put(geography)
            return _ray.get(actor.answer.remote(question, geo_ref))
        for agent in self.agents:
            if agent.agent_id.lower() == agent_id.lower():
                return agent.answer(question, geography)
        return None

    def to_persisted(self) -> list[dict]:
        if self._use_ray:
            import ray as _ray
            futures = [a.to_persisted.remote() for a in self._actors]
            return _ray.get(futures)
        return [agent.to_persisted() for agent in self.agents]

    def country_coverage(self, geography: Geography) -> set[str]:
        """Return represented admin_level_1 countries across current agents."""
        countries: set[str] = set()
        for agent in self.agents:
            loc = geography.get_location(agent.location_id)
            if loc and loc.admin_level_1:
                country = loc.admin_level_1.strip()
                if country:
                    countries.add(country)
        return countries

    def redeploy_to_spawn_pool(self, spawn_pool: list[LocationData]) -> int:
        """Reassign all agent locations across a balanced spawn pool.

        Preserves agent identity and learned state while re-homing placement
        to match current available geography coverage.
        """
        if not self.agents or not spawn_pool:
            return 0

        moved = 0
        pool_size = len(spawn_pool)
        for i, agent in enumerate(self.agents):
            target = spawn_pool[i % pool_size]
            if agent.location_id != target.id:
                moved += 1
            agent.location_id = target.id
            agent.last_action = "spawned"
            agent.last_move_distance_km = 0.0
            agent.last_move_connection_type = "road"
            agent.knowledge.visited_locations.add(target.id)
            agent.knowledge.visit_counts[target.id] = max(
                1, agent.knowledge.visit_counts.get(target.id, 0)
            )

        return moved

    @classmethod
    def from_persisted(cls, payloads: list[dict], seed: int) -> "AgentSystem":
        agents = [AutonomousAgent.from_persisted(payload) for payload in payloads]
        return cls(agents=agents, _rng=random.Random(seed))

    def _social_learn(self, geography: Geography, tick_count: int | None = None, sim_time: datetime | None = None) -> None:
        # If agents share a location, they exchange policy and factual knowledge.
        by_location: dict[int, list[AutonomousAgent]] = {}
        for agent in self.agents:
            by_location.setdefault(agent.location_id, []).append(agent)

        # --- Track co-location duration (A60) ---
        current_pairs: set[frozenset] = set()
        for location_id, group in by_location.items():
            if len(group) < 2:
                continue
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    pair = frozenset({a.agent_id, b.agent_id})
                    current_pairs.add(pair)
                    self._colocation_ticks[pair] = self._colocation_ticks.get(pair, 0) + 1

        # Expire pairs that are no longer co-located
        expired = [pair for pair in self._colocation_ticks if pair not in current_pairs]
        for pair in expired:
            del self._colocation_ticks[pair]

        for location_id, group in by_location.items():
            if len(group) < 2:
                continue

            # --- Record encounters & update social memory (A53/A54/A60) ---
            self._record_encounters(group, location_id, tick_count, sim_time)

            # --- Initiate proactive conversations (A55) ---
            self._initiate_conversations(group, location_id, geography, tick_count, sim_time)

            # --- Process multi-turn conversations (A56) ---
            self._process_conversations(group, location_id, geography, tick_count, sim_time)

            # --- Deliberate teaching (A57) ---
            self._teach_knowledge_gaps(group, location_id, geography, tick_count, sim_time)

            # --- Structured learning from peers (A58) ---
            self._request_knowledge(group, location_id, geography, tick_count, sim_time)

            # --- Track interactions for social network (A59) ---
            for agent in group:
                for peer in group:
                    if agent.agent_id != peer.agent_id:
                        self._update_interaction_tracking(agent, peer.agent_id, f"location:{location_id}", tick_count)

            self._social_dialogue(group, location_id=location_id, tick_count=tick_count, sim_time=sim_time)
            self._social_exchange_facts(group)

            peer_best_q = 0.0
            peer_best_action: int | None = None
            for peer in group:
                action_map = peer.knowledge.q_values.get(location_id, {})
                if not action_map:
                    continue
                action_id, q_val = max(action_map.items(), key=lambda item: item[1])
                if peer_best_action is None or q_val > peer_best_q:
                    peer_best_action = action_id
                    peer_best_q = q_val

            if peer_best_action is None:
                continue

            for learner in group:
                learner_map = learner.knowledge.q_values.setdefault(location_id, {})
                old_q = learner_map.get(peer_best_action, 0.0)
                learner_map[peer_best_action] = round(
                    old_q + self.social_learning_rate * (peer_best_q - old_q),
                    6,
                )

        # --- Periodic network analysis (A59) - run every 10 ticks ---
        if tick_count is not None and tick_count % 10 == 0:
            self._update_specialist_detection(self.agents, tick_count)

        # --- Centrality computation (A59) - run every 50 ticks ---
        if tick_count is not None and tick_count % 50 == 0:
            self._compute_network_centrality(self.agents, tick_count)

    # --- Encounter recording & social memory (A53/A54/A60) ----------------

    def _record_encounters(
        self,
        group: list[AutonomousAgent],
        location_id: int,
        tick_count: int | None,
        sim_time: datetime | None,
    ) -> None:
        """Record encounters and update social memory for all co-located pairs."""
        time_iso = sim_time.isoformat() if sim_time else None
        for i, agent_a in enumerate(group):
            for agent_b in group[i + 1:]:
                pair = frozenset({agent_a.agent_id, agent_b.agent_id})
                duration = self._colocation_ticks.get(pair, 1)

                # Update both agents' social memory
                self._update_social_memory(agent_a, agent_b, location_id, tick_count, duration)
                self._update_social_memory(agent_b, agent_a, location_id, tick_count, duration)

                # Log encounter event on both agents (only on first tick of co-location)
                if duration == 1:
                    encounter = {
                        "agent_id": agent_b.agent_id,
                        "agent_name": agent_b.name,
                        "location_id": location_id,
                        "tick": tick_count,
                        "time": time_iso,
                    }
                    agent_a.knowledge.encounters.append(encounter)
                    if len(agent_a.knowledge.encounters) > 2000:
                        agent_a.knowledge.encounters = agent_a.knowledge.encounters[-2000:]

                    encounter_b = {
                        "agent_id": agent_a.agent_id,
                        "agent_name": agent_a.name,
                        "location_id": location_id,
                        "tick": tick_count,
                        "time": time_iso,
                    }
                    agent_b.knowledge.encounters.append(encounter_b)
                    if len(agent_b.knowledge.encounters) > 2000:
                        agent_b.knowledge.encounters = agent_b.knowledge.encounters[-2000:]

    def _update_social_memory(
        self,
        agent: AutonomousAgent,
        other: AutonomousAgent,
        location_id: int,
        tick_count: int | None,
        duration_ticks: int,
    ) -> None:
        """Update agent's social memory record for another agent."""
        mem = agent.knowledge.social_memory.get(other.agent_id)
        if mem is None:
            # First encounter — create record
            mem = {
                "name": other.name,
                "first_met_tick": tick_count,
                "first_met_location": location_id,
                "last_met_tick": tick_count,
                "last_met_location": location_id,
                "encounter_count": 1,
                "total_colocation_ticks": duration_ticks,
                "familiarity": 0.0,
                "trust": 0.5,
                "facts_received": 0,
                "facts_shared": 0,
            }
            agent.knowledge.social_memory[other.agent_id] = mem
        else:
            # Subsequent encounter — update
            if tick_count is not None and tick_count != mem.get("last_met_tick"):
                mem["encounter_count"] = mem.get("encounter_count", 1) + 1
            mem["last_met_tick"] = tick_count
            mem["last_met_location"] = location_id
            mem["total_colocation_ticks"] = mem.get("total_colocation_ticks", 0) + 1

        # Familiarity: grows with encounters, diminishing returns (asymptotic to 1.0)
        encounters = mem.get("encounter_count", 1)
        mem["familiarity"] = round(1.0 - (1.0 / (1.0 + 0.15 * encounters)), 3)

    def _social_dialogue(
        self,
        group: list[AutonomousAgent],
        location_id: int,
        tick_count: int | None,
        sim_time: datetime | None,
    ) -> None:
        for speaker in group:
            fact = self._dialogue_fact(speaker)
            if fact is None:
                continue

            text = str(fact.get("text", "")).strip()
            if not text:
                continue

            predicate = fact.get("predicate")
            value = fact.get("value")
            confidence = self._sender_fact_confidence(speaker, fact)

            for listener in group:
                if listener.agent_id == speaker.agent_id:
                    continue

                message = {
                    "from_agent": speaker.agent_id,
                    "to_agent": listener.agent_id,
                    "location_id": location_id,
                    "predicate": predicate,
                    "text": text,
                    "confidence": round(confidence, 3),
                    "tick": tick_count,
                    "time": sim_time.isoformat() if sim_time else None,
                }

                speaker.knowledge.dialogue_events.append({**message, "direction": "out"})
                listener.knowledge.dialogue_events.append({**message, "direction": "in"})

                if len(speaker.knowledge.dialogue_events) > 5000:
                    speaker.knowledge.dialogue_events = speaker.knowledge.dialogue_events[-5000:]
                if len(listener.knowledge.dialogue_events) > 5000:
                    listener.knowledge.dialogue_events = listener.knowledge.dialogue_events[-5000:]

                listener.knowledge.facts.append(
                    {
                        "predicate": predicate,
                        "location_id": fact.get("location_id"),
                        "text": text,
                        "value": value,
                        "source": "dialogue",
                        "source_agent": speaker.agent_id,
                        "source_confidence": round(confidence, 3),
                    }
                )
                if len(listener.knowledge.facts) > 5000:
                    listener.knowledge.facts = listener.knowledge.facts[-5000:]

                # Update social memory fact counts (A53/A54)
                speaker_mem = speaker.knowledge.social_memory.get(listener.agent_id)
                if speaker_mem:
                    speaker_mem["facts_shared"] = speaker_mem.get("facts_shared", 0) + 1
                listener_mem = listener.knowledge.social_memory.get(speaker.agent_id)
                if listener_mem:
                    listener_mem["facts_received"] = listener_mem.get("facts_received", 0) + 1
                    # Trust nudge: receiving information increases trust slightly
                    old_trust = listener_mem.get("trust", 0.5)
                    listener_mem["trust"] = round(min(1.0, old_trust + 0.01 * confidence), 3)

                if (
                    isinstance(predicate, str)
                    and predicate in AutonomousAgent._BELIEF_PREDICATES
                    and isinstance(fact.get("location_id"), int)
                    and value is not None
                ):
                    self._merge_social_belief(
                        learner=listener,
                        predicate=predicate,
                        location_id=int(fact["location_id"]),
                        value=value,
                        sender_confidence=confidence,
                    )

    def _dialogue_fact(self, speaker: AutonomousAgent) -> dict | None:
        candidates = [fact for fact in speaker.knowledge.facts[-60:] if isinstance(fact, dict) and str(fact.get("text", "")).strip()]
        if not candidates:
            return None

        scored: list[tuple[float, dict]] = []
        for fact in candidates:
            confidence = self._sender_fact_confidence(speaker, fact)
            scored.append((confidence, fact))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_conf = scored[0][0]
        top_facts = [fact for conf, fact in scored if conf >= top_conf - 1e-9]
        return self._rng.choice(top_facts)

    def _social_exchange_facts(self, group: list[AutonomousAgent]) -> None:
        for sender in group:
            shared_facts = sender.knowledge.facts[-40:]
            for learner in group:
                if learner.agent_id == sender.agent_id:
                    continue

                learner_recent_texts = {
                    str(fact.get("text", "")).strip()
                    for fact in learner.knowledge.facts[-500:]
                    if isinstance(fact, dict)
                }

                for fact in shared_facts:
                    if not isinstance(fact, dict):
                        continue

                    text = str(fact.get("text", "")).strip()
                    if not text or text in learner_recent_texts:
                        continue

                    sender_conf = self._sender_fact_confidence(sender, fact)
                    transfer_strength = self.social_learning_rate * sender_conf
                    if transfer_strength < 0.08:
                        continue

                    copied = {
                        "predicate": fact.get("predicate"),
                        "location_id": fact.get("location_id"),
                        "text": text,
                        "value": fact.get("value"),
                        "source": "social",
                        "source_agent": sender.agent_id,
                        "source_confidence": round(sender_conf, 3),
                    }
                    learner.knowledge.facts.append(copied)
                    learner_recent_texts.add(text)

                    if len(learner.knowledge.facts) > 5000:
                        learner.knowledge.facts = learner.knowledge.facts[-5000:]

                    # Update social memory fact counts (A53/A54)
                    sender_mem = sender.knowledge.social_memory.get(learner.agent_id)
                    if sender_mem:
                        sender_mem["facts_shared"] = sender_mem.get("facts_shared", 0) + 1
                    learner_mem = learner.knowledge.social_memory.get(sender.agent_id)
                    if learner_mem:
                        learner_mem["facts_received"] = learner_mem.get("facts_received", 0) + 1
                        old_trust = learner_mem.get("trust", 0.5)
                        learner_mem["trust"] = round(min(1.0, old_trust + 0.005 * sender_conf), 3)

                    predicate = copied.get("predicate")
                    location_id = copied.get("location_id")
                    value = copied.get("value")
                    if (
                        isinstance(predicate, str)
                        and predicate in AutonomousAgent._BELIEF_PREDICATES
                        and isinstance(location_id, int)
                        and value is not None
                    ):
                        self._merge_social_belief(learner, predicate, location_id, value, sender_conf)

    def _sender_fact_confidence(self, sender: AutonomousAgent, fact: dict) -> float:
        predicate = fact.get("predicate")
        location_id = fact.get("location_id")
        if isinstance(predicate, str) and isinstance(location_id, int):
            belief_key = f"{predicate}:{location_id}"
            belief = sender.knowledge.beliefs.get(belief_key)
            if isinstance(belief, dict):
                return max(0.0, min(1.0, float(belief.get("confidence", 0.6))))
        return 0.55

    def _merge_social_belief(
        self,
        learner: AutonomousAgent,
        predicate: str,
        location_id: int,
        value: str | int | float,
        sender_confidence: float,
    ) -> None:
        belief_key = f"{predicate}:{location_id}"
        existing = learner.knowledge.beliefs.get(belief_key)

        if existing is None:
            learner.knowledge.beliefs[belief_key] = {
                "predicate": predicate,
                "location_id": location_id,
                "value": value,
                "confidence": round(max(0.35, min(0.95, 0.35 + sender_confidence * 0.45)), 3),
                "evidence_count": 1,
            }
            return

        existing_value = existing.get("value")
        existing_confidence = float(existing.get("confidence", 0.6))
        if existing_value == value:
            existing["evidence_count"] = int(existing.get("evidence_count", 1)) + 1
            existing["confidence"] = round(min(1.0, existing_confidence + 0.1 * sender_confidence), 3)
            return

        if sender_confidence > existing_confidence + 0.12:
            learner.knowledge.belief_conflicts.append(
                {
                    "predicate": predicate,
                    "location_id": location_id,
                    "old_value": existing_value,
                    "new_value": value,
                    "source": "social",
                }
            )
            if len(learner.knowledge.belief_conflicts) > 1000:
                learner.knowledge.belief_conflicts = learner.knowledge.belief_conflicts[-1000:]

            existing["value"] = value
            existing["evidence_count"] = int(existing.get("evidence_count", 1)) + 1
            existing["confidence"] = round(max(0.35, min(0.95, (existing_confidence + sender_confidence) / 2.0)), 3)
        else:
            existing["confidence"] = round(max(0.2, existing_confidence - 0.03 * sender_confidence), 3)

    # --- A55: Proactive Communication Helper Methods ---------------------

    def _compute_conversation_initiation_score(
        self,
        initiator: AutonomousAgent,
        peer: AutonomousAgent,
        topic: str,
        geography: Geography,
        tick_count: int | None = None,
    ) -> float:
        """Compute score for whether initiator should start conversation with peer about topic."""
        # Knowledge gap component (0-1)
        gap_score = 0.0
        if topic.startswith("location:"):
            try:
                loc_id = int(topic.split(":")[1])
                gap_score = initiator._knowledge_gap_score(loc_id, geography)
            except (ValueError, IndexError):
                gap_score = 0.0
        else:
            # For predicate topics, estimate gap by checking if initiator has few facts about it
            predicate = topic.split(":")[1] if ":" in topic else topic
            initiator_fact_count = sum(
                1 for f in initiator.knowledge.facts if f.get("predicate") == predicate
            )
            gap_score = max(0.0, min(1.0, (10 - initiator_fact_count) / 10.0))

        # Social component (0-1)
        mem = initiator.knowledge.social_memory.get(peer.agent_id, {})
        familiarity = mem.get("familiarity", 0.0)
        trust = mem.get("trust", 0.5)
        social_score = (familiarity + trust) / 2.0

        # Recency penalty (avoid spamming same agent)
        last_interaction = mem.get("last_met_tick")
        if last_interaction is not None and tick_count is not None:
            ticks_since = max(1, tick_count - last_interaction)
            recency_penalty = min(1.0, ticks_since / 10.0)
        else:
            recency_penalty = 1.0  # No penalty if no history

        return gap_score * social_score * recency_penalty

    def _initiate_conversations(
        self,
        group: list[AutonomousAgent],
        location_id: int,
        geography: Geography,
        tick_count: int | None,
        sim_time: datetime | None,
    ) -> None:
        """Initiate proactive conversations based on knowledge gaps (A55)."""
        for agent in group:
            # Limit: max 3 active conversations per agent
            if len(agent.knowledge.active_conversations) >= 3:
                continue

            # Identify knowledge gap (use current location as topic)
            topic = f"location:{location_id}"
            gap_score = agent._knowledge_gap_score(location_id, geography)
            if gap_score < 0.3:  # No significant gap
                continue

            # Find best peer to ask
            peers = [p for p in group if p.agent_id != agent.agent_id]
            if not peers:
                continue

            # Score all peers
            scored_peers = [
                (self._compute_conversation_initiation_score(agent, peer, topic, geography, tick_count), peer)
                for peer in peers
            ]
            scored_peers.sort(key=lambda x: x[0], reverse=True)

            if scored_peers[0][0] < 0.5:  # Score too low
                continue

            best_peer = scored_peers[0][1]

            # Create conversation
            conv_id = f"conv_{min(agent.agent_id, best_peer.agent_id)}_{max(agent.agent_id, best_peer.agent_id)}_{tick_count or 0}"
            if conv_id in agent.knowledge.active_conversations:
                continue  # Conversation already exists

            conversation = {
                "participants": [agent.agent_id, best_peer.agent_id],
                "topic": topic,
                "initiator": agent.agent_id,
                "started_tick": tick_count or 0,
                "turn_count": 0,
                "last_tick": tick_count or 0,
                "messages": [],
                "status": "active",
            }

            # Add to both agents
            agent.knowledge.active_conversations[conv_id] = conversation
            best_peer.knowledge.active_conversations[conv_id] = conversation

            # Update social memory
            mem = agent.knowledge.social_memory.get(best_peer.agent_id)
            if mem:
                mem["requests_sent"] = mem.get("requests_sent", 0) + 1
                mem["conversations_initiated"] = mem.get("conversations_initiated", 0) + 1

            peer_mem = best_peer.knowledge.social_memory.get(agent.agent_id)
            if peer_mem:
                peer_mem["requests_received"] = peer_mem.get("requests_received", 0) + 1

            # Limit: only 1 initiation per agent per tick
            break

    # --- A56: Multi-turn Dialogue -----------------------------------------

    def _process_conversations(
        self,
        group: list[AutonomousAgent],
        location_id: int,
        geography: Geography,
        tick_count: int | None,
        sim_time: datetime | None,
    ) -> None:
        """Process active conversations for multi-turn dialogue (A56)."""
        agent_ids = {a.agent_id for a in group}

        for agent in group:
            # Check each active conversation
            for conv_id in list(agent.knowledge.active_conversations.keys()):
                conv = agent.knowledge.active_conversations[conv_id]

                # Skip if conversation already concluded
                if conv.get("status") != "active":
                    continue

                # Check if conversation partners still co-located
                participants = set(conv["participants"])
                if not participants.issubset(agent_ids):
                    self._conclude_conversation(agent, conv_id, "agents_separated", tick_count)
                    continue

                # Max turn limit (5 turns)
                if conv["turn_count"] >= 5:
                    self._conclude_conversation(agent, conv_id, "max_turns_reached", tick_count)
                    continue

                # Determine if this agent should respond
                messages = conv.get("messages", [])
                if not messages:
                    continue  # No messages yet, wait for initiator

                last_message = messages[-1]
                last_speaker = last_message.get("from")

                # If last speaker was this agent, wait for other agent's turn
                if last_speaker == agent.agent_id:
                    continue

                # Generate response
                response_text = self._generate_conversation_response(agent, conv, geography)

                if response_text:
                    # Add response message
                    message = {
                        "from": agent.agent_id,
                        "text": response_text,
                        "tick": tick_count or 0,
                    }
                    conv["messages"].append(message)
                    conv["turn_count"] += 1
                    conv["last_tick"] = tick_count or 0

                    # Update conversation for both participants
                    for peer in group:
                        if peer.agent_id in participants:
                            if conv_id in peer.knowledge.active_conversations:
                                peer.knowledge.active_conversations[conv_id] = conv
                else:
                    # No response possible, topic exhausted
                    self._conclude_conversation(agent, conv_id, "topic_exhausted", tick_count)

    def _generate_conversation_response(
        self,
        agent: AutonomousAgent,
        conversation: dict,
        geography: Geography,
    ) -> str | None:
        """Generate contextual response in multi-turn conversation (A56)."""
        topic = conversation.get("topic", "")
        messages = conversation.get("messages", [])

        if not messages:
            return None

        last_message = messages[-1]
        last_text = last_message.get("text", "")

        # If last message was a question (contains ?), provide an answer
        if "?" in last_text:
            # Find relevant fact about topic
            if topic.startswith("location:"):
                try:
                    loc_id = int(topic.split(":")[1])
                    # Get facts about this location
                    relevant_facts = [
                        f for f in agent.knowledge.facts
                        if f.get("location_id") == loc_id
                    ]
                    if relevant_facts:
                        # Return highest confidence fact
                        best_fact = max(
                            relevant_facts,
                            key=lambda f: agent.knowledge.beliefs.get(
                                f"{f.get('predicate')}:{loc_id}", {}
                            ).get("confidence", 0.0),
                            default=relevant_facts[0]
                        )
                        return f"Based on my knowledge, {best_fact.get('text', '')}"
                except (ValueError, IndexError):
                    pass

            return None  # Don't know answer

        # If last message was a statement, check for belief conflict or add complementary info
        else:
            # Check if agent has conflicting belief
            for belief_key, belief in agent.knowledge.beliefs.items():
                if belief.get("predicate") in last_text.lower():
                    # Found potentially relevant belief
                    # Check recent conflicts
                    conflicts = agent.knowledge.belief_conflicts[-10:]
                    for conflict in conflicts:
                        if conflict.get("predicate") == belief.get("predicate"):
                            return f"Interesting, but I observed {conflict.get('new_value')}"

            # Add complementary information if available
            if topic.startswith("location:"):
                try:
                    loc_id = int(topic.split(":")[1])
                    # Find fact not yet mentioned in conversation
                    mentioned_texts = {msg.get("text", "") for msg in messages}
                    unused_facts = [
                        f for f in agent.knowledge.facts
                        if f.get("location_id") == loc_id and f.get("text") not in mentioned_texts
                    ]
                    if unused_facts:
                        fact = unused_facts[0]
                        return f"Also, {fact.get('text', '')}"
                except (ValueError, IndexError):
                    pass

        return None  # No response to generate

    def _conclude_conversation(
        self,
        agent: AutonomousAgent,
        conv_id: str,
        reason: str,
        tick_count: int | None,
    ) -> None:
        """Conclude an active conversation and archive it (A56)."""
        if conv_id not in agent.knowledge.active_conversations:
            return

        conv = agent.knowledge.active_conversations[conv_id]
        conv["status"] = "concluded"
        conv["conclusion_reason"] = reason
        conv["concluded_tick"] = tick_count or 0

        # Move to history
        agent.knowledge.conversation_history.append(conv)
        if len(agent.knowledge.conversation_history) > 1000:
            agent.knowledge.conversation_history = agent.knowledge.conversation_history[-1000:]

        # Remove from active
        del agent.knowledge.active_conversations[conv_id]

        # Update social memory
        for participant_id in conv.get("participants", []):
            if participant_id != agent.agent_id:
                mem = agent.knowledge.social_memory.get(participant_id)
                if mem:
                    turn_count = conv.get("turn_count", 0)
                    current_avg = mem.get("avg_conversation_length", 0.0)
                    conv_count = mem.get("conversations_initiated", 0) + mem.get("conversations_participated", 0)
                    if conv_count > 0:
                        mem["avg_conversation_length"] = round(
                            (current_avg * (conv_count - 1) + turn_count) / conv_count,
                            2
                        )

    # --- A57: Deliberate Teaching ------------------------------------------

    def _assess_peer_knowledge_gaps(
        self,
        teacher: AutonomousAgent,
        learner: AutonomousAgent,
        location_id: int,
    ) -> list[str]:
        """Identify what learner doesn't know about a location that teacher knows (A57)."""
        # Teacher's predicates for this location
        teacher_predicates = set()
        for fact in teacher.knowledge.facts:
            if fact.get("location_id") == location_id:
                pred = fact.get("predicate")
                if pred:
                    teacher_predicates.add(pred)

        # Learner's predicates for this location
        learner_predicates = set()
        for fact in learner.knowledge.facts:
            if fact.get("location_id") == location_id:
                pred = fact.get("predicate")
                if pred:
                    learner_predicates.add(pred)

        # Return missing predicates
        missing = teacher_predicates - learner_predicates
        return list(missing)

    def _teach_knowledge_gaps(
        self,
        group: list[AutonomousAgent],
        location_id: int,
        geography: Geography,
        tick_count: int | None,
        sim_time: datetime | None,
    ) -> None:
        """Deliberate teaching based on knowledge gap assessment (A57)."""
        for teacher in group:
            for learner in group:
                if teacher.agent_id == learner.agent_id:
                    continue

                # Assess gaps
                missing_predicates = self._assess_peer_knowledge_gaps(teacher, learner, location_id)
                if not missing_predicates:
                    continue  # Learner knows everything teacher knows

                # Select facts to teach (top 3 highest confidence)
                facts_to_teach = []
                for predicate in missing_predicates[:3]:  # Limit to 3 predicates
                    # Find teacher's facts with this predicate at this location
                    relevant_facts = [
                        f for f in teacher.knowledge.facts
                        if f.get("predicate") == predicate and f.get("location_id") == location_id
                    ]
                    if relevant_facts:
                        # Pick highest confidence fact
                        belief_key = f"{predicate}:{location_id}"
                        confidence = teacher.knowledge.beliefs.get(belief_key, {}).get("confidence", 0.5)
                        if confidence > 0.4:  # Only teach if confident
                            facts_to_teach.append((relevant_facts[0], confidence))

                if not facts_to_teach:
                    continue

                # Teach facts to learner
                for fact, confidence in facts_to_teach:
                    taught_fact = {
                        **fact,
                        "source": "taught",
                        "source_agent": teacher.agent_id,
                        "source_confidence": confidence,
                    }
                    learner.knowledge.facts.append(taught_fact)

                if len(learner.knowledge.facts) > 5000:
                    learner.knowledge.facts = learner.knowledge.facts[-5000:]

                # Record teaching event
                teaching_event = {
                    "taught_agent": learner.agent_id,
                    "topic": f"location:{location_id}",
                    "facts_taught": len(facts_to_teach),
                    "tick": tick_count or 0,
                    "effectiveness": None,  # Will be assessed later
                }
                teacher.knowledge.teaching_events.append(teaching_event)
                if len(teacher.knowledge.teaching_events) > 1000:
                    teacher.knowledge.teaching_events = teacher.knowledge.teaching_events[-1000:]

                # Update social memory
                teacher_mem = teacher.knowledge.social_memory.get(learner.agent_id)
                if teacher_mem:
                    teacher_mem["times_taught"] = teacher_mem.get("times_taught", 0) + 1

                learner_mem = learner.knowledge.social_memory.get(teacher.agent_id)
                if learner_mem:
                    learner_mem["times_learned_from"] = learner_mem.get("times_learned_from", 0) + 1
                    learner_mem["learning_value"] = learner_mem.get("learning_value", 0.0) + len(facts_to_teach)

                # Limit: 1 teaching event per pair per tick
                break

    # --- A58: Structured Learning from Peers ------------------------------

    def _request_knowledge(
        self,
        group: list[AutonomousAgent],
        location_id: int,
        geography: Geography,
        tick_count: int | None,
        sim_time: datetime | None,
    ) -> None:
        """Handle structured knowledge requests from peers (A58)."""
        for requester in group:
            # Check if requester has knowledge gap
            gap_score = requester._knowledge_gap_score(location_id, geography)
            if gap_score < 0.5:  # No significant gap
                continue

            # Find best provider based on social network
            best_provider = None
            best_score = 0.0

            for provider in group:
                if provider.agent_id == requester.agent_id:
                    continue

                # Score provider
                mem = requester.knowledge.social_memory.get(provider.agent_id)
                if not mem:
                    continue

                trust = mem.get("trust", 0.5)
                teaching_eff = mem.get("teaching_effectiveness", 0.5)
                times_learned = mem.get("times_learned_from", 0)

                # Higher score for trusted, effective teachers
                score = trust * 0.5 + teaching_eff * 0.3 + min(times_learned / 10.0, 0.2)

                if score > best_score:
                    best_score = score
                    best_provider = provider

            if not best_provider or best_score < 0.3:
                continue  # No good provider

            # Request knowledge (get provider's facts about this location)
            provided_facts = [
                f for f in best_provider.knowledge.facts
                if f.get("location_id") == location_id
            ][:5]  # Top 5 facts

            if not provided_facts:
                continue

            # Integrate with confidence boost (requested knowledge is more trusted)
            confidence_multiplier = 1.2 * best_score

            for fact in provided_facts:
                base_conf = fact.get("source_confidence", 0.5)
                adjusted_conf = min(1.0, base_conf * confidence_multiplier)

                requested_fact = {
                    **fact,
                    "source": "requested",
                    "source_agent": best_provider.agent_id,
                    "source_confidence": round(adjusted_conf, 3),
                }
                requester.knowledge.facts.append(requested_fact)

            if len(requester.knowledge.facts) > 5000:
                requester.knowledge.facts = requester.knowledge.facts[-5000:]

            # Record learning request
            learning_request = {
                "from_agent": requester.agent_id,
                "to_agent": best_provider.agent_id,
                "topic": f"location:{location_id}",
                "tick": tick_count or 0,
                "fulfilled": True,
                "facts_received": len(provided_facts),
            }
            requester.knowledge.learning_requests.append(learning_request)
            if len(requester.knowledge.learning_requests) > 1000:
                requester.knowledge.learning_requests = requester.knowledge.learning_requests[-1000:]

            # Limit: 1 request per agent per tick
            break

    # --- A59: Social Network Formation (complete) -------------------------

    def _update_specialist_detection(
        self,
        agents: list[AutonomousAgent],
        tick_count: int | None,
    ) -> None:
        """Detect knowledge specialists and update social memory (A59)."""
        # Build predicate expertise map for each agent
        for observer in agents:
            for peer in agents:
                if observer.agent_id == peer.agent_id:
                    continue

                mem = observer.knowledge.social_memory.get(peer.agent_id)
                if not mem:
                    continue

                # Count facts per predicate for this peer
                predicate_counts = {}
                for fact in peer.knowledge.facts:
                    pred = fact.get("predicate")
                    if pred:
                        predicate_counts[pred] = predicate_counts.get(pred, 0) + 1

                # Update topic expertise scores (0-1 based on fact count)
                topic_expertise = {}
                for predicate, count in predicate_counts.items():
                    topic_expertise[predicate] = min(1.0, count / 10.0)

                mem["topic_expertise"] = topic_expertise

                # Update specialty predicates (top 3)
                top_predicates = sorted(
                    predicate_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                mem["specialty_predicates"] = [p for p, _ in top_predicates]

    def _compute_network_centrality(
        self,
        agents: list[AutonomousAgent],
        tick_count: int | None,
    ) -> None:
        """Compute centrality scores and identify hubs (A59)."""
        for agent in agents:
            if not agent.knowledge.interaction_network:
                continue

            # Compute centrality (simple: interaction count / max)
            max_interactions = max(
                (entry.get("interaction_count", 0) for entry in agent.knowledge.interaction_network.values()),
                default=1
            )

            for peer_id, network_entry in agent.knowledge.interaction_network.items():
                interaction_count = network_entry.get("interaction_count", 0)
                topics_count = len(network_entry.get("topics_discussed", {}))

                centrality = interaction_count / max_interactions if max_interactions > 0 else 0.0
                network_entry["centrality_score"] = round(centrality, 3)

                # Mark as hub if high centrality and diverse topics
                is_hub = centrality > 0.7 and topics_count >= 3
                network_entry["is_hub"] = is_hub

                # Update social memory
                mem = agent.knowledge.social_memory.get(peer_id)
                if mem:
                    mem["is_hub"] = is_hub

    def _update_interaction_tracking(
        self,
        agent: AutonomousAgent,
        peer_id: str,
        topic: str,
        tick_count: int | None,
    ) -> None:
        """Track interaction for social network analysis (A59)."""
        network_entry = agent.knowledge.interaction_network.setdefault(
            peer_id,
            {
                "interaction_count": 0,
                "last_interaction_tick": 0,
                "topics_discussed": {},
                "knowledge_quality": 0.0,
                "centrality_score": 0.0,
                "is_hub": False,
            },
        )

        network_entry["interaction_count"] += 1
        network_entry["last_interaction_tick"] = tick_count or 0
        topics_dict = network_entry["topics_discussed"]
        topics_dict[topic] = topics_dict.get(topic, 0) + 1
