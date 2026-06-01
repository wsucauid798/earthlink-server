"""Ray actor wrapper for AutonomousAgent.

Each actor owns a **batch** of agents (not one), so N≈cores actors cover the
whole population with true multi-process parallelism while staying inside the
VPS memory budget — 1000 single-agent actors (~150–250 MB each) OOM a 24 GB
host (S94), whereas ~N batched actors hold all agents in N resident processes.
State stays inside the actor across ticks (no per-tick re-pickling), unlike a
ProcessPoolExecutor. Geography and Weather are shared via Ray's object store
(zero-copy on same node). The AgentSystem coordinator fans out tick() calls and
handles social learning. See server-design-plan S100.

Falls back gracefully to sequential mode if Ray is unavailable.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.info("Ray not available — agents will tick sequentially")


# ---------------------------------------------------------------------------
# Actor implementation — defined always, decorated only when Ray present.
# ---------------------------------------------------------------------------

class _AgentActorImpl:
    """A batch of autonomous agents running as one Ray actor.

    The actor owns several AutonomousAgent instances (keyed by agent_id) and
    ticks them in-process. All state mutations happen here; the coordinator
    (AgentSystem) sends tick parameters and collects per-agent results via
    remote calls. Per-agent methods (`get_detail`, `answer`,
    `apply_social_update`) address an agent within the batch by id.
    """

    def __init__(self, batch_persisted: list[dict], seed: int):
        from agents.system import AutonomousAgent
        # Preserve order for stable iteration; index by id for addressed calls.
        self._agents = [AutonomousAgent.from_persisted(p) for p in batch_persisted]
        self._by_id = {a.agent_id: a for a in self._agents}
        # One RNG per agent, deterministically derived from the actor seed and
        # the agent's position in the batch — keeps decisions reproducible and
        # independent across agents (mirrors the sequential per-agent RNG).
        self._rngs = {a.agent_id: random.Random(seed + i) for i, a in enumerate(self._agents)}

    @property
    def agent_ids(self) -> list[str]:
        return [a.agent_id for a in self._agents]

    # --- Core tick --------------------------------------------------------

    def tick(
        self,
        geography,
        weather,
        sim_time: datetime,
        tick_count: int | None,
        earth_facts_map: dict,
        wind=None,
        tick_interval: float = 1.0,
    ) -> list[dict]:
        """Tick every agent in the batch; return one event dict per agent.

        Args:
            geography: Geography object (resolved from ObjectRef).
            weather: Weather object (resolved from ObjectRef).
            sim_time: Current simulation datetime.
            tick_count: World tick counter.
            earth_facts_map: {location_id: [EarthFact, ...]} for all
                resolved locations. Each agent looks up facts for its
                current and post-move locations.
            wind: Wind object (resolved from ObjectRef), or None.

        Returns:
            List of per-agent event dicts (batch order).
        """
        return [
            self._tick_one(
                agent, geography, weather, sim_time, tick_count, earth_facts_map, wind, tick_interval,
            )
            for agent in self._agents
        ]

    def _tick_one(
        self, agent, geography, weather, sim_time, tick_count, earth_facts_map, wind, tick_interval,
    ) -> dict:
        rng = self._rngs[agent.agent_id]

        # Pre-move perception with earth facts for current location
        current_earth_facts = earth_facts_map.get(agent.location_id, [])
        observation = agent.perceive(
            geography, weather, sim_time, earth_facts=current_earth_facts, wind=wind,
        )
        agent.refresh_goal(observation, geography, rng)
        next_location = agent.choose_next_location(
            observation, geography, weather, rng,
        )
        previous_location = agent.location_id
        agent.apply_action(next_location, geography, tick_interval_seconds=tick_interval)

        # Post-move perception
        post_earth_facts = earth_facts_map.get(agent.location_id, [])
        next_observation = agent.perceive(
            geography, weather, sim_time, earth_facts=post_earth_facts, wind=wind,
        )
        agent.learn(next_observation, tick_count=tick_count)

        q_value = 0.0
        if agent._last_state_id is not None and agent._last_action_id is not None:
            q_value = agent.knowledge.q_values.get(
                agent._last_state_id, {},
            ).get(agent._last_action_id, 0.0)

        return {
            "agent_id": agent.agent_id,
            "from_location_id": previous_location,
            "to_location_id": agent.location_id,
            "action": agent.last_action,
            "knowledge_score": agent.knowledge.knowledge_score,
            "reward": agent.last_reward,
            "q_value": round(q_value, 6),
            "goal": agent.current_goal.to_dict() if agent.current_goal else None,
        }

    # --- Read-only queries ------------------------------------------------

    def get_summary(self, geography) -> list[dict]:
        """Summaries for every agent in the batch."""
        return [agent.to_summary(geography) for agent in self._agents]

    def get_detail(self, agent_id: str, geography) -> dict | None:
        agent = self._by_id.get(agent_id)
        return agent.to_detail(geography) if agent is not None else None

    def answer(self, agent_id: str, question: str, geography) -> dict | None:
        agent = self._by_id.get(agent_id)
        return agent.answer(question, geography) if agent is not None else None

    def to_persisted(self) -> list[dict]:
        """Persisted state for every agent in the batch."""
        return [agent.to_persisted() for agent in self._agents]

    # --- Social learning interface ----------------------------------------

    def get_social_snapshot(self) -> list[dict]:
        """Social-learning snapshots for every agent in the batch."""
        return [self._snapshot_one(agent) for agent in self._agents]

    def _snapshot_one(self, agent) -> dict:
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "location_id": agent.location_id,
            "recent_facts": list(agent.knowledge.facts[-60:]),
            "recent_facts_texts_500": {
                str(f.get("text", "")).strip()
                for f in agent.knowledge.facts[-500:]
                if isinstance(f, dict)
            },
            "beliefs": dict(agent.knowledge.beliefs),
            "q_values": {
                loc_id: dict(action_map)
                for loc_id, action_map in agent.knowledge.q_values.items()
            },
            "social_memory": {k: dict(v) for k, v in agent.knowledge.social_memory.items()},
            # A55/A56 - Conversation state
            "active_conversations": {k: dict(v) for k, v in agent.knowledge.active_conversations.items()},
            # A57/A58 - Teaching/learning tracking (for effectiveness computation)
            "teaching_events": list(agent.knowledge.teaching_events[-100:]),  # Recent 100
            # A59 - Interaction network
            "interaction_network": {k: dict(v) for k, v in agent.knowledge.interaction_network.items()},
        }

    def apply_social_updates(self, updates_by_id: dict[str, dict]) -> None:
        """Apply social-learning updates to the batch's agents.

        ``updates_by_id`` maps ``agent_id`` → an update dict for the agents in
        this actor that participated in an exchange (agents with no exchange
        this tick are simply absent). Each update dict has keys:
            new_facts:             list[dict] — facts to append
            belief_updates:        dict[key, belief_dict] — beliefs to set/merge
            belief_conflicts:      list[dict] — conflicts to append
            q_updates:             dict[location_id_str, dict[action_id_str, float]]
            dialogue_events:       list[dict] — dialogue entries to append
            social_memory:         dict[agent_id, record] — social memory to merge
            encounters:            list[dict] — encounter events to append
            active_conversations:  dict[conv_id, conv_record] — conversations to merge (A56)
            conversation_history:  list[dict] — archived conversations to append (A56)
            teaching_events:       list[dict] — teaching events to append (A57)
            learning_requests:     list[dict] — learning requests to append (A58)
            interaction_network:   dict[agent_id, network_record] — network data to merge (A59)
        """
        for agent_id, update in updates_by_id.items():
            agent = self._by_id.get(agent_id)
            if agent is not None:
                self._apply_one_update(agent, update)

    def _apply_one_update(self, agent, update: dict) -> None:
        knowledge = agent.knowledge

        # Append new facts
        for fact in update.get("new_facts", []):
            knowledge.facts.append(fact)
        if len(knowledge.facts) > 5000:
            knowledge.facts = knowledge.facts[-5000:]

        # Set/merge beliefs
        for key, belief in update.get("belief_updates", {}).items():
            knowledge.beliefs[key] = belief

        # Append belief conflicts
        for conflict in update.get("belief_conflicts", []):
            knowledge.belief_conflicts.append(conflict)
        if len(knowledge.belief_conflicts) > 1000:
            knowledge.belief_conflicts = knowledge.belief_conflicts[-1000:]

        # Merge Q-values
        for loc_str, action_map in update.get("q_updates", {}).items():
            loc_id = int(loc_str)
            q_map = knowledge.q_values.setdefault(loc_id, {})
            for action_str, q_val in action_map.items():
                q_map[int(action_str)] = q_val

        # Append dialogue events
        for event in update.get("dialogue_events", []):
            knowledge.dialogue_events.append(event)
        if len(knowledge.dialogue_events) > 5000:
            knowledge.dialogue_events = knowledge.dialogue_events[-5000:]

        # Merge social memory (A53/A54/A60)
        for agent_id, record in update.get("social_memory", {}).items():
            knowledge.social_memory[agent_id] = record

        # Append encounters (A60)
        for encounter in update.get("encounters", []):
            knowledge.encounters.append(encounter)
        if len(knowledge.encounters) > 2000:
            knowledge.encounters = knowledge.encounters[-2000:]

        # Merge active conversations (A56)
        for conv_id, conv_data in update.get("active_conversations", {}).items():
            knowledge.active_conversations[conv_id] = conv_data

        # Append conversation history (A56)
        for conv in update.get("conversation_history", []):
            knowledge.conversation_history.append(conv)
        if len(knowledge.conversation_history) > 1000:
            knowledge.conversation_history = knowledge.conversation_history[-1000:]

        # Append teaching events (A57)
        for event in update.get("teaching_events", []):
            knowledge.teaching_events.append(event)
        if len(knowledge.teaching_events) > 1000:
            knowledge.teaching_events = knowledge.teaching_events[-1000:]

        # Append learning requests (A58)
        for request in update.get("learning_requests", []):
            knowledge.learning_requests.append(request)
        if len(knowledge.learning_requests) > 1000:
            knowledge.learning_requests = knowledge.learning_requests[-1000:]

        # Merge interaction network (A59)
        for peer_id, network_entry in update.get("interaction_network", {}).items():
            knowledge.interaction_network[peer_id] = network_entry


# ---------------------------------------------------------------------------
# Apply @ray.remote decorator conditionally
# ---------------------------------------------------------------------------

if RAY_AVAILABLE:
    # Each actor gets a tiny CPU slice — agents are lightweight compute.
    # This lets hundreds of actors coexist without each claiming a full core.
    AgentActor = ray.remote(num_cpus=0.01)(_AgentActorImpl)
else:
    AgentActor = None
