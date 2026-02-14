"""Ray actor wrapper for AutonomousAgent.

Each agent runs as an independent Ray actor — true parallel ticking.
Geography and Weather are shared via Ray's object store (zero-copy on same node).
The AgentSystem coordinator fans out tick() calls and handles social learning.

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
    """A single autonomous agent running as a Ray actor.

    The actor owns the AutonomousAgent instance. All state mutations happen
    here. The coordinator (AgentSystem) sends tick parameters and collects
    results via remote calls.
    """

    def __init__(self, agent_persisted: dict, seed: int):
        from agents.system import AutonomousAgent
        self._agent = AutonomousAgent.from_persisted(agent_persisted)
        self._rng = random.Random(seed)

    # --- Core tick --------------------------------------------------------

    def tick(
        self,
        geography,
        weather,
        sim_time: datetime,
        tick_count: int | None,
        earth_facts_map: dict,
    ) -> dict:
        """Run one full agent tick: perceive → decide → act → learn.

        Args:
            geography: Geography object (resolved from ObjectRef).
            weather: Weather object (resolved from ObjectRef).
            sim_time: Current simulation datetime.
            tick_count: World tick counter.
            earth_facts_map: {location_id: [EarthFact, ...]} for all
                resolved locations. The actor looks up facts for its
                current and post-move locations.

        Returns:
            Event dict for this tick.
        """
        agent = self._agent

        # Pre-move perception with earth facts for current location
        current_earth_facts = earth_facts_map.get(agent.location_id, [])
        observation = agent.perceive(
            geography, weather, sim_time, earth_facts=current_earth_facts,
        )
        agent.refresh_goal(observation, geography, self._rng)
        next_location = agent.choose_next_location(
            observation, geography, weather, self._rng,
        )
        previous_location = agent.location_id
        agent.apply_action(next_location, geography)

        # Post-move perception
        post_earth_facts = earth_facts_map.get(agent.location_id, [])
        next_observation = agent.perceive(
            geography, weather, sim_time, earth_facts=post_earth_facts,
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

    def get_agent_id(self) -> str:
        return self._agent.agent_id

    def get_location_id(self) -> int:
        return self._agent.location_id

    def get_summary(self, geography) -> dict:
        return self._agent.to_summary(geography)

    def get_detail(self, geography) -> dict:
        return self._agent.to_detail(geography)

    def answer(self, question: str, geography) -> dict:
        return self._agent.answer(question, geography)

    def to_persisted(self) -> dict:
        return self._agent.to_persisted()

    # --- Social learning interface ----------------------------------------

    def get_social_snapshot(self) -> dict:
        """Return data needed by the coordinator for social learning.

        Kept lean — only the slices the social algorithms need.
        """
        agent = self._agent
        return {
            "agent_id": agent.agent_id,
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
        }

    def apply_social_update(self, update: dict) -> None:
        """Apply social learning updates computed by the coordinator.

        The update dict has keys:
            new_facts:        list[dict] — facts to append
            belief_updates:   dict[key, belief_dict] — beliefs to set/merge
            belief_conflicts: list[dict] — conflicts to append
            q_updates:        dict[location_id_str, dict[action_id_str, float]]
            dialogue_events:  list[dict] — dialogue entries to append
        """
        agent = self._agent
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


# ---------------------------------------------------------------------------
# Apply @ray.remote decorator conditionally
# ---------------------------------------------------------------------------

if RAY_AVAILABLE:
    AgentActor = ray.remote(_AgentActorImpl)
else:
    AgentActor = None
