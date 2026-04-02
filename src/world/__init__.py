"""
The World.

Bootstraps geography, weather, wind, and time from the database.
Ticks forward independently. Knows nothing about agents.
"""

__version__ = "0.0.2"

import asyncio
import logging
import time as _time
from datetime import datetime, timezone

from db.engine import async_session
from db.models import AgentState, WorldState

from sqlalchemy import select

from .config import RefreshPolicy, WorldConfig
from .geography import Geography, load_geography
from .time_system import TimeSystem
from .weather import Weather
from .wind import Wind
from .atmosphere import Atmosphere
from .earth_proxy import EarthProxy
from .geophysics import Geophysics
from .orbital import Orbital
from .data_feeds import DataFeeds

logger = logging.getLogger(__name__)


class World:
    """
    The virtual world. It exists. It ticks. It is Earth.

    The world loads its state from real data in the database, advances
    through time, and updates weather — all independently. It doesn't
    know about agents, APIs, or anything outside itself.
    """

    def __init__(self, config: WorldConfig | None = None):
        self.config = config or WorldConfig()
        self.geography: Geography | None = None
        self.weather: Weather | None = None
        self.wind: Wind | None = None
        self.atmosphere: Atmosphere = Atmosphere()
        self.time: TimeSystem | None = None
        self.geophysics: Geophysics = Geophysics()
        self.orbital: Orbital = Orbital()
        self.data_feeds: DataFeeds = DataFeeds()
        self.earth_proxy: EarthProxy | None = None
        self.agents: AgentSystem | None = None
        self._running = False
        self._tick_task: asyncio.Task | None = None
        self._refresh_tasks: dict[str, asyncio.Task] = {}
        self._last_refresh: dict[str, datetime] = {}
        self._refresh_counts: dict[str, int] = {}
        self._tick_callbacks: list = []
        self._refreshed_domains: set[str] = set()  # domains refreshed since last tick
        self._resolve_task: asyncio.Task | None = None  # background location resolution

    @property
    def is_running(self) -> bool:
        return self._running

    def on_tick(self, callback) -> None:
        """Register a callback to be called on each tick. Used by API layer for streaming."""
        self._tick_callbacks.append(callback)

    async def load(self) -> None:
        """Load the world from the database."""
        logger.info("Loading world...")

        # Load geography (locations in memory, connections on demand)
        self.geography = await load_geography(async_session)

        async with async_session() as session:
            # Initialize time system — always real Earth time
            self.time = TimeSystem()

            # Restore tick count from database if it exists
            result = await session.execute(select(WorldState).where(WorldState.id == 1))
            saved_state = result.scalar_one_or_none()
            if saved_state:
                self.time.tick_count = saved_state.tick_count
            logger.info(
                f"World time: {self.time.local_time.isoformat()} "
                f"{self.time.timezone_abbr}, tick={self.time.tick_count}"
            )

            # Get top populated locations for weather/astronomy tracking
            weather_locations = await self.geography.get_top_locations_by_population(
                types=["capital", "city", "town"], limit=50
            )
            weather_location_ids = [loc.id for loc in weather_locations]

            # Load weather
            self.weather = Weather()
            await self.weather.load_history(session, weather_location_ids)

            # Load astronomy
            await self.time.load_astronomy(session, weather_location_ids)

            # Select wind stations — broader geographic coverage than weather
            wind_station_ids = await self._select_wind_stations(
                self.geography, self.config.wind_station_count
            )

            # Load wind
            self.wind = Wind()
            await self.wind.load_history(session, wind_station_ids)

            # Restore agents if previously persisted
            persisted_agents_result = await session.execute(select(AgentState).order_by(AgentState.id))
            persisted_agents = persisted_agents_result.scalars().all()

        # Set initial weather and wind state
        self.weather.update(self.time.current_time)
        self.wind.update(self.time.current_time)

        # Populate wind station coordinates for interpolation
        for loc_id in self.wind._history:
            loc = self.geography.get_location(loc_id)
            if loc:
                self.wind._station_coords[loc_id] = (loc.lat, loc.lng, loc.terrain)

        from agents.system import AgentSystem

        if persisted_agents:
            payloads = [
                {
                    "id": row.id,
                    "name": row.name,
                    "location_id": row.location_id,
                    "energy": row.energy,
                    "last_move_distance_km": row.last_move_distance_km,
                    "last_move_connection_type": row.last_move_connection_type or "road",
                    "last_action": row.last_action,
                    "learning_rate": row.learning_rate,
                    "exploration_bias": row.exploration_bias,
                    "risk_tolerance": row.risk_tolerance,
                    "stamina": row.stamina,
                    "comfort_temperature_c": row.comfort_temperature_c,
                    "discount_factor": (row.knowledge or {}).get("_policy", {}).get("discount_factor", 0.92),
                    "goal": (row.knowledge or {}).get("_goal"),
                    "goal_age_ticks": (row.knowledge or {}).get("_goal_age_ticks", 0),
                    "knowledge": row.knowledge,
                }
                for row in persisted_agents
            ]
            self.agents = AgentSystem.from_persisted(payloads, seed=self.config.agent_random_seed)

            # Top up if configured count is higher than persisted count
            existing_count = len(self.agents.agents)
            if existing_count < self.config.agent_count:
                topup = await AgentSystem.bootstrap(
                    geography=self.geography,
                    count=self.config.agent_count,
                    learning_rate=self.config.agent_learning_rate,
                    exploration_bias=self.config.agent_exploration_bias,
                    seed=self.config.agent_random_seed,
                )
                # Only take the new agents (skip indices that overlap with existing)
                existing_ids = {a.agent_id for a in self.agents.agents}
                for agent in topup.agents:
                    if agent.agent_id not in existing_ids and len(self.agents.agents) < self.config.agent_count:
                        self.agents.agents.append(agent)
                added = len(self.agents.agents) - existing_count
                if added > 0:
                    logger.info(f"Topped up {added} new agents (total: {len(self.agents.agents)})")
        else:
            # Bootstrap autonomous agents from scratch
            self.agents = await AgentSystem.bootstrap(
                geography=self.geography,
                count=self.config.agent_count,
                learning_rate=self.config.agent_learning_rate,
                exploration_bias=self.config.agent_exploration_bias,
                seed=self.config.agent_random_seed,
            )

        logger.info(
            f"World loaded: {self.geography.location_count} locations, "
            f"time={self.time.current_time}, season={self.time.season}"
        )

        # Initialise agents as Ray actors (parallel ticking).
        # Falls back gracefully to sequential mode if Ray is unavailable.
        try:
            ray_active = self.agents.init_ray(seed=self.config.agent_random_seed)
            if ray_active:
                logger.info("Agents running as Ray actors (parallel mode)")
            else:
                logger.info("Agents running in sequential mode")
        except Exception as e:
            logger.warning(f"Ray init failed, using sequential mode: {e}")

        # Initialise Earth proxy — the world's live connection to civilisation
        self.earth_proxy = EarthProxy(
            policies=self.config.adapters,
            default_timeout_seconds=self.config.adapter_timeout_seconds,
            max_concurrent_adapters=self.config.max_concurrent_adapters,
        )
        try:
            from config import settings
            await self.earth_proxy.connect_redis(settings.redis_url)
        except Exception as e:
            logger.warning(f"Redis not available for Earth proxy: {e}")
        try:
            from adapters import all_key_free_adapters
            for adapter in all_key_free_adapters():
                self.earth_proxy.register_adapter(adapter)
        except Exception as e:
            logger.warning(f"Failed to register adapters: {e}")
        backend = "Redis" if self.earth_proxy.using_redis else "in-memory fallback"
        enabled_count = len([a for a in self.earth_proxy.adapters if self.earth_proxy.policies.get_policy(a.name).enabled])
        logger.info(
            f"Earth proxy ready: {self.earth_proxy.adapter_count} adapters "
            f"({enabled_count} enabled), backend={backend}"
        )

    async def tick(self) -> dict:
        """
        Advance the world by one step.
        Time moves — the world lives.

        Phases are timed for observability (R6). The agent phase is
        wall-clock capped so a slow agent tick (e.g. Chroma ingestion)
        never prevents the WS broadcast from firing on schedule (R1/R5).
        """
        tick_wall_start = _time.monotonic()

        # Phase 1: advance time (essentially free)
        self.time.advance()

        # Phase 2: schedule background LOD resolve (fire-and-forget)
        if self.earth_proxy and self.agents:
            if self._resolve_task is None or self._resolve_task.done():
                self._resolve_task = asyncio.create_task(
                    self._resolve_agent_locations()
                )

        # Phase 2.5: warm location + connection caches for agent positions (LOD)
        if self.agents and self.geography:
            agent_locs = {a.location_id for a in self.agents.agents}
            await self.geography.warm(agent_locs)

        # Phase 3: agent tick — capped at remaining wall budget
        agent_events = []
        agent_phase_exceeded = False
        if self.agents:
            budget = max(0.1, self.config.max_tick_wall_seconds - (_time.monotonic() - tick_wall_start))
            try:
                agent_events = await asyncio.wait_for(
                    self.agents.tick(
                        self.geography,
                        self.weather,
                        self.time.current_time,
                        tick_count=self.time.tick_count,
                        earth_proxy=self.earth_proxy,
                        wind=self.wind,
                        tick_interval=self.config.tick_interval_seconds,
                    ),
                    timeout=budget,
                )
            except asyncio.TimeoutError:
                agent_phase_exceeded = True
                logger.warning(
                    f"Tick {self.time.tick_count}: agent phase exceeded budget "
                    f"({budget:.2f}s), skipped to keep WS on schedule"
                )

        # Build tick summary — include which domains were refreshed since last tick
        refreshed = self._refreshed_domains.copy()
        self._refreshed_domains.clear()

        # Earth rotation state for frontend globe rendering
        from world.celestial import earth_rotation_state
        rotation = earth_rotation_state(self.time.current_time.replace(tzinfo=None))

        tick_data = {
            "tick": self.time.tick_count,
            "time": self.time.to_dict(),
            "rotation": rotation,
            "weather_updated": "weather" in refreshed,
            "wind_updated": "wind" in refreshed,
            "atmosphere_updated": "atmosphere" in refreshed,
            "astronomy_updated": "astronomy" in refreshed,
            "data_feeds_updated": "data_feeds" in refreshed,
            "agent_events": agent_events,
            "agent_phase_exceeded": agent_phase_exceeded,
            "earth_proxy_resolves": self.earth_proxy.total_resolves if self.earth_proxy else 0,
        }

        # Persist world state periodically (every 10 ticks)
        if self.time.tick_count % 10 == 0:
            await self._save_state()

        # Phase 4: notify listeners (WS broadcast) — always runs
        for callback in self._tick_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tick_data)
                else:
                    callback(tick_data)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")

        # R6: structured timing log (every 50 ticks to avoid noise)
        tick_wall_ms = (_time.monotonic() - tick_wall_start) * 1000
        if self.time.tick_count % 50 == 0:
            resolving = "yes" if (self._resolve_task and not self._resolve_task.done()) else "no"
            logger.info(
                f"Tick {self.time.tick_count}: wall={tick_wall_ms:.0f}ms "
                f"agents={len(agent_events)} events "
                f"lod_resolving={resolving} "
                f"proxy_resolves={self.earth_proxy.total_resolves if self.earth_proxy else 0}"
            )

        return tick_data

    async def start(self) -> None:
        """Start the world ticking."""
        if self._running:
            return
        self._running = True
        self._tick_task = asyncio.create_task(self._tick_loop())
        self._start_refresh_schedulers()
        logger.info("World started")

    async def pause(self) -> None:
        """Pause the world."""
        self._running = False
        # Cancel tick loop
        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None
        # Cancel background resolve task
        if self._resolve_task and not self._resolve_task.done():
            self._resolve_task.cancel()
            try:
                await self._resolve_task
            except asyncio.CancelledError:
                pass
            self._resolve_task = None
        # Cancel all refresh schedulers
        for domain, task in self._refresh_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._refresh_tasks.clear()
        await self._save_state()
        # Shut down Ray actors
        if self.agents:
            self.agents.shutdown_ray()
        logger.info("World paused")

    async def reset(self) -> None:
        """Reset the world — agents and tick count restart, time stays real."""
        await self.pause()
        self.time.sync()
        self.time.tick_count = 0
        self.weather.update(self.time.current_time)
        if self.wind:
            self.wind.update(self.time.current_time)
        self.atmosphere._current.clear()
        from agents.system import AgentSystem
        self.agents = AgentSystem.bootstrap(
            geography=self.geography,
            count=self.config.agent_count,
            learning_rate=self.config.agent_learning_rate,
            exploration_bias=self.config.agent_exploration_bias,
            seed=self.config.agent_random_seed,
        )
        # Re-init Ray actors for the new agents
        try:
            self.agents.init_ray(seed=self.config.agent_random_seed)
        except Exception as e:
            logger.warning(f"Ray re-init after reset failed: {e}")
        await self._save_state()
        logger.info("World reset")

    async def _tick_loop(self) -> None:
        """The heartbeat of the world."""
        while self._running:
            await self.tick()
            await asyncio.sleep(self.config.tick_interval_seconds)

    # --- Refresh Schedulers ---

    def _tracked_locations(self) -> list[tuple[int, float, float]]:
        """Build (location_id, lat, lng) list for weather/astronomy locations."""
        if not self.geography or not self.weather:
            return []
        return [
            (loc_id, self.geography.locations[loc_id].lat, self.geography.locations[loc_id].lng)
            for loc_id in self.weather._history
            if loc_id in self.geography.locations
        ]

    def _start_refresh_schedulers(self) -> None:
        """Start a background refresh task for each enabled domain policy."""
        policies = self.config.refresh

        if policies.weather.enabled:
            self._refresh_tasks["weather"] = asyncio.create_task(
                self._refresh_loop("weather", policies.weather, self._refresh_weather)
            )
        if policies.wind.enabled:
            self._refresh_tasks["wind"] = asyncio.create_task(
                self._refresh_loop("wind", policies.wind, self._refresh_wind)
            )
        if policies.astronomy.enabled:
            self._refresh_tasks["astronomy"] = asyncio.create_task(
                self._refresh_loop("astronomy", policies.astronomy, self._refresh_astronomy)
            )
        if policies.atmosphere.enabled:
            self._refresh_tasks["atmosphere"] = asyncio.create_task(
                self._refresh_loop("atmosphere", policies.atmosphere, self._refresh_atmosphere)
            )
        if policies.geography.enabled:
            self._refresh_tasks["geography"] = asyncio.create_task(
                self._refresh_loop("geography", policies.geography, self._refresh_geography)
            )
        if policies.data_feeds.enabled:
            self._refresh_tasks["data_feeds"] = asyncio.create_task(
                self._refresh_loop("data_feeds", policies.data_feeds, self._refresh_data_feeds)
            )

    async def _refresh_loop(self, domain: str, policy: RefreshPolicy, action) -> None:
        """Generic refresh loop: run action immediately, then every interval."""
        interval = policy.interval_minutes * 60
        logger.info(f"Refresh scheduler [{domain}]: every {policy.interval_minutes} min")

        # First refresh immediately
        try:
            await action()
            self._last_refresh[domain] = datetime.now(timezone.utc)
            self._refresh_counts[domain] = self._refresh_counts.get(domain, 0) + 1
            self._refreshed_domains.add(domain)
        except Exception as e:
            logger.error(f"Initial refresh [{domain}] failed: {e}")

        while self._running:
            await asyncio.sleep(interval)
            try:
                await action()
                self._last_refresh[domain] = datetime.now(timezone.utc)
                self._refresh_counts[domain] = self._refresh_counts.get(domain, 0) + 1
                self._refreshed_domains.add(domain)
            except Exception as e:
                logger.error(f"Refresh [{domain}] failed: {e}")

    async def _resolve_agent_locations(self) -> None:
        """LOD: resolve civilisation content for locations agents are at.

        Like a video game loading terrain as the player approaches — the world
        progressively resolves civilisation for places agents have reached.
        Once resolved, content lives in Redis with TTL. When it expires,
        the world asks Earth again. The internet is always the source of truth.

        Runs as a background task — never blocks the tick loop. Locations
        are resolved concurrently up to max_concurrent_locations.
        """
        # Gather unique agent locations
        agent_location_ids = {agent.location_id for agent in self.agents.agents}

        # Check which ones need resolution (concurrently)
        check_results = await asyncio.gather(
            *(self.earth_proxy.is_location_resolved(loc_id) for loc_id in agent_location_ids)
        )
        unresolved = [
            loc_id for loc_id, resolved in zip(agent_location_ids, check_results)
            if not resolved
        ]
        if not unresolved:
            return

        # Location-level concurrency limit
        loc_sem = asyncio.Semaphore(self.config.max_concurrent_locations)

        async def _safe_resolve(loc_id: int) -> None:
            loc = self.geography.get_location(loc_id)
            if loc:
                try:
                    async with loc_sem:
                        await self.earth_proxy.resolve_for_location(loc_id, loc.name)
                except Exception as e:
                    logger.warning(f"LOD resolve failed for {loc.name}: {e}")

        logger.info(f"LOD: resolving {len(unresolved)} locations (max concurrent={self.config.max_concurrent_locations})")
        await asyncio.gather(*(_safe_resolve(loc_id) for loc_id in unresolved))

    async def _refresh_weather(self) -> None:
        """Refresh weather: fetch live current conditions from Open-Meteo."""
        locations = self._tracked_locations()
        if locations:
            await self.weather.refresh(locations)

    def _tracked_wind_locations(self) -> list[tuple[int, float, float, str | None]]:
        """Build (location_id, lat, lng, terrain) list for wind stations."""
        if not self.geography or not self.wind:
            return []
        return [
            (
                loc_id,
                self.geography.locations[loc_id].lat,
                self.geography.locations[loc_id].lng,
                self.geography.locations[loc_id].terrain,
            )
            for loc_id in self.wind._history
            if loc_id in self.geography.locations
        ]

    async def _refresh_wind(self) -> None:
        """Refresh wind: fetch live current conditions from Open-Meteo."""
        locations = self._tracked_wind_locations()
        if locations:
            await self.wind.refresh(locations)

    async def _refresh_atmosphere(self) -> None:
        """Refresh atmosphere: fetch air quality from Open-Meteo AQ API."""
        locations = self._tracked_locations()
        if locations:
            await self.atmosphere.refresh(locations)

    async def _refresh_astronomy(self) -> None:
        """Refresh astronomy: recompute sunrise/sunset for today from math."""
        locations = self._tracked_locations()
        if locations:
            self.time.refresh_astronomy(locations)

    async def _refresh_data_feeds(self) -> None:
        """Refresh world data feeds (solar activity from NOAA SWPC)."""
        await self.data_feeds.refresh()

    async def _refresh_geography(self) -> None:
        """Refresh geography: reload locations from the database, reset connection cache."""
        self.geography = await load_geography(async_session)
        logger.info(
            f"Geography refreshed: {self.geography.location_count} locations, "
            f"{self.geography.connection_count} connections"
        )

    @staticmethod
    async def _select_wind_stations(geography: Geography, count: int) -> list[int]:
        """Select wind station locations — top populated places from DB."""
        locations = await geography.get_top_locations_by_population(
            types=["capital", "city", "town", "village", "settlement"],
            limit=count,
        )
        return [loc.id for loc in locations]

    async def _save_state(self) -> None:
        """Persist the current world state to the database."""
        # Strip timezone info for naive DateTime column
        current_time = self.time.current_time.replace(tzinfo=None)
        async with async_session() as session:
            result = await session.execute(select(WorldState).where(WorldState.id == 1))
            state = result.scalar_one_or_none()
            if state:
                state.current_time = current_time
                state.tick_count = self.time.tick_count
                state.is_running = self._running
            else:
                state = WorldState(
                    id=1,
                    current_time=current_time,
                    tick_count=self.time.tick_count,
                    is_running=self._running,
                )
                session.add(state)

            if self.agents:
                result_agents = await session.execute(select(AgentState))
                existing = {row.id: row for row in result_agents.scalars().all()}
                persisted_agents = self.agents.to_persisted()
                active_ids = {item["id"] for item in persisted_agents}

                for agent_payload in persisted_agents:
                    row = existing.get(agent_payload["id"])
                    if row is None:
                        row = AgentState(id=agent_payload["id"])
                        session.add(row)
                    row.name = agent_payload["name"]
                    row.location_id = agent_payload["location_id"]
                    row.energy = agent_payload["energy"]
                    row.last_move_distance_km = agent_payload["last_move_distance_km"]
                    row.last_move_connection_type = agent_payload.get("last_move_connection_type", "road")
                    row.last_action = agent_payload["last_action"]
                    row.learning_rate = agent_payload["learning_rate"]
                    row.exploration_bias = agent_payload["exploration_bias"]
                    row.risk_tolerance = agent_payload["risk_tolerance"]
                    row.stamina = agent_payload["stamina"]
                    row.comfort_temperature_c = agent_payload["comfort_temperature_c"]
                    persisted_knowledge = dict(agent_payload["knowledge"])
                    persisted_knowledge["_policy"] = {
                        "discount_factor": agent_payload["discount_factor"],
                    }
                    persisted_knowledge["_goal"] = agent_payload.get("goal")
                    persisted_knowledge["_goal_age_ticks"] = agent_payload.get("goal_age_ticks", 0)
                    row.knowledge = persisted_knowledge

                for stale_id, stale_row in existing.items():
                    if stale_id not in active_ids:
                        await session.delete(stale_row)

            await session.commit()

    def get_state_summary(self) -> dict:
        """Get a summary of the current world state."""
        weather_summary = {}
        if self.weather:
            for loc_id, ws in self.weather.get_all_weather().items():
                loc = self.geography.get_location(loc_id)
                if loc:
                    weather_summary[loc.name] = {
                        "temperature_c": round(ws.temperature_c, 1) if ws.temperature_c else None,
                        "conditions": ws.conditions,
                        "wind_speed_kmh": round(ws.wind_speed_kmh, 1) if ws.wind_speed_kmh else None,
                        "is_daylight": self.time.is_daytime(loc_id),
                    }

        # Earth geography statistics — aggregated from DB
        location_types: dict[str, int] = {}
        regions: dict[str, int] = {}
        countries: dict[str, int] = {}
        total_population = 0
        elevations: list[float] = []

        return {
            "time": self.time.to_dict() if self.time else None,
            "is_running": self._running,
            "location_count": self.geography.location_count if self.geography else 0,
            "connection_count": self.geography.connection_count if self.geography else 0,
            "weather_stations": len(weather_summary),
            "wind_stations": self.wind.station_count if self.wind else 0,
            "weather": weather_summary,
            "geography_stats": {
                "location_types": dict(sorted(location_types.items(), key=lambda x: -x[1])),
                "countries": dict(sorted(countries.items(), key=lambda x: -x[1])),
                "regions": dict(sorted(regions.items(), key=lambda x: -x[1])),
                "total_population": total_population,
                "elevation_min": round(min(elevations), 1) if elevations else None,
                "elevation_max": round(max(elevations), 1) if elevations else None,
            },
            "refresh": {
                domain: {
                    "enabled": policy.enabled,
                    "interval_minutes": policy.interval_minutes,
                    "last_refresh": self._last_refresh.get(domain, None)
                        and self._last_refresh[domain].isoformat(),
                    "refresh_count": self._refresh_counts.get(domain, 0),
                }
                for domain, policy in [
                    ("weather", self.config.refresh.weather),
                    ("wind", self.config.refresh.wind),
                    ("atmosphere", self.config.refresh.atmosphere),
                    ("astronomy", self.config.refresh.astronomy),
                    ("geography", self.config.refresh.geography),
                    ("data_feeds", self.config.refresh.data_feeds),
                ]
            },
            "agent_count": len(self.agents.agents) if self.agents else 0,
            "agent_backend": "ray" if (self.agents and self.agents.using_ray) else "sequential",
            "agents": self.agents.summaries(self.geography) if self.agents else [],
            "earth_proxy": {
                "adapters": self.earth_proxy.adapter_count if self.earth_proxy else 0,
                "total_resolves": self.earth_proxy.total_resolves if self.earth_proxy else 0,
                "ttl_seconds": self.earth_proxy.ttl_seconds if self.earth_proxy else 0,
                "backend": "redis" if (self.earth_proxy and self.earth_proxy.using_redis) else "in-memory",
            },
        }

    def list_agents(self) -> list[dict]:
        if not self.agents or not self.geography:
            return []
        return self.agents.summaries(self.geography)

    def get_agent(self, agent_id: str) -> dict | None:
        if not self.agents or not self.geography:
            return None
        return self.agents.detail(agent_id, self.geography)

    def ask_agent(self, agent_id: str, question: str) -> dict | None:
        if not self.agents or not self.geography:
            return None
        return self.agents.answer(agent_id, question, self.geography)
