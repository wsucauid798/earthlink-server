"""
The World.

Bootstraps geography, weather, and time from the database.
Ticks forward independently. Knows nothing about agents.
"""

__version__ = "0.0.1"

import asyncio
import logging
from datetime import timedelta

from db.engine import async_session
from db.models import WorldState

from sqlalchemy import select

from .config import WorldConfig
from .geography import Geography, load_geography
from .time_system import TimeSystem
from .weather import Weather

logger = logging.getLogger(__name__)


class World:
    """
    The virtual world. It exists. It ticks. It is the United Kingdom.

    The world loads its state from real data in the database, advances
    through time, and updates weather — all independently. It doesn't
    know about agents, APIs, or anything outside itself.
    """

    def __init__(self, config: WorldConfig | None = None):
        self.config = config or WorldConfig()
        self.geography: Geography | None = None
        self.weather: Weather | None = None
        self.time: TimeSystem | None = None
        self._running = False
        self._tick_task: asyncio.Task | None = None
        self._tick_callbacks: list = []

    @property
    def is_running(self) -> bool:
        return self._running

    def on_tick(self, callback) -> None:
        """Register a callback to be called on each tick. Used by API layer for streaming."""
        self._tick_callbacks.append(callback)

    async def load(self) -> None:
        """Load the world from the database."""
        logger.info("Loading world...")

        async with async_session() as session:
            # Load geography
            self.geography = await load_geography(session)

            # Initialize time system
            self.time = TimeSystem(
                current_time=self.config.start_time,
                time_step=timedelta(minutes=self.config.time_scale_minutes),
            )

            # Restore world state from database if it exists
            result = await session.execute(select(WorldState).where(WorldState.id == 1))
            saved_state = result.scalar_one_or_none()
            if saved_state:
                self.time.current_time = saved_state.current_time
                self.time.tick_count = saved_state.tick_count
                logger.info(f"Restored world state: time={self.time.current_time}, tick={self.time.tick_count}")

            # Get location IDs that have weather/astronomy data
            weather_location_ids = [
                loc_id for loc_id in self.geography.locations
                if self.geography.locations[loc_id].type in ("capital", "city", "town")
                and self.geography.locations[loc_id].population
                and self.geography.locations[loc_id].population > 0
            ]
            # Limit to top 50 by population (matching data acquisition)
            weather_location_ids = sorted(
                weather_location_ids,
                key=lambda lid: self.geography.locations[lid].population or 0,
                reverse=True,
            )[:50]

            # Load weather
            self.weather = Weather()
            await self.weather.load_history(session, weather_location_ids)

            # Load astronomy
            await self.time.load_astronomy(session, weather_location_ids)

        # Set initial weather state
        self.weather.update(self.time.current_time)

        logger.info(
            f"World loaded: {self.geography.location_count} locations, "
            f"time={self.time.current_time}, season={self.time.season}"
        )

    async def tick(self) -> dict:
        """
        Advance the world by one step.
        Time moves, weather changes — the world lives.
        Returns a summary of what changed.
        """
        # Advance time
        self.time.advance()

        # Update weather periodically
        weather_updated = False
        if self.time.tick_count % self.config.weather_update_interval_ticks == 0:
            self.weather.update(self.time.current_time)
            weather_updated = True

        # Build tick summary
        tick_data = {
            "tick": self.time.tick_count,
            "time": self.time.to_dict(),
            "weather_updated": weather_updated,
        }

        # Persist world state periodically (every 10 ticks)
        if self.time.tick_count % 10 == 0:
            await self._save_state()

        # Notify listeners
        for callback in self._tick_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(tick_data)
                else:
                    callback(tick_data)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")

        return tick_data

    async def start(self) -> None:
        """Start the world ticking."""
        if self._running:
            return
        self._running = True
        self._tick_task = asyncio.create_task(self._tick_loop())
        logger.info("World started")

    async def pause(self) -> None:
        """Pause the world."""
        self._running = False
        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None
        await self._save_state()
        logger.info("World paused")

    async def reset(self) -> None:
        """Reset the world to its initial state."""
        await self.pause()
        self.time.current_time = self.config.start_time
        self.time.tick_count = 0
        self.weather.update(self.time.current_time)
        await self._save_state()
        logger.info("World reset")

    async def _tick_loop(self) -> None:
        """The heartbeat of the world."""
        while self._running:
            await self.tick()
            await asyncio.sleep(self.config.tick_interval_seconds)

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

        return {
            "time": self.time.to_dict() if self.time else None,
            "is_running": self._running,
            "location_count": self.geography.location_count if self.geography else 0,
            "connection_count": self.geography.connection_count if self.geography else 0,
            "weather_stations": len(weather_summary),
            "weather": weather_summary,
        }
