"""Time system — the temporal dimension of the world."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AstronomyData as AstronomyModel

logger = logging.getLogger(__name__)


@dataclass
class AstronomyState:
    """Astronomical state at a location for a given date."""
    sunrise: datetime | None = None
    sunset: datetime | None = None
    day_length_hours: float | None = None
    moon_phase: float | None = None
    is_daylight: bool = True


@dataclass
class TimeSystem:
    """
    The world's clock. Tracks the current date and time, computes day/night
    and astronomical state per location from real data.
    """
    current_time: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    tick_count: int = 0
    time_step: timedelta = field(default_factory=lambda: timedelta(minutes=15))

    # Astronomy data: location_id -> date -> AstronomyState
    _astronomy: dict[int, dict[str, AstronomyState]] = field(default_factory=dict)

    @property
    def date(self):
        return self.current_time.date()

    @property
    def hour(self):
        return self.current_time.hour

    @property
    def minute(self):
        return self.current_time.minute

    @property
    def season(self) -> str:
        """Current season based on month (Northern Hemisphere)."""
        month = self.current_time.month
        if month in (3, 4, 5):
            return "spring"
        elif month in (6, 7, 8):
            return "summer"
        elif month in (9, 10, 11):
            return "autumn"
        else:
            return "winter"

    def advance(self) -> None:
        """Advance the world clock by one tick."""
        self.current_time += self.time_step
        self.tick_count += 1

    def get_astronomy(self, location_id: int) -> AstronomyState | None:
        """Get the astronomical state at a location for the current date."""
        date_key = self.current_time.date().isoformat()
        loc_data = self._astronomy.get(location_id, {})
        state = loc_data.get(date_key)

        if state and state.sunrise and state.sunset:
            state.is_daylight = state.sunrise <= self.current_time <= state.sunset

        return state

    def is_daytime(self, location_id: int) -> bool:
        """Check if it's currently daytime at a location."""
        astro = self.get_astronomy(location_id)
        if astro:
            return astro.is_daylight
        # Fallback: rough estimate based on hour
        hour = self.current_time.hour
        return 6 <= hour <= 20

    async def load_astronomy(self, session: AsyncSession, location_ids: list[int]) -> None:
        """Load astronomy data from the database."""
        for loc_id in location_ids:
            result = await session.execute(
                select(AstronomyModel)
                .where(AstronomyModel.location_id == loc_id)
            )
            loc_data = {}
            for row in result.scalars().all():
                loc_data[row.date.isoformat()] = AstronomyState(
                    sunrise=row.sunrise,
                    sunset=row.sunset,
                    day_length_hours=row.day_length_hours,
                    moon_phase=row.moon_phase,
                )
            self._astronomy[loc_id] = loc_data

        logger.info(f"Loaded astronomy data for {len(location_ids)} locations")

    def to_dict(self) -> dict:
        """Serialize current time state."""
        return {
            "current_time": self.current_time.isoformat(),
            "tick_count": self.tick_count,
            "date": self.current_time.date().isoformat(),
            "hour": self.hour,
            "minute": self.minute,
            "season": self.season,
        }
