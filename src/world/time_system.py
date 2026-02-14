"""Time system — the temporal dimension of the world.

The world IS Earth. Time is real Earth time — always. There is no
simulated clock, no alternative era, no fast-forward. Every tick,
the world reads the real wall clock and that is the time.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

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
    is_daylight: bool = True


@dataclass
class TimeSystem:
    """
    The world's clock. Always real Earth time.

    Internally stores UTC. Exposes local time in the configured
    timezone (default Europe/London — handles GMT/BST automatically).
    """
    current_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tick_count: int = 0
    timezone_name: str = "Europe/London"
    _tz: ZoneInfo | None = field(default=None, repr=False)

    # Astronomy data: location_id -> date -> AstronomyState
    _astronomy: dict[int, dict[str, AstronomyState]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._tz = ZoneInfo(self.timezone_name)
        # Always start at real time
        self.sync()

    @property
    def tz(self) -> ZoneInfo:
        """The active timezone object."""
        if self._tz is None:
            self._tz = ZoneInfo(self.timezone_name)
        return self._tz

    @property
    def local_time(self) -> datetime:
        """Current time expressed in the configured timezone (e.g. Europe/London)."""
        return self.current_time.astimezone(self.tz)

    @property
    def utc_offset(self) -> str:
        """Current UTC offset string (e.g. '+00:00' for GMT, '+01:00' for BST)."""
        offset = self.local_time.utcoffset()
        if offset is None:
            return "+00:00"
        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes = remainder // 60
        return f"{sign}{hours:02d}:{minutes:02d}"

    @property
    def timezone_abbr(self) -> str:
        """Current timezone abbreviation (e.g. 'GMT' or 'BST')."""
        return self.local_time.strftime("%Z")

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

    def sync(self) -> None:
        """Sync the world clock to real Earth time (UTC)."""
        self.current_time = datetime.now(timezone.utc)

    def advance(self) -> None:
        """Advance the world by one tick — read the real clock."""
        self.sync()
        self.tick_count += 1

    def get_astronomy(self, location_id: int) -> AstronomyState | None:
        """Get the astronomical state at a location for the current date."""
        date_key = self.current_time.date().isoformat()
        loc_data = self._astronomy.get(location_id, {})
        state = loc_data.get(date_key)

        if state and state.sunrise and state.sunset:
            # Compare as naive datetimes to avoid offset-naive vs offset-aware mismatch.
            # sunrise/sunset from math are naive; current_time is timezone-aware (UTC).
            now_naive = self.current_time.replace(tzinfo=None)
            state.is_daylight = state.sunrise <= now_naive <= state.sunset

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
                )
            self._astronomy[loc_id] = loc_data

        logger.info(f"Loaded astronomy data for {len(location_ids)} locations")

    def refresh_astronomy(self, locations: list[tuple[int, float, float]]) -> int:
        """Recompute astronomy (sunrise/sunset) for the current date from math.

        Pure computation — no DB, no API. Uses the same algorithms as
        data_acquisition/fetch_astronomy.py.

        Args:
            locations: list of (location_id, lat, lng) tuples

        Returns:
            Number of locations refreshed.
        """
        from data_acquisition.fetch_astronomy import sunrise_sunset

        today = self.current_time.date()
        refreshed = 0

        for loc_id, lat, lng in locations:
            rise, sett = sunrise_sunset(lat, lng, today)
            day_length = None
            if rise and sett:
                day_length = round((sett - rise).total_seconds() / 3600, 2)

            if loc_id not in self._astronomy:
                self._astronomy[loc_id] = {}

            self._astronomy[loc_id][today.isoformat()] = AstronomyState(
                sunrise=rise,
                sunset=sett,
                day_length_hours=day_length,
            )
            refreshed += 1

        if refreshed > 0:
            logger.info(f"Astronomy refreshed: {refreshed} locations for {today}")

        return refreshed

    def to_dict(self) -> dict:
        """Serialize current time state."""
        return {
            "current_time": self.current_time.isoformat(),
            "local_time": self.local_time.isoformat(),
            "tick_count": self.tick_count,
            "date": self.current_time.date().isoformat(),
            "hour": self.hour,
            "minute": self.minute,
            "season": self.season,
            "timezone": self.timezone_name,
            "timezone_abbr": self.timezone_abbr,
            "utc_offset": self.utc_offset,
        }
