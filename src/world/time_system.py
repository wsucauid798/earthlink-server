"""Time system — the temporal dimension of the world.

The world IS Earth. Time is real Earth time — always. There is no
simulated clock, no alternative era, no fast-forward. Every tick,
the world reads the real wall clock and that is the time.

Time is per-location. Every place on Earth has its own local time,
timezone, and season based on its coordinates.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from timezonefinder import TimezoneFinder

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AstronomyData as AstronomyModel
from world.celestial import (
    julian_day_from_datetime,
    moon_age,
    moon_illumination,
    moon_phase,
    moon_phase_emoji,
    moon_phase_name,
    moonrise_moonset,
    season_from_ecliptic_longitude,
    solar_ecliptic_longitude,
    solar_noon_time,
    solar_position,
    sunrise_sunset,
    twilight_times,
)

logger = logging.getLogger(__name__)


@dataclass
class AstronomyState:
    """Astronomical state at a location for a given date."""
    # Sun (existing)
    sunrise: datetime | None = None
    sunset: datetime | None = None
    day_length_hours: float | None = None
    is_daylight: bool = True

    # Sun (expanded)
    solar_noon: datetime | None = None
    solar_elevation_deg: float | None = None
    solar_azimuth_deg: float | None = None

    # Twilight
    civil_dawn: datetime | None = None
    civil_dusk: datetime | None = None
    nautical_dawn: datetime | None = None
    nautical_dusk: datetime | None = None

    # Moon
    moon_phase: float | None = None
    moon_phase_name: str | None = None
    moon_phase_emoji: str | None = None
    moon_illumination_pct: float | None = None
    moon_age_days: float | None = None
    moonrise: datetime | None = None
    moonset: datetime | None = None


@dataclass
class TimeSystem:
    """
    The world's clock. Always real Earth time.

    Internally stores UTC. Per-location local time, timezone, and
    season are resolved from coordinates using real timezone boundaries.
    """
    current_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tick_count: int = 0
    _tf: TimezoneFinder | None = field(default=None, repr=False)

    # Timezone cache: (lat, lng) rounded -> IANA timezone name
    _tz_cache: dict[tuple[float, float], str] = field(default_factory=dict)

    # Astronomy data: location_id -> date -> AstronomyState
    _astronomy: dict[int, dict[str, AstronomyState]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._tf = TimezoneFinder()
        self.sync()

    @property
    def date(self):
        return self.current_time.date()

    @property
    def hour(self):
        return self.current_time.hour

    @property
    def minute(self):
        return self.current_time.minute

    def timezone_at(self, lat: float, lng: float) -> str:
        """Resolve IANA timezone name from coordinates. Cached."""
        key = (round(lat, 2), round(lng, 2))
        if key not in self._tz_cache:
            tz_name = self._tf.timezone_at(lat=lat, lng=lng) if self._tf else None
            self._tz_cache[key] = tz_name or "UTC"
        return self._tz_cache[key]

    def local_time_at(self, lat: float, lng: float) -> datetime:
        """Current local time at a specific location on Earth."""
        tz_name = self.timezone_at(lat, lng)
        tz = ZoneInfo(tz_name)
        return self.current_time.astimezone(tz)

    def time_at(self, lat: float, lng: float) -> dict:
        """Full time info for a location: local time, timezone, offset, abbreviation."""
        tz_name = self.timezone_at(lat, lng)
        tz = ZoneInfo(tz_name)
        local = self.current_time.astimezone(tz)
        offset = local.utcoffset()
        if offset is None:
            offset_str = "+00:00"
        else:
            total_seconds = int(offset.total_seconds())
            sign = "+" if total_seconds >= 0 else "-"
            total_seconds = abs(total_seconds)
            hours, remainder = divmod(total_seconds, 3600)
            minutes = remainder // 60
            offset_str = f"{sign}{hours:02d}:{minutes:02d}"

        return {
            "local_time": local.isoformat(),
            "timezone": tz_name,
            "timezone_abbr": local.strftime("%Z"),
            "utc_offset": offset_str,
            "hour": local.hour,
            "minute": local.minute,
            "date": local.date().isoformat(),
        }

    def season_at(self, lat: float) -> str:
        """Current season at a latitude. Hemisphere and tropical aware."""
        jd = julian_day_from_datetime(self.current_time.replace(tzinfo=None))
        lon = solar_ecliptic_longitude(jd)
        name, _, _, _ = season_from_ecliptic_longitude(lon, latitude=lat)
        return name.lower()

    # --- Backward-compatible properties (default to London for global display) ---

    @property
    def local_time(self) -> datetime:
        """Current time in Europe/London. For global display / backward compat."""
        return self.current_time.astimezone(ZoneInfo("Europe/London"))

    @property
    def utc_offset(self) -> str:
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
        return self.local_time.strftime("%Z")

    @property
    def season(self) -> str:
        """Current season at default latitude (London). For backward compat."""
        return self.season_at(51.5)

    def sync(self) -> None:
        """Sync the world clock to real Earth time (UTC)."""
        self.current_time = datetime.now(timezone.utc)

    def advance(self) -> None:
        """Advance the world by one tick — read the real clock."""
        self.sync()
        self.tick_count += 1

    def get_astronomy(self, location_id: int, lat: float | None = None, lng: float | None = None) -> AstronomyState | None:
        """Get the astronomical state at a location for the current date.

        If lat/lng are provided, also computes real-time solar elevation
        and azimuth (these change continuously, unlike daily fields).
        """
        date_key = self.current_time.date().isoformat()
        loc_data = self._astronomy.get(location_id, {})
        state = loc_data.get(date_key)

        if state and state.sunrise and state.sunset:
            # Compare as naive datetimes to avoid offset-naive vs offset-aware mismatch.
            # sunrise/sunset from math are naive; current_time is timezone-aware (UTC).
            now_naive = self.current_time.replace(tzinfo=None)
            state.is_daylight = state.sunrise <= now_naive <= state.sunset

        # Compute real-time solar position if coordinates available
        if state and lat is not None and lng is not None:
            now_naive = self.current_time.replace(tzinfo=None)
            elev, az = solar_position(lat, lng, now_naive)
            state.solar_elevation_deg = elev
            state.solar_azimuth_deg = az

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
                    solar_noon=getattr(row, "solar_noon", None),
                    civil_dawn=getattr(row, "civil_dawn", None),
                    civil_dusk=getattr(row, "civil_dusk", None),
                    nautical_dawn=getattr(row, "nautical_dawn", None),
                    nautical_dusk=getattr(row, "nautical_dusk", None),
                    moon_phase=getattr(row, "moon_phase", None),
                    moon_illumination_pct=getattr(row, "moon_illumination_pct", None),
                    moon_age_days=getattr(row, "moon_age_days", None),
                    moonrise=getattr(row, "moonrise", None),
                    moonset=getattr(row, "moonset", None),
                    moon_phase_name=moon_phase_name(row.moon_phase) if getattr(row, "moon_phase", None) is not None else None,
                    moon_phase_emoji=moon_phase_emoji(row.moon_phase) if getattr(row, "moon_phase", None) is not None else None,
                )
            self._astronomy[loc_id] = loc_data

        logger.info(f"Loaded astronomy data for {len(location_ids)} locations")

    def refresh_astronomy(self, locations: list[tuple[int, float, float]]) -> int:
        """Recompute all astronomy for the current date from celestial math.

        Pure computation — no DB, no API. Uses Jean Meeus algorithms
        from world/celestial.py.

        Args:
            locations: list of (location_id, lat, lng) tuples

        Returns:
            Number of locations refreshed.
        """
        today = self.current_time.date()
        now_naive = self.current_time.replace(tzinfo=None)
        refreshed = 0

        # Moon phase/age/illumination are the same globally for a given moment
        m_phase = moon_phase(now_naive)
        m_name = moon_phase_name(m_phase)
        m_emoji = moon_phase_emoji(m_phase)
        m_illum = moon_illumination(now_naive)
        m_age = moon_age(now_naive)

        for loc_id, lat, lng in locations:
            # Sun
            rise, sett = sunrise_sunset(lat, lng, today)
            day_length = None
            if rise and sett:
                day_length = round((sett - rise).total_seconds() / 3600, 2)

            noon = solar_noon_time(lat, lng, today)

            # Twilight
            civil_dawn, civil_dusk = twilight_times(lat, lng, today, 96.0)
            nautical_dawn, nautical_dusk = twilight_times(lat, lng, today, 102.0)

            # Moon rise/set (location-dependent)
            m_rise, m_set = moonrise_moonset(lat, lng, today)

            if loc_id not in self._astronomy:
                self._astronomy[loc_id] = {}

            self._astronomy[loc_id][today.isoformat()] = AstronomyState(
                sunrise=rise,
                sunset=sett,
                day_length_hours=day_length,
                solar_noon=noon,
                civil_dawn=civil_dawn,
                civil_dusk=civil_dusk,
                nautical_dawn=nautical_dawn,
                nautical_dusk=nautical_dusk,
                moon_phase=m_phase,
                moon_phase_name=m_name,
                moon_phase_emoji=m_emoji,
                moon_illumination_pct=m_illum,
                moon_age_days=m_age,
                moonrise=m_rise,
                moonset=m_set,
            )
            refreshed += 1

        if refreshed > 0:
            logger.info(
                f"Astronomy refreshed: {refreshed} locations for {today} "
                f"(moon: {m_name} {m_illum:.0f}%)"
            )

        return refreshed

    def to_dict(self) -> dict:
        """Serialize current time state (global / UTC-centric)."""
        return {
            "current_time": self.current_time.isoformat(),
            "local_time": self.local_time.isoformat(),
            "tick_count": self.tick_count,
            "date": self.current_time.date().isoformat(),
            "hour": self.hour,
            "minute": self.minute,
            "season": self.season,
            "timezone": "UTC",
            "timezone_abbr": "UTC",
            "utc_offset": "+00:00",
        }
