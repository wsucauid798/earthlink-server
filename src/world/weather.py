"""Weather — the atmospheric state of the world, from real Earth data.

Weather comes from Open-Meteo. On startup, historical data from the
database provides the initial state. A background scheduler then
periodically fetches live current conditions so the world's weather
matches what is actually happening in the UK right now.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import WeatherData as WeatherModel

logger = logging.getLogger(__name__)

# Open-Meteo Forecast API — free, no key, current conditions
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]

# WMO weather interpretation codes
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@dataclass
class WeatherState:
    """Current weather at a specific location."""
    location_id: int
    temperature_c: float | None = None
    precipitation_mm: float | None = None
    humidity_pct: float | None = None
    wind_speed_kmh: float | None = None
    wind_direction_deg: float | None = None
    cloud_cover_pct: float | None = None
    visibility_km: float | None = None
    pressure_hpa: float | None = None
    conditions: str | None = None


@dataclass
class Weather:
    """
    The atmospheric state of the world.

    On startup, historical data provides initial weather. A background
    refresh scheduler then fetches live current conditions from Open-Meteo
    so the world stays aligned with real Earth weather.
    """
    # Current weather per location
    _current: dict[int, WeatherState] = field(default_factory=dict)

    # Cached historical data for interpolation: location_id -> list of (datetime, WeatherState)
    _history: dict[int, list[tuple[datetime, WeatherState]]] = field(default_factory=dict)

    # Live refresh tracking
    _last_refresh: datetime | None = field(default=None)
    _refresh_count: int = 0
    _refresh_failures: int = 0

    @property
    def last_refresh(self) -> datetime | None:
        """When weather was last refreshed from Open-Meteo (UTC), or None."""
        return self._last_refresh

    @property
    def refresh_count(self) -> int:
        """Total number of successful refreshes since startup."""
        return self._refresh_count

    def get_weather(self, location_id: int) -> WeatherState | None:
        """Get current weather at a location."""
        return self._current.get(location_id)

    def get_all_weather(self) -> dict[int, WeatherState]:
        """Get current weather for all locations."""
        return dict(self._current)

    async def refresh(self, locations: list[tuple[int, float, float]]) -> int:
        """Fetch current weather from Open-Meteo for tracked locations.

        Args:
            locations: list of (location_id, lat, lng) tuples

        Returns:
            Number of locations successfully refreshed.
        """
        if not locations:
            return 0

        refreshed = 0
        async with httpx.AsyncClient(timeout=30) as client:
            for loc_id, lat, lng in locations:
                try:
                    state = await self._fetch_current(client, loc_id, lat, lng)
                    if state:
                        self._current[loc_id] = state
                        refreshed += 1
                except Exception as e:
                    logger.warning(f"Weather refresh failed for location {loc_id}: {e}")

        if refreshed > 0:
            self._last_refresh = datetime.now(timezone.utc)
            self._refresh_count += 1
            logger.info(f"Weather refreshed: {refreshed}/{len(locations)} locations updated")
        else:
            self._refresh_failures += 1
            logger.warning("Weather refresh: no locations updated")

        return refreshed

    async def _fetch_current(
        self,
        client: httpx.AsyncClient,
        location_id: int,
        lat: float,
        lng: float,
    ) -> WeatherState | None:
        """Fetch current weather for a single location from Open-Meteo."""
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": ",".join(CURRENT_WEATHER_PARAMS),
            "timezone": "Europe/London",
        }

        resp = await client.get(OPEN_METEO_FORECAST_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        if not current:
            return None

        weather_code = current.get("weather_code")
        return WeatherState(
            location_id=location_id,
            temperature_c=current.get("temperature_2m"),
            precipitation_mm=current.get("precipitation"),
            humidity_pct=current.get("relative_humidity_2m"),
            wind_speed_kmh=current.get("wind_speed_10m"),
            wind_direction_deg=current.get("wind_direction_10m"),
            cloud_cover_pct=current.get("cloud_cover"),
            visibility_km=None,  # Not available in current weather
            pressure_hpa=current.get("surface_pressure"),
            conditions=WMO_CODES.get(weather_code, f"Code {weather_code}") if weather_code is not None else None,
        )

    async def load_history(self, session: AsyncSession, location_ids: list[int]) -> None:
        """Load historical weather data from the database for interpolation."""
        for loc_id in location_ids:
            result = await session.execute(
                select(WeatherModel)
                .where(WeatherModel.location_id == loc_id)
                .order_by(WeatherModel.datetime)
            )
            rows = result.scalars().all()
            self._history[loc_id] = [
                (
                    row.datetime,
                    WeatherState(
                        location_id=loc_id,
                        temperature_c=row.temperature_c,
                        precipitation_mm=row.precipitation_mm,
                        humidity_pct=row.humidity_pct,
                        wind_speed_kmh=row.wind_speed_kmh,
                        wind_direction_deg=row.wind_direction_deg,
                        cloud_cover_pct=row.cloud_cover_pct,
                        visibility_km=row.visibility_km,
                        pressure_hpa=row.pressure_hpa,
                        conditions=row.conditions,
                    ),
                )
                for row in rows
            ]

        logger.info(f"Loaded weather history for {len(location_ids)} locations")

    def update(self, sim_time: datetime) -> None:
        """
        Update weather for all locations based on simulation time.

        Maps the simulation time to the same time-of-day and day-of-year
        in our historical data, then interpolates between the two nearest
        data points with small random variation for natural feel.
        """
        for loc_id, history in self._history.items():
            if not history:
                continue

            # Find the closest historical data point by matching month, day, hour
            state = self._interpolate(loc_id, sim_time, history)
            if state:
                self._current[loc_id] = state

    def _interpolate(
        self,
        loc_id: int,
        sim_time: datetime,
        history: list[tuple[datetime, WeatherState]],
    ) -> WeatherState | None:
        """
        Find the best matching historical weather for the given simulation time.
        Matches by month/day/hour to preserve seasonal/diurnal patterns.
        """
        target_month = sim_time.month
        target_day = sim_time.day
        target_hour = sim_time.hour

        # Find closest match by month, day, hour
        best = None
        best_score = float("inf")

        for dt, state in history:
            # Score: lower is better match
            month_diff = abs(dt.month - target_month)
            if month_diff > 6:
                month_diff = 12 - month_diff
            day_diff = abs(dt.day - target_day)
            if day_diff > 15:
                day_diff = 30 - day_diff
            hour_diff = abs(dt.hour - target_hour)
            if hour_diff > 12:
                hour_diff = 24 - hour_diff

            score = month_diff * 1000 + day_diff * 10 + hour_diff
            if score < best_score:
                best_score = score
                best = state

        if best is None:
            return None

        # Add small natural variation
        return WeatherState(
            location_id=loc_id,
            temperature_c=self._vary(best.temperature_c, 1.5),
            precipitation_mm=max(0, self._vary(best.precipitation_mm, 0.3)) if best.precipitation_mm else 0,
            humidity_pct=self._clamp(self._vary(best.humidity_pct, 3.0), 0, 100),
            wind_speed_kmh=max(0, self._vary(best.wind_speed_kmh, 2.0)),
            wind_direction_deg=self._vary(best.wind_direction_deg, 15.0) if best.wind_direction_deg else None,
            cloud_cover_pct=self._clamp(self._vary(best.cloud_cover_pct, 5.0), 0, 100),
            visibility_km=max(0.1, self._vary(best.visibility_km, 1.0)) if best.visibility_km else None,
            pressure_hpa=self._vary(best.pressure_hpa, 2.0),
            conditions=best.conditions,
        )

    @staticmethod
    def _vary(value: float | None, magnitude: float) -> float | None:
        if value is None:
            return None
        return value + random.gauss(0, magnitude)

    @staticmethod
    def _clamp(value: float | None, lo: float, hi: float) -> float | None:
        if value is None:
            return None
        return max(lo, min(hi, value))
