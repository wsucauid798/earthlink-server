"""Weather — the atmospheric state of the world, evolving from real data."""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import WeatherData as WeatherModel

logger = logging.getLogger(__name__)


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
    Grounded in real historical weather data — interpolates between known observations
    to produce weather at any point in simulation time.
    """
    # Current weather per location
    _current: dict[int, WeatherState] = field(default_factory=dict)

    # Cached historical data for interpolation: location_id -> list of (datetime, WeatherState)
    _history: dict[int, list[tuple[datetime, WeatherState]]] = field(default_factory=dict)

    def get_weather(self, location_id: int) -> WeatherState | None:
        """Get current weather at a location."""
        return self._current.get(location_id)

    def get_all_weather(self) -> dict[int, WeatherState]:
        """Get current weather for all locations."""
        return dict(self._current)

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
