"""Wind — the global wind field of the world, from real Earth data.

Wind comes from Open-Meteo. It covers more locations than weather,
supports terrain-aware interpolation for any location, and provides
richer data (gusts, multi-height, pressure).

This is the first Earth-science system — it sets the pattern for
atmosphere, orbital, geophysics modules.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import WindData as WindModel

logger = logging.getLogger(__name__)

# Open-Meteo Forecast API — free, no key, current conditions
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WIND_PARAMS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "wind_speed_80m",
    "wind_direction_80m",
    "surface_pressure",
]

# Beaufort scale descriptions
BEAUFORT_DESCRIPTIONS = [
    "Calm",            # 0
    "Light air",       # 1
    "Light breeze",    # 2
    "Gentle breeze",   # 3
    "Moderate breeze",  # 4
    "Fresh breeze",    # 5
    "Strong breeze",   # 6
    "High wind",       # 7
    "Gale",            # 8
    "Strong gale",     # 9
    "Storm",           # 10
    "Violent storm",   # 11
    "Hurricane force",  # 12
]


# --- Terrain Modifiers ---

@dataclass
class TerrainModifier:
    """How terrain type affects wind at a location."""
    speed_factor: float        # Multiplier for wind speed (< 1 = sheltered, > 1 = exposed)
    gust_factor: float         # Multiplier for gust speed
    direction_variance: float  # Degrees of random direction shift from terrain channeling


TERRAIN_WIND_MODIFIERS: dict[str | None, TerrainModifier] = {
    "urban":      TerrainModifier(speed_factor=0.70, gust_factor=1.20, direction_variance=20.0),
    "rural":      TerrainModifier(speed_factor=0.90, gust_factor=1.05, direction_variance=8.0),
    "highland":   TerrainModifier(speed_factor=1.30, gust_factor=1.35, direction_variance=15.0),
    "water":      TerrainModifier(speed_factor=1.15, gust_factor=1.10, direction_variance=5.0),
    "woodland":   TerrainModifier(speed_factor=0.55, gust_factor=0.80, direction_variance=25.0),
    "vegetation": TerrainModifier(speed_factor=0.75, gust_factor=0.90, direction_variance=12.0),
    "parkland":   TerrainModifier(speed_factor=0.80, gust_factor=1.00, direction_variance=10.0),
    "natural":    TerrainModifier(speed_factor=1.00, gust_factor=1.10, direction_variance=10.0),
    None:         TerrainModifier(speed_factor=1.00, gust_factor=1.00, direction_variance=10.0),
}


# --- Wind State ---

@dataclass
class WindState:
    """Current wind conditions at a specific location."""
    location_id: int

    # Surface (10m)
    speed_kmh: float | None = None
    direction_deg: float | None = None
    gust_kmh: float | None = None

    # Upper level
    speed_upper_kmh: float | None = None
    direction_upper_deg: float | None = None
    upper_height_m: int | None = None

    # Atmospheric
    pressure_hpa: float | None = None

    # Metadata
    terrain_modifier: str | None = None
    is_interpolated: bool = False
    interpolation_distance_km: float | None = None

    @property
    def beaufort(self) -> int:
        """Beaufort wind force scale (0-12) from speed."""
        if self.speed_kmh is None:
            return 0
        s = self.speed_kmh
        if s < 1: return 0
        if s < 6: return 1
        if s < 12: return 2
        if s < 20: return 3
        if s < 29: return 4
        if s < 39: return 5
        if s < 50: return 6
        if s < 62: return 7
        if s < 75: return 8
        if s < 89: return 9
        if s < 103: return 10
        if s < 118: return 11
        return 12

    @property
    def beaufort_description(self) -> str:
        """Human-readable Beaufort description."""
        return BEAUFORT_DESCRIPTIONS[self.beaufort]


# --- Wind System ---

@dataclass
class Wind:
    """The global wind field of the world.

    On startup, historical data provides initial wind patterns. A background
    refresh scheduler fetches live conditions from Open-Meteo. For locations
    without a wind station, inverse-distance weighted interpolation provides
    terrain-aware estimates from nearby stations.
    """

    # Direct station data: location_id -> WindState
    _stations: dict[int, WindState] = field(default_factory=dict)

    # Cached interpolated results: location_id -> WindState (cleared on each refresh)
    _interpolated_cache: dict[int, WindState] = field(default_factory=dict)

    # Historical data for interpolation during startup
    _history: dict[int, list[tuple[datetime, WindState]]] = field(default_factory=dict)

    # Station metadata: location_id -> (lat, lng, terrain)
    _station_coords: dict[int, tuple[float, float, str | None]] = field(default_factory=dict)

    # Refresh tracking
    _last_refresh: datetime | None = field(default=None)
    _refresh_count: int = 0
    _refresh_failures: int = 0

    @property
    def station_count(self) -> int:
        """Number of direct wind monitoring stations."""
        return len(self._stations)

    @property
    def last_refresh(self) -> datetime | None:
        """When wind was last refreshed from Open-Meteo (UTC)."""
        return self._last_refresh

    @property
    def refresh_count(self) -> int:
        """Total successful refreshes since startup."""
        return self._refresh_count

    # --- Public API ---

    def get_wind(
        self,
        location_id: int,
        lat: float | None = None,
        lng: float | None = None,
        terrain: str | None = None,
    ) -> WindState | None:
        """Get wind at any location — station data or interpolated.

        If the location is a wind station, returns direct data.
        Otherwise, interpolates from nearby stations with terrain adjustment.
        lat/lng/terrain needed for interpolation of non-station locations.
        """
        # Direct station hit
        if location_id in self._stations:
            return self._stations[location_id]

        # Check interpolation cache
        if location_id in self._interpolated_cache:
            return self._interpolated_cache[location_id]

        # Interpolate if we have coordinates
        if lat is not None and lng is not None:
            result = self._interpolate_at(location_id, lat, lng, terrain)
            if result:
                self._interpolated_cache[location_id] = result
            return result

        return None

    def get_station_wind(self, location_id: int) -> WindState | None:
        """Get wind only if this location is a direct station."""
        return self._stations.get(location_id)

    def get_all_stations(self) -> dict[int, WindState]:
        """Get wind data for all direct stations."""
        return dict(self._stations)

    # --- Refresh from Open-Meteo ---

    async def refresh(self, locations: list[tuple[int, float, float, str | None]]) -> int:
        """Fetch current wind from Open-Meteo for tracked stations.

        Args:
            locations: list of (location_id, lat, lng, terrain) tuples

        Returns:
            Number of stations successfully refreshed.
        """
        if not locations:
            return 0

        # Update station coordinates
        for loc_id, lat, lng, terrain in locations:
            self._station_coords[loc_id] = (lat, lng, terrain)

        refreshed = 0
        # Clear interpolation cache — station data is about to change
        self._interpolated_cache.clear()

        async with httpx.AsyncClient(timeout=30) as client:
            for loc_id, lat, lng, terrain in locations:
                try:
                    state = await self._fetch_current(client, loc_id, lat, lng, terrain)
                    if state:
                        self._stations[loc_id] = state
                        refreshed += 1
                except Exception as e:
                    logger.warning(f"Wind refresh failed for location {loc_id}: {e}")

        if refreshed > 0:
            self._last_refresh = datetime.now(timezone.utc)
            self._refresh_count += 1
            logger.info(f"Wind refreshed: {refreshed}/{len(locations)} stations updated")
        else:
            self._refresh_failures += 1
            logger.warning("Wind refresh: no stations updated")

        return refreshed

    async def _fetch_current(
        self,
        client: httpx.AsyncClient,
        location_id: int,
        lat: float,
        lng: float,
        terrain: str | None,
    ) -> WindState | None:
        """Fetch current wind for a single station from Open-Meteo."""
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": ",".join(WIND_PARAMS),
            "timezone": "Europe/London",
        }

        resp = await client.get(OPEN_METEO_FORECAST_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        if not current:
            return None

        return WindState(
            location_id=location_id,
            speed_kmh=current.get("wind_speed_10m"),
            direction_deg=current.get("wind_direction_10m"),
            gust_kmh=current.get("wind_gusts_10m"),
            speed_upper_kmh=current.get("wind_speed_80m"),
            direction_upper_deg=current.get("wind_direction_80m"),
            upper_height_m=80,
            pressure_hpa=current.get("surface_pressure"),
            terrain_modifier=terrain,
            is_interpolated=False,
        )

    # --- Historical Data ---

    async def load_history(self, session: AsyncSession, location_ids: list[int]) -> None:
        """Load historical wind data from the database for initial state."""
        for loc_id in location_ids:
            result = await session.execute(
                select(WindModel)
                .where(WindModel.location_id == loc_id)
                .order_by(WindModel.datetime)
            )
            rows = result.scalars().all()
            self._history[loc_id] = [
                (
                    row.datetime,
                    WindState(
                        location_id=loc_id,
                        speed_kmh=row.wind_speed_10m,
                        direction_deg=row.wind_direction_10m,
                        gust_kmh=row.wind_gusts_10m,
                        speed_upper_kmh=row.wind_speed_upper,
                        direction_upper_deg=row.wind_direction_upper,
                        upper_height_m=row.upper_height_m,
                        pressure_hpa=row.pressure_hpa,
                    ),
                )
                for row in rows
            ]

        loaded = sum(len(h) for h in self._history.values())
        logger.info(f"Loaded wind history: {len(location_ids)} stations, {loaded} records")

    def update(self, sim_time: datetime) -> None:
        """Update wind from historical data (startup / fallback).

        Maps simulation time to the same time-of-day and day-of-year
        in historical data, with small random variation.
        """
        for loc_id, history in self._history.items():
            if not history:
                continue
            state = self._interpolate_historical(loc_id, sim_time, history)
            if state:
                self._stations[loc_id] = state
        # Clear interpolation cache after bulk update
        self._interpolated_cache.clear()

    def _interpolate_historical(
        self,
        loc_id: int,
        sim_time: datetime,
        history: list[tuple[datetime, WindState]],
    ) -> WindState | None:
        """Find best matching historical wind by month/day/hour."""
        target_month = sim_time.month
        target_day = sim_time.day
        target_hour = sim_time.hour

        best = None
        best_score = float("inf")

        for dt, state in history:
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

        # Small natural variation
        return WindState(
            location_id=loc_id,
            speed_kmh=max(0, self._vary(best.speed_kmh, 2.0)),
            direction_deg=self._wrap_deg(self._vary(best.direction_deg, 15.0)),
            gust_kmh=max(0, self._vary(best.gust_kmh, 3.0)) if best.gust_kmh else None,
            speed_upper_kmh=max(0, self._vary(best.speed_upper_kmh, 3.0)) if best.speed_upper_kmh else None,
            direction_upper_deg=self._wrap_deg(self._vary(best.direction_upper_deg, 10.0)),
            upper_height_m=best.upper_height_m,
            pressure_hpa=self._vary(best.pressure_hpa, 2.0),
            terrain_modifier=best.terrain_modifier,
        )

    # --- Spatial Interpolation (IDW) ---

    def _interpolate_at(
        self,
        location_id: int,
        lat: float,
        lng: float,
        terrain: str | None,
    ) -> WindState | None:
        """Inverse-distance weighted interpolation from nearby stations."""
        if not self._station_coords:
            return None

        # Find nearest stations within 200 km
        distances: list[tuple[int, float]] = []
        for sid, (slat, slng, _) in self._station_coords.items():
            if sid not in self._stations:
                continue
            d = self._haversine_km(lat, lng, slat, slng)
            if d < 200:
                distances.append((sid, d))

        if not distances:
            return None

        distances.sort(key=lambda x: x[1])
        nearest = distances[:4]

        # IDW: weight = 1 / distance^2
        total_weight = 0.0
        weighted_speed = 0.0
        weighted_gust = 0.0
        weighted_pressure = 0.0
        # Direction needs vector averaging
        weighted_dx = 0.0
        weighted_dy = 0.0
        # Upper level
        weighted_speed_upper = 0.0
        weighted_dx_upper = 0.0
        weighted_dy_upper = 0.0
        has_upper = False

        for sid, dist in nearest:
            ws = self._stations[sid]
            w = 1.0 / max(dist, 0.5) ** 2
            total_weight += w

            if ws.speed_kmh is not None:
                weighted_speed += w * ws.speed_kmh
            if ws.gust_kmh is not None:
                weighted_gust += w * ws.gust_kmh
            if ws.pressure_hpa is not None:
                weighted_pressure += w * ws.pressure_hpa
            if ws.direction_deg is not None:
                rad = math.radians(ws.direction_deg)
                weighted_dx += w * math.cos(rad)
                weighted_dy += w * math.sin(rad)
            if ws.speed_upper_kmh is not None:
                has_upper = True
                weighted_speed_upper += w * ws.speed_upper_kmh
            if ws.direction_upper_deg is not None:
                rad = math.radians(ws.direction_upper_deg)
                weighted_dx_upper += w * math.cos(rad)
                weighted_dy_upper += w * math.sin(rad)

        if total_weight == 0:
            return None

        raw_speed = weighted_speed / total_weight
        raw_gust = weighted_gust / total_weight
        raw_direction = math.degrees(math.atan2(weighted_dy, weighted_dx)) % 360
        raw_pressure = weighted_pressure / total_weight if weighted_pressure else None

        # Apply terrain modifier
        modifier = TERRAIN_WIND_MODIFIERS.get(terrain, TERRAIN_WIND_MODIFIERS[None])
        speed = raw_speed * modifier.speed_factor
        gust = raw_gust * modifier.gust_factor
        direction = (raw_direction + random.gauss(0, modifier.direction_variance)) % 360

        avg_dist = sum(d for _, d in nearest) / len(nearest)

        result = WindState(
            location_id=location_id,
            speed_kmh=round(speed, 1),
            direction_deg=round(direction, 1),
            gust_kmh=round(gust, 1),
            pressure_hpa=round(raw_pressure, 1) if raw_pressure else None,
            terrain_modifier=terrain,
            is_interpolated=True,
            interpolation_distance_km=round(avg_dist, 1),
        )

        if has_upper and total_weight > 0:
            result.speed_upper_kmh = round(weighted_speed_upper / total_weight, 1)
            result.direction_upper_deg = round(
                math.degrees(math.atan2(weighted_dy_upper, weighted_dx_upper)) % 360, 1
            )
            result.upper_height_m = 80

        return result

    # --- Helpers ---

    @staticmethod
    def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Great-circle distance between two points in km."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlng / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _vary(value: float | None, magnitude: float) -> float | None:
        if value is None:
            return None
        return value + random.gauss(0, magnitude)

    @staticmethod
    def _wrap_deg(value: float | None) -> float | None:
        """Wrap degree value to 0-360 range."""
        if value is None:
            return None
        return value % 360
