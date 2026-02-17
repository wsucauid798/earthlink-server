"""Atmosphere — air quality and derived atmospheric quantities.

Air quality data fetched from Open-Meteo Air Quality API (free, no key).
Derived quantities (dew point, feels-like, UV index, air density) computed
from existing weather and astronomy data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import exp, log, cos, radians

import httpx

logger = logging.getLogger(__name__)

# Open-Meteo Air Quality API — free, no key
OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

CURRENT_AQ_PARAMS = [
    "european_aqi",
    "us_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

# European AQI labels
EUROPEAN_AQI_LABELS = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor",
}

# Specific gas constant for dry air (J/(kg·K))
_R_SPECIFIC = 287.05


@dataclass
class AtmosphereState:
    """Atmospheric state at a specific location."""
    location_id: int
    # Air quality (fetched)
    european_aqi: int | None = None
    european_aqi_label: str | None = None
    us_aqi: int | None = None
    pm2_5: float | None = None
    pm10: float | None = None
    ozone: float | None = None
    nitrogen_dioxide: float | None = None
    sulphur_dioxide: float | None = None
    carbon_monoxide: float | None = None
    # Derived (computed)
    dew_point_c: float | None = None
    feels_like_c: float | None = None
    uv_index: float | None = None
    air_density_kgm3: float | None = None


@dataclass
class Atmosphere:
    """
    The atmosphere system — air quality + derived atmospheric quantities.

    Air quality is fetched from Open-Meteo AQ API on a 30-minute refresh.
    Derived values (dew point, feels-like, UV, density) are computed on
    demand from current weather and astronomy data.
    """
    # Cached air quality per location (fetched data only)
    _current: dict[int, AtmosphereState] = field(default_factory=dict)

    # Refresh tracking
    _last_refresh: datetime | None = None
    _refresh_count: int = 0
    _refresh_failures: int = 0

    @property
    def last_refresh(self) -> datetime | None:
        return self._last_refresh

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    @property
    def station_count(self) -> int:
        return len(self._current)

    def get_atmosphere(
        self,
        location_id: int,
        temperature_c: float | None = None,
        humidity_pct: float | None = None,
        wind_speed_kmh: float | None = None,
        pressure_hpa: float | None = None,
        cloud_cover_pct: float | None = None,
        solar_elevation_deg: float | None = None,
    ) -> AtmosphereState | None:
        """Get full atmosphere state, merging cached AQ with computed values.

        Computed values require weather/astronomy data passed in.
        If no cached AQ exists, still returns computed values.
        """
        cached = self._current.get(location_id)

        # Start from cached or fresh state
        state = AtmosphereState(
            location_id=location_id,
            european_aqi=cached.european_aqi if cached else None,
            european_aqi_label=cached.european_aqi_label if cached else None,
            us_aqi=cached.us_aqi if cached else None,
            pm2_5=cached.pm2_5 if cached else None,
            pm10=cached.pm10 if cached else None,
            ozone=cached.ozone if cached else None,
            nitrogen_dioxide=cached.nitrogen_dioxide if cached else None,
            sulphur_dioxide=cached.sulphur_dioxide if cached else None,
            carbon_monoxide=cached.carbon_monoxide if cached else None,
        )

        # Compute derived values from weather/astronomy
        if temperature_c is not None and humidity_pct is not None:
            state.dew_point_c = self._dew_point(temperature_c, humidity_pct)

        if temperature_c is not None:
            state.feels_like_c = self._feels_like(
                temperature_c,
                wind_speed_kmh or 0.0,
                humidity_pct or 50.0,
            )

        if solar_elevation_deg is not None:
            state.uv_index = self._uv_index(
                solar_elevation_deg,
                cloud_cover_pct or 0.0,
            )

        if pressure_hpa is not None and temperature_c is not None:
            state.air_density_kgm3 = self._air_density(pressure_hpa, temperature_c)

        # If we have no AQ data and no computed data, return None
        if cached is None and all(
            v is None for v in [state.dew_point_c, state.feels_like_c, state.uv_index, state.air_density_kgm3]
        ):
            return None

        return state

    async def refresh(self, locations: list[tuple[int, float, float]]) -> int:
        """Fetch current air quality from Open-Meteo for tracked locations.

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
                    logger.warning(f"Atmosphere refresh failed for location {loc_id}: {e}")

        if refreshed > 0:
            self._last_refresh = datetime.now(timezone.utc)
            self._refresh_count += 1
            logger.info(f"Atmosphere refreshed: {refreshed}/{len(locations)} locations updated")
        else:
            self._refresh_failures += 1
            logger.warning("Atmosphere refresh: no locations updated")

        return refreshed

    async def _fetch_current(
        self,
        client: httpx.AsyncClient,
        location_id: int,
        lat: float,
        lng: float,
    ) -> AtmosphereState | None:
        """Fetch current air quality for a single location from Open-Meteo."""
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": ",".join(CURRENT_AQ_PARAMS),
        }

        resp = await client.get(OPEN_METEO_AQ_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        if not current:
            return None

        eu_aqi = current.get("european_aqi")
        eu_aqi_int = int(eu_aqi) if eu_aqi is not None else None

        return AtmosphereState(
            location_id=location_id,
            european_aqi=eu_aqi_int,
            european_aqi_label=EUROPEAN_AQI_LABELS.get(eu_aqi_int) if eu_aqi_int else None,
            us_aqi=int(current["us_aqi"]) if current.get("us_aqi") is not None else None,
            pm2_5=current.get("pm2_5"),
            pm10=current.get("pm10"),
            ozone=current.get("ozone"),
            nitrogen_dioxide=current.get("nitrogen_dioxide"),
            sulphur_dioxide=current.get("sulphur_dioxide"),
            carbon_monoxide=current.get("carbon_monoxide"),
        )

    # --- Derived atmospheric quantities ---

    @staticmethod
    def _dew_point(temp_c: float, humidity_pct: float) -> float:
        """Dew point from Magnus formula.

        Td = 243.04 × α / (17.625 − α)
        where α = ln(RH/100) + 17.625 × T / (243.04 + T)

        Accuracy: ±0.4°C for T in [-40, 50]°C and RH in [1, 100]%.
        """
        rh = max(1.0, min(100.0, humidity_pct))  # clamp to avoid log(0)
        alpha = log(rh / 100.0) + 17.625 * temp_c / (243.04 + temp_c)
        dew = 243.04 * alpha / (17.625 - alpha)
        return round(dew, 1)

    @staticmethod
    def _feels_like(temp_c: float, wind_kmh: float, humidity_pct: float) -> float:
        """Perceived temperature: wind chill or heat index.

        - Below 10°C with wind: Canadian wind chill formula
        - Above 27°C with humidity: simplified Steadman heat index
        - Otherwise: actual temperature
        """
        if temp_c <= 10.0 and wind_kmh > 4.8:
            # Wind chill (Environment Canada formula)
            wc = 13.12 + 0.6215 * temp_c - 11.37 * (wind_kmh ** 0.16) + 0.3965 * temp_c * (wind_kmh ** 0.16)
            return round(wc, 1)

        if temp_c >= 27.0 and humidity_pct > 40.0:
            # Simplified Steadman heat index
            t = temp_c
            r = humidity_pct
            hi = (
                -8.785
                + 1.611 * t
                + 2.339 * r
                - 0.1461 * t * r
                - 0.01231 * t * t
                - 0.01642 * r * r
                + 0.002212 * t * t * r
                + 0.000725 * t * r * r
                - 0.000003582 * t * t * r * r
            )
            return round(hi, 1)

        return round(temp_c, 1)

    @staticmethod
    def _uv_index(solar_elevation_deg: float, cloud_cover_pct: float) -> float:
        """UV index from solar elevation and cloud cover.

        Clear-sky UV approximation: UV ≈ 12 × sin(elevation)^1.3
        Cloud attenuation: UV × (1 - 0.75 × (cloud_cover/100)^3.4)

        Based on empirical models (Madronich 1993, WHO UV guidelines).
        """
        if solar_elevation_deg <= 0:
            return 0.0

        # Clear-sky UV from solar elevation
        sin_elev = max(0.0, sin(radians(solar_elevation_deg)))
        uv_clear = 12.0 * (sin_elev ** 1.3)

        # Cloud attenuation (cubic relationship — thin clouds transmit most UV)
        cloud_factor = 1.0 - 0.75 * ((cloud_cover_pct / 100.0) ** 3.4)
        cloud_factor = max(0.05, cloud_factor)  # minimum 5% transmission

        uv = uv_clear * cloud_factor
        return round(max(0.0, uv), 1)

    @staticmethod
    def _air_density(pressure_hpa: float, temp_c: float) -> float:
        """Air density from ideal gas law.

        ρ = P / (R_specific × T)
        where P in Pa, T in Kelvin, R_specific = 287.05 J/(kg·K)

        At sea level, standard conditions: ~1.225 kg/m³
        """
        p_pa = pressure_hpa * 100.0  # hPa to Pa
        t_k = temp_c + 273.15
        density = p_pa / (_R_SPECIFIC * t_k)
        return round(density, 4)
