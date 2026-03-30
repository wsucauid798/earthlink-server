"""Climate — long-term atmospheric patterns at a location.

Climate is not weather. Weather is what's happening now. Climate is
what normally happens here — average temperatures, rainfall patterns,
climate zone. Derived from real historical data via Open-Meteo Archive API.

Climate data is fetched once per location and cached permanently.
It does not change on a tick-by-tick basis.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REFERENCE_YEAR_START = "2023-01-01"
REFERENCE_YEAR_END = "2023-12-31"


@dataclass
class ClimateState:
    """Climate profile for a location."""
    # Monthly averages (index 0 = January, 11 = December)
    monthly_temp_c: list[float] = field(default_factory=list)
    monthly_precip_mm: list[float] = field(default_factory=list)

    # Annual summaries
    annual_temp_c: float = 0.0
    annual_precip_mm: float = 0.0
    coldest_month_temp_c: float = 0.0
    warmest_month_temp_c: float = 0.0

    # Classification
    koppen_zone: str = ""
    koppen_description: str = ""
    climate_type: str = ""

    def to_dict(self) -> dict:
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        monthly = []
        for i in range(min(12, len(self.monthly_temp_c))):
            monthly.append({
                "month": months[i],
                "avg_temp_c": round(self.monthly_temp_c[i], 1),
                "precip_mm": round(self.monthly_precip_mm[i], 0),
            })
        return {
            "monthly": monthly,
            "annual_temp_c": round(self.annual_temp_c, 1),
            "annual_precip_mm": round(self.annual_precip_mm, 0),
            "coldest_month_temp_c": round(self.coldest_month_temp_c, 1),
            "warmest_month_temp_c": round(self.warmest_month_temp_c, 1),
            "koppen_zone": self.koppen_zone,
            "koppen_description": self.koppen_description,
            "climate_type": self.climate_type,
        }


def _classify_koppen(
    monthly_temps: list[float],
    monthly_precips: list[float],
    lat: float,
) -> tuple[str, str, str]:
    """Classify climate using simplified Koppen system.

    Returns (zone_code, description, climate_type).
    """
    if len(monthly_temps) < 12 or len(monthly_precips) < 12:
        return "?", "Insufficient data", "unknown"

    annual_temp = sum(monthly_temps) / 12
    annual_precip = sum(monthly_precips)
    coldest = min(monthly_temps)
    warmest = max(monthly_temps)

    # Determine precipitation threshold for B climates
    # (depends on whether rain falls mostly in winter or summer)
    winter_months = [10, 11, 0, 1, 2, 3] if lat >= 0 else [4, 5, 6, 7, 8, 9]
    winter_precip = sum(monthly_precips[m] for m in winter_months)
    winter_pct = winter_precip / max(1, annual_precip)

    if winter_pct >= 0.7:
        threshold = 20 * annual_temp
    elif winter_pct <= 0.3:
        threshold = 20 * annual_temp + 280
    else:
        threshold = 20 * annual_temp + 140

    # E — Polar
    if warmest < 10:
        if warmest < 0:
            return "EF", "Ice cap", "polar"
        return "ET", "Tundra", "polar"

    # B — Arid
    if annual_precip < threshold:
        if annual_precip < threshold / 2:
            if annual_temp >= 18:
                return "BWh", "Hot desert", "arid"
            return "BWk", "Cold desert", "arid"
        if annual_temp >= 18:
            return "BSh", "Hot semi-arid", "arid"
        return "BSk", "Cold semi-arid", "arid"

    # A — Tropical
    if coldest >= 18:
        driest = min(monthly_precips)
        if driest >= 60:
            return "Af", "Tropical rainforest", "tropical"
        if driest >= 100 - annual_precip / 25:
            return "Am", "Tropical monsoon", "tropical"
        return "Aw", "Tropical savanna", "tropical"

    # D — Continental
    if coldest < -3 and warmest >= 10:
        driest_summer = min(monthly_precips[m] for m in ([6, 7, 8] if lat >= 0 else [0, 1, 2]))
        driest_winter = min(monthly_precips[m] for m in ([0, 1, 2] if lat >= 0 else [6, 7, 8]))
        wettest_summer = max(monthly_precips[m] for m in ([6, 7, 8] if lat >= 0 else [0, 1, 2]))
        wettest_winter = max(monthly_precips[m] for m in ([0, 1, 2] if lat >= 0 else [6, 7, 8]))

        if driest_summer < 30 and driest_summer < wettest_winter / 3:
            precip_code = "s"
        elif driest_winter < wettest_summer / 10:
            precip_code = "w"
        else:
            precip_code = "f"

        if warmest >= 22:
            return f"D{precip_code}a", f"Continental, hot summer", "continental"
        warm_months = sum(1 for t in monthly_temps if t >= 10)
        if warm_months >= 4:
            return f"D{precip_code}b", f"Continental, warm summer", "continental"
        if coldest < -38:
            return f"D{precip_code}d", f"Continental, very cold winter", "continental"
        return f"D{precip_code}c", f"Continental, cold summer", "continental"

    # C — Temperate
    if coldest >= -3 and coldest < 18 and warmest >= 10:
        driest_summer = min(monthly_precips[m] for m in ([6, 7, 8] if lat >= 0 else [0, 1, 2]))
        driest_winter = min(monthly_precips[m] for m in ([0, 1, 2] if lat >= 0 else [6, 7, 8]))
        wettest_summer = max(monthly_precips[m] for m in ([6, 7, 8] if lat >= 0 else [0, 1, 2]))
        wettest_winter = max(monthly_precips[m] for m in ([0, 1, 2] if lat >= 0 else [6, 7, 8]))

        if driest_summer < 30 and driest_summer < wettest_winter / 3:
            precip_code = "s"
        elif driest_winter < wettest_summer / 10:
            precip_code = "w"
        else:
            precip_code = "f"

        if warmest >= 22:
            return f"C{precip_code}a", f"Temperate, hot summer", "temperate"
        warm_months = sum(1 for t in monthly_temps if t >= 10)
        if warm_months >= 4:
            return f"C{precip_code}b", f"Temperate, warm summer", "temperate"
        return f"C{precip_code}c", f"Temperate, cold summer", "temperate"

    return "?", "Unclassified", "unknown"


class Climate:
    """Climate system — resolves and caches climate profiles for locations."""

    def __init__(self):
        self._cache: dict[int, ClimateState] = {}
        self._http = httpx.AsyncClient(timeout=15.0)

    async def get_climate(self, location_id: int, lat: float, lng: float) -> ClimateState:
        """Get climate profile for a location. Fetches from API on first call, cached after."""
        if location_id in self._cache:
            return self._cache[location_id]

        state = await self._fetch_climate(lat, lng)
        self._cache[location_id] = state
        return state

    async def _fetch_climate(self, lat: float, lng: float) -> ClimateState:
        """Fetch climate normals from Open-Meteo Archive API."""
        try:
            r = await self._http.get(ARCHIVE_URL, params={
                "latitude": lat,
                "longitude": lng,
                "start_date": REFERENCE_YEAR_START,
                "end_date": REFERENCE_YEAR_END,
                "daily": "temperature_2m_mean,precipitation_sum",
                "timezone": "UTC",
            })
            r.raise_for_status()
        except Exception as e:
            logger.warning(f"Climate fetch failed for ({lat}, {lng}): {e}")
            return ClimateState()

        data = r.json()
        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_mean", [])
        precip = daily.get("precipitation_sum", [])
        dates = daily.get("time", [])

        monthly_temp = defaultdict(list)
        monthly_precip = defaultdict(float)
        for i, d in enumerate(dates):
            m = int(d[5:7]) - 1  # 0-indexed
            if i < len(temps) and temps[i] is not None:
                monthly_temp[m].append(temps[i])
            if i < len(precip) and precip[i] is not None:
                monthly_precip[m] += precip[i]

        avg_temps = []
        precips = []
        for m in range(12):
            avg_temps.append(
                sum(monthly_temp[m]) / len(monthly_temp[m]) if monthly_temp[m] else 0.0
            )
            precips.append(monthly_precip[m])

        annual_temp = sum(avg_temps) / 12
        annual_precip = sum(precips)

        zone, description, climate_type = _classify_koppen(avg_temps, precips, lat)

        state = ClimateState(
            monthly_temp_c=avg_temps,
            monthly_precip_mm=precips,
            annual_temp_c=annual_temp,
            annual_precip_mm=annual_precip,
            coldest_month_temp_c=min(avg_temps),
            warmest_month_temp_c=max(avg_temps),
            koppen_zone=zone,
            koppen_description=description,
            climate_type=climate_type,
        )

        logger.info(
            f"Climate resolved: ({lat:.2f}, {lng:.2f}) -> {zone} ({description}), "
            f"annual {annual_temp:.1f}C, {annual_precip:.0f}mm"
        )
        return state

    async def close(self):
        await self._http.aclose()
