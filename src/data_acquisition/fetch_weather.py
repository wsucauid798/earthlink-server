"""
Fetch historical weather data from Open-Meteo and insert into PostgreSQL.

Uses Open-Meteo's Historical Weather API to download real weather observations
for key UK locations. The data covers the past year of hourly observations.

Open-Meteo Historical API:
  https://archive-api.open-meteo.com/v1/archive
  Free, no API key, <10k calls/day for non-commercial use.

WMO Weather Codes:
  0: Clear sky, 1-3: Mainly clear/partly cloudy/overcast,
  45-48: Fog, 51-55: Drizzle, 56-57: Freezing drizzle,
  61-65: Rain, 66-67: Freezing rain, 71-75: Snowfall,
  77: Snow grains, 80-82: Rain showers, 85-86: Snow showers,
  95: Thunderstorm, 96-99: Thunderstorm with hail
"""

import asyncio
import logging
from datetime import date, datetime, timedelta

import httpx
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import async_session, engine
from db.models import Base, Location, WeatherData

logger = logging.getLogger(__name__)

OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

# WMO weather code descriptions
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


async def get_representative_locations(session: AsyncSession, max_locations: int = 50) -> list[Location]:
    """
    Select representative locations for weather data.
    Picks major cities/towns spread across the UK to give good coverage.
    """
    result = await session.execute(
        select(Location)
        .where(Location.type.in_(["capital", "city", "town"]))
        .where(Location.population.isnot(None))
        .where(Location.population > 0)
        .order_by(Location.population.desc())
        .limit(max_locations)
    )
    locations = list(result.scalars().all())
    logger.info(f"Selected {len(locations)} representative locations for weather data")
    return locations


async def fetch_weather_for_location(
    client: httpx.AsyncClient,
    lat: float,
    lng: float,
    start_date: date,
    end_date: date,
) -> dict | None:
    """Fetch historical hourly weather data from Open-Meteo for a location."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "visibility",
            "surface_pressure",
        ]),
        "timezone": "Europe/London",
    }

    try:
        resp = await client.get(OPEN_METEO_HISTORICAL_URL, params=params)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch weather for ({lat}, {lng}): {e}")
        return None


def parse_weather_response(location_id: int, data: dict) -> list[dict]:
    """Parse Open-Meteo JSON response into WeatherData rows."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        return []

    rows = []
    for i, time_str in enumerate(times):
        dt = datetime.fromisoformat(time_str)
        weather_code = hourly.get("weather_code", [None])[i]

        rows.append({
            "location_id": location_id,
            "datetime": dt,
            "temperature_c": hourly.get("temperature_2m", [None])[i],
            "precipitation_mm": hourly.get("precipitation", [None])[i],
            "humidity_pct": hourly.get("relative_humidity_2m", [None])[i],
            "wind_speed_kmh": hourly.get("wind_speed_10m", [None])[i],
            "wind_direction_deg": hourly.get("wind_direction_10m", [None])[i],
            "cloud_cover_pct": hourly.get("cloud_cover", [None])[i],
            "visibility_km": (
                hourly.get("visibility", [None])[i] / 1000.0
                if hourly.get("visibility", [None])[i] is not None
                else None
            ),
            "pressure_hpa": hourly.get("surface_pressure", [None])[i],
            "conditions": WMO_CODES.get(weather_code, f"Code {weather_code}") if weather_code is not None else None,
        })

    return rows


async def insert_weather_data(session: AsyncSession, rows: list[dict]) -> int:
    """Batch insert weather data rows."""
    batch_size = 2000
    inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        for row in batch:
            session.add(WeatherData(**row))
        await session.flush()
        inserted += len(batch)
    await session.commit()
    return inserted


async def run():
    """Main entry point: download and insert historical UK weather data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    logger.info("=== Fetching UK Weather Data ===")

    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Clear existing weather data
    async with async_session() as session:
        await session.execute(text("DELETE FROM weather_data"))
        await session.commit()
        logger.info("Cleared existing weather data")

    # Get representative locations
    async with async_session() as session:
        locations = await get_representative_locations(session)

    if not locations:
        logger.error("No locations found in database. Run fetch_geography.py first.")
        return

    # Fetch 90 days of historical data (to stay within API limits)
    end_date = date.today() - timedelta(days=5)  # Open-Meteo historical has ~5 day lag
    start_date = end_date - timedelta(days=89)

    logger.info(f"Fetching weather data from {start_date} to {end_date}")

    async with httpx.AsyncClient(timeout=60) as client:
        for i, loc in enumerate(locations):
            logger.info(f"[{i + 1}/{len(locations)}] Fetching weather for {loc.name} ({loc.lat}, {loc.lng})")

            data = await fetch_weather_for_location(client, loc.lat, loc.lng, start_date, end_date)
            if not data:
                continue

            rows = parse_weather_response(loc.id, data)
            if rows:
                async with async_session() as session:
                    count = await insert_weather_data(session, rows)
                    logger.info(f"  Inserted {count} hourly weather records for {loc.name}")

            # Rate limiting: be respectful to Open-Meteo
            await asyncio.sleep(1.0)

    logger.info("=== Weather data acquisition complete ===")


if __name__ == "__main__":
    asyncio.run(run())
