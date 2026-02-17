"""
Fetch historical wind data from Open-Meteo and insert into PostgreSQL.

Uses Open-Meteo's Historical Weather API for wind-specific parameters.
Covers ~100 locations with 90 days of hourly wind data.

Open-Meteo Historical API:
  https://archive-api.open-meteo.com/v1/archive
  Free, no API key, <10k calls/day for non-commercial use.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta

import httpx
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import async_session, engine
from db.models import Base, Location, WindData

logger = logging.getLogger(__name__)

OPEN_METEO_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

WIND_HOURLY_PARAMS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "wind_speed_100m",      # Historical API uses 100m, not 80m
    "wind_direction_100m",
    "surface_pressure",
]


async def get_wind_stations(session: AsyncSession, max_stations: int = 100) -> list[Location]:
    """Select wind station locations with broad geographic coverage.

    Strategy:
    1. Top 50 by population (overlap with weather stations).
    2. Fill geographic gaps using a grid over UK/Ireland.
    """
    # Phase 1: top populated places
    result = await session.execute(
        select(Location)
        .where(Location.type.in_(["capital", "city", "town", "village", "settlement"]))
        .where(Location.population.isnot(None))
        .where(Location.population > 0)
        .order_by(Location.population.desc())
        .limit(50)
    )
    top_by_pop = list(result.scalars().all())
    selected = list(top_by_pop)
    selected_ids = {loc.id for loc in selected}

    if len(selected) >= max_stations:
        logger.info(f"Selected {len(selected)} wind stations (population only)")
        return selected[:max_stations]

    # Phase 2: geographic grid fill
    result = await session.execute(
        select(Location)
        .where(Location.type.in_(["capital", "city", "town", "village", "settlement"]))
        .where(Location.population.isnot(None))
        .where(Location.population > 0)
    )
    all_candidates = list(result.scalars().all())

    LAT_MIN, LAT_MAX = 49.5, 60.5
    LNG_MIN, LNG_MAX = -11.0, 2.0
    GRID_ROWS, GRID_COLS = 10, 10
    lat_step = (LAT_MAX - LAT_MIN) / GRID_ROWS
    lng_step = (LNG_MAX - LNG_MIN) / GRID_COLS

    preferred_terrain = {"highland", "water", "woodland"}

    for row in range(GRID_ROWS):
        if len(selected) >= max_stations:
            break
        for col in range(GRID_COLS):
            if len(selected) >= max_stations:
                break

            cell_lat_min = LAT_MIN + row * lat_step
            cell_lat_max = cell_lat_min + lat_step
            cell_lng_min = LNG_MIN + col * lng_step
            cell_lng_max = cell_lng_min + lng_step

            # Check if any selected station covers this cell
            has_station = any(
                cell_lat_min <= loc.lat < cell_lat_max
                and cell_lng_min <= loc.lng < cell_lng_max
                for loc in selected
            )
            if has_station:
                continue

            # Find candidates in this cell
            in_cell = [
                loc for loc in all_candidates
                if cell_lat_min <= loc.lat < cell_lat_max
                and cell_lng_min <= loc.lng < cell_lng_max
                and loc.id not in selected_ids
            ]
            if not in_cell:
                continue

            # Prefer interesting terrain, then highest population
            preferred = [l for l in in_cell if l.terrain in preferred_terrain]
            pick = max(
                preferred if preferred else in_cell,
                key=lambda l: l.population or 0,
            )
            selected.append(pick)
            selected_ids.add(pick.id)

    logger.info(f"Selected {len(selected)} wind stations ({len(top_by_pop)} by population + {len(selected) - len(top_by_pop)} by geography)")
    return selected[:max_stations]


async def fetch_wind_for_location(
    client: httpx.AsyncClient,
    lat: float,
    lng: float,
    start_date: date,
    end_date: date,
) -> dict | None:
    """Fetch historical hourly wind data from Open-Meteo for a location."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(WIND_HOURLY_PARAMS),
        "timezone": "Europe/London",
    }

    try:
        resp = await client.get(OPEN_METEO_HISTORICAL_URL, params=params)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch wind for ({lat}, {lng}): {e}")
        return None


def parse_wind_response(location_id: int, data: dict) -> list[dict]:
    """Parse Open-Meteo JSON response into WindData rows."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        return []

    rows = []
    for i, time_str in enumerate(times):
        dt = datetime.fromisoformat(time_str)
        rows.append({
            "location_id": location_id,
            "datetime": dt,
            "wind_speed_10m": hourly.get("wind_speed_10m", [None])[i],
            "wind_direction_10m": hourly.get("wind_direction_10m", [None])[i],
            "wind_gusts_10m": hourly.get("wind_gusts_10m", [None])[i],
            "wind_speed_upper": hourly.get("wind_speed_100m", [None])[i],
            "wind_direction_upper": hourly.get("wind_direction_100m", [None])[i],
            "upper_height_m": 100,  # Historical API provides 100m
            "pressure_hpa": hourly.get("surface_pressure", [None])[i],
        })

    return rows


async def insert_wind_data(session: AsyncSession, rows: list[dict]) -> int:
    """Batch insert wind data rows."""
    batch_size = 2000
    inserted = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        for row in batch:
            session.add(WindData(**row))
        await session.flush()
        inserted += len(batch)
    await session.commit()
    return inserted


async def run():
    """Main entry point: download and insert historical wind data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    logger.info("=== Fetching Wind Data ===")

    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Clear existing wind data
    async with async_session() as session:
        await session.execute(text("DELETE FROM wind_data"))
        await session.commit()
        logger.info("Cleared existing wind data")

    # Get wind station locations
    async with async_session() as session:
        locations = await get_wind_stations(session)

    if not locations:
        logger.error("No locations found in database. Run fetch_geography.py first.")
        return

    # Fetch 90 days of historical data (Open-Meteo historical has ~5 day lag)
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=89)

    logger.info(f"Fetching wind data from {start_date} to {end_date}")

    async with httpx.AsyncClient(timeout=60) as client:
        for i, loc in enumerate(locations):
            logger.info(f"[{i + 1}/{len(locations)}] Fetching wind for {loc.name} ({loc.lat}, {loc.lng})")

            data = await fetch_wind_for_location(client, loc.lat, loc.lng, start_date, end_date)
            if not data:
                continue

            rows = parse_wind_response(loc.id, data)
            if rows:
                async with async_session() as session:
                    count = await insert_wind_data(session, rows)
                    logger.info(f"  Inserted {count} hourly wind records for {loc.name}")

            # Rate limiting: be respectful to Open-Meteo
            await asyncio.sleep(1.0)

    logger.info("=== Wind data acquisition complete ===")


if __name__ == "__main__":
    asyncio.run(run())
