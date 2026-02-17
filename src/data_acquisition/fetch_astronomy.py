"""
Compute astronomy data (sun, moon, twilight) for UK locations
and insert into PostgreSQL.

Uses Jean Meeus algorithms from world/celestial.py — pure math from
well-established formulas. No external API needed.

Based on NOAA solar calculator algorithms and Jean Meeus' "Astronomical Algorithms".
"""

import asyncio
import logging
from datetime import date, datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import async_session, engine
from db.models import AstronomyData, Base, Location
from world.celestial import (
    moon_age,
    moon_illumination,
    moon_phase,
    moonrise_moonset,
    solar_noon_time,
    sunrise_sunset,
    twilight_times,
)

logger = logging.getLogger(__name__)


async def get_representative_locations(session: AsyncSession, max_locations: int = 50) -> list[Location]:
    """Select the same representative locations used for weather data."""
    result = await session.execute(
        select(Location)
        .where(Location.type.in_(["capital", "city", "town"]))
        .where(Location.population.isnot(None))
        .where(Location.population > 0)
        .order_by(Location.population.desc())
        .limit(max_locations)
    )
    return list(result.scalars().all())


async def run():
    """Main entry point: compute and insert astronomy data for UK locations."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    logger.info("=== Computing UK Astronomy Data ===")

    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Clear existing astronomy data
    async with async_session() as session:
        await session.execute(text("DELETE FROM astronomy_data"))
        await session.commit()
        logger.info("Cleared existing astronomy data")

    # Get representative locations
    async with async_session() as session:
        locations = await get_representative_locations(session)

    if not locations:
        logger.error("No locations found in database. Run fetch_geography.py first.")
        return

    # Compute for one full year
    today = date.today()
    start_date = today - timedelta(days=90)
    end_date = today + timedelta(days=275)  # ~1 year total coverage
    num_days = (end_date - start_date).days

    logger.info(f"Computing astronomy data from {start_date} to {end_date} ({num_days} days)")

    total_rows = 0
    async with async_session() as session:
        for i, loc in enumerate(locations):
            rows = []
            for day_offset in range(num_days):
                d = start_date + timedelta(days=day_offset)

                # Sun
                rise, sett = sunrise_sunset(loc.lat, loc.lng, d)
                day_length = None
                if rise and sett:
                    day_length = round((sett - rise).total_seconds() / 3600, 2)

                noon = solar_noon_time(loc.lat, loc.lng, d)

                # Twilight
                civil_dawn, civil_dusk = twilight_times(loc.lat, loc.lng, d, 96.0)
                nautical_dawn, nautical_dusk = twilight_times(loc.lat, loc.lng, d, 102.0)

                # Moon (use noon UTC as reference time for phase on this date)
                dt_noon = datetime(d.year, d.month, d.day, 12, 0)
                m_phase = moon_phase(dt_noon)
                m_illum = moon_illumination(dt_noon)
                m_age = moon_age(dt_noon)
                m_rise, m_set = moonrise_moonset(loc.lat, loc.lng, d)

                rows.append(AstronomyData(
                    location_id=loc.id,
                    date=d,
                    sunrise=rise,
                    sunset=sett,
                    solar_noon=noon,
                    day_length_hours=day_length,
                    civil_dawn=civil_dawn,
                    civil_dusk=civil_dusk,
                    nautical_dawn=nautical_dawn,
                    nautical_dusk=nautical_dusk,
                    moon_phase=m_phase,
                    moon_illumination_pct=m_illum,
                    moon_age_days=m_age,
                    moonrise=m_rise,
                    moonset=m_set,
                ))

            session.add_all(rows)
            await session.flush()
            total_rows += len(rows)

            if (i + 1) % 10 == 0:
                logger.info(f"  [{i + 1}/{len(locations)}] Computed astronomy for {loc.name}")

        await session.commit()

    logger.info(f"Inserted {total_rows} astronomy records for {len(locations)} locations")
    logger.info("=== Astronomy data acquisition complete ===")


if __name__ == "__main__":
    asyncio.run(run())
