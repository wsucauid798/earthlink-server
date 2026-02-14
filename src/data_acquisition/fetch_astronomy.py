"""
Compute astronomy data (sunrise, sunset, day length) for UK locations
and insert into PostgreSQL.

Uses astronomical algorithms to compute these values from coordinates and date.
No external API needed — pure math from well-established formulas.

Based on NOAA solar calculator algorithms and Jean Meeus' "Astronomical Algorithms".
"""

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from math import acos, asin, atan2, cos, degrees, fmod, pi, radians, sin

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import async_session, engine
from db.models import AstronomyData, Base, Location

logger = logging.getLogger(__name__)


def julian_day(d: date) -> float:
    """Calculate Julian Day Number from a calendar date."""
    y = d.year
    m = d.month
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d.day + B - 1524.5


def solar_declination(jd: float) -> float:
    """Calculate solar declination in radians."""
    n = jd - 2451545.0  # days since J2000.0
    L = fmod(280.460 + 0.9856474 * n, 360.0)  # mean longitude
    g = radians(fmod(357.528 + 0.9856003 * n, 360.0))  # mean anomaly
    ecliptic_lon = radians(L + 1.915 * sin(g) + 0.020 * sin(2 * g))
    obliquity = radians(23.439 - 0.0000004 * n)
    return asin(sin(obliquity) * sin(ecliptic_lon))


def equation_of_time(jd: float) -> float:
    """Calculate the equation of time in minutes."""
    n = jd - 2451545.0
    L = fmod(280.460 + 0.9856474 * n, 360.0)
    g = radians(fmod(357.528 + 0.9856003 * n, 360.0))
    ecliptic_lon = L + 1.915 * sin(g) * 180 / pi + 0.020 * sin(2 * g) * 180 / pi
    obliquity = radians(23.439 - 0.0000004 * n)

    RA = degrees(atan2(cos(obliquity) * sin(radians(ecliptic_lon)), cos(radians(ecliptic_lon))))
    RA = fmod(RA + 360, 360.0)

    eot = (L - RA) * 4  # in minutes (1 degree = 4 minutes of time)
    if eot > 720:
        eot -= 1440
    elif eot < -720:
        eot += 1440
    return eot


def sunrise_sunset(lat: float, lng: float, d: date) -> tuple[datetime | None, datetime | None]:
    """
    Calculate sunrise and sunset times for a given location and date.
    Returns (sunrise, sunset) as UTC datetime objects.
    Returns (None, None) for polar day/night.
    """
    jd = julian_day(d)
    decl = solar_declination(jd)
    eot = equation_of_time(jd)

    lat_rad = radians(lat)

    # Hour angle at sunrise/sunset (when solar zenith = 90.833 degrees)
    cos_ha = (cos(radians(90.833)) - sin(lat_rad) * sin(decl)) / (cos(lat_rad) * cos(decl))

    if cos_ha > 1:
        return None, None  # Polar night — sun never rises
    if cos_ha < -1:
        return None, None  # Polar day — sun never sets

    ha = degrees(acos(cos_ha))

    # Solar noon in minutes from midnight UTC
    solar_noon = 720 - 4 * lng - eot

    sunrise_min = solar_noon - ha * 4
    sunset_min = solar_noon + ha * 4

    base = datetime(d.year, d.month, d.day)
    sunrise = base + timedelta(minutes=sunrise_min)
    sunset = base + timedelta(minutes=sunset_min)

    return sunrise, sunset


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
                rise, sett = sunrise_sunset(loc.lat, loc.lng, d)

                day_length = None
                if rise and sett:
                    day_length = round((sett - rise).total_seconds() / 3600, 2)

                rows.append(AstronomyData(
                    location_id=loc.id,
                    date=d,
                    sunrise=rise,
                    sunset=sett,
                    day_length_hours=day_length,
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
