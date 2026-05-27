"""Orbital mechanics — Earth's position in the solar system.

Keplerian orbital elements with first-order corrections from Jean Meeus'
"Astronomical Algorithms". Pure math — datetime -> orbital state.

The Earth IS orbiting the Sun. These calculations give its true position.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import (
    asin,
    cos,
    degrees,
    radians,
    sin,
    sqrt,
)

from world.celestial import (
    julian_day_from_datetime,
    _sun_mean_elements,
    solar_ecliptic_longitude,
    season_from_ecliptic_longitude,
)
from world.earth_constants import (
    AU_KM,
    EARTH_ECCENTRICITY,
    GM_SUN_KM3S2,
    J2000_JD,
    JULIAN_CENTURY_DAYS,
    OBLIQUITY_J2000_DEG,
    OBLIQUITY_RATE_DEG_PER_CENTURY,
)


@dataclass
class OrbitalState:
    """Earth's orbital state at a given moment."""
    # Position
    earth_sun_distance_km: float      # current distance to Sun
    earth_sun_distance_au: float      # same in AU
    orbital_position_deg: float       # ecliptic longitude (0 = vernal equinox)
    true_anomaly_deg: float           # angle from perihelion
    mean_anomaly_deg: float           # mean angular position

    # Velocity
    orbital_speed_kms: float          # current orbital speed (km/s)

    # Axial tilt
    axial_tilt_deg: float             # current obliquity
    solar_declination_deg: float      # sun's declination (tilt effect)

    # Season (from orbital state, not calendar)
    season: str
    season_progress: float            # 0.0-1.0 progress through current season
    days_to_perihelion: float
    days_to_next_event: float         # days to next solstice/equinox
    next_event: str                   # name of next astronomical event

    # Orbital elements
    eccentricity: float
    semi_major_axis_km: float

    def to_dict(self) -> dict:
        return {
            "earth_sun_distance_km": self.earth_sun_distance_km,
            "earth_sun_distance_au": self.earth_sun_distance_au,
            "orbital_position_deg": self.orbital_position_deg,
            "true_anomaly_deg": self.true_anomaly_deg,
            "mean_anomaly_deg": self.mean_anomaly_deg,
            "orbital_speed_kms": self.orbital_speed_kms,
            "axial_tilt_deg": self.axial_tilt_deg,
            "solar_declination_deg": self.solar_declination_deg,
            "season": self.season,
            "season_progress": self.season_progress,
            "days_to_perihelion": self.days_to_perihelion,
            "days_to_next_event": self.days_to_next_event,
            "next_event": self.next_event,
            "eccentricity": self.eccentricity,
            "semi_major_axis_km": self.semi_major_axis_km,
        }


class Orbital:
    """Stateless orbital mechanics engine — pure computation per request.

    Same pattern as Geophysics: no state, no cache, no refresh.
    Give it a datetime, get back orbital state.
    """

    def compute(self, dt: datetime) -> OrbitalState:
        """Compute Earth's orbital state for a given datetime (UTC)."""
        jd = julian_day_from_datetime(dt)
        n = jd - J2000_JD
        T = n / JULIAN_CENTURY_DAYS

        # Mean orbital elements from celestial math
        _, g, _ = _sun_mean_elements(jd)
        mean_anomaly_deg = degrees(g) % 360

        # Obliquity (axial tilt) with secular variation (~47"/century decrease)
        obliquity_deg = OBLIQUITY_J2000_DEG + OBLIQUITY_RATE_DEG_PER_CENTURY * T

        # True anomaly via equation of center (first-order Kepler)
        e = EARTH_ECCENTRICITY
        eoc = 2 * e * sin(g) + 1.25 * e**2 * sin(2 * g)
        true_anomaly = g + eoc  # radians
        true_anomaly_deg_val = degrees(true_anomaly) % 360

        # Earth-Sun distance: r = a(1 - e^2) / (1 + e cos v)
        r_au = (1 - e**2) / (1 + e * cos(true_anomaly))
        r_km = r_au * AU_KM

        # Orbital speed via vis-viva: v = sqrt(GM * (2/r - 1/a))
        v_kms = sqrt(GM_SUN_KM3S2 * (2.0 / r_km - 1.0 / AU_KM))

        # Sun's ecliptic longitude — shared computation from celestial.py
        ecliptic_lon = solar_ecliptic_longitude(jd)

        # Solar declination from ecliptic longitude and obliquity
        solar_decl = degrees(
            asin(sin(radians(obliquity_deg)) * sin(radians(ecliptic_lon)))
        )

        # Season from ecliptic longitude — shared logic from celestial.py
        season, progress, days_to_event, next_event = season_from_ecliptic_longitude(ecliptic_lon)

        # Days to perihelion (true anomaly = 0)
        days_to_perihelion = ((360 - true_anomaly_deg_val) % 360) / 0.9856

        return OrbitalState(
            earth_sun_distance_km=round(r_km, 0),
            earth_sun_distance_au=round(r_au, 9),
            orbital_position_deg=round(ecliptic_lon, 5),
            true_anomaly_deg=round(true_anomaly_deg_val, 5),
            mean_anomaly_deg=round(mean_anomaly_deg, 5),
            orbital_speed_kms=round(v_kms, 6),
            axial_tilt_deg=round(obliquity_deg, 4),
            solar_declination_deg=round(solar_decl, 4),
            season=season,
            season_progress=round(progress, 4),
            days_to_perihelion=round(days_to_perihelion, 1),
            days_to_next_event=round(days_to_event, 1),
            next_event=next_event,
            eccentricity=e,
            semi_major_axis_km=round(AU_KM, 0),
        )
