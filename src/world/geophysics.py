"""Geophysics — gravity, magnetic field, rotation speed.

Pure computation from physics formulas. No external data needed.

Gravity: WGS84 Somigliana formula + free-air correction + tidal variation.
Magnetic field: Simplified IGRF dipole model.
Rotation speed: v = 1674.4 × cos(lat) at sea level, altitude-corrected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import atan, atan2, cos, degrees, pi, radians, sin, sqrt

from world.celestial import (
    julian_day_from_datetime,
    local_sidereal_time,
    moon_declination,
    moon_right_ascension,
)
from world.earth_constants import (
    EARTH_EQUATORIAL_SPEED_KMH,
    EARTH_RADIUS_KM,
)

# WGS84 constants
_G_EQUATOR = 9.7803253359          # m/s² at equator
_K_SOMIGLIANA = 0.00193185265241   # Somigliana constant
_E2 = 0.00669437999014             # first eccentricity squared
_FREE_AIR = 3.086e-6               # m/s² per metre of elevation

# Magnetic dipole model (simplified IGRF)
_B0 = 30.0  # µT — equatorial surface field strength


@dataclass
class GeophysicsState:
    """Geophysical state at a location for a given moment."""
    gravity_ms2: float           # total gravity (base + tidal)
    gravity_base_ms2: float      # WGS84 base gravity (no tidal)
    tidal_variation_ms2: float   # lunar/solar tidal perturbation
    magnetic_field_ut: float     # total field strength (µT)
    magnetic_declination_deg: float  # deviation from true north
    magnetic_inclination_deg: float  # dip angle
    rotation_speed_kmh: float    # surface rotation speed


class Geophysics:
    """Stateless geophysics engine — pure computation per request."""

    def compute(
        self,
        lat: float,
        lng: float,
        elevation: float | None,
        dt: datetime,
    ) -> GeophysicsState:
        """Compute all geophysical quantities for a location and time."""
        elev = elevation if elevation is not None else 0.0

        g_base = self._wgs84_gravity(lat, elev)
        g_tidal = self._tidal_gravity(lat, lng, dt)
        field_ut, decl, incl = self._magnetic_field(lat, lng)
        rotation = self._rotation_speed(lat, elev)

        return GeophysicsState(
            gravity_ms2=round(g_base + g_tidal, 6),
            gravity_base_ms2=round(g_base, 6),
            tidal_variation_ms2=round(g_tidal, 7),
            magnetic_field_ut=round(field_ut, 1),
            magnetic_declination_deg=round(decl, 2),
            magnetic_inclination_deg=round(incl, 2),
            rotation_speed_kmh=round(rotation, 1),
        )

    @staticmethod
    def _wgs84_gravity(lat: float, elevation: float) -> float:
        """WGS84 Somigliana gravity formula with free-air correction.

        g(φ) = 9.7803253359 × (1 + 0.00193185265241 × sin²φ) / √(1 − 0.00669437999014 × sin²φ)
        g(φ, h) = g(φ) − 3.086×10⁻⁶ × h
        """
        sin2 = sin(radians(lat)) ** 2
        g = _G_EQUATOR * (1 + _K_SOMIGLIANA * sin2) / sqrt(1 - _E2 * sin2)
        # Free-air correction: gravity decreases with altitude
        g -= _FREE_AIR * elevation
        return g

    @staticmethod
    def _tidal_gravity(lat: float, lng: float, dt: datetime) -> float:
        """Tidal gravity variation from lunar position.

        Uses actual lunar coordinates from celestial mechanics and Earth's
        rotation (via GMST → LST) to compute the Moon's zenith distance
        at the observation point.

        Tidal acceleration follows the P₂(cos z) Legendre polynomial —
        the dominant term in the tidal potential expansion.

        Peak tidal acceleration ≈ 1.1 µm/s² (lunar component only).
        """
        jd = julian_day_from_datetime(dt)

        # Moon's equatorial coordinates from celestial mechanics
        moon_decl = moon_declination(jd)
        moon_ra = moon_right_ascension(jd)

        # Moon's hour angle from Earth's rotation: HA = LST - RA
        lst_deg = local_sidereal_time(jd, lng)
        moon_ha = radians(lst_deg) - moon_ra

        lat_rad = radians(lat)

        # Moon's zenith distance at the observer
        cos_z = (sin(lat_rad) * sin(moon_decl)
                 + cos(lat_rad) * cos(moon_decl) * cos(moon_ha))

        # Tidal acceleration: P₂(cos z) = (3cos²z - 1) / 2
        # Ranges from -0.5 (Moon on horizon) to +1.0 (Moon overhead/nadir)
        return 1.1e-6 * (3 * cos_z**2 - 1) / 2

    @staticmethod
    def _magnetic_field(lat: float, lng: float) -> tuple[float, float, float]:
        """Simplified IGRF dipole magnetic field model.

        Returns (field_strength_ut, declination_deg, inclination_deg).

        B(φ) = B₀ × √(1 + 3 sin²φ)  — total field strength
        I = atan(2 × tan φ)           — inclination (dip angle)
        D = simplified declination     — for UK, ~0° to -2°
        """
        sin2 = sin(radians(lat)) ** 2
        field = _B0 * sqrt(1 + 3 * sin2)

        # Inclination (dip angle)
        inclination = degrees(atan(2 * sin(radians(lat)) / cos(radians(lat)))) if abs(lat) < 89.9 else (90.0 if lat > 0 else -90.0)

        # Simplified declination model for UK/Europe
        # The magnetic north pole is in northern Canada (~80°N, 73°W as of 2025)
        # For UK (50-60°N, -11 to 2°E): declination ranges ~0° to -2°
        # Simple linear model: D ≈ -1.0 + 0.05 × (lng + 5)
        declination = -1.0 + 0.05 * (lng + 5.0)
        # Clamp to reasonable range
        declination = max(-15.0, min(15.0, declination))

        return field, declination, inclination

    @staticmethod
    def _rotation_speed(lat: float, elevation: float) -> float:
        """Surface rotation speed at a latitude and elevation.

        v = 1674.4 × cos(φ) km/h at sea level
        Altitude correction: v_h = (R + h) / R × v
        """
        v = EARTH_EQUATORIAL_SPEED_KMH * cos(radians(lat))
        # Altitude correction
        h_km = elevation / 1000.0
        v *= (EARTH_RADIUS_KM + h_km) / EARTH_RADIUS_KM
        return v
