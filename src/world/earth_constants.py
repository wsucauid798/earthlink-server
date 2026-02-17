"""Shared physical constants for Earth and the solar system.

Single source of truth for constants used across multiple world modules.
All values are standard reference values (J2000 epoch where applicable).
"""

# === Earth rotation ===
EARTH_EQUATORIAL_SPEED_KMH = 1674.4     # equatorial rotation speed at sea level (km/h)
EARTH_RADIUS_KM = 6371.0                # mean radius (km)

# Sidereal time (IAU 2006)
GMST_EPOCH_DEG = 280.46061837           # GMST at J2000.0 epoch (degrees)
GMST_RATE_DEG_PER_DAY = 360.98564736629  # sidereal rotation rate (deg/day)

# === Earth orbital ===
AU_KM = 149_597_870.7                   # 1 AU in km
EARTH_ECCENTRICITY = 0.0167086          # orbital eccentricity (J2000)
GM_SUN_KM3S2 = 1.32712440018e11        # heliocentric gravitational parameter (km³/s²)

# === Obliquity ===
OBLIQUITY_J2000_DEG = 23.4393           # mean obliquity at J2000.0
OBLIQUITY_RATE_DEG_PER_CENTURY = -0.0130  # secular change (~47"/century decrease)

# === Moon ===
SYNODIC_PERIOD_DAYS = 29.53059          # synodic month — new moon to new moon

# === Julian Day ===
J2000_JD = 2451545.0                    # Julian Day of J2000.0 epoch
JULIAN_CENTURY_DAYS = 36525.0           # days in a Julian century

# === Solar motion ===
SOLAR_MEAN_MOTION_DEG_PER_DAY = 0.9856  # 360° / 365.25 days
