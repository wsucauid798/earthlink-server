"""Celestial mechanics — pure math engine for sun and moon positions.

All algorithms based on Jean Meeus' "Astronomical Algorithms" and NOAA
solar calculator formulas. No external APIs — coordinate + datetime → result.

For celestial mechanics, the math IS the real-world data. These are the
same algorithms used by observatories and navigation systems.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from math import (
    acos,
    asin,
    atan2,
    cos,
    degrees,
    floor,
    fmod,
    pi,
    radians,
    sin,
    sqrt,
    tan,
)

from world.earth_constants import (
    GMST_EPOCH_DEG,
    GMST_RATE_DEG_PER_DAY,
    J2000_JD,
    JULIAN_CENTURY_DAYS,
    SYNODIC_PERIOD_DAYS,
)

# ---------------------------------------------------------------------------
# Julian Day helpers
# ---------------------------------------------------------------------------

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


def julian_day_from_datetime(dt: datetime) -> float:
    """Julian Day from a full datetime (includes fractional day)."""
    jd = julian_day(dt.date())
    frac = (dt.hour + dt.minute / 60 + dt.second / 3600) / 24.0
    return jd + frac


# ---------------------------------------------------------------------------
# Earth rotation — sidereal time
# ---------------------------------------------------------------------------

def greenwich_mean_sidereal_time(jd: float) -> float:
    """Greenwich Mean Sidereal Time in degrees for any Julian Day.

    The fundamental quantity encoding Earth's rotation. GMST tells you
    how far Earth has rotated from the vernal equinox direction.

    IAU formula with quadratic correction.

    Returns: GMST in degrees [0, 360)
    """
    T = (jd - J2000_JD) / JULIAN_CENTURY_DAYS
    gmst = fmod(
        GMST_EPOCH_DEG
        + GMST_RATE_DEG_PER_DAY * (jd - J2000_JD)
        + 0.000387933 * T * T
        - T * T * T / 38710000.0,
        360.0,
    )
    if gmst < 0:
        gmst += 360.0
    return gmst


def local_sidereal_time(jd: float, lng: float) -> float:
    """Local Sidereal Time in degrees for a Julian Day and longitude.

    LST = GMST + longitude. This is the right ascension currently
    on the local meridian — the key to converting between celestial
    coordinates and local horizontal coordinates.

    Returns: LST in degrees [0, 360)
    """
    lst = (greenwich_mean_sidereal_time(jd) + lng) % 360
    if lst < 0:
        lst += 360.0
    return lst


# ---------------------------------------------------------------------------
# Earth rotation state
# ---------------------------------------------------------------------------

def sub_solar_point(dt: datetime) -> tuple[float, float]:
    """The latitude and longitude where the sun is directly overhead.

    Sub-solar latitude = solar declination.
    Sub-solar longitude = where solar noon is right now = -(GMST + RA_sun)
    converted to [-180, 180].

    Args:
        dt: UTC datetime

    Returns:
        (latitude_deg, longitude_deg)
    """
    jd = julian_day_from_datetime(dt)
    L, g, obliquity = _sun_mean_elements(jd)
    ecliptic_lon_rad = radians(L + 1.915 * sin(g) + 0.020 * sin(2 * g))

    # Solar declination = sub-solar latitude
    decl = degrees(asin(sin(obliquity) * sin(ecliptic_lon_rad)))

    # Solar right ascension
    ra_deg = degrees(atan2(cos(obliquity) * sin(ecliptic_lon_rad), cos(ecliptic_lon_rad)))

    # Sub-solar longitude: the meridian where local solar time = noon
    gmst = greenwich_mean_sidereal_time(jd)
    lng = -(gmst - ra_deg)
    # Normalise to [-180, 180]
    lng = ((lng + 180) % 360) - 180

    return decl, lng


def earth_rotation_state(dt: datetime) -> dict:
    """Full Earth rotation state for a given UTC moment.

    Returns everything a frontend needs to render the globe correctly:
    rotation angle, sub-solar point, solar declination.

    Args:
        dt: UTC datetime

    Returns:
        dict with gmst_deg, sub_solar_lat, sub_solar_lng, solar_declination_deg
    """
    jd = julian_day_from_datetime(dt)
    gmst = greenwich_mean_sidereal_time(jd)
    ss_lat, ss_lng = sub_solar_point(dt)
    decl = degrees(solar_declination(jd))

    return {
        "gmst_deg": round(gmst, 4),
        "sub_solar_lat": round(ss_lat, 4),
        "sub_solar_lng": round(ss_lng, 4),
        "solar_declination_deg": round(decl, 4),
    }


# ---------------------------------------------------------------------------
# Sun: core orbital elements
# ---------------------------------------------------------------------------

def _sun_mean_elements(jd: float) -> tuple[float, float, float]:
    """Return (mean_longitude_deg, mean_anomaly_rad, obliquity_rad) for a JD."""
    n = jd - 2451545.0  # days since J2000.0
    L = fmod(280.460 + 0.9856474 * n, 360.0)  # mean longitude
    g = radians(fmod(357.528 + 0.9856003 * n, 360.0))  # mean anomaly
    obliquity = radians(23.439 - 0.0000004 * n)
    return L, g, obliquity


def solar_declination(jd: float) -> float:
    """Solar declination in radians for a given Julian Day."""
    L, g, obliquity = _sun_mean_elements(jd)
    ecliptic_lon = radians(L + 1.915 * sin(g) + 0.020 * sin(2 * g))
    return asin(sin(obliquity) * sin(ecliptic_lon))


def equation_of_time(jd: float) -> float:
    """Equation of time in minutes (difference between solar time and clock time)."""
    L, g, obliquity = _sun_mean_elements(jd)
    ecliptic_lon = L + 1.915 * sin(g) + 0.020 * sin(2 * g)

    RA = degrees(atan2(cos(obliquity) * sin(radians(ecliptic_lon)), cos(radians(ecliptic_lon))))
    RA = fmod(RA + 360, 360.0)

    eot = (L - RA) * 4  # 1 degree = 4 minutes of time
    if eot > 720:
        eot -= 1440
    elif eot < -720:
        eot += 1440
    return eot


def solar_ecliptic_longitude(jd: float) -> float:
    """Sun's ecliptic longitude in degrees for a given Julian Day.

    This is the Sun's position along the ecliptic, measured from the
    vernal equinox. Drives seasons, solar declination, and the
    equation of time.

    Returns: ecliptic longitude in degrees [0, 360)
    """
    L, g, _ = _sun_mean_elements(jd)
    lon = (L + 1.915 * sin(g) + 0.020 * sin(2 * g)) % 360
    if lon < 0:
        lon += 360.0
    return lon


def season_from_ecliptic_longitude(lon_deg: float, latitude: float = 51.5) -> tuple[str, float, float, str]:
    """Determine season from solar ecliptic longitude and latitude.

    Hemisphere-aware:
        Northern (lat >= 23.5):  Spring/Summer/Autumn/Winter
        Southern (lat <= -23.5): Seasons are reversed
        Tropical (|lat| < 23.5): Wet/Dry based on solar declination proximity

    Returns:
        (season_name, progress_0_to_1, days_to_next_event, next_event_name)
    """
    lon = lon_deg % 360
    rate = 0.9856  # Sun's mean daily motion (deg/day)

    # Northern hemisphere seasons
    if lon < 90:
        n_season, n_progress = "Spring", lon / 90
        n_days, n_next = (90 - lon) / rate, "Summer Solstice"
    elif lon < 180:
        n_season, n_progress = "Summer", (lon - 90) / 90
        n_days, n_next = (180 - lon) / rate, "Autumnal Equinox"
    elif lon < 270:
        n_season, n_progress = "Autumn", (lon - 180) / 90
        n_days, n_next = (270 - lon) / rate, "Winter Solstice"
    else:
        n_season, n_progress = "Winter", (lon - 270) / 90
        n_days, n_next = (360 - lon) / rate, "Vernal Equinox"

    if abs(latitude) < 23.5:
        # Tropical: wet/dry based on whether sun is near this latitude
        # Sun declination ranges from -23.4 to +23.4 over the year
        # When sun is near the location's latitude = wet season (more direct heating)
        jd_approx = 2451545.0 + lon / rate  # rough JD for this ecliptic longitude
        decl_deg = degrees(solar_declination(jd_approx))
        sun_proximity = abs(decl_deg - latitude)
        if sun_proximity < 20:
            return "Wet", n_progress, n_days, n_next
        else:
            return "Dry", n_progress, n_days, n_next
    elif latitude < -23.5:
        # Southern hemisphere: flip seasons
        flip = {"Spring": "Autumn", "Summer": "Winter", "Autumn": "Spring", "Winter": "Summer"}
        return flip[n_season], n_progress, n_days, n_next
    else:
        # Northern hemisphere
        return n_season, n_progress, n_days, n_next


# ---------------------------------------------------------------------------
# Sun: rise/set and twilight
# ---------------------------------------------------------------------------

def _hour_angle_for_zenith(lat: float, decl: float, zenith_deg: float) -> float | None:
    """Hour angle (degrees) for the sun to reach a given zenith angle.

    Returns None if the event doesn't occur (polar day/night for that zenith).
    """
    lat_rad = radians(lat)
    cos_ha = (cos(radians(zenith_deg)) - sin(lat_rad) * sin(decl)) / (cos(lat_rad) * cos(decl))
    if cos_ha > 1 or cos_ha < -1:
        return None
    return degrees(acos(cos_ha))


def sunrise_sunset(lat: float, lng: float, d: date) -> tuple[datetime | None, datetime | None]:
    """Sunrise and sunset as UTC datetimes. Returns (None, None) for polar day/night."""
    jd = julian_day(d)
    decl = solar_declination(jd)
    eot = equation_of_time(jd)
    ha = _hour_angle_for_zenith(lat, decl, 90.833)
    if ha is None:
        return None, None

    solar_noon_min = 720 - 4 * lng - eot
    base = datetime(d.year, d.month, d.day)
    rise = base + timedelta(minutes=solar_noon_min - ha * 4)
    sett = base + timedelta(minutes=solar_noon_min + ha * 4)
    return rise, sett


def solar_noon_time(lat: float, lng: float, d: date) -> datetime:
    """Solar noon as a UTC datetime."""
    jd = julian_day(d)
    eot = equation_of_time(jd)
    noon_min = 720 - 4 * lng - eot
    return datetime(d.year, d.month, d.day) + timedelta(minutes=noon_min)


def twilight_times(
    lat: float, lng: float, d: date, zenith_deg: float,
) -> tuple[datetime | None, datetime | None]:
    """Dawn and dusk for a given zenith angle.

    Standard angles:
        civil      = 96°  (sun 6° below horizon)
        nautical   = 102° (sun 12° below)
        astronomical = 108° (sun 18° below)

    Returns (dawn, dusk) as UTC datetimes. None if event doesn't occur.
    """
    jd = julian_day(d)
    decl = solar_declination(jd)
    eot = equation_of_time(jd)
    ha = _hour_angle_for_zenith(lat, decl, zenith_deg)
    if ha is None:
        return None, None

    solar_noon_min = 720 - 4 * lng - eot
    base = datetime(d.year, d.month, d.day)
    dawn = base + timedelta(minutes=solar_noon_min - ha * 4)
    dusk = base + timedelta(minutes=solar_noon_min + ha * 4)
    return dawn, dusk


# ---------------------------------------------------------------------------
# Sun: real-time position
# ---------------------------------------------------------------------------

def solar_position(lat: float, lng: float, dt: datetime) -> tuple[float, float]:
    """Solar elevation and azimuth at a specific moment.

    The hour angle is derived from Earth's rotation angle (GMST -> LST)
    and the Sun's right ascension, making the rotation coupling explicit.

    Args:
        lat: latitude in degrees
        lng: longitude in degrees
        dt: UTC datetime

    Returns:
        (elevation_deg, azimuth_deg) where azimuth 0=N, 90=E, 180=S, 270=W
    """
    jd = julian_day_from_datetime(dt)

    # Sun's ecliptic coordinates
    L, g, obliquity = _sun_mean_elements(jd)
    ecliptic_lon_rad = radians(L + 1.915 * sin(g) + 0.020 * sin(2 * g))

    # Sun's equatorial coordinates (ecliptic -> equatorial transform)
    decl = asin(sin(obliquity) * sin(ecliptic_lon_rad))
    ra = atan2(cos(obliquity) * sin(ecliptic_lon_rad), cos(ecliptic_lon_rad))

    # Hour angle from Earth's rotation: HA = LST - RA
    lst_deg = local_sidereal_time(jd, lng)
    hour_angle = radians(lst_deg) - ra
    # Normalize to [-pi, pi]
    hour_angle = (hour_angle + pi) % (2 * pi) - pi

    lat_rad = radians(lat)

    # Elevation
    sin_elev = sin(lat_rad) * sin(decl) + cos(lat_rad) * cos(decl) * cos(hour_angle)
    elevation = degrees(asin(max(-1.0, min(1.0, sin_elev))))

    # Azimuth (0=N, 90=E, 180=S, 270=W)
    cos_elev = cos(radians(elevation))
    if abs(cos_elev) < 1e-10 or abs(cos(lat_rad)) < 1e-10:
        azimuth = 0.0  # at pole or zenith, azimuth is undefined
    else:
        cos_az = (sin(decl) - sin(lat_rad) * sin_elev) / (cos(lat_rad) * cos_elev)
        cos_az = max(-1.0, min(1.0, cos_az))
        azimuth = degrees(acos(cos_az))
        if hour_angle > 0:
            azimuth = 360 - azimuth

    return round(elevation, 2), round(azimuth, 2)


# ---------------------------------------------------------------------------
# Moon: phase, illumination, age
# ---------------------------------------------------------------------------

# Reference new moon: 2000-01-06 18:14 UTC (well-established astronomical reference)
_NEW_MOON_REF = datetime(2000, 1, 6, 18, 14)


def moon_age(dt: datetime) -> float:
    """Days since the last new moon (0 = new moon, ~14.77 = full moon).

    Uses a simple synodic period calculation from a known new moon reference.
    Accuracy: ±0.5 days, sufficient for phase display.
    """
    days_since_ref = (dt - _NEW_MOON_REF).total_seconds() / 86400.0
    age = days_since_ref % SYNODIC_PERIOD_DAYS
    return round(age, 2)


def moon_phase(dt: datetime) -> float:
    """Moon phase as 0.0–1.0 (0=new, 0.25=first quarter, 0.5=full, 0.75=last quarter)."""
    return round(moon_age(dt) / SYNODIC_PERIOD_DAYS, 4)


def moon_illumination(dt: datetime) -> float:
    """Moon illumination percentage (0–100). 0 = new moon, 100 = full moon."""
    phase = moon_phase(dt)
    # Illumination follows a cosine curve: 0% at new, 100% at full
    illum = (1 - cos(2 * pi * phase)) / 2 * 100
    return round(illum, 1)


def moon_phase_name(phase: float) -> str:
    """Human-readable moon phase name from phase value (0.0–1.0)."""
    if phase < 0.0625:
        return "New Moon"
    elif phase < 0.1875:
        return "Waxing Crescent"
    elif phase < 0.3125:
        return "First Quarter"
    elif phase < 0.4375:
        return "Waxing Gibbous"
    elif phase < 0.5625:
        return "Full Moon"
    elif phase < 0.6875:
        return "Waning Gibbous"
    elif phase < 0.8125:
        return "Last Quarter"
    elif phase < 0.9375:
        return "Waning Crescent"
    else:
        return "New Moon"


def moon_phase_emoji(phase: float) -> str:
    """Unicode moon phase emoji from phase value (0.0–1.0)."""
    if phase < 0.0625:
        return "\U0001F311"  # 🌑 New Moon
    elif phase < 0.1875:
        return "\U0001F312"  # 🌒 Waxing Crescent
    elif phase < 0.3125:
        return "\U0001F313"  # 🌓 First Quarter
    elif phase < 0.4375:
        return "\U0001F314"  # 🌔 Waxing Gibbous
    elif phase < 0.5625:
        return "\U0001F315"  # 🌕 Full Moon
    elif phase < 0.6875:
        return "\U0001F316"  # 🌖 Waning Gibbous
    elif phase < 0.8125:
        return "\U0001F317"  # 🌗 Last Quarter
    elif phase < 0.9375:
        return "\U0001F318"  # 🌘 Waning Crescent
    else:
        return "\U0001F311"  # 🌑 New Moon


# ---------------------------------------------------------------------------
# Moon: approximate declination and rise/set
# ---------------------------------------------------------------------------

def moon_ecliptic_position(jd: float) -> tuple[float, float]:
    """Approximate ecliptic longitude and latitude of the Moon (degrees).

    Simplified from Meeus Ch. 47. Accuracy ~1° in longitude, sufficient
    for moonrise/moonset calculations.
    """
    T = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000

    # Moon's mean longitude (degrees)
    Lp = fmod(218.3165 + 481267.8813 * T, 360.0)
    # Moon's mean anomaly (degrees)
    M = fmod(134.9634 + 477198.8676 * T, 360.0)
    # Sun's mean anomaly (degrees)
    Ms = fmod(357.5291 + 35999.0503 * T, 360.0)
    # Moon's mean elongation (degrees)
    D = fmod(297.8502 + 445267.1115 * T, 360.0)
    # Moon's argument of latitude (degrees)
    F = fmod(93.2720 + 483202.0175 * T, 360.0)

    Mr = radians(M)
    Msr = radians(Ms)
    Dr = radians(D)
    Fr = radians(F)

    # Ecliptic longitude (simplified — main terms only)
    lon = Lp + (
        6.289 * sin(Mr)
        - 1.274 * sin(2 * Dr - Mr)
        + 0.658 * sin(2 * Dr)
        + 0.214 * sin(2 * Mr)
        - 0.186 * sin(Msr)
        - 0.114 * sin(2 * Fr)
    )

    # Ecliptic latitude (simplified)
    lat = (
        5.128 * sin(Fr)
        + 0.281 * sin(Mr + Fr)
        + 0.278 * sin(Mr - Fr)
    )

    return fmod(lon + 360, 360.0), lat


def moon_declination(jd: float) -> float:
    """Moon's declination in radians for a given Julian Day."""
    lon, lat = moon_ecliptic_position(jd)
    n = jd - 2451545.0
    obliquity = radians(23.439 - 0.0000004 * n)

    lon_r = radians(lon)
    lat_r = radians(lat)

    decl = asin(
        sin(lat_r) * cos(obliquity) + cos(lat_r) * sin(obliquity) * sin(lon_r)
    )
    return decl


def moon_right_ascension(jd: float) -> float:
    """Moon's right ascension in radians for a given Julian Day."""
    lon, lat = moon_ecliptic_position(jd)
    n = jd - 2451545.0
    obliquity = radians(23.439 - 0.0000004 * n)

    lon_r = radians(lon)
    lat_r = radians(lat)

    ra = atan2(
        sin(lon_r) * cos(obliquity) - tan(lat_r) * sin(obliquity),
        cos(lon_r),
    )
    return ra


def moonrise_moonset(
    lat: float, lng: float, d: date,
) -> tuple[datetime | None, datetime | None]:
    """Approximate moonrise and moonset as UTC datetimes.

    Uses an iterative approach because the Moon moves ~13° per day,
    making a simple hour-angle calculation insufficient. We iterate
    3 times to converge on the correct rise/set time.

    Returns (moonrise, moonset). Either can be None if the event
    doesn't occur on this date (Moon can be above/below horizon all day).
    """
    lat_rad = radians(lat)
    # Moon's apparent angular radius + atmospheric refraction
    h0 = radians(0.125)  # ~0.125° accounts for Moon's mean radius + refraction

    base = datetime(d.year, d.month, d.day)

    def _find_event(is_rise: bool) -> datetime | None:
        # Start with an estimate based on noon declination
        jd_noon = julian_day(d) + 0.5
        decl = moon_declination(jd_noon)
        ra = moon_right_ascension(jd_noon)

        cos_H = (sin(h0) - sin(lat_rad) * sin(decl)) / (cos(lat_rad) * cos(decl))
        if abs(cos_H) > 1:
            return None

        H0 = acos(cos_H)

        # Greenwich sidereal time at 0h UT for this date
        jd0 = julian_day(d)
        theta0_rad = radians(greenwich_mean_sidereal_time(jd0))

        if is_rise:
            m = (degrees(ra) - degrees(theta0_rad) - lng - degrees(H0)) / 360.0
        else:
            m = (degrees(ra) - degrees(theta0_rad) - lng + degrees(H0)) / 360.0

        m = m % 1.0

        # Iterate to refine (Moon moves fast)
        for _ in range(3):
            jd_event = jd0 + m
            decl_i = moon_declination(jd_event)
            ra_i = moon_right_ascension(jd_event)

            # Local sidereal time at the event
            theta = theta0_rad + radians(GMST_RATE_DEG_PER_DAY * m)
            H = theta - radians(lng) - ra_i

            # Altitude at this hour angle
            sin_alt = sin(lat_rad) * sin(decl_i) + cos(lat_rad) * cos(decl_i) * cos(H)
            alt = asin(max(-1.0, min(1.0, sin_alt)))

            # Correction
            dm = (alt - h0) / (2 * pi * cos(decl_i) * cos(lat_rad) * sin(H))
            m += dm

        if m < 0 or m > 1:
            return None

        return base + timedelta(days=m)

    rise = _find_event(is_rise=True)
    sett = _find_event(is_rise=False)
    return rise, sett
