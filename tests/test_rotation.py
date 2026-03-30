"""Tests for the Earth rotation system (S59).

Sub-solar point, GMST, and rotation state for frontend rendering.
"""

from datetime import datetime

from world.celestial import (
    earth_rotation_state,
    greenwich_mean_sidereal_time,
    julian_day_from_datetime,
    sub_solar_point,
)


def test_sub_solar_lat_in_range():
    """Sub-solar latitude should be within Earth's axial tilt (~23.4 degrees)."""
    dt = datetime(2026, 3, 28, 12, 0, 0)
    lat, lng = sub_solar_point(dt)
    assert -23.5 <= lat <= 23.5


def test_sub_solar_lng_in_range():
    dt = datetime(2026, 3, 28, 12, 0, 0)
    lat, lng = sub_solar_point(dt)
    assert -180 <= lng <= 180


def test_sub_solar_at_noon_utc_near_zero_longitude():
    """At 12:00 UTC, sun should be roughly over the prime meridian (0 longitude)."""
    dt = datetime(2026, 3, 21, 12, 0, 0)  # near equinox
    lat, lng = sub_solar_point(dt)
    # Should be within ~20 degrees of 0 longitude (equation of time variation)
    assert -25 <= lng <= 25


def test_sub_solar_at_equinox_near_equator():
    """Near the equinox, sub-solar latitude should be near zero."""
    dt = datetime(2026, 3, 21, 12, 0, 0)
    lat, lng = sub_solar_point(dt)
    assert -5 <= lat <= 5


def test_sub_solar_at_summer_solstice_is_north():
    """At northern summer solstice, sun is over ~23.4N."""
    dt = datetime(2026, 6, 21, 12, 0, 0)
    lat, lng = sub_solar_point(dt)
    assert lat > 20


def test_sub_solar_at_winter_solstice_is_south():
    """At northern winter solstice, sun is over ~23.4S."""
    dt = datetime(2026, 12, 21, 12, 0, 0)
    lat, lng = sub_solar_point(dt)
    assert lat < -20


def test_gmst_in_range():
    dt = datetime(2026, 3, 28, 12, 0, 0)
    jd = julian_day_from_datetime(dt)
    gmst = greenwich_mean_sidereal_time(jd)
    assert 0 <= gmst < 360


def test_earth_rotation_state_has_all_fields():
    dt = datetime(2026, 3, 28, 12, 0, 0)
    state = earth_rotation_state(dt)
    assert "gmst_deg" in state
    assert "sub_solar_lat" in state
    assert "sub_solar_lng" in state
    assert "solar_declination_deg" in state


def test_rotation_state_values_are_reasonable():
    dt = datetime(2026, 3, 28, 15, 0, 0)
    state = earth_rotation_state(dt)
    assert 0 <= state["gmst_deg"] < 360
    assert -23.5 <= state["sub_solar_lat"] <= 23.5
    assert -180 <= state["sub_solar_lng"] <= 180
    assert -23.5 <= state["solar_declination_deg"] <= 23.5
