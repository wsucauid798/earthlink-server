"""Basic tests for the world module."""

from datetime import date, datetime, timezone

from world.time_system import TimeSystem, AstronomyState
from world.weather import Weather, WeatherState
from world.config import WorldConfig
from data_acquisition.fetch_geography import haversine_km, compass_direction
from data_acquisition.fetch_astronomy import sunrise_sunset, moon_phase


def test_haversine_london_edinburgh():
    """London to Edinburgh should be roughly 530 km."""
    dist = haversine_km(51.5074, -0.1278, 55.9533, -3.1883)
    assert 500 < dist < 560


def test_compass_direction():
    """Edinburgh is roughly north of London."""
    direction = compass_direction(51.5074, -0.1278, 55.9533, -3.1883)
    assert direction in ("N", "NW")


def test_sunrise_sunset_london_summer():
    """Sunrise in London in June should be early, sunset late."""
    rise, sett = sunrise_sunset(51.5074, -0.1278, date(2025, 6, 21))
    assert rise is not None and sett is not None
    assert rise.hour <= 5  # sunrise before 5 AM UTC
    assert sett.hour >= 20  # sunset after 8 PM UTC


def test_sunrise_sunset_london_winter():
    """Sunrise in London in December should be late, sunset early."""
    rise, sett = sunrise_sunset(51.5074, -0.1278, date(2025, 12, 21))
    assert rise is not None and sett is not None
    assert rise.hour >= 6  # sunrise after 6 AM UTC
    assert sett.hour <= 16  # sunset before 4 PM UTC


def test_moon_phase_range():
    """Moon phase should always be between 0 and 1."""
    for day_offset in range(365):
        d = date(2025, 1, 1) + __import__("datetime").timedelta(days=day_offset)
        phase = moon_phase(d)
        assert 0.0 <= phase <= 1.0


def test_time_system_advance():
    """Time system should advance correctly."""
    ts = TimeSystem(
        current_time=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert ts.tick_count == 0
    ts.advance()
    assert ts.tick_count == 1
    assert ts.current_time.minute == 15  # default 15-min step


def test_time_system_season():
    """Season should match the month."""
    ts = TimeSystem(current_time=datetime(2025, 1, 15, tzinfo=timezone.utc))
    assert ts.season == "winter"

    ts.current_time = datetime(2025, 4, 15, tzinfo=timezone.utc)
    assert ts.season == "spring"

    ts.current_time = datetime(2025, 7, 15, tzinfo=timezone.utc)
    assert ts.season == "summer"

    ts.current_time = datetime(2025, 10, 15, tzinfo=timezone.utc)
    assert ts.season == "autumn"


def test_weather_state():
    """WeatherState should hold values."""
    ws = WeatherState(location_id=1, temperature_c=12.5, conditions="Overcast")
    assert ws.temperature_c == 12.5
    assert ws.conditions == "Overcast"


def test_world_config_defaults():
    """WorldConfig should have sensible defaults."""
    cfg = WorldConfig()
    assert cfg.tick_interval_seconds == 1.0
    assert cfg.time_scale_minutes == 15
    assert cfg.region == "GB"
