"""Basic tests for the world module."""

import asyncio
import json
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

from world.time_system import TimeSystem, AstronomyState
from world.weather import Weather, WeatherState, WMO_CODES
from world.config import WorldConfig, RefreshPolicy, RefreshPolicies
from data_acquisition.fetch_geography import haversine_km, compass_direction
from data_acquisition.fetch_astronomy import sunrise_sunset


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


def test_time_system_is_real_time():
    """TimeSystem should initialise to real Earth time, not a simulated era."""
    ts = TimeSystem()
    now_utc = datetime.now(timezone.utc)
    diff = abs((ts.current_time - now_utc).total_seconds())
    assert diff < 5, f"TimeSystem drifted {diff}s from wall clock on init"


def test_time_system_advance():
    """advance() should read the real clock and increment the tick count."""
    ts = TimeSystem()
    assert ts.tick_count == 0
    ts.advance()
    assert ts.tick_count == 1

    now_utc = datetime.now(timezone.utc)
    diff = abs((ts.current_time - now_utc).total_seconds())
    assert diff < 5, f"advance() drifted {diff}s from wall clock"


def test_time_system_season():
    """Season should match the month."""
    ts = TimeSystem()
    # Override current_time to test each season
    ts.current_time = datetime(2025, 1, 15, tzinfo=timezone.utc)
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
    assert cfg.region == "GB"
    assert cfg.timezone == "Europe/London"
    assert cfg.refresh.weather.enabled is True
    assert cfg.refresh.weather.interval_minutes == 30
    assert cfg.refresh.astronomy.enabled is True
    assert cfg.refresh.astronomy.interval_minutes == 1440
    assert cfg.refresh.geography.enabled is True
    assert cfg.refresh.geography.interval_minutes == 10080


# --- Real Earth Time (S31) ---


def test_time_system_sync():
    """sync() should set current_time to real wall-clock UTC."""
    ts = TimeSystem()
    # Force to an old time, then sync
    ts.current_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    ts.sync()
    now_utc = datetime.now(timezone.utc)
    diff = abs((ts.current_time - now_utc).total_seconds())
    assert diff < 5


def test_time_system_timezone_gmt_bst():
    """Timezone abbreviation should reflect GMT in winter, BST in summer."""
    ts = TimeSystem(timezone_name="Europe/London")

    # January = GMT
    ts.current_time = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
    assert ts.timezone_abbr == "GMT"
    assert ts.utc_offset == "+00:00"

    # July = BST (+01:00)
    ts.current_time = datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc)
    assert ts.timezone_abbr == "BST"
    assert ts.utc_offset == "+01:00"


def test_time_system_local_time_bst():
    """local_time should show BST offset during summer."""
    ts = TimeSystem(timezone_name="Europe/London")
    ts.current_time = datetime(2025, 7, 15, 12, 0, tzinfo=timezone.utc)
    assert ts.local_time.hour == 13  # 12:00 UTC = 13:00 BST


def test_time_system_local_time_gmt():
    """local_time should match UTC during winter (GMT)."""
    ts = TimeSystem(timezone_name="Europe/London")
    ts.current_time = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
    assert ts.local_time.hour == 12  # 12:00 UTC = 12:00 GMT


def test_time_system_to_dict():
    """to_dict should include timezone, local_time, and related fields."""
    ts = TimeSystem(timezone_name="Europe/London")
    ts.current_time = datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc)
    d = ts.to_dict()
    assert d["timezone"] == "Europe/London"
    assert d["timezone_abbr"] == "GMT"
    assert d["utc_offset"] == "+00:00"
    assert "local_time" in d
    assert "time_mode" not in d  # No simulated mode — always real


# --- Weather Refresh (S32) ---


def _mock_open_meteo_response(temperature: float = 8.5, weather_code: int = 3) -> dict:
    """Build a fake Open-Meteo Forecast API current-weather response."""
    return {
        "current": {
            "temperature_2m": temperature,
            "relative_humidity_2m": 82.0,
            "precipitation": 0.1,
            "weather_code": weather_code,
            "cloud_cover": 90.0,
            "wind_speed_10m": 15.5,
            "wind_direction_10m": 230.0,
            "surface_pressure": 1013.2,
        }
    }


@pytest.mark.asyncio
async def test_weather_refresh_updates_current():
    """refresh() should update _current with live data from Open-Meteo."""
    w = Weather()
    locations = [(1, 51.5, -0.12), (2, 53.8, -1.55)]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _mock_open_meteo_response(temperature=10.0, weather_code=61)

    with patch("world.weather.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        count = await w.refresh(locations)

    assert count == 2
    assert w.last_refresh is not None
    assert w.refresh_count == 1

    # Both locations should have live data
    ws1 = w.get_weather(1)
    ws2 = w.get_weather(2)
    assert ws1 is not None
    assert ws1.temperature_c == 10.0
    assert ws1.conditions == "Slight rain"
    assert ws2 is not None
    assert ws2.temperature_c == 10.0


@pytest.mark.asyncio
async def test_weather_refresh_empty_locations():
    """refresh() with no locations should return 0 and not call the API."""
    w = Weather()
    count = await w.refresh([])
    assert count == 0
    assert w.last_refresh is None


@pytest.mark.asyncio
async def test_weather_refresh_handles_api_failure():
    """refresh() should gracefully handle API errors without crashing."""
    w = Weather()
    locations = [(1, 51.5, -0.12)]

    with patch("world.weather.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=MagicMock()
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        count = await w.refresh(locations)

    assert count == 0
    assert w.last_refresh is None
    assert w._refresh_failures == 1


@pytest.mark.asyncio
async def test_weather_refresh_partial_failure():
    """If some locations fail, the successful ones should still update."""
    w = Weather()
    locations = [(1, 51.5, -0.12), (2, 53.8, -1.55)]

    call_count = 0

    async def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call succeeds
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = _mock_open_meteo_response(temperature=7.0)
            return resp
        else:
            # Second call fails
            raise httpx.ConnectError("Connection refused")

    with patch("world.weather.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        count = await w.refresh(locations)

    assert count == 1
    assert w.get_weather(1) is not None
    assert w.get_weather(1).temperature_c == 7.0
    assert w.get_weather(2) is None  # Failed location


def test_wmo_codes_coverage():
    """WMO_CODES should include all standard weather codes."""
    assert WMO_CODES[0] == "Clear sky"
    assert WMO_CODES[3] == "Overcast"
    assert WMO_CODES[61] == "Slight rain"
    assert WMO_CODES[95] == "Thunderstorm"


# --- Refresh Policies (S33) ---


def test_refresh_policy_model():
    """RefreshPolicy should hold enabled flag and interval."""
    p = RefreshPolicy(enabled=True, interval_minutes=60)
    assert p.enabled is True
    assert p.interval_minutes == 60


def test_refresh_policies_defaults():
    """Default policies: weather=30m, astronomy=daily, geography=weekly."""
    policies = RefreshPolicies()
    assert policies.weather.interval_minutes == 30
    assert policies.astronomy.interval_minutes == 1440
    assert policies.geography.interval_minutes == 10080
    assert all(p.enabled for p in [policies.weather, policies.astronomy, policies.geography])


def test_refresh_policies_custom():
    """Policies should be overridable."""
    cfg = WorldConfig(refresh=RefreshPolicies(
        weather=RefreshPolicy(enabled=True, interval_minutes=15),
        astronomy=RefreshPolicy(enabled=False, interval_minutes=1440),
        geography=RefreshPolicy(enabled=True, interval_minutes=43200),  # monthly
    ))
    assert cfg.refresh.weather.interval_minutes == 15
    assert cfg.refresh.astronomy.enabled is False
    assert cfg.refresh.geography.interval_minutes == 43200


def test_astronomy_refresh_computes_for_today():
    """refresh_astronomy should compute sunrise/sunset for the current date."""
    ts = TimeSystem(timezone_name="Europe/London")

    # London coordinates
    locations = [(1, 51.5074, -0.1278)]
    count = ts.refresh_astronomy(locations)

    assert count == 1
    today_key = ts.current_time.date().isoformat()
    assert today_key in ts._astronomy.get(1, {})

    state = ts._astronomy[1][today_key]
    assert state.sunrise is not None
    assert state.sunset is not None
    assert state.day_length_hours is not None
    assert state.day_length_hours > 0


def test_astronomy_refresh_multiple_locations():
    """refresh_astronomy should handle multiple locations."""
    ts = TimeSystem()
    locations = [
        (1, 51.5074, -0.1278),  # London
        (2, 55.9533, -3.1883),  # Edinburgh
        (3, 51.4816, -3.1791),  # Cardiff
    ]
    count = ts.refresh_astronomy(locations)
    assert count == 3
    today_key = ts.current_time.date().isoformat()
    for loc_id, _, _ in locations:
        assert today_key in ts._astronomy[loc_id]
