"""Tests for location-relative temporal features (S57).

Every location on Earth has its own local time, timezone, and season.
"""

from world.time_system import TimeSystem


def test_timezone_at_london():
    ts = TimeSystem()
    tz = ts.timezone_at(51.5, -0.13)
    assert tz == "Europe/London"


def test_timezone_at_tokyo():
    ts = TimeSystem()
    tz = ts.timezone_at(35.68, 139.65)
    assert tz == "Asia/Tokyo"


def test_timezone_at_new_york():
    ts = TimeSystem()
    tz = ts.timezone_at(40.71, -74.01)
    assert tz == "America/New_York"


def test_timezone_at_lagos():
    ts = TimeSystem()
    tz = ts.timezone_at(6.52, 3.38)
    assert tz == "Africa/Lagos"


def test_timezone_cached():
    ts = TimeSystem()
    tz1 = ts.timezone_at(51.5, -0.13)
    tz2 = ts.timezone_at(51.5, -0.13)
    assert tz1 == tz2
    # Same rounded key should be cached
    assert (51.5, -0.13) in ts._tz_cache


def test_local_time_differs_by_location():
    ts = TimeSystem()
    london = ts.local_time_at(51.5, -0.13)
    tokyo = ts.local_time_at(35.68, 139.65)
    # Tokyo is ahead of London
    assert tokyo.utcoffset() > london.utcoffset()


def test_time_at_returns_full_info():
    ts = TimeSystem()
    info = ts.time_at(51.5, -0.13)
    assert "local_time" in info
    assert "timezone" in info
    assert "timezone_abbr" in info
    assert "utc_offset" in info
    assert "hour" in info
    assert "minute" in info
    assert "date" in info
    assert info["timezone"] == "Europe/London"


def test_season_at_northern_hemisphere():
    ts = TimeSystem()
    # March = spring in northern hemisphere
    season = ts.season_at(51.5)
    assert season in ("spring", "summer", "autumn", "winter")


def test_season_at_southern_hemisphere_is_opposite():
    ts = TimeSystem()
    north = ts.season_at(51.5)
    south = ts.season_at(-33.87)
    # In March: north=spring, south=autumn (opposite)
    opposites = {"spring": "autumn", "summer": "winter", "autumn": "spring", "winter": "summer"}
    assert south == opposites.get(north, south)


def test_season_at_tropical_is_wet_or_dry():
    ts = TimeSystem()
    season = ts.season_at(6.5)  # Lagos latitude
    assert season in ("wet", "dry")


def test_to_dict_has_utc():
    ts = TimeSystem()
    d = ts.to_dict()
    assert d["timezone"] == "UTC"
    assert d["utc_offset"] == "+00:00"
    assert "tick_count" in d
    assert "season" in d
