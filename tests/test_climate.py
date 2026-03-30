"""Tests for the climate system (S63).

Koppen classification from real monthly temperature and precipitation data.
"""

from world.climate import _classify_koppen


def test_tropical_rainforest():
    """Af: all months warm (>18C), driest month >= 60mm."""
    temps = [26, 26, 27, 27, 27, 26, 25, 25, 26, 26, 26, 26]
    precip = [250, 200, 300, 280, 200, 100, 80, 60, 100, 200, 250, 300]
    zone, desc, ctype = _classify_koppen(temps, precip, lat=0)
    assert zone == "Af"
    assert ctype == "tropical"


def test_tropical_savanna():
    """Aw: all months warm, but driest month < 60mm."""
    temps = [27, 28, 28, 28, 27, 26, 25, 26, 27, 27, 27, 27]
    precip = [20, 30, 80, 150, 200, 180, 100, 80, 120, 150, 50, 15]
    zone, desc, ctype = _classify_koppen(temps, precip, lat=6)
    assert zone == "Aw"
    assert ctype == "tropical"


def test_hot_desert():
    """BWh: very low precipitation, warm annual temp."""
    temps = [14, 16, 20, 25, 30, 33, 35, 34, 31, 26, 20, 15]
    precip = [5, 3, 2, 1, 0, 0, 0, 0, 0, 1, 3, 5]
    zone, desc, ctype = _classify_koppen(temps, precip, lat=30)
    assert zone.startswith("B")
    assert ctype == "arid"


def test_temperate_oceanic():
    """Cfb: coldest > -3C, warmest < 22C, no dry season."""
    temps = [5, 6, 7, 9, 13, 16, 18, 17, 15, 11, 8, 6]
    precip = [60, 50, 55, 50, 50, 50, 45, 55, 60, 70, 70, 65]
    zone, desc, ctype = _classify_koppen(temps, precip, lat=51)
    assert zone == "Cfb"
    assert ctype == "temperate"


def test_continental():
    """Dfa/Dfb: coldest < -3C, warmest >= 10C."""
    temps = [-10, -8, -1, 8, 15, 20, 23, 22, 16, 8, 0, -7]
    precip = [30, 25, 35, 50, 70, 80, 90, 80, 60, 50, 40, 30]
    zone, desc, ctype = _classify_koppen(temps, precip, lat=45)
    assert zone.startswith("D")
    assert ctype == "continental"


def test_tundra():
    """ET: warmest month < 10C but >= 0C."""
    temps = [-20, -18, -12, -5, 2, 6, 9, 8, 3, -5, -12, -18]
    precip = [15, 10, 10, 15, 20, 30, 35, 30, 25, 20, 15, 15]
    zone, desc, ctype = _classify_koppen(temps, precip, lat=70)
    assert zone == "ET"
    assert ctype == "polar"


def test_ice_cap():
    """EF: warmest month < 0C."""
    temps = [-30, -35, -30, -25, -15, -10, -5, -8, -15, -25, -30, -35]
    precip = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    zone, desc, ctype = _classify_koppen(temps, precip, lat=80)
    assert zone == "EF"
    assert ctype == "polar"


def test_southern_hemisphere_reversal():
    """Southern hemisphere temperate should still be C zone."""
    temps = [22, 21, 19, 16, 13, 11, 10, 11, 13, 16, 18, 20]
    precip = [100, 110, 90, 60, 50, 50, 40, 40, 50, 60, 70, 80]
    zone, desc, ctype = _classify_koppen(temps, precip, lat=-34)
    assert zone.startswith("C")
    assert ctype == "temperate"


def test_insufficient_data():
    zone, desc, ctype = _classify_koppen([10, 20], [50, 60], lat=0)
    assert zone == "?"
    assert ctype == "unknown"
