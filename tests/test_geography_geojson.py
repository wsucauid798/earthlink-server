from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_GEO_PATH = Path(__file__).resolve().parents[1] / "src" / "world" / "geography.py"
_SPEC = spec_from_file_location("earthlink_world_geography", _GEO_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
Geography = _MOD.Geography


def test_geojson_display_type_promotes_canada_town_at_low_zoom():
    got = Geography._geojson_display_type("town", "Canada", 2)
    assert got == "city"


def test_geojson_display_type_keeps_canada_town_at_higher_zoom():
    got = Geography._geojson_display_type("town", "Canada", 6)
    assert got == "town"


def test_geojson_display_type_keeps_non_canada_town():
    got = Geography._geojson_display_type("town", "United States", 2)
    assert got == "town"
