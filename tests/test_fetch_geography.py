from data_acquisition.fetch_geography import parse_geonames_row


def _build_row(feature_class: str, feature_code: str) -> list[str]:
    # GeoNames row has 19 columns; only a subset is used by parse_geonames_row.
    return [
        "12345",           # geonameid
        "Sample Place",    # name
        "Sample Place",    # asciiname
        "",                # alternatenames
        "45.0",            # latitude
        "-75.0",           # longitude
        feature_class,     # feature_class
        feature_code,      # feature_code
        "CA",              # country_code in file (unused here)
        "",                # cc2
        "08",              # admin1
        "001",             # admin2
        "",                # admin3
        "",                # admin4
        "1000",            # population
        "",                # elevation
        "",                # dem
        "America/Toronto", # timezone
        "2026-01-01",      # modification_date
    ]


def test_parse_geonames_row_ca_ppl_is_town():
    row = _build_row("P", "PPL")
    loc = parse_geonames_row(row, admin_codes={}, country_code="CA")
    assert loc is not None
    assert loc["type"] == "town"


def test_parse_geonames_row_non_ca_ppl_remains_town():
    row = _build_row("P", "PPL")
    loc = parse_geonames_row(row, admin_codes={}, country_code="US")
    assert loc is not None
    assert loc["type"] == "town"
