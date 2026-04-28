"""
Fetch geography data from GeoNames and insert into PostgreSQL.

Downloads country data files (GB.zip, IE.zip, etc.) from GeoNames, parses the
tab-delimited files, and populates the `locations` and `location_connections`
tables with real places.

GeoNames data format (tab-delimited):
  0: geonameid, 1: name, 2: asciiname, 3: alternatenames, 4: latitude,
  5: longitude, 6: feature_class, 7: feature_code, 8: country_code,
  9: cc2, 10: admin1, 11: admin2, 12: admin3, 13: admin4,
  14: population, 15: elevation, 16: dem, 17: timezone, 18: modification_date

Feature classes:
  A: country, state, region   H: stream, lake   L: parks, area
  P: city, village            R: road, railroad  S: spot, building, farm
  T: mountain, hill, rock     U: undersea        V: forest, heath
"""

import asyncio
import csv
import io
import logging
import zipfile
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.engine import async_session, engine
from db.models import Base, Location, LocationConnection

logger = logging.getLogger(__name__)

# GeoNames country URLs
GEONAMES_URLS = {
    "GB": "https://download.geonames.org/export/dump/GB.zip",
    "IE": "https://download.geonames.org/export/dump/IE.zip",
    "FR": "https://download.geonames.org/export/dump/FR.zip",
    "DE": "https://download.geonames.org/export/dump/DE.zip",
    "NL": "https://download.geonames.org/export/dump/NL.zip",
    "BE": "https://download.geonames.org/export/dump/BE.zip",
    "LU": "https://download.geonames.org/export/dump/LU.zip",
    "ES": "https://download.geonames.org/export/dump/ES.zip",
    "PT": "https://download.geonames.org/export/dump/PT.zip",
    "IT": "https://download.geonames.org/export/dump/IT.zip",
    "CH": "https://download.geonames.org/export/dump/CH.zip",
    "AT": "https://download.geonames.org/export/dump/AT.zip",
    "DK": "https://download.geonames.org/export/dump/DK.zip",
    "NO": "https://download.geonames.org/export/dump/NO.zip",
    "SE": "https://download.geonames.org/export/dump/SE.zip",
    "IS": "https://download.geonames.org/export/dump/IS.zip",
    "FI": "https://download.geonames.org/export/dump/FI.zip",
    # --- North Africa ---
    "MA": "https://download.geonames.org/export/dump/MA.zip",
    "DZ": "https://download.geonames.org/export/dump/DZ.zip",
    "TN": "https://download.geonames.org/export/dump/TN.zip",
    "LY": "https://download.geonames.org/export/dump/LY.zip",
    "EG": "https://download.geonames.org/export/dump/EG.zip",
    "EH": "https://download.geonames.org/export/dump/EH.zip",
    # --- Eastern Europe ---
    "PL": "https://download.geonames.org/export/dump/PL.zip",
    "CZ": "https://download.geonames.org/export/dump/CZ.zip",
    "SK": "https://download.geonames.org/export/dump/SK.zip",
    "HU": "https://download.geonames.org/export/dump/HU.zip",
    "RO": "https://download.geonames.org/export/dump/RO.zip",
    "BG": "https://download.geonames.org/export/dump/BG.zip",
    "MD": "https://download.geonames.org/export/dump/MD.zip",
    "UA": "https://download.geonames.org/export/dump/UA.zip",
    "BY": "https://download.geonames.org/export/dump/BY.zip",
    # --- Baltic states ---
    "EE": "https://download.geonames.org/export/dump/EE.zip",
    "LV": "https://download.geonames.org/export/dump/LV.zip",
    "LT": "https://download.geonames.org/export/dump/LT.zip",
    # --- Balkans ---
    "RS": "https://download.geonames.org/export/dump/RS.zip",
    "HR": "https://download.geonames.org/export/dump/HR.zip",
    "SI": "https://download.geonames.org/export/dump/SI.zip",
    "BA": "https://download.geonames.org/export/dump/BA.zip",
    "ME": "https://download.geonames.org/export/dump/ME.zip",
    "MK": "https://download.geonames.org/export/dump/MK.zip",
    "AL": "https://download.geonames.org/export/dump/AL.zip",
    "XK": "https://download.geonames.org/export/dump/XK.zip",
    # --- Mediterranean Europe ---
    "GR": "https://download.geonames.org/export/dump/GR.zip",
    "CY": "https://download.geonames.org/export/dump/CY.zip",
    "MT": "https://download.geonames.org/export/dump/MT.zip",
    # --- European microstates ---
    "MC": "https://download.geonames.org/export/dump/MC.zip",
    "SM": "https://download.geonames.org/export/dump/SM.zip",
    "VA": "https://download.geonames.org/export/dump/VA.zip",
    "AD": "https://download.geonames.org/export/dump/AD.zip",
    "LI": "https://download.geonames.org/export/dump/LI.zip",
    # --- European territories / dependencies ---
    "FO": "https://download.geonames.org/export/dump/FO.zip",
    "AX": "https://download.geonames.org/export/dump/AX.zip",
    "SJ": "https://download.geonames.org/export/dump/SJ.zip",
    "GL": "https://download.geonames.org/export/dump/GL.zip",
    "GI": "https://download.geonames.org/export/dump/GI.zip",
    "GG": "https://download.geonames.org/export/dump/GG.zip",
    "JE": "https://download.geonames.org/export/dump/JE.zip",
    "IM": "https://download.geonames.org/export/dump/IM.zip",
    # --- North America (continental) ---
    "US": "https://download.geonames.org/export/dump/US.zip",
    "CA": "https://download.geonames.org/export/dump/CA.zip",
    "MX": "https://download.geonames.org/export/dump/MX.zip",
    # --- Central America ---
    "BZ": "https://download.geonames.org/export/dump/BZ.zip",
    "GT": "https://download.geonames.org/export/dump/GT.zip",
    "SV": "https://download.geonames.org/export/dump/SV.zip",
    "HN": "https://download.geonames.org/export/dump/HN.zip",
    "NI": "https://download.geonames.org/export/dump/NI.zip",
    "CR": "https://download.geonames.org/export/dump/CR.zip",
    "PA": "https://download.geonames.org/export/dump/PA.zip",
    # --- Atlantic / N.A. dependencies ---
    "BM": "https://download.geonames.org/export/dump/BM.zip",
    "PM": "https://download.geonames.org/export/dump/PM.zip",
    # --- Caribbean (sovereign) ---
    "BS": "https://download.geonames.org/export/dump/BS.zip",
    "BB": "https://download.geonames.org/export/dump/BB.zip",
    "CU": "https://download.geonames.org/export/dump/CU.zip",
    "DM": "https://download.geonames.org/export/dump/DM.zip",
    "DO": "https://download.geonames.org/export/dump/DO.zip",
    "GD": "https://download.geonames.org/export/dump/GD.zip",
    "HT": "https://download.geonames.org/export/dump/HT.zip",
    "JM": "https://download.geonames.org/export/dump/JM.zip",
    "KN": "https://download.geonames.org/export/dump/KN.zip",
    "LC": "https://download.geonames.org/export/dump/LC.zip",
    "TT": "https://download.geonames.org/export/dump/TT.zip",
    "VC": "https://download.geonames.org/export/dump/VC.zip",
    "AG": "https://download.geonames.org/export/dump/AG.zip",
    # --- Caribbean (US / UK territories) ---
    "PR": "https://download.geonames.org/export/dump/PR.zip",
    "VI": "https://download.geonames.org/export/dump/VI.zip",
    "KY": "https://download.geonames.org/export/dump/KY.zip",
    "TC": "https://download.geonames.org/export/dump/TC.zip",
    "VG": "https://download.geonames.org/export/dump/VG.zip",
    "AI": "https://download.geonames.org/export/dump/AI.zip",
    "MS": "https://download.geonames.org/export/dump/MS.zip",
    # --- Caribbean (Netherlands) ---
    "AW": "https://download.geonames.org/export/dump/AW.zip",
    "CW": "https://download.geonames.org/export/dump/CW.zip",
    "SX": "https://download.geonames.org/export/dump/SX.zip",
    "BQ": "https://download.geonames.org/export/dump/BQ.zip",
    # --- Caribbean (France) ---
    "MQ": "https://download.geonames.org/export/dump/MQ.zip",
    "GP": "https://download.geonames.org/export/dump/GP.zip",
    "MF": "https://download.geonames.org/export/dump/MF.zip",
    "BL": "https://download.geonames.org/export/dump/BL.zip",
}
ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"
ADMIN2_URL = "https://download.geonames.org/export/dump/admin2Codes.txt"

DATA_DIR = Path(__file__).parent.parent / "data" / "geography"

# Countries to fetch (can be configured).
# `run()` is idempotent: it filters out countries already in the DB, so adding
# new ISO codes here triggers an incremental fetch on the next startup
# (handled by main.py's missing-country detection).
COUNTRIES_TO_FETCH = [
    # --- Western / Northern Europe ---
    "GB", "IE", "FR", "DE", "NL", "BE", "LU", "ES", "PT", "IT", "CH", "AT",
    "DK", "NO", "SE", "IS", "FI",
    # --- North Africa ---
    "MA", "DZ", "TN", "LY", "EG", "EH",
    # --- ROLLED BACK: full Europe + North America expansion ---
    # The 75-country bulk-seed in main lifespan blew through 24 GB RAM
    # (millions of in-memory location objects + 50–100M in-memory connection
    # objects before any DB write) and OOM-killed the host. Recovery: VPS
    # was hard-rebooted, server container manually stopped to prevent the
    # seed from re-firing on every startup.
    #
    # The URL and name dictionaries above retain entries for all the
    # additional countries (so re-adding them is just a list change here),
    # but we are NOT re-enabling the bulk seed until the seed flow is
    # rewritten to:
    #   (a) parse + insert + connection-generate per country (commit before
    #       moving to the next country, no cross-country in-memory accumulation),
    #   (b) stream connection generation rather than building millions of
    #       objects in memory,
    #   (c) optionally run as a separate one-shot container with a memory
    #       cap so OOM kills the seed cleanly without taking out the host.
    # Tracked as future work; see server-design-plan.md.
]

# Feature codes that represent meaningful locations for our world
FEATURE_CLASSES_KEEP = {"A", "H", "L", "P", "R", "S", "T", "V"}

# Map GeoNames feature codes to our terrain/type categories
FEATURE_CODE_TO_TYPE = {
    # Administrative
    "PCLI": "country", "PCLH": "country", "ADM1": "admin_region", "ADM2": "admin_region",
    "ADM3": "admin_region", "ADM4": "admin_region",
    # Populated places
    "PPLC": "capital", "PPLA": "city", "PPLA2": "city", "PPLA3": "town",
    "PPLA4": "town", "PPL": "town", "PPLS": "settlement", "PPLX": "neighbourhood",
    "PPLL": "village", "PPLF": "village",
    # Hydrographic
    "STM": "river", "LK": "lake", "RSV": "reservoir", "BAY": "bay",
    "HBR": "harbour", "CHN": "channel", "FLLS": "waterfall", "MRSH": "marsh",
    "GULF": "gulf", "ESTY": "estuary", "COVE": "cove",
    # Terrain
    "MT": "mountain", "MTS": "mountains", "HLL": "hill", "HLLS": "hills",
    "PLN": "plain", "PLT": "plateau", "VLC": "volcano", "CLF": "cliff",
    "PEN": "peninsula", "ISL": "island", "ISLS": "islands", "CAPE": "cape",
    "BCH": "beach", "RDGE": "ridge", "VAL": "valley",
    # Vegetation
    "FRST": "forest", "GRSLD": "grassland", "HTH": "heath", "SCRB": "scrubland",
    # Spots/buildings
    "CSTL": "castle", "CH": "church", "UNIV": "university", "SCH": "school",
    "HSP": "hospital", "MUS": "museum", "LIBR": "library", "AIRP": "airport",
    "RSTN": "railway_station", "BUSTN": "bus_station", "FRM": "farm",
    # Parks/areas
    "PRK": "park", "RES": "reserve", "RESN": "nature_reserve",
    "AMUS": "amusement_park", "RECG": "recreation_ground",
    # Roads
    "RD": "road", "RR": "railway", "TRL": "trail",
}

# Country code to name mapping
COUNTRY_NAMES = {
    # Western / Northern Europe
    "GB": "United Kingdom", "IE": "Ireland", "FR": "France",
    "DE": "Germany", "NL": "Netherlands", "BE": "Belgium",
    "LU": "Luxembourg", "ES": "Spain", "PT": "Portugal",
    "IT": "Italy", "CH": "Switzerland", "AT": "Austria",
    "DK": "Denmark", "NO": "Norway", "SE": "Sweden", "IS": "Iceland",
    "FI": "Finland",
    # North Africa
    "MA": "Morocco", "DZ": "Algeria", "TN": "Tunisia",
    "LY": "Libya", "EG": "Egypt", "EH": "Western Sahara",
    # Eastern Europe
    "PL": "Poland", "CZ": "Czech Republic", "SK": "Slovakia",
    "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria",
    "MD": "Moldova", "UA": "Ukraine", "BY": "Belarus",
    # Baltic states
    "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania",
    # Balkans
    "RS": "Serbia", "HR": "Croatia", "SI": "Slovenia",
    "BA": "Bosnia and Herzegovina", "ME": "Montenegro",
    "MK": "North Macedonia", "AL": "Albania", "XK": "Kosovo",
    # Mediterranean Europe
    "GR": "Greece", "CY": "Cyprus", "MT": "Malta",
    # European microstates
    "MC": "Monaco", "SM": "San Marino", "VA": "Vatican City",
    "AD": "Andorra", "LI": "Liechtenstein",
    # European territories / dependencies
    "FO": "Faroe Islands", "AX": "Åland Islands",
    "SJ": "Svalbard and Jan Mayen", "GL": "Greenland",
    "GI": "Gibraltar", "GG": "Guernsey", "JE": "Jersey", "IM": "Isle of Man",
    # North America
    "US": "United States", "CA": "Canada", "MX": "Mexico",
    # Central America
    "BZ": "Belize", "GT": "Guatemala", "SV": "El Salvador",
    "HN": "Honduras", "NI": "Nicaragua", "CR": "Costa Rica", "PA": "Panama",
    # Atlantic dependencies
    "BM": "Bermuda", "PM": "Saint Pierre and Miquelon",
    # Caribbean (sovereign)
    "BS": "Bahamas", "BB": "Barbados", "CU": "Cuba",
    "DM": "Dominica", "DO": "Dominican Republic", "GD": "Grenada",
    "HT": "Haiti", "JM": "Jamaica", "KN": "Saint Kitts and Nevis",
    "LC": "Saint Lucia", "TT": "Trinidad and Tobago",
    "VC": "Saint Vincent and the Grenadines", "AG": "Antigua and Barbuda",
    # Caribbean (US / UK territories)
    "PR": "Puerto Rico", "VI": "U.S. Virgin Islands",
    "KY": "Cayman Islands", "TC": "Turks and Caicos Islands",
    "VG": "British Virgin Islands", "AI": "Anguilla", "MS": "Montserrat",
    # Caribbean (Netherlands)
    "AW": "Aruba", "CW": "Curaçao", "SX": "Sint Maarten",
    "BQ": "Bonaire, Sint Eustatius and Saba",
    # Caribbean (France)
    "MQ": "Martinique", "GP": "Guadeloupe",
    "MF": "Saint Martin", "BL": "Saint Barthélemy",
}

# UK admin1 code mapping (GeoNames uses ENG, SCT, WLS, NIR)
UK_NATIONS = {
    "ENG": "England",
    "SCT": "Scotland",
    "WLS": "Wales",
    "NIR": "Northern Ireland",
}

# Ireland admin1 code mapping (26 counties in Republic, some consolidated)
IE_COUNTIES = {
    "01": "Carlow", "02": "Cavan", "03": "Clare", "04": "Cork",
    "06": "Donegal", "07": "Dublin", "10": "Galway", "11": "Kerry",
    "12": "Kildare", "13": "Kilkenny", "14": "Laois", "15": "Leitrim",
    "16": "Limerick", "18": "Longford", "19": "Louth", "20": "Mayo",
    "21": "Meath", "22": "Monaghan", "23": "Offaly", "24": "Roscommon",
    "25": "Sligo", "26": "Tipperary", "27": "Waterford", "29": "Westmeath",
    "30": "Wexford", "31": "Wicklow",
}

# France admin1 code mapping (régions)
FR_REGIONS = {
    "75": "Nouvelle-Aquitaine", "76": "Occitanie", "44": "Grand Est",
    "32": "Hauts-de-France", "28": "Normandie", "53": "Bretagne",
    "52": "Pays de la Loire", "24": "Centre-Val de Loire",
    "27": "Bourgogne-Franche-Comté", "84": "Auvergne-Rhône-Alpes",
    "93": "Provence-Alpes-Côte d'Azur", "94": "Corse",
    "11": "Île-de-France",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def compass_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Calculate compass direction from point 1 to point 2."""
    dlon = radians(lon2 - lon1)
    lat1, lat2 = radians(lat1), radians(lat2)
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = (atan2(x, y) * 180 / 3.14159265 + 360) % 360

    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(bearing / 45) % 8
    return directions[idx]


async def download_file(url: str, dest: Path) -> Path:
    """Download a file if it doesn't already exist locally."""
    if dest.exists():
        logger.info(f"Using cached file: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} ...")
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    logger.info(f"Downloaded to {dest} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


async def load_admin_codes(country_codes: list[str]) -> dict[str, str]:
    """Download and parse admin1 and admin2 code files for specified countries."""
    admin1_path = await download_file(ADMIN1_URL, DATA_DIR / "admin1CodesASCII.txt")
    admin2_path = await download_file(ADMIN2_URL, DATA_DIR / "admin2Codes.txt")

    codes = {}
    for path in [admin1_path, admin2_path]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    # Filter for specified countries
                    for country_code in country_codes:
                        if parts[0].startswith(f"{country_code}."):
                            codes[parts[0]] = parts[1]
                            break
    return codes


def parse_geonames_row(row: list[str], admin_codes: dict[str, str], country_code: str) -> dict | None:
    """Parse a single GeoNames row into a location dict."""
    if len(row) < 19:
        return None

    feature_class = row[6]
    feature_code = row[7]

    if feature_class not in FEATURE_CLASSES_KEEP:
        return None

    geonameid = int(row[0])
    name = row[1]
    lat = float(row[4])
    lng = float(row[5])
    population = int(row[14]) if row[14] else None
    elevation = float(row[15]) if row[15] else (float(row[16]) if row[16] else None)
    admin1 = row[10]
    admin2 = row[11]

    # Resolve type
    loc_type = FEATURE_CODE_TO_TYPE.get(feature_code, feature_class.lower())

    # Resolve admin hierarchy based on country
    if country_code == "GB":
        country_name = "United Kingdom"
        nation = UK_NATIONS.get(admin1, admin_codes.get(f"GB.{admin1}", admin1) if admin1 else None)
        county = admin_codes.get(f"GB.{admin1}.{admin2}", admin2) if admin2 else None
    elif country_code == "IE":
        country_name = "Ireland"
        nation = None  # Ireland doesn't have sub-national divisions like UK
        county = IE_COUNTIES.get(admin1, admin_codes.get(f"IE.{admin1}", admin1) if admin1 else None)
    elif country_code == "FR":
        country_name = "France"
        nation = FR_REGIONS.get(admin1, admin_codes.get(f"FR.{admin1}", admin1) if admin1 else None)
        county = admin_codes.get(f"FR.{admin1}.{admin2}", admin2) if admin2 else None
    else:
        country_name = COUNTRY_NAMES.get(country_code, country_code)
        nation = admin_codes.get(f"{country_code}.{admin1}", admin1) if admin1 else None
        county = admin_codes.get(f"{country_code}.{admin1}.{admin2}", admin2) if admin2 else None

    # Infer terrain from feature class and code
    terrain = None
    if feature_class == "P":
        terrain = "urban" if (population and population > 10000) else "rural"
    elif feature_class == "T":
        terrain = "highland" if feature_code in ("MT", "MTS", "HLL", "HLLS", "PLT", "RDGE") else "natural"
    elif feature_class == "H":
        terrain = "water"
    elif feature_class == "V":
        terrain = "woodland" if feature_code == "FRST" else "vegetation"
    elif feature_class == "L":
        terrain = "parkland"

    return {
        "geonameid": geonameid,
        "name": name,
        "type": loc_type,
        "lat": lat,
        "lng": lng,
        "elevation": elevation,
        "terrain": terrain,
        "admin_level_1": country_name,
        "admin_level_2": nation,
        "admin_level_3": county,
        "admin_level_4": None,
        "population": population,
        "metadata_": {
            "geonameid": geonameid,
            "feature_class": feature_class,
            "feature_code": feature_code,
            "asciiname": row[2],
            "country_code": country_code,
        },
    }


async def fetch_and_parse_geography(country_code: str, admin_codes: dict[str, str]) -> list[dict]:
    """Download and parse geography data for a specific country."""
    if country_code not in GEONAMES_URLS:
        raise ValueError(f"Unsupported country code: {country_code}")

    url = GEONAMES_URLS[country_code]
    zip_path = await download_file(url, DATA_DIR / f"{country_code}.zip")

    locations = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(f"{country_code}.txt") as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
            for row in reader:
                loc = parse_geonames_row(row, admin_codes, country_code)
                if loc:
                    locations.append(loc)

    logger.info(f"Parsed {len(locations)} locations from {country_code} GeoNames data")
    return locations


def generate_connections(locations: list[dict], max_distance_km: float = 15.0) -> list[dict]:
    """
    Generate connections between nearby locations.
    Connects each location to its neighbours within max_distance_km.
    Only connects populated places (cities, towns, villages) to each other
    and to nearby features.

    Uses a spatial grid to avoid O(n²) pairwise checks.
    """
    from collections import defaultdict
    import math

    # Index populated places for connection generation
    populated_types = {"capital", "city", "town", "village", "settlement", "neighbourhood"}
    populated = [loc for loc in locations if loc["type"] in populated_types]

    # Build spatial grid — each cell ~30km wide (0.3 degrees latitude)
    cell_size = 0.3  # degrees
    grid: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for loc in populated:
        cx = int(math.floor(loc["lng"] / cell_size))
        cy = int(math.floor(loc["lat"] / cell_size))
        grid[(cx, cy)].append(loc)

    connections = []
    seen: set[tuple[int, int]] = set()

    for i, loc_a in enumerate(populated):
        cx = int(math.floor(loc_a["lng"] / cell_size))
        cy = int(math.floor(loc_a["lat"] / cell_size))

        # Check this cell and 8 neighbours
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for loc_b in grid.get((cx + dx, cy + dy), []):
                    if loc_a["geonameid"] == loc_b["geonameid"]:
                        continue

                    pair = (min(loc_a["geonameid"], loc_b["geonameid"]),
                            max(loc_a["geonameid"], loc_b["geonameid"]))
                    if pair in seen:
                        continue

                    dist = haversine_km(loc_a["lat"], loc_a["lng"], loc_b["lat"], loc_b["lng"])
                    if dist <= max_distance_km:
                        direction = compass_direction(loc_a["lat"], loc_a["lng"], loc_b["lat"], loc_b["lng"])
                        connections.append({
                            "from_geonameid": loc_a["geonameid"],
                            "to_geonameid": loc_b["geonameid"],
                            "distance_km": round(dist, 2),
                            "connection_type": "proximity",
                            "route_name": None,
                            "direction": direction,
                        })
                        seen.add(pair)

        if (i + 1) % 5000 == 0:
            logger.info(f"Connection generation progress: {i + 1}/{len(populated)} places processed")

    logger.info(f"Generated {len(connections)} connections between locations")
    return connections


async def insert_locations(session: AsyncSession, locations: list[dict]) -> dict[int, int]:
    """Insert locations into the database. Returns mapping of geonameid -> db id."""
    geoname_to_db_id = {}

    # Batch insert
    batch_size = 1000
    for i in range(0, len(locations), batch_size):
        batch = locations[i : i + batch_size]
        for loc_data in batch:
            geonameid = loc_data.pop("geonameid")
            loc = Location(**loc_data)
            session.add(loc)
            await session.flush()
            geoname_to_db_id[geonameid] = loc.id

        if (i + batch_size) <= len(locations):
            logger.info(f"Inserted {min(i + batch_size, len(locations))}/{len(locations)} locations")

    await session.commit()
    logger.info(f"Committed {len(locations)} locations to database")
    return geoname_to_db_id


async def insert_connections(
    session: AsyncSession,
    connections: list[dict],
    geoname_to_db_id: dict[int, int],
) -> None:
    """Insert location connections into the database using bulk insert."""
    from sqlalchemy import insert

    batch_size = 10000
    inserted = 0

    for i in range(0, len(connections), batch_size):
        batch = connections[i : i + batch_size]
        rows = []
        for conn_data in batch:
            from_id = geoname_to_db_id.get(conn_data["from_geonameid"])
            to_id = geoname_to_db_id.get(conn_data["to_geonameid"])
            if from_id and to_id:
                rows.append({
                    "from_id": from_id,
                    "to_id": to_id,
                    "distance_km": conn_data["distance_km"],
                    "connection_type": conn_data["connection_type"],
                    "route_name": conn_data["route_name"],
                    "direction": conn_data["direction"],
                })
        if rows:
            await session.execute(insert(LocationConnection), rows)
            inserted += len(rows)

        if (i + batch_size) % 50000 < batch_size:
            logger.info(f"Inserted {inserted} connections so far")

    await session.commit()
    logger.info(f"Committed {inserted} connections to database")


async def run():
    """Main entry point: download, parse, and insert geography data for configured countries."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    logger.info(f"=== Fetching Geography Data for {', '.join(COUNTRIES_TO_FETCH)} ===")

    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Check which countries already exist in database
    async with async_session() as session:
        result = await session.execute(
            text("SELECT DISTINCT metadata->>'country_code' FROM locations WHERE metadata->>'country_code' IS NOT NULL")
        )
        existing_countries = {row[0] for row in result.fetchall()}

    if existing_countries:
        logger.info(f"Existing countries in database: {', '.join(sorted(existing_countries))}")

    # Filter out countries already in database
    countries_to_fetch = [c for c in COUNTRIES_TO_FETCH if c not in existing_countries]

    if not countries_to_fetch:
        logger.info("All requested countries already in database. Nothing to fetch.")
        return

    logger.info(f"Will fetch: {', '.join(countries_to_fetch)}")

    # Load admin codes for countries we're actually fetching
    admin_codes = await load_admin_codes(countries_to_fetch)

    # Fetch and parse data for each country
    all_locations = []
    for country_code in countries_to_fetch:
        logger.info(f"Fetching data for {country_code}...")
        locations = await fetch_and_parse_geography(country_code, admin_codes)
        all_locations.extend(locations)

    logger.info(f"Total locations parsed: {len(all_locations)}")

    # Generate connections across all locations
    connections = generate_connections(all_locations)

    # Insert into database
    async with async_session() as session:
        geoname_to_db_id = await insert_locations(session, all_locations)

    async with async_session() as session:
        await insert_connections(session, connections, geoname_to_db_id)

    logger.info("=== Geography data acquisition complete ===")


if __name__ == "__main__":
    asyncio.run(run())
