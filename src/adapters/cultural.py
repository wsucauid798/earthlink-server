"""Cultural / social adapters — UK landmarks, holidays, and local features.

Key-free sources:
  - UK Bank Holidays API — official government list
  - OpenStreetMap Overpass — POIs, transport, landmarks near a location
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UK Bank Holidays
# ---------------------------------------------------------------------------

class BankHolidaysAdapter(EarthAdapter):
    """Official UK bank holidays from gov.uk. No API key required."""

    URL = "https://www.gov.uk/bank-holidays.json"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "bank_holidays"

    @property
    def domains(self) -> list[str]:
        return ["culture", "calendar"]

    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
    ) -> list[EarthFact]:
        facts: list[EarthFact] = []
        now = datetime.now(timezone.utc)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(self.URL)
                resp.raise_for_status()
                data = resp.json()

            # Pick the right division based on location context
            division = self._pick_division(location_name)
            events = data.get(division, {}).get("events", [])

            # Get upcoming holidays (today or later)
            today = now.strftime("%Y-%m-%d")
            upcoming = [e for e in events if e.get("date", "") >= today][:max_results]
            if not upcoming:
                upcoming = events[-max_results:]  # fallback: most recent

            for event in upcoming:
                title = event.get("title", "")
                date = event.get("date", "")
                bunting = event.get("bunting", False)

                text = f"{title} — {date}"
                if bunting:
                    text += " (bunting day)"

                facts.append(EarthFact(
                    domain="culture",
                    topic=title.lower().replace(" ", "_"),
                    text=text,
                    source="gov.uk_bank_holidays",
                    location_id=location_id,
                    location_name=location_name,
                    confidence=1.0,
                    timestamp=now,
                    metadata={"date": date, "bunting": bunting, "division": division},
                ))
        except Exception as e:
            logger.warning(f"Bank Holidays adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(self.URL)
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _pick_division(location_name: str | None) -> str:
        if not location_name:
            return "england-and-wales"
        loc = location_name.lower()
        if "scotland" in loc or "edinburgh" in loc or "glasgow" in loc:
            return "scotland"
        if "northern ireland" in loc or "belfast" in loc:
            return "northern-ireland"
        return "england-and-wales"


# ---------------------------------------------------------------------------
# OpenStreetMap Overpass — local features and landmarks
# ---------------------------------------------------------------------------

class OverpassAdapter(EarthAdapter):
    """Points of interest from OpenStreetMap via Overpass API. No API key required."""

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    def __init__(self, timeout: float = 20.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "openstreetmap"

    @property
    def domains(self) -> list[str]:
        return ["geography", "culture", "transport"]

    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
        lat: float | None = None,
        lng: float | None = None,
    ) -> list[EarthFact]:
        facts: list[EarthFact] = []
        now = datetime.now(timezone.utc)

        if lat is None or lng is None:
            logger.debug("Overpass adapter requires lat/lng — skipping")
            return facts

        # Build Overpass QL query for POIs around coordinate
        radius = 2000  # 2 km
        overpass_query = self._build_query(query, lat, lng, radius, max_results)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self.OVERPASS_URL,
                    data={"data": overpass_query},
                )
                resp.raise_for_status()
                data = resp.json()

            for element in data.get("elements", [])[:max_results]:
                tags = element.get("tags", {})
                name = tags.get("name", "")
                if not name:
                    continue

                poi_type = tags.get("amenity", tags.get("tourism", tags.get("historic", tags.get("shop", ""))))
                addr = tags.get("addr:street", "")
                wiki = tags.get("wikipedia", "")

                text = name
                if poi_type:
                    text += f" ({poi_type.replace('_', ' ')})"
                if addr:
                    text += f", {addr}"

                facts.append(EarthFact(
                    domain="geography",
                    topic=name[:80].lower().replace(" ", "_"),
                    text=text,
                    source="openstreetmap",
                    location_id=location_id,
                    location_name=location_name,
                    confidence=0.85,
                    timestamp=now,
                    metadata={
                        "osm_type": element.get("type"),
                        "osm_id": element.get("id"),
                        "wikipedia": wiki,
                    },
                ))
        except Exception as e:
            logger.warning(f"Overpass adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get("https://overpass-api.de/api/status")
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _build_query(query: str, lat: float, lng: float, radius: int, limit: int) -> str:
        """Build an Overpass QL query for POIs near a coordinate."""
        q_lower = query.lower()

        # Choose tag filter based on query context
        if any(w in q_lower for w in ("museum", "gallery", "heritage", "historic")):
            tag_filter = '["tourism"~"museum|gallery|attraction"]["name"]'
        elif any(w in q_lower for w in ("restaurant", "food", "eat", "pub", "cafe")):
            tag_filter = '["amenity"~"restaurant|pub|cafe|fast_food"]["name"]'
        elif any(w in q_lower for w in ("transport", "station", "bus", "train")):
            tag_filter = '["public_transport"~"station|stop_position"]["name"]'
        elif any(w in q_lower for w in ("park", "garden", "green")):
            tag_filter = '["leisure"~"park|garden"]["name"]'
        elif any(w in q_lower for w in ("church", "cathedral", "mosque", "temple")):
            tag_filter = '["amenity"~"place_of_worship"]["name"]'
        else:
            # General: get named amenities, tourism, and historic features
            tag_filter = '["name"]'

        return (
            f"[out:json][timeout:10];\n"
            f"(\n"
            f"  node(around:{radius},{lat},{lng}){tag_filter};\n"
            f"  way(around:{radius},{lat},{lng}){tag_filter};\n"
            f");\n"
            f"out center {limit};"
        )
