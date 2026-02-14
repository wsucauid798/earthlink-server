"""Wikidata adapter — structured facts and entity relationships.

Queries the Wikidata SPARQL endpoint for structured data about locations,
institutions, people, and events. Complements Wikipedia's prose with
machine-readable facts. No API key required.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# SPARQL query: get key facts about a place by name
LOCATION_QUERY = """
SELECT ?item ?itemLabel ?itemDescription ?population ?area ?inception ?countryLabel ?adminLabel WHERE {{
  ?item rdfs:label "{name}"@en .
  ?item wdt:P17 wd:Q145 .
  OPTIONAL {{ ?item wdt:P1082 ?population . }}
  OPTIONAL {{ ?item wdt:P2046 ?area . }}
  OPTIONAL {{ ?item wdt:P571 ?inception . }}
  OPTIONAL {{ ?item wdt:P17 ?country . }}
  OPTIONAL {{ ?item wdt:P131 ?admin . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT 5
"""

# SPARQL query: general entity search
GENERAL_QUERY = """
SELECT ?item ?itemLabel ?itemDescription WHERE {{
  ?item rdfs:label "{query}"@en .
  OPTIONAL {{ ?item schema:description ?itemDescription . FILTER(LANG(?itemDescription) = "en") }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT {limit}
"""


class WikidataAdapter(EarthAdapter):
    """Structured facts from Wikidata SPARQL. No API key required."""

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "wikidata"

    @property
    def domains(self) -> list[str]:
        return ["geography", "history", "institution", "people", "culture"]

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
            if location_name:
                results = await self._query_location(location_name)
            else:
                results = await self._query_general(query, max_results)

            for result in results[:max_results]:
                label = result.get("itemLabel", {}).get("value", "")
                description = result.get("itemDescription", {}).get("value", "")
                item_uri = result.get("item", {}).get("value", "")

                parts = [f"{label}: {description}" if description else label]

                population = result.get("population", {}).get("value")
                if population:
                    parts.append(f"Population: {int(float(population)):,}")

                inception = result.get("inception", {}).get("value", "")
                if inception:
                    parts.append(f"Founded: {inception[:10]}")

                area = result.get("area", {}).get("value")
                if area:
                    parts.append(f"Area: {float(area):.1f} km²")

                text = ". ".join(parts)
                if not text.strip():
                    continue

                facts.append(EarthFact(
                    domain=self._classify(description),
                    topic=label.lower().replace(" ", "_") if label else query.lower().replace(" ", "_"),
                    text=text,
                    source="wikidata",
                    location_id=location_id,
                    location_name=location_name or label,
                    confidence=0.9,
                    timestamp=now,
                    metadata={"wikidata_uri": item_uri, "description": description},
                ))

        except Exception as e:
            logger.warning(f"Wikidata adapter error for '{query}': {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(WIKIDATA_SPARQL_URL, params={
                    "query": "SELECT ?item WHERE { ?item wdt:P31 wd:Q515 . } LIMIT 1",
                    "format": "json",
                })
                return resp.status_code == 200
        except Exception:
            return False

    async def _sparql(self, query: str) -> list[dict]:
        """Execute a SPARQL query and return bindings."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(WIKIDATA_SPARQL_URL, params={
                "query": query,
                "format": "json",
            }, headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "EarthLink/0.1 (https://github.com/earthlink; earthlink@example.com)",
            })
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", {}).get("bindings", [])

    async def _query_location(self, name: str) -> list[dict]:
        sparql = LOCATION_QUERY.format(name=name.replace('"', '\\"'))
        return await self._sparql(sparql)

    async def _query_general(self, query: str, limit: int) -> list[dict]:
        sparql = GENERAL_QUERY.format(query=query.replace('"', '\\"'), limit=limit)
        return await self._sparql(sparql)

    @staticmethod
    def _classify(description: str) -> str:
        desc = description.lower() if description else ""
        if any(w in desc for w in ("city", "town", "village", "county", "river")):
            return "geography"
        if any(w in desc for w in ("university", "school", "parliament", "government")):
            return "institution"
        if any(w in desc for w in ("politician", "king", "queen", "writer", "scientist")):
            return "people"
        if any(w in desc for w in ("battle", "war", "historical")):
            return "history"
        return "culture"
