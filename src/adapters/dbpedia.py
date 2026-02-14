"""DBpedia adapter — structured data extracted from Wikipedia.

Uses the DBpedia Lookup API for entity search and description.
Complements the Wikidata adapter with Wikipedia-derived typed entities.
No API key required.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)

DBPEDIA_LOOKUP_URL = "https://lookup.dbpedia.org/api/search"


class DBpediaAdapter(EarthAdapter):
    """Structured Wikipedia-derived knowledge from DBpedia Lookup. No API key required."""

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "dbpedia"

    @property
    def domains(self) -> list[str]:
        return ["geography", "history", "culture", "people", "institution"]

    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
    ) -> list[EarthFact]:
        facts: list[EarthFact] = []
        now = datetime.now(timezone.utc)
        search_term = location_name or query

        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.get(
                    DBPEDIA_LOOKUP_URL,
                    params={"query": search_term, "format": "json", "maxResults": max_results},
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "EarthLink/0.1 (https://github.com/earthlink)",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            for doc in data.get("docs", [])[:max_results]:
                resource = _first(doc.get("resource", []))
                label = _first(doc.get("label", []))
                comment = _first(doc.get("comment", []))
                type_names = doc.get("typeName", [])

                if not label:
                    continue

                # Strip HTML bold tags from label
                label = re.sub(r"</?[Bb]>", "", label)

                text = label
                if type_names:
                    text += f" ({', '.join(type_names[:3])})"
                if comment:
                    clean_comment = re.sub(r"</?[Bb]>", "", comment)
                    text += f". {clean_comment[:400]}"
                    if len(clean_comment) > 400:
                        text += "..."

                facts.append(EarthFact(
                    domain=_classify_types(type_names),
                    topic=label[:80].lower().replace(" ", "_"),
                    text=text,
                    source="dbpedia",
                    location_id=location_id,
                    location_name=location_name,
                    confidence=0.85,
                    timestamp=now,
                    metadata={
                        "resource": resource,
                        "types": type_names[:5],
                    },
                ))

        except Exception as e:
            logger.warning(f"DBpedia adapter error for '{search_term}': {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.get(
                    DBPEDIA_LOOKUP_URL,
                    params={"query": "London", "format": "json", "maxResults": 1},
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "EarthLink/0.1",
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False


def _first(lst: list) -> str:
    """Return first element of a list or empty string."""
    return lst[0] if lst else ""


def _classify_types(type_names: list[str]) -> str:
    """Classify domain from DBpedia type names."""
    types_lower = " ".join(type_names).lower()
    if any(w in types_lower for w in ("city", "settlement", "place", "river", "mountain")):
        return "geography"
    if any(w in types_lower for w in ("person", "athlete", "artist", "politician")):
        return "people"
    if any(w in types_lower for w in ("organisation", "company", "university", "school")):
        return "institution"
    if any(w in types_lower for w in ("event", "battle", "war")):
        return "history"
    return "culture"
