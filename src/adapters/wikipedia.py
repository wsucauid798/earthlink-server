"""Wikipedia adapter — encyclopedia and reference knowledge from Earth.

Queries the Wikipedia REST API to resolve historical, cultural, and
institutional content about locations and topics. No API key required.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)

WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1"
WIKIPEDIA_SEARCH_BASE = "https://en.wikipedia.org/w/api.php"


class WikipediaAdapter(EarthAdapter):
    """Resolves civilisation content from Wikipedia."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def domains(self) -> list[str]:
        return ["history", "culture", "institution", "geography", "science", "people"]

    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
    ) -> list[EarthFact]:
        """Search Wikipedia and return structured facts."""
        facts: list[EarthFact] = []
        now = datetime.now(timezone.utc)

        try:
            titles = await self._search(query, limit=max_results)
            for title in titles:
                summary = await self._get_summary(title)
                if summary:
                    facts.append(
                        EarthFact(
                            domain=self._classify_domain(summary.get("description", "")),
                            topic=title.lower().replace(" ", "_"),
                            text=summary.get("extract", ""),
                            source="wikipedia",
                            location_id=location_id,
                            location_name=location_name or summary.get("title"),
                            confidence=0.85,
                            timestamp=now,
                            metadata={
                                "wikipedia_title": summary.get("title", title),
                                "page_url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                "description": summary.get("description", ""),
                            },
                        )
                    )
        except Exception as e:
            logger.warning(f"Wikipedia adapter error for query '{query}': {e}")

        return facts

    async def health_check(self) -> bool:
        """Check Wikipedia API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{WIKIPEDIA_API_BASE}/page/summary/London")
                return resp.status_code == 200
        except Exception:
            return False

    async def _search(self, query: str, limit: int = 5) -> list[str]:
        """Search Wikipedia for article titles matching a query."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(WIKIPEDIA_SEARCH_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            return [r["title"] for r in results]

    async def _get_summary(self, title: str) -> dict | None:
        """Get the summary extract for a Wikipedia article."""
        encoded_title = title.replace(" ", "_")
        url = f"{WIKIPEDIA_API_BASE}/page/summary/{encoded_title}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            logger.debug(f"Wikipedia summary not found for '{title}': {resp.status_code}")
            return None

    @staticmethod
    def _classify_domain(description: str) -> str:
        """Best-effort domain classification from Wikipedia description."""
        desc_lower = description.lower() if description else ""
        if any(w in desc_lower for w in ("city", "town", "village", "borough", "county", "river", "mountain")):
            return "geography"
        if any(w in desc_lower for w in ("history", "historical", "battle", "war", "medieval", "ancient")):
            return "history"
        if any(w in desc_lower for w in ("university", "school", "parliament", "government", "law", "court")):
            return "institution"
        if any(w in desc_lower for w in ("artist", "writer", "musician", "actor", "poet")):
            return "culture"
        if any(w in desc_lower for w in ("scientist", "physicist", "chemist", "biologist", "engineer")):
            return "science"
        if any(w in desc_lower for w in ("politician", "king", "queen", "monarch", "leader", "general")):
            return "people"
        return "culture"
