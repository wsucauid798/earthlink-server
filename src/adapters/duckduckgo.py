"""DuckDuckGo adapter — the world's ability to search for anything.

Uses the DuckDuckGo Instant Answer API for quick factual lookups.
No API key required.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)

DDG_API_URL = "https://api.duckduckgo.com/"


class DuckDuckGoAdapter(EarthAdapter):
    """Quick factual answers from DuckDuckGo. No API key required."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "duckduckgo"

    @property
    def domains(self) -> list[str]:
        return ["search", "geography", "history", "culture", "science", "people"]

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
                resp = await client.get(DDG_API_URL, params={
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                })
                resp.raise_for_status()
                data = resp.json()

            # Abstract (main answer)
            abstract = data.get("AbstractText", "").strip()
            abstract_source = data.get("AbstractSource", "")
            abstract_url = data.get("AbstractURL", "")
            if abstract:
                facts.append(EarthFact(
                    domain="search",
                    topic=query.lower().replace(" ", "_"),
                    text=abstract,
                    source="duckduckgo",
                    location_id=location_id,
                    location_name=location_name,
                    confidence=0.8,
                    timestamp=now,
                    metadata={"abstract_source": abstract_source, "url": abstract_url},
                ))

            # Answer (direct factual answer)
            answer = data.get("Answer", "").strip()
            if answer and answer != abstract:
                facts.append(EarthFact(
                    domain="search",
                    topic=f"{query.lower().replace(' ', '_')}_answer",
                    text=answer,
                    source="duckduckgo",
                    location_id=location_id,
                    location_name=location_name,
                    confidence=0.85,
                    timestamp=now,
                    metadata={"answer_type": data.get("AnswerType", "")},
                ))

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results - len(facts)]:
                if isinstance(topic, dict) and "Text" in topic:
                    text = topic["Text"].strip()
                    if text:
                        facts.append(EarthFact(
                            domain="search",
                            topic=text[:60].lower().replace(" ", "_"),
                            text=text,
                            source="duckduckgo",
                            location_id=location_id,
                            location_name=location_name,
                            confidence=0.7,
                            timestamp=now,
                            metadata={"url": topic.get("FirstURL", "")},
                        ))

        except Exception as e:
            logger.warning(f"DuckDuckGo adapter error for '{query}': {e}")

        return facts[:max_results]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(DDG_API_URL, params={"q": "London", "format": "json"})
                return resp.status_code == 200
        except Exception:
            return False
