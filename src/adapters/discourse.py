"""Public discourse adapters — community knowledge and discussion.

Key-free sources:
  - Stack Exchange — Q&A across hundreds of topics
  - Mastodon / Fediverse — decentralised social media
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stack Exchange
# ---------------------------------------------------------------------------

class StackExchangeAdapter(EarthAdapter):
    """Q&A knowledge from Stack Exchange network. No API key required.

    Without a key the API is throttled (300 reqs/day per IP) which is
    fine for a world-knowledge adapter that queries on demand.
    """

    BASE = "https://api.stackexchange.com/2.3"

    # Relevant sites for a UK-centric world
    SITES = {
        "general": "stackoverflow",
        "geography": "gis",
        "travel": "travel",
        "history": "history",
        "english": "english",
        "politics": "politics",
        "science": "physics",
    }

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "stack_exchange"

    @property
    def domains(self) -> list[str]:
        return ["knowledge", "technology", "science", "history", "geography"]

    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
    ) -> list[EarthFact]:
        facts: list[EarthFact] = []
        now = datetime.now(timezone.utc)
        site = self._pick_site(query)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self.BASE}/search/excerpts",
                    params={
                        "order": "desc",
                        "sort": "relevance",
                        "q": query,
                        "site": site,
                        "pagesize": max_results,
                        "filter": "default",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            for item in data.get("items", [])[:max_results]:
                title = item.get("title", "")
                excerpt = item.get("excerpt", "")
                item_type = item.get("item_type", "question")
                question_id = item.get("question_id", "")

                import html as html_mod
                title = html_mod.unescape(title)
                excerpt = html_mod.unescape(excerpt)
                # Strip residual HTML tags from excerpts
                import re
                excerpt = re.sub(r"<[^>]+>", "", excerpt)
                text = f"{title}. {excerpt}" if excerpt else title
                if text.strip():
                    facts.append(EarthFact(
                        domain="knowledge",
                        topic=title[:80].lower().replace(" ", "_"),
                        text=text,
                        source=f"stackexchange/{site}",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.75,
                        timestamp=now,
                        metadata={
                            "site": site,
                            "type": item_type,
                            "url": f"https://{site}.stackexchange.com/q/{question_id}" if question_id else "",
                        },
                    ))
        except Exception as e:
            logger.warning(f"Stack Exchange adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self.BASE}/info", params={"site": "stackoverflow"})
                return resp.status_code == 200
        except Exception:
            return False

    def _pick_site(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ("map", "gis", "coordinate", "spatial")):
            return self.SITES["geography"]
        if any(w in q for w in ("travel", "visit", "tourism")):
            return self.SITES["travel"]
        if any(w in q for w in ("history", "medieval", "war", "ancient")):
            return self.SITES["history"]
        if any(w in q for w in ("politics", "government", "election", "parliament")):
            return self.SITES["politics"]
        if any(w in q for w in ("physics", "chemistry", "biology", "science")):
            return self.SITES["science"]
        return self.SITES["general"]


# ---------------------------------------------------------------------------
# Mastodon / Fediverse
# ---------------------------------------------------------------------------

class MastodonAdapter(EarthAdapter):
    """Public posts from Mastodon instances. No API key required for public timelines.

    Defaults to mastodon.social but can target UK-centric instances.
    """

    DEFAULT_INSTANCE = "https://mastodon.social"

    def __init__(self, instance_url: str = DEFAULT_INSTANCE, timeout: float = 10.0):
        self._instance_url = instance_url.rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "mastodon"

    @property
    def domains(self) -> list[str]:
        return ["social", "discourse", "news"]

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
            # Search public posts (v2 search endpoint, no auth for public)
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self._instance_url}/api/v2/search",
                    params={"q": query, "type": "statuses", "limit": max_results},
                )
                resp.raise_for_status()
                data = resp.json()

            for status in data.get("statuses", [])[:max_results]:
                content = status.get("content", "")
                # Strip basic HTML tags
                import re
                text = re.sub(r"<[^>]+>", "", content).strip()
                account = status.get("account", {}).get("acct", "")
                created = status.get("created_at", "")
                url = status.get("url", "")

                if text:
                    facts.append(EarthFact(
                        domain="social",
                        topic=text[:60].lower().replace(" ", "_"),
                        text=f"@{account}: {text}" if account else text,
                        source="mastodon",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.5,  # social media — lower confidence
                        timestamp=now,
                        metadata={"url": url, "created_at": created, "instance": self._instance_url},
                    ))
        except Exception as e:
            logger.warning(f"Mastodon adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._instance_url}/api/v1/instance")
                return resp.status_code == 200
        except Exception:
            return False
