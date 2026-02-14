"""Social / real-time adapter — Bluesky (AT Protocol).

Key-free public API for searching public posts on Bluesky.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)


class BlueskyAdapter(EarthAdapter):
    """Public posts from Bluesky via AT Protocol. No API key required for public search."""

    PUBLIC_API = "https://public.api.bsky.app"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout
        self._headers = {"User-Agent": "EarthLink/0.1"}

    @property
    def name(self) -> str:
        return "bluesky"

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
        search_q = f"{query} {location_name}" if location_name else query

        try:
            async with httpx.AsyncClient(timeout=self._timeout, headers=self._headers) as client:
                resp = await client.get(
                    f"{self.PUBLIC_API}/xrpc/app.bsky.feed.searchPosts",
                    params={"q": search_q, "limit": max_results},
                )
                resp.raise_for_status()
                data = resp.json()

            for post in data.get("posts", [])[:max_results]:
                record = post.get("record", {})
                text = record.get("text", "").strip()
                author = post.get("author", {})
                handle = author.get("handle", "")
                display_name = author.get("displayName", handle)
                created_at = record.get("createdAt", "")
                uri = post.get("uri", "")

                if text:
                    facts.append(EarthFact(
                        domain="social",
                        topic=text[:60].lower().replace(" ", "_"),
                        text=f"{display_name} (@{handle}): {text}",
                        source="bluesky",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.5,  # social posts — lower confidence
                        timestamp=now,
                        metadata={"uri": uri, "created_at": created_at, "handle": handle},
                    ))
        except Exception as e:
            logger.warning(f"Bluesky adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self.PUBLIC_API}/xrpc/app.bsky.feed.searchPosts", params={"q": "test", "limit": 1})
                return resp.status_code == 200
        except Exception:
            return False
