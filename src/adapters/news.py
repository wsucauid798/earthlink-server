"""News / current-events adapter — what is happening on Earth right now.

Fetches UK news from BBC and Reuters RSS feeds. No API key required.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)

BBC_RSS_FEEDS = {
    "top": "https://feeds.bbci.co.uk/news/rss.xml",
    "uk": "https://feeds.bbci.co.uk/news/uk/rss.xml",
    "england": "https://feeds.bbci.co.uk/news/england/rss.xml",
    "scotland": "https://feeds.bbci.co.uk/news/scotland/rss.xml",
    "wales": "https://feeds.bbci.co.uk/news/wales/rss.xml",
    "northern_ireland": "https://feeds.bbci.co.uk/news/northern_ireland/rss.xml",
    "politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
}

REUTERS_RSS_FEEDS = {
    "world": "https://www.reutersagency.com/feed/?best-topics=world&post_type=best",
    "uk": "https://www.reutersagency.com/feed/?best-regions=europe&post_type=best",
}


def _parse_rss(xml_text: str, source: str, location_id: int | None, location_name: str | None, max_items: int) -> list[EarthFact]:
    """Parse an RSS XML feed into EarthFacts."""
    facts: list[EarthFact] = []
    now = datetime.now(timezone.utc)

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return facts

    # RSS items live under channel/item
    channel = root.find("channel")
    if channel is None:
        return facts

    for item in channel.findall("item")[:max_items]:
        title = item.findtext("title", "").strip()
        description = item.findtext("description", "").strip()
        link = item.findtext("link", "").strip()

        if not title:
            continue

        text = f"{title}. {description}" if description else title

        facts.append(EarthFact(
            domain="news",
            topic=title[:80].lower().replace(" ", "_"),
            text=text,
            source=source,
            location_id=location_id,
            location_name=location_name,
            confidence=0.9,
            timestamp=now,
            metadata={"url": link, "title": title},
        ))

    return facts


class BBCNewsAdapter(EarthAdapter):
    """UK news from BBC RSS feeds. No API key required."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "bbc_news"

    @property
    def domains(self) -> list[str]:
        return ["news", "politics", "science"]

    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
    ) -> list[EarthFact]:
        """Fetch BBC news. Picks feed based on query/location context."""
        feed_url = self._pick_feed(query, location_name)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(feed_url)
                resp.raise_for_status()
                return _parse_rss(resp.text, "bbc_news", location_id, location_name, max_results)
        except Exception as e:
            logger.warning(f"BBC News adapter error: {e}")
            return []

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(BBC_RSS_FEEDS["top"])
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _pick_feed(query: str, location_name: str | None) -> str:
        """Select the most relevant BBC feed for the query context."""
        context = f"{query} {location_name or ''}".lower()
        if "scotland" in context:
            return BBC_RSS_FEEDS["scotland"]
        if "wales" in context or "cardiff" in context:
            return BBC_RSS_FEEDS["wales"]
        if "northern ireland" in context or "belfast" in context:
            return BBC_RSS_FEEDS["northern_ireland"]
        if "politic" in context or "parliament" in context or "government" in context:
            return BBC_RSS_FEEDS["politics"]
        if "science" in context or "environment" in context or "climate" in context:
            return BBC_RSS_FEEDS["science"]
        return BBC_RSS_FEEDS["uk"]


class ReutersAdapter(EarthAdapter):
    """International and UK news from Reuters RSS. No API key required."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "reuters"

    @property
    def domains(self) -> list[str]:
        return ["news"]

    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
    ) -> list[EarthFact]:
        feed_url = REUTERS_RSS_FEEDS["uk"]
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(feed_url)
                resp.raise_for_status()
                return _parse_rss(resp.text, "reuters", location_id, location_name, max_results)
        except Exception as e:
            logger.warning(f"Reuters adapter error: {e}")
            return []

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(REUTERS_RSS_FEEDS["uk"])
                return resp.status_code == 200
        except Exception:
            return False
