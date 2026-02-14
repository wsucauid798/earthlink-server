"""Historical / archival adapters — the deep memory of civilisation.

Key-free sources:
  - Internet Archive / Wayback Machine — archived web pages and full-text search
  - Project Gutenberg — public-domain books (full text via plain URLs)
  - British Museum Collection — objects and artefacts
  - Historic England — listed buildings and heritage assets
  - National Archives — UK government historical records
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internet Archive
# ---------------------------------------------------------------------------

class InternetArchiveAdapter(EarthAdapter):
    """Full-text search across the Internet Archive. No API key required."""

    SEARCH_URL = "https://archive.org/advancedsearch.php"

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "internet_archive"

    @property
    def domains(self) -> list[str]:
        return ["history", "culture", "knowledge", "archive"]

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
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    self.SEARCH_URL,
                    params={
                        "q": search_q,
                        "fl[]": "identifier,title,description,creator,date,mediatype",
                        "rows": max_results,
                        "page": 1,
                        "output": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            for doc in data.get("response", {}).get("docs", [])[:max_results]:
                title = doc.get("title", "")
                desc = doc.get("description", "")
                if isinstance(desc, list):
                    desc = desc[0] if desc else ""
                creator = doc.get("creator", "")
                if isinstance(creator, list):
                    creator = creator[0] if creator else ""
                date = doc.get("date", "")
                identifier = doc.get("identifier", "")
                mediatype = doc.get("mediatype", "")

                text = title
                if creator:
                    text += f" by {creator}"
                if date:
                    text += f" ({date})"
                if desc:
                    text += f". {desc[:200]}"

                if text.strip():
                    facts.append(EarthFact(
                        domain="archive",
                        topic=title[:80].lower().replace(" ", "_"),
                        text=text,
                        source="internet_archive",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.85,
                        timestamp=now,
                        metadata={
                            "url": f"https://archive.org/details/{identifier}",
                            "mediatype": mediatype,
                        },
                    ))
        except Exception as e:
            logger.warning(f"Internet Archive adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(self.SEARCH_URL, params={"q": "test", "rows": 1, "output": "json"})
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Project Gutenberg
# ---------------------------------------------------------------------------

class GutenbergAdapter(EarthAdapter):
    """Public-domain literature from Project Gutenberg. No API key required.

    Uses the Gutendex API (gutendex.com) for search.
    """

    SEARCH_URL = "https://gutendex.com/books/"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "gutenberg"

    @property
    def domains(self) -> list[str]:
        return ["literature", "history", "culture"]

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
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.get(
                    self.SEARCH_URL,
                    params={"search": query, "page": 1},
                )
                resp.raise_for_status()
                data = resp.json()

            for book in data.get("results", [])[:max_results]:
                title = book.get("title", "")
                authors = ", ".join(a.get("name", "") for a in book.get("authors", []))
                download_count = book.get("download_count", 0)
                book_id = book.get("id", "")
                subjects = book.get("subjects", [])

                text = title
                if authors:
                    text += f" by {authors}"
                if subjects:
                    text += f". Subjects: {', '.join(subjects[:3])}"
                text += f". Downloads: {download_count:,}"

                if text.strip():
                    facts.append(EarthFact(
                        domain="literature",
                        topic=title[:80].lower().replace(" ", "_"),
                        text=text,
                        source="gutenberg",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.9,
                        timestamp=now,
                        metadata={
                            "url": f"https://www.gutenberg.org/ebooks/{book_id}",
                            "gutenberg_id": book_id,
                        },
                    ))
        except Exception as e:
            logger.warning(f"Gutenberg adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(self.SEARCH_URL, params={"search": "shakespeare", "page": 1})
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# British Museum Collection
# ---------------------------------------------------------------------------

class BritishMuseumAdapter(EarthAdapter):
    """Artefact data from the British Museum collection. No API key required."""

    SEARCH_URL = "https://www.britishmuseum.org/api/_search"

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "british_museum"

    @property
    def domains(self) -> list[str]:
        return ["history", "culture", "archaeology"]

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
                resp = await client.get(
                    "https://www.britishmuseum.org/api/_search",
                    params={"keyword": query, "view": "grid", "page": 1, "size": max_results},
                )
                resp.raise_for_status()
                data = resp.json()

            for hit in data.get("hits", data.get("results", []))[:max_results]:
                source = hit.get("_source", hit) if isinstance(hit, dict) else {}
                title = source.get("title", source.get("name", ""))
                description = source.get("description", source.get("summary", ""))
                period = source.get("period", "")
                material = source.get("material", "")

                text = title
                if period:
                    text += f" ({period})"
                if material:
                    text += f" [{material}]"
                if description:
                    text += f". {description[:200]}"

                if text.strip():
                    facts.append(EarthFact(
                        domain="history",
                        topic=title[:80].lower().replace(" ", "_") if title else query[:40].lower().replace(" ", "_"),
                        text=text,
                        source="british_museum",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.85,
                        timestamp=now,
                        metadata={},
                    ))
        except Exception as e:
            logger.warning(f"British Museum adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get("https://www.britishmuseum.org/")
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Historic England — listed buildings and heritage
# ---------------------------------------------------------------------------

class HistoricEnglandAdapter(EarthAdapter):
    """Listed buildings and heritage assets from Historic England. No API key required."""

    BASE = "https://historicengland.org.uk/api"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "historic_england"

    @property
    def domains(self) -> list[str]:
        return ["heritage", "history", "architecture"]

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
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                # Search the heritage list via their search endpoint
                resp = await client.get(
                    f"{self.BASE}/search",
                    params={"query": search_q, "pageSize": max_results},
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()

            for item in data.get("results", data.get("items", []))[:max_results]:
                name = item.get("name", item.get("title", ""))
                grade = item.get("grade", "")
                list_entry = item.get("listEntry", item.get("id", ""))
                location = item.get("location", "")

                text = name
                if grade:
                    text += f" (Grade {grade})"
                if location:
                    text += f", {location}"

                if text.strip():
                    facts.append(EarthFact(
                        domain="heritage",
                        topic=name[:80].lower().replace(" ", "_") if name else query[:40].lower().replace(" ", "_"),
                        text=text,
                        source="historic_england",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.9,
                        timestamp=now,
                        metadata={"list_entry": list_entry},
                    ))
        except Exception as e:
            logger.warning(f"Historic England adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get("https://historicengland.org.uk/")
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# National Archives
# ---------------------------------------------------------------------------

class NationalArchivesAdapter(EarthAdapter):
    """UK government historical records from The National Archives. No API key required."""

    DISCOVERY_URL = "https://discovery.nationalarchives.gov.uk/API/search/v1/records"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "national_archives"

    @property
    def domains(self) -> list[str]:
        return ["history", "government", "archive"]

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
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    self.DISCOVERY_URL,
                    params={"sps.searchQuery": search_q, "sps.resultsPageSize": max_results},
                )
                resp.raise_for_status()
                data = resp.json()

            for record in data.get("records", [])[:max_results]:
                title = record.get("title", "")
                reference = record.get("reference", "")
                date_range = record.get("coveringDates", "")
                held_by = record.get("heldBy", "")
                description = record.get("description", "")

                text = title
                if reference:
                    text += f" [{reference}]"
                if date_range:
                    text += f" ({date_range})"
                if held_by:
                    text += f" — held by {held_by}"
                if description:
                    text += f". {description[:200]}"

                if text.strip():
                    facts.append(EarthFact(
                        domain="history",
                        topic=title[:80].lower().replace(" ", "_") if title else query[:40].lower().replace(" ", "_"),
                        text=text,
                        source="national_archives",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.9,
                        timestamp=now,
                        metadata={"reference": reference},
                    ))
        except Exception as e:
            logger.warning(f"National Archives adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(self.DISCOVERY_URL, params={"sps.searchQuery": "test", "sps.resultsPageSize": 1})
                return resp.status_code == 200
        except Exception:
            return False
