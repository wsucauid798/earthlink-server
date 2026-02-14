"""Government / institutional adapters — UK public-sector open data.

All five sources are key-free public APIs:
  - legislation.gov.uk — UK laws and statutory instruments
  - ONS (Office for National Statistics) — demographic and economic data
  - data.gov.uk — CKAN open-data catalogue
  - UK Parliament API — Bills, members, debates
  - Police Data API — street-level crime data
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# legislation.gov.uk
# ---------------------------------------------------------------------------

class LegislationAdapter(EarthAdapter):
    """UK law from legislation.gov.uk. No API key required."""

    BASE = "https://www.legislation.gov.uk"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "legislation_gov_uk"

    @property
    def domains(self) -> list[str]:
        return ["law", "institution"]

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
                    f"{self.BASE}/search",
                    params={"text": query, "page": 1},
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("searchResults", [])
            if isinstance(results, dict):
                results = results.get("results", [])
            if isinstance(results, list):
                for item in results[:max_results]:
                    title = item.get("title", "")
                    year = item.get("year", "")
                    leg_type = item.get("type", "")
                    uri = item.get("uri", "")
                    text = f"{title} ({year})" if year else title
                    if leg_type:
                        text += f" [{leg_type}]"
                    if text.strip():
                        facts.append(EarthFact(
                            domain="law",
                            topic=title[:80].lower().replace(" ", "_"),
                            text=text,
                            source="legislation.gov.uk",
                            location_id=location_id,
                            location_name=location_name,
                            confidence=0.95,
                            timestamp=now,
                            metadata={"uri": uri},
                        ))
        except Exception as e:
            logger.warning(f"Legislation adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self.BASE}/", headers={"Accept": "application/json"})
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# ONS (Office for National Statistics)
# ---------------------------------------------------------------------------

class ONSAdapter(EarthAdapter):
    """Population and economic statistics from ONS. No API key required."""

    BASE = "https://www.ons.gov.uk/search"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "ons"

    @property
    def domains(self) -> list[str]:
        return ["demographics", "economy", "institution"]

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
                    self.BASE,
                    params={"q": search_q, "size": max_results},
                    headers={"Accept": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()

            for item in data.get("items", data.get("results", []))[:max_results]:
                title = item.get("title", item.get("description", {}).get("title", ""))
                summary = item.get("summary", item.get("description", {}).get("summary", ""))
                uri = item.get("uri", "")

                text = f"{title}. {summary}" if summary else title
                if text.strip():
                    facts.append(EarthFact(
                        domain="demographics",
                        topic=title[:80].lower().replace(" ", "_"),
                        text=text,
                        source="ons",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.9,
                        timestamp=now,
                        metadata={"uri": f"https://www.ons.gov.uk{uri}"},
                    ))
        except Exception as e:
            logger.warning(f"ONS adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get("https://www.ons.gov.uk/", headers={"Accept": "text/html"})
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# data.gov.uk (CKAN)
# ---------------------------------------------------------------------------

class DataGovUKAdapter(EarthAdapter):
    """Open datasets from data.gov.uk (CKAN). No API key required."""

    BASE = "https://ckan.publishing.service.gov.uk/api/3/action"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "data_gov_uk"

    @property
    def domains(self) -> list[str]:
        return ["open_data", "institution", "geography"]

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
                    f"{self.BASE}/package_search",
                    params={"q": search_q, "rows": max_results},
                )
                resp.raise_for_status()
                data = resp.json()

            for pkg in data.get("result", {}).get("results", [])[:max_results]:
                title = pkg.get("title", "")
                notes = pkg.get("notes", "")
                org = pkg.get("organization", {}).get("title", "")
                pkg_name = pkg.get("name", "")

                text = f"{title}. {notes[:200]}" if notes else title
                if org:
                    text += f" (Published by {org})"

                if text.strip():
                    facts.append(EarthFact(
                        domain="open_data",
                        topic=title[:80].lower().replace(" ", "_"),
                        text=text,
                        source="data.gov.uk",
                        location_id=location_id,
                        location_name=location_name,
                        confidence=0.9,
                        timestamp=now,
                        metadata={"url": f"https://www.data.gov.uk/dataset/{pkg_name}"},
                    ))
        except Exception as e:
            logger.warning(f"data.gov.uk adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self.BASE}/status_show")
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# UK Parliament API
# ---------------------------------------------------------------------------

class ParliamentAdapter(EarthAdapter):
    """Bills, members, and debates from UK Parliament API. No API key required."""

    BASE = "https://members-api.parliament.uk/api"
    BILLS_BASE = "https://bills-api.parliament.uk/api/v1"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "uk_parliament"

    @property
    def domains(self) -> list[str]:
        return ["politics", "institution", "law"]

    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
    ) -> list[EarthFact]:
        facts: list[EarthFact] = []
        now = datetime.now(timezone.utc)

        q_lower = query.lower()
        if any(w in q_lower for w in ("bill", "law", "act", "legislation")):
            facts.extend(await self._search_bills(query, location_id, location_name, max_results, now))
        else:
            facts.extend(await self._search_members(query, location_id, location_name, max_results, now))

        return facts

    async def _search_members(self, query: str, location_id: int | None, location_name: str | None, max_results: int, now: datetime) -> list[EarthFact]:
        facts: list[EarthFact] = []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self.BASE}/Members/Search",
                    params={"Name": query, "skip": 0, "take": max_results},
                )
                resp.raise_for_status()
                data = resp.json()

            for item in data.get("items", [])[:max_results]:
                value = item.get("value", {})
                name = value.get("nameDisplayAs", "")
                party = value.get("latestParty", {}).get("name", "")
                house = value.get("latestHouseMembership", {}).get("membershipFrom", "")
                membership_house = "Commons" if value.get("latestHouseMembership", {}).get("house") == 1 else "Lords"

                text = f"{name}, {party}"
                if house:
                    text += f", MP for {house}" if membership_house == "Commons" else f", House of Lords"

                facts.append(EarthFact(
                    domain="politics",
                    topic=name[:80].lower().replace(" ", "_"),
                    text=text,
                    source="uk_parliament",
                    location_id=location_id,
                    location_name=location_name,
                    confidence=0.95,
                    timestamp=now,
                    metadata={"member_id": value.get("id")},
                ))
        except Exception as e:
            logger.warning(f"Parliament members search error: {e}")
        return facts

    async def _search_bills(self, query: str, location_id: int | None, location_name: str | None, max_results: int, now: datetime) -> list[EarthFact]:
        facts: list[EarthFact] = []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self.BILLS_BASE}/Bills",
                    params={"SearchTerm": query, "Skip": 0, "Take": max_results},
                )
                resp.raise_for_status()
                data = resp.json()

            for item in data.get("items", [])[:max_results]:
                title = item.get("shortTitle", item.get("longTitle", ""))
                stage = item.get("currentStage", {}).get("description", "")
                session = item.get("currentSession", "")

                text = title
                if stage:
                    text += f" — Stage: {stage}"
                if session:
                    text += f" (Session: {session})"

                facts.append(EarthFact(
                    domain="law",
                    topic=title[:80].lower().replace(" ", "_"),
                    text=text,
                    source="uk_parliament",
                    location_id=location_id,
                    location_name=location_name,
                    confidence=0.95,
                    timestamp=now,
                    metadata={"bill_id": item.get("billId")},
                ))
        except Exception as e:
            logger.warning(f"Parliament bills search error: {e}")
        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self.BASE}/Members/Search", params={"Name": "test", "take": 1})
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Police Data API
# ---------------------------------------------------------------------------

class PoliceDataAdapter(EarthAdapter):
    """Street-level crime data from data.police.uk. No API key required."""

    BASE = "https://data.police.uk/api"

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "police_data"

    @property
    def domains(self) -> list[str]:
        return ["crime", "safety"]

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
            logger.debug("Police Data adapter requires lat/lng — skipping")
            return facts

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self.BASE}/crimes-street/all-crime",
                    params={"lat": lat, "lng": lng},
                )
                resp.raise_for_status()
                crimes = resp.json()

            # Aggregate by category
            category_counts: dict[str, int] = {}
            for crime in crimes:
                cat = crime.get("category", "other")
                category_counts[cat] = category_counts.get(cat, 0) + 1

            total = len(crimes)
            top_categories = sorted(category_counts.items(), key=lambda x: -x[1])[:max_results]

            if total > 0:
                summary_parts = [f"Total incidents near coordinates: {total}"]
                for cat, count in top_categories:
                    label = cat.replace("-", " ").title()
                    summary_parts.append(f"  {label}: {count}")

                facts.append(EarthFact(
                    domain="crime",
                    topic=f"crime_near_{location_name or 'location'}".lower().replace(" ", "_"),
                    text=". ".join(summary_parts),
                    source="police_data",
                    location_id=location_id,
                    location_name=location_name,
                    confidence=0.9,
                    timestamp=now,
                    metadata={"total_crimes": total, "categories": dict(top_categories)},
                ))

        except Exception as e:
            logger.warning(f"Police Data adapter error: {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self.BASE}/forces")
                return resp.status_code == 200
        except Exception:
            return False
