"""Base adapter interface.

Every Earth adapter implements this contract. The world's EarthProxy
calls adapters to resolve civilisation content on demand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EarthFact:
    """A single piece of civilisation content resolved from Earth.

    This is what the world presents to agents as native world content.
    Agents see EarthFacts the same way they see weather or geography.
    """

    domain: str  # e.g. "history", "culture", "news", "institution", "geography"
    topic: str  # short label, e.g. "founding", "population_2024", "rail_strike"
    text: str  # human-readable content the agent will observe
    source: str  # adapter name, e.g. "wikipedia", "bbc_rss", "ollama"
    location_id: int | None = None  # location this fact is relevant to (if any)
    location_name: str | None = None  # readable name for provenance
    confidence: float = 0.8  # how much to trust this (0.0–1.0)
    timestamp: datetime | None = None  # when this was resolved from Earth
    metadata: dict = field(default_factory=dict)  # adapter-specific extras

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "topic": self.topic,
            "text": self.text,
            "source": self.source,
            "location_id": self.location_id,
            "location_name": self.location_name,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }


class EarthAdapter(ABC):
    """Interface for all Earth data source adapters.

    Each adapter knows how to query one class of Earth data source
    and return structured EarthFacts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique adapter name, e.g. 'wikipedia', 'bbc_rss'."""
        ...

    @property
    @abstractmethod
    def domains(self) -> list[str]:
        """Content domains this adapter can serve, e.g. ['history', 'culture']."""
        ...

    @abstractmethod
    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
    ) -> list[EarthFact]:
        """Resolve Earth content for a query.

        Args:
            query: What to look up (e.g. "history of London", "current news Manchester").
            location_name: Optional location name for context.
            location_id: Optional location ID for tagging results.
            max_results: Maximum number of facts to return.

        Returns:
            List of EarthFacts resolved from this source.
        """
        ...

    async def health_check(self) -> bool:
        """Check if this adapter's data source is reachable.

        Returns True if the source is available, False otherwise.
        Default implementation returns True (optimistic).
        """
        return True
