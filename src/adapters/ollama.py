"""Ollama adapter — LLM as civilisation substrate.

Large language models are compressed representations of civilisation's
written output. The world uses a local LLM (via Ollama) as another
Earth adapter — not as agent brains, but as a world-side knowledge
source for general civilisation knowledge.

One adapter, one API, any model. No external API keys, no rate limits,
no token costs, full control. Model choice is a config decision.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from .base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.2"


class OllamaAdapter(EarthAdapter):
    """Local LLM civilisation knowledge via Ollama. No API key required."""

    def __init__(
        self,
        base_url: str = OLLAMA_DEFAULT_URL,
        model: str = OLLAMA_DEFAULT_MODEL,
        timeout: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def domains(self) -> list[str]:
        return ["history", "culture", "institution", "geography", "science", "people", "philosophy"]

    async def resolve(
        self,
        query: str,
        location_name: str | None = None,
        location_id: int | None = None,
        max_results: int = 5,
    ) -> list[EarthFact]:
        facts: list[EarthFact] = []
        now = datetime.now(timezone.utc)

        prompt = self._build_prompt(query, location_name)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 512},
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            response_text = data.get("response", "").strip()
            if response_text:
                facts.append(EarthFact(
                    domain=self._classify(query),
                    topic=query[:80].lower().replace(" ", "_"),
                    text=response_text,
                    source="ollama",
                    location_id=location_id,
                    location_name=location_name,
                    confidence=0.7,  # LLM knowledge is broad but not always precise
                    timestamp=now,
                    metadata={"model": self._model, "prompt": prompt},
                ))

        except httpx.ConnectError:
            logger.debug("Ollama not available (connection refused) — skipping")
        except Exception as e:
            logger.warning(f"Ollama adapter error for '{query}': {e}")

        return facts

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _build_prompt(query: str, location_name: str | None) -> str:
        context = f" about {location_name} in the United Kingdom" if location_name else ""
        return (
            f"You are a knowledgeable reference source{context}. "
            f"Answer concisely and factually in 2-3 sentences. "
            f"Question: {query}"
        )

    @staticmethod
    def _classify(query: str) -> str:
        q = query.lower()
        if any(w in q for w in ("history", "founded", "origin", "medieval", "ancient")):
            return "history"
        if any(w in q for w in ("university", "school", "government", "parliament", "law")):
            return "institution"
        if any(w in q for w in ("population", "geography", "river", "mountain", "city")):
            return "geography"
        if any(w in q for w in ("science", "physics", "biology", "chemistry")):
            return "science"
        if any(w in q for w in ("who", "person", "king", "queen", "writer")):
            return "people"
        return "culture"
