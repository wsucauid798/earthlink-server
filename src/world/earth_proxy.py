"""Earth proxy — the world's live connection to Earth civilisation.

The VW doesn't contain civilisation. The VW is connected to civilisation.
Civilisation stays where it is — on the internet. The world maintains
live connections to it through adapters.

When the world needs to present civilisation content, it queries Earth
through those connections in real time — like a child walking into a
library. The book doesn't get copied into the child's house first.
The library exists. The child goes to it.

Resolved content is held briefly in Redis with native TTL expiration.
Content flows through, gets served, expires automatically. Redis
handles memory limits with LRU eviction. The internet is always the
source of truth.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from adapters.base import EarthAdapter, EarthFact

logger = logging.getLogger(__name__)

# How long resolved content stays before the world asks Earth again.
# Short enough to stay live, long enough to not hammer APIs within
# the same tick window.
DEFAULT_TTL_SECONDS = 600  # 10 minutes


def _fact_to_json(fact: EarthFact) -> str:
    """Serialise an EarthFact for Redis storage."""
    return json.dumps(fact.to_dict(), default=str)


def _json_to_fact(raw: str) -> EarthFact:
    """Deserialise an EarthFact from Redis."""
    d = json.loads(raw)
    return EarthFact(
        domain=d.get("domain", ""),
        topic=d.get("topic", ""),
        text=d.get("text", ""),
        source=d.get("source", ""),
        location_id=d.get("location_id"),
        location_name=d.get("location_name"),
        confidence=d.get("confidence", 0.8),
        timestamp=datetime.fromisoformat(d["timestamp"]) if d.get("timestamp") else None,
        metadata=d.get("metadata", {}),
    )


@dataclass
class EarthProxy:
    """The world's sense organs pointed at the real Earth.

    This is a live proxy, not a download-and-store layer. Content is
    resolved from the internet on demand, held briefly in Redis with
    TTL, and evicted automatically. Redis handles memory limits with
    LRU eviction. The internet is the database. The proxy is a pipe.

    Falls back to in-memory dicts when Redis is not available (dev/test).
    """

    adapters: list[EarthAdapter] = field(default_factory=list)
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    _redis: object | None = field(default=None, repr=False)

    # Fallback in-memory store (used when Redis is not available)
    _fallback_entries: dict[str, tuple[list[EarthFact], float]] = field(default_factory=dict)

    # Stats
    _resolve_count: int = field(default=0)

    async def connect_redis(self, redis_url: str) -> None:
        """Connect to Redis. Call during server startup."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info(f"Earth proxy connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"Redis not available ({e}), falling back to in-memory TTL")
            self._redis = None

    async def disconnect_redis(self) -> None:
        """Disconnect from Redis. Call during server shutdown."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    def register_adapter(self, adapter: EarthAdapter) -> None:
        """Register an Earth data source adapter."""
        self.adapters.append(adapter)
        logger.info(f"Earth adapter registered: {adapter.name} (domains: {adapter.domains})")

    async def get_resolved_facts(self, location_id: int) -> list[EarthFact]:
        """Get live civilisation content for a location.

        Returns currently-resolved content if it hasn't expired.
        Returns empty if expired or never resolved.
        """
        key = f"earthlink:location:{location_id}"
        return await self._get_facts(key)

    async def resolve_for_location(
        self,
        location_id: int,
        location_name: str,
        force: bool = False,
        max_results_per_adapter: int = 3,
    ) -> list[EarthFact]:
        """Resolve civilisation content for a location from the live internet.

        If content was resolved recently (within TTL), returns that.
        Otherwise queries Earth through adapters in real time.
        The results live for TTL then evict — the library is always
        the source of truth, not the proxy.
        """
        key = f"earthlink:location:{location_id}"

        # Serve from Redis if still live
        if not force:
            cached = await self._get_facts(key)
            if cached:
                return cached

        if not self.adapters:
            return []

        all_facts: list[EarthFact] = []
        query = f"{location_name} United Kingdom"

        for adapter in self.adapters:
            try:
                facts = await adapter.resolve(
                    query=query,
                    location_name=location_name,
                    location_id=location_id,
                    max_results=max_results_per_adapter,
                )
                all_facts.extend(facts)
                logger.debug(
                    f"Earth proxy: {adapter.name} resolved {len(facts)} facts for {location_name}"
                )
            except Exception as e:
                logger.warning(f"Earth proxy: adapter {adapter.name} failed for {location_name}: {e}")

        await self._set_facts(key, all_facts)
        self._resolve_count += 1

        logger.info(
            f"Earth proxy: resolved {len(all_facts)} facts for {location_name} "
            f"(TTL={self.ttl_seconds}s, total resolves={self._resolve_count})"
        )
        return all_facts

    async def resolve_query(
        self,
        query: str,
        location_id: int | None = None,
        location_name: str | None = None,
        max_results_per_adapter: int = 3,
    ) -> list[EarthFact]:
        """Resolve civilisation content for a free-form query.

        Same live proxy behaviour — TTL eviction, internet is source of truth.
        """
        key = f"earthlink:query:{query.lower().strip()}"

        cached = await self._get_facts(key)
        if cached:
            return cached

        if not self.adapters:
            return []

        all_facts: list[EarthFact] = []
        for adapter in self.adapters:
            try:
                facts = await adapter.resolve(
                    query=query,
                    location_name=location_name,
                    location_id=location_id,
                    max_results=max_results_per_adapter,
                )
                all_facts.extend(facts)
            except Exception as e:
                logger.warning(f"Earth proxy: adapter {adapter.name} failed for query '{query}': {e}")

        await self._set_facts(key, all_facts)
        self._resolve_count += 1
        return all_facts

    async def is_location_resolved(self, location_id: int) -> bool:
        """Check if live civilisation content is available for a location."""
        key = f"earthlink:location:{location_id}"
        return await self._has_key(key)

    @property
    def adapter_count(self) -> int:
        return len(self.adapters)

    @property
    def total_resolves(self) -> int:
        """How many times the proxy has queried Earth since startup."""
        return self._resolve_count

    @property
    def using_redis(self) -> bool:
        return self._redis is not None

    async def health(self) -> dict[str, bool]:
        """Check health of all registered adapters."""
        results: dict[str, bool] = {}
        for adapter in self.adapters:
            try:
                results[adapter.name] = await adapter.health_check()
            except Exception:
                results[adapter.name] = False
        return results

    # ------------------------------------------------------------------
    # Storage layer — Redis when available, in-memory fallback otherwise
    # ------------------------------------------------------------------

    async def _get_facts(self, key: str) -> list[EarthFact]:
        """Get facts from Redis (or fallback). Returns [] if expired/missing."""
        if self._redis:
            try:
                raw = await self._redis.get(key)
                if raw is None:
                    return []
                items = json.loads(raw)
                return [_json_to_fact(json.dumps(item)) for item in items]
            except Exception as e:
                logger.warning(f"Redis GET failed for {key}: {e}")
                return []

        # Fallback: in-memory with manual TTL check
        entry = self._fallback_entries.get(key)
        if entry is None:
            return []
        facts, stored_at = entry
        import time
        if time.time() - stored_at > self.ttl_seconds:
            del self._fallback_entries[key]
            return []
        return list(facts)

    async def _set_facts(self, key: str, facts: list[EarthFact]) -> None:
        """Store facts in Redis with TTL (or fallback)."""
        if self._redis:
            try:
                serialised = json.dumps([f.to_dict() for f in facts], default=str)
                await self._redis.set(key, serialised, ex=self.ttl_seconds)
                return
            except Exception as e:
                logger.warning(f"Redis SET failed for {key}: {e}")

        # Fallback: in-memory
        import time
        self._fallback_entries[key] = (list(facts), time.time())

    async def _has_key(self, key: str) -> bool:
        """Check if a key exists and hasn't expired."""
        if self._redis:
            try:
                return bool(await self._redis.exists(key))
            except Exception:
                return False

        # Fallback
        entry = self._fallback_entries.get(key)
        if entry is None:
            return False
        import time
        _, stored_at = entry
        if time.time() - stored_at > self.ttl_seconds:
            del self._fallback_entries[key]
            return False
        return True
