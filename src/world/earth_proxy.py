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

import asyncio
import json
import logging
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from adapters.base import EarthAdapter, EarthFact
from world.config import AdapterPolicies, FallbackBehavior
from world.rate_limiter import RateLimiter

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
    policies: AdapterPolicies = field(default_factory=AdapterPolicies)
    default_timeout_seconds: float = 10.0  # hard wall-clock cap per adapter call
    max_concurrent_adapters: int = 8       # semaphore width for adapter fan-out per location
    _redis: object | None = field(default=None, repr=False)
    _rate_limiter: RateLimiter | None = field(default=None, repr=False)
    _adapter_semaphore: asyncio.Semaphore | None = field(default=None, repr=False)

    # Fallback in-memory store (used when Redis is not available)
    _fallback_entries: dict[str, tuple[list[EarthFact], float]] = field(default_factory=dict)

    # Stats
    _resolve_count: int = field(default=0)
    _rate_limited_count: dict[str, int] = field(default_factory=dict)
    _disabled_adapters: set[str] = field(default_factory=set)
    _timeout_count: dict[str, int] = field(default_factory=dict)
    _adapter_latency_sum: dict[str, float] = field(default_factory=dict)
    _adapter_call_count: dict[str, int] = field(default_factory=dict)

    def _get_adapter_semaphore(self) -> asyncio.Semaphore:
        """Lazy-init the adapter semaphore (must be created inside a running loop)."""
        if self._adapter_semaphore is None:
            self._adapter_semaphore = asyncio.Semaphore(self.max_concurrent_adapters)
        return self._adapter_semaphore

    async def connect_redis(self, redis_url: str) -> None:
        """Connect to Redis. Call during server startup."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            await self._redis.ping()
            # Initialize rate limiter with Redis connection
            self._rate_limiter = RateLimiter(self._redis)  # NEW
            logger.info(f"Earth proxy connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"Redis not available ({e}), falling back to in-memory TTL")
            self._redis = None
            # Initialize rate limiter with in-memory fallback
            self._rate_limiter = RateLimiter(None)  # NEW

    async def disconnect_redis(self) -> None:
        """Disconnect from Redis. Call during server shutdown."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    @property
    def redis(self):
        """The shared async Redis client (or None when unavailable). Reused by
        the WT broadcast fan-out (S90) so it doesn't open a second connection."""
        return self._redis

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

        query = location_name

        # Run all adapters concurrently — the single biggest throughput win.
        # Each adapter is independent (different HTTP endpoint), so there is
        # no reason to wait for one before starting the next.
        results = await asyncio.gather(
            *(
                self._resolve_with_policy(
                    adapter, query, location_name, location_id, max_results_per_adapter
                )
                for adapter in self.adapters
            ),
            return_exceptions=True,
        )

        all_facts: list[EarthFact] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(f"Adapter {self.adapters[i].name} raised: {result}")
            else:
                all_facts.extend(result)

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

        results = await asyncio.gather(
            *(
                self._resolve_with_policy(
                    adapter, query, location_name, location_id, max_results_per_adapter
                )
                for adapter in self.adapters
            ),
            return_exceptions=True,
        )

        all_facts: list[EarthFact] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(f"Adapter {self.adapters[i].name} raised: {result}")
            else:
                all_facts.extend(result)

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

    def get_policy_stats(self) -> dict:
        """Get statistics about policy enforcement.

        Returns metrics on rate limiting, disabled adapters, timeouts,
        and per-adapter latency for monitoring and debugging.
        """
        adapter_stats = {}
        for a in self.adapters:
            name = a.name
            calls = self._adapter_call_count.get(name, 0)
            adapter_stats[name] = {
                "calls": calls,
                "avg_latency_ms": round(
                    (self._adapter_latency_sum.get(name, 0) / calls) * 1000, 1
                ) if calls else 0,
                "timeouts": self._timeout_count.get(name, 0),
                "rate_limited": self._rate_limited_count.get(name, 0),
            }
        return {
            "total_resolves": self._resolve_count,
            "rate_limited_counts": dict(self._rate_limited_count),
            "timeout_counts": dict(self._timeout_count),
            "disabled_adapters": list(self._disabled_adapters),
            "enabled_adapter_count": len([
                a for a in self.adapters
                if self.policies.get_policy(a.name).enabled
            ]),
            "adapters": adapter_stats,
        }

    # ------------------------------------------------------------------
    # Policy enforcement — rate limits, fallback behavior, enable/disable
    # ------------------------------------------------------------------

    async def _resolve_with_policy(
        self,
        adapter: EarthAdapter,
        query: str,
        location_name: str | None,
        location_id: int | None,
        max_results: int,
    ) -> list[EarthFact]:
        """Resolve from adapter with policy enforcement.

        Wraps adapter.resolve() with all policy checks:
        1. Check if adapter is enabled
        2. Enforce rate limits
        3. Acquire adapter concurrency semaphore
        4. Call adapter.resolve() with a hard wall-clock timeout
        5. Track latency and timeout stats

        This is the single enforcement point for all adapter policies.
        """
        policy = self.policies.get_policy(adapter.name)

        # 1. Check if adapter is enabled
        if not policy.enabled:
            if adapter.name not in self._disabled_adapters:
                logger.info(f"Adapter {adapter.name} is disabled by policy")
                self._disabled_adapters.add(adapter.name)
            return []

        # 2. Check rate limits
        if self._rate_limiter:
            rate_check = await self._rate_limiter.check_rate_limit(
                adapter.name,
                policy.max_requests_per_second,
                policy.max_requests_per_minute,
            )
            if not rate_check.allowed:
                self._rate_limited_count[adapter.name] = \
                    self._rate_limited_count.get(adapter.name, 0) + 1
                return await self._handle_rate_limit(
                    adapter, policy, rate_check.retry_after_seconds
                )

        # 3. Resolve timeout: per-adapter policy overrides proxy default
        timeout = policy.timeout_seconds or self.default_timeout_seconds

        # 4. Call adapter behind semaphore + hard timeout
        sem = self._get_adapter_semaphore()
        t0 = _time.monotonic()
        try:
            async with sem:
                facts = await asyncio.wait_for(
                    adapter.resolve(
                        query=query,
                        location_name=location_name,
                        location_id=location_id,
                        max_results=max_results,
                    ),
                    timeout=timeout,
                )
            elapsed = _time.monotonic() - t0
            self._adapter_latency_sum[adapter.name] = \
                self._adapter_latency_sum.get(adapter.name, 0) + elapsed
            self._adapter_call_count[adapter.name] = \
                self._adapter_call_count.get(adapter.name, 0) + 1
            return facts

        except asyncio.TimeoutError:
            elapsed = _time.monotonic() - t0
            self._timeout_count[adapter.name] = \
                self._timeout_count.get(adapter.name, 0) + 1
            self._adapter_call_count[adapter.name] = \
                self._adapter_call_count.get(adapter.name, 0) + 1
            logger.warning(
                f"Adapter {adapter.name} timed out after {elapsed:.1f}s "
                f"(limit={timeout}s, total timeouts={self._timeout_count[adapter.name]})"
            )
            return []

        except Exception as e:
            elapsed = _time.monotonic() - t0
            self._adapter_latency_sum[adapter.name] = \
                self._adapter_latency_sum.get(adapter.name, 0) + elapsed
            self._adapter_call_count[adapter.name] = \
                self._adapter_call_count.get(adapter.name, 0) + 1
            return await self._handle_adapter_failure(adapter, policy, e)

    async def _handle_rate_limit(
        self,
        adapter: EarthAdapter,
        policy,
        retry_after: float,
    ) -> list[EarthFact]:
        """Handle rate limit based on fallback policy.

        Args:
            adapter: The adapter that was rate limited
            policy: AdapterPolicy with fallback strategy
            retry_after: Seconds until rate limit resets

        Returns:
            list[EarthFact]: Empty list (future: USE_STALE could return cached content)
        """
        if policy.fallback == FallbackBehavior.LOG_WARNING:
            logger.warning(
                f"Adapter {adapter.name} rate limited, retry after {retry_after:.1f}s "
                f"(total rate limits: {self._rate_limited_count.get(adapter.name, 0)})"
            )
        elif policy.fallback == FallbackBehavior.SKIP:
            # Silent skip
            pass
        elif policy.fallback == FallbackBehavior.USE_STALE:
            # Future: attempt to retrieve stale cache
            logger.warning(
                f"Adapter {adapter.name} rate limited, USE_STALE not yet implemented"
            )
        # RETRY_AFTER not implemented (would block)

        return []

    async def _handle_adapter_failure(
        self,
        adapter: EarthAdapter,
        policy,
        error: Exception,
    ) -> list[EarthFact]:
        """Handle adapter failure based on fallback policy.

        Args:
            adapter: The adapter that failed
            policy: AdapterPolicy with fallback strategy
            error: The exception that was raised

        Returns:
            list[EarthFact]: Empty list (future: USE_STALE could return cached content)
        """
        if policy.fallback == FallbackBehavior.LOG_WARNING:
            logger.warning(f"Earth proxy: adapter {adapter.name} failed: {error}")
        elif policy.fallback == FallbackBehavior.SKIP:
            # Silent skip
            pass
        elif policy.fallback == FallbackBehavior.USE_STALE:
            # Future: attempt to retrieve stale cache
            logger.debug(f"Adapter {adapter.name} failed, USE_STALE not yet implemented")
        # RETRY_AFTER not applicable to failures

        return []

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
        import time
        if self._redis:
            try:
                serialised = json.dumps([f.to_dict() for f in facts], default=str)
                await self._redis.set(key, serialised, ex=self.ttl_seconds)
                # S80/S89: announce the resolved facts on a stream so the
                # embed worker can index them into the world semantic layer
                # off the resolution path. Fire-and-forget, never blocks.
                await self._publish_resolved(key, time.time() + self.ttl_seconds)
                return
            except Exception as e:
                logger.warning(f"Redis SET failed for {key}: {e}")

        # Fallback: in-memory
        self._fallback_entries[key] = (list(facts), time.time())

    async def _publish_resolved(self, key: str, expires_at: float) -> None:
        """Publish a resolved-fact event to the embed-pipeline stream (S89)."""
        if not self._redis:
            return
        try:
            await self._redis.xadd(
                "earthlink:stream:resolved-facts",
                {"key": key, "expires_at": str(expires_at)},
                maxlen=10000,
                approximate=True,
            )
        except Exception as e:
            logger.debug(f"resolved-facts xadd failed for {key}: {e}")

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
