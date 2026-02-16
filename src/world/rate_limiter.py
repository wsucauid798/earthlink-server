"""Rate limiting for Earth adapters.

Token bucket algorithm with Redis-backed state for distributed rate limiting.
Falls back to in-memory buckets when Redis is unavailable.
"""

import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        allowed: True if the request is allowed, False if rate limited
        retry_after_seconds: If not allowed, how long to wait before retrying
    """
    allowed: bool
    retry_after_seconds: float = 0.0


class RateLimiter:
    """Token bucket rate limiter with Redis backend.

    Uses Redis for persistent, distributed-safe rate limiting. Falls back to
    in-memory token buckets when Redis is unavailable.

    The token bucket algorithm allows bursts up to the capacity while enforcing
    average rate limits over time.
    """

    def __init__(self, redis_client=None):
        """Initialize rate limiter.

        Args:
            redis_client: Optional Redis async client. If None, uses in-memory fallback.
        """
        self._redis = redis_client
        # Fallback: in-memory buckets (key -> (tokens, last_refill_time))
        self._fallback_buckets: dict[str, tuple[float, float]] = {}

    async def check_rate_limit(
        self,
        adapter_name: str,
        max_per_second: float | None,
        max_per_minute: float | None,
    ) -> RateLimitResult:
        """Check if a request is allowed under rate limits.

        Checks both per-second and per-minute limits. If either limit is
        exceeded, returns not allowed.

        Args:
            adapter_name: Name of the adapter being rate limited
            max_per_second: Maximum requests per second (None = unlimited)
            max_per_minute: Maximum requests per minute (None = unlimited)

        Returns:
            RateLimitResult: allowed=True if request can proceed, False if rate limited
        """
        if max_per_second is None and max_per_minute is None:
            return RateLimitResult(allowed=True)

        # Check per-second limit first (stricter)
        if max_per_second is not None:
            result = await self._check_limit(
                adapter_name, "second", max_per_second, 1.0
            )
            if not result.allowed:
                return result

        # Check per-minute limit
        if max_per_minute is not None:
            result = await self._check_limit(
                adapter_name, "minute", max_per_minute, 60.0
            )
            if not result.allowed:
                return result

        return RateLimitResult(allowed=True)

    async def _check_limit(
        self,
        adapter_name: str,
        window: str,
        max_requests: float,
        window_seconds: float,
    ) -> RateLimitResult:
        """Check a single rate limit window using token bucket.

        Args:
            adapter_name: Name of the adapter
            window: Window name ("second" or "minute")
            max_requests: Maximum requests in this window (capacity)
            window_seconds: Window duration in seconds

        Returns:
            RateLimitResult: allowed=True if under limit
        """
        key = f"earthlink:ratelimit:{adapter_name}:{window}"

        if self._redis:
            return await self._check_redis(key, max_requests, window_seconds)
        else:
            return self._check_fallback(key, max_requests, window_seconds)

    async def _check_redis(
        self, key: str, capacity: float, window_seconds: float
    ) -> RateLimitResult:
        """Redis-backed token bucket using atomic operations.

        Uses Redis INCR and EXPIRE for atomic token consumption.
        Key expires after window_seconds to auto-refill the bucket.

        Args:
            key: Redis key for this rate limit bucket
            capacity: Token capacity (max requests)
            window_seconds: Window duration

        Returns:
            RateLimitResult: allowed=True if tokens available
        """
        try:
            # Get current count
            current = await self._redis.get(key)
            count = float(current) if current else 0.0

            if count >= capacity:
                # Rate limited - calculate retry_after from TTL
                ttl = await self._redis.ttl(key)
                retry_after = max(0.0, float(ttl)) if ttl > 0 else window_seconds
                return RateLimitResult(allowed=False, retry_after_seconds=retry_after)

            # Increment and set/refresh expiry
            await self._redis.incr(key)
            await self._redis.expire(key, int(window_seconds))
            return RateLimitResult(allowed=True)

        except Exception as e:
            logger.warning(f"Redis rate limit check failed: {e}, allowing request (fail-open)")
            return RateLimitResult(allowed=True)  # Fail open

    def _check_fallback(
        self, key: str, capacity: float, window_seconds: float
    ) -> RateLimitResult:
        """In-memory fallback token bucket.

        Simpler implementation for development/testing when Redis unavailable.
        Not distributed-safe (each server instance has independent state).

        Args:
            key: Bucket key
            capacity: Token capacity
            window_seconds: Window duration

        Returns:
            RateLimitResult: allowed=True if tokens available
        """
        now = time.time()

        if key in self._fallback_buckets:
            tokens, last_refill = self._fallback_buckets[key]
            # Refill tokens based on elapsed time
            elapsed = now - last_refill
            if elapsed >= window_seconds:
                # Full refill
                tokens = capacity
                last_refill = now
        else:
            # New bucket - start with full capacity
            tokens = capacity
            last_refill = now

        if tokens < 1.0:
            # Rate limited
            retry_after = window_seconds - (now - last_refill)
            return RateLimitResult(allowed=False, retry_after_seconds=retry_after)

        # Consume one token
        self._fallback_buckets[key] = (tokens - 1.0, last_refill)
        return RateLimitResult(allowed=True)
