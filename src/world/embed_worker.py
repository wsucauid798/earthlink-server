"""Resolved-fact embedding worker (S89, feeds S80).

Decouples embedding from the resolution/tick path. EarthProxy XADDs a tiny
event each time it resolves civilisation content into Redis; this background
worker consumes the stream, reads the facts back, embeds them via TEI, and
pushes them into the world semantic layer (Chroma) with a TTL aligned to the
Redis TTL. Embedding never blocks resolution or the tick loop.

Redis consumer group so the work survives restarts and is processed once.
Degrades gracefully: if Redis/TEI/Chroma are unavailable the loop idles.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

STREAM = "earthlink:stream:resolved-facts"
GROUP = "world-embedder"
CONSUMER = "embedder-1"


class EmbedWorker:
    def __init__(self, redis, earth_proxy, world_store, block_ms: int = 2000, batch: int = 32):
        self._redis = redis
        self._earth_proxy = earth_proxy
        self._store = world_store
        self._block_ms = block_ms
        self._batch = batch
        self._running = False

    async def ensure_group(self) -> bool:
        if self._redis is None:
            return False
        try:
            await self._redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except Exception as e:
            # BUSYGROUP = already exists, which is fine.
            if "BUSYGROUP" not in str(e):
                logger.debug("xgroup_create: %s", e)
        return True

    async def run(self) -> None:
        if not await self.ensure_group():
            logger.info("EmbedWorker idle — no Redis stream available")
            return
        self._running = True
        logger.info("EmbedWorker started on %s/%s", STREAM, GROUP)
        while self._running:
            try:
                resp = await self._redis.xreadgroup(
                    GROUP, CONSUMER, {STREAM: ">"}, count=self._batch, block=self._block_ms
                )
            except Exception as e:
                logger.debug("xreadgroup failed: %s", e)
                await asyncio.sleep(1.0)
                continue
            if not resp:
                continue
            for _stream, entries in resp:
                ack_ids = []
                for entry_id, fields in entries:
                    try:
                        await self._process(fields)
                    except Exception as e:
                        logger.debug("embed worker process failed: %s", e)
                    ack_ids.append(entry_id)
                if ack_ids:
                    try:
                        await self._redis.xack(STREAM, GROUP, *ack_ids)
                    except Exception:
                        pass

    async def _process(self, fields: dict) -> None:
        """Read the resolved facts back from Redis and push them embedded into
        the world semantic layer with the resolution's TTL."""
        key = fields.get("key")
        if not key:
            return
        expires_at = float(fields.get("expires_at", 0) or 0)
        facts = await self._earth_proxy._get_facts(key)
        if not facts:
            return
        await self._store.add_facts(facts, expires_at=expires_at)

    def stop(self) -> None:
        self._running = False
