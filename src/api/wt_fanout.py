"""Redis Streams fan-out for WebTransport broadcast (S90).

The world tick loop runs in ONE process, but WT subscribers may be served by
several. Publishing each tick once to a Redis stream lets every WT-serving
process tail the stream and fan the tick out to its own local subscribers
(S69's per-session streams/datagrams).

This is **broadcast, not work-sharing**: every process must see *every* tick,
so each tails the stream with ``XREAD`` from its own last-seen id. A consumer
group would be wrong here — it splits entries across consumers, so each process
would serve only a fraction of the ticks and subscribers would miss frames.
(Contrast S89's embed worker, which *does* want a group: each resolved-fact
event should be embedded once, not once per process.)

The stream is length-capped: only recent ticks matter, a late-joining process
just starts at the tail, and the stream can't grow without bound.

Degrades gracefully: with no Redis the publisher is a no-op and the caller
broadcasts in-process directly (the single-process dev path).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

STREAM = "earthlink:stream:wt-broadcast"
# Keep roughly this many recent ticks on the stream (approximate trimming is
# cheaper for Redis and exactness doesn't matter for a broadcast tail).
STREAM_MAXLEN = 1024


async def publish_tick(redis, tick_data: dict) -> bool:
    """Publish one tick to the broadcast stream. Returns True if published.

    A no-op returning False when Redis is unavailable or the XADD fails, so the
    caller can fall back to an in-process broadcast.
    """
    if redis is None:
        return False
    try:
        await redis.xadd(
            STREAM,
            {"tick": json.dumps(tick_data, default=str)},
            maxlen=STREAM_MAXLEN,
            approximate=True,
        )
        return True
    except Exception as e:
        logger.debug("wt publish_tick xadd failed: %s", e)
        return False


class WtBroadcastConsumer:
    """Tails the broadcast stream and fans each tick out to local subscribers.

    ``broadcast`` is an async callable taking the decoded tick dict (in
    practice ``_WtConnectionManager.broadcast``). Starts at the stream tail
    (``$``) so a freshly-started process serves ticks from now on rather than
    replaying history.
    """

    def __init__(
        self,
        redis,
        broadcast: Callable[[dict], Awaitable[None]],
        *,
        block_ms: int = 2000,
        count: int = 16,
        start_id: str = "$",
    ):
        self._redis = redis
        self._broadcast = broadcast
        self._block_ms = block_ms
        self._count = count
        self._last_id = start_id
        self._running = False

    async def run(self) -> None:
        if self._redis is None:
            logger.info("WtBroadcastConsumer idle — no Redis; broadcasting in-process")
            return
        self._running = True
        logger.info("WtBroadcastConsumer tailing %s", STREAM)
        while self._running:
            try:
                resp = await self._redis.xread(
                    {STREAM: self._last_id}, block=self._block_ms, count=self._count
                )
            except Exception as e:
                logger.debug("wt xread failed: %s", e)
                await asyncio.sleep(1.0)
                continue
            if not resp:
                continue
            for _stream, entries in resp:
                for entry_id, fields in entries:
                    self._last_id = entry_id
                    raw = fields.get("tick")
                    if not raw:
                        continue
                    try:
                        tick_data = json.loads(raw)
                    except Exception:
                        continue
                    try:
                        await self._broadcast(tick_data)
                    except Exception as e:
                        logger.debug("wt fan-out broadcast failed: %s", e)

    def stop(self) -> None:
        self._running = False
