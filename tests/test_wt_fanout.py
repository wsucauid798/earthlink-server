"""Tests for the S90 Redis-Streams WebTransport fan-out (api/wt_fanout.py).

A fake Redis exercises the publish + tail-and-broadcast logic without a live
server. The key behaviours: publish is a graceful no-op without Redis, the
consumer tails with XREAD (not a consumer group) so every process sees every
tick, last-seen id advances, and malformed entries are skipped.
"""

import json

import pytest

from api.wt_fanout import STREAM, WtBroadcastConsumer, publish_tick


class _FakeRedis:
    def __init__(self, read_batches=None):
        self._batches = list(read_batches or [])
        self.xadds = []
        self.xread_args = []
        self.fail_xadd = False

    async def xadd(self, stream, fields, maxlen=None, approximate=None):
        if self.fail_xadd:
            raise RuntimeError("redis down")
        self.xadds.append((stream, fields, maxlen, approximate))
        return "1-0"

    async def xread(self, streams, block=None, count=None):
        self.xread_args.append(dict(streams))
        return self._batches.pop(0) if self._batches else []


# -- publish_tick --------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_tick_xadds_json_and_caps_length():
    redis = _FakeRedis()
    ok = await publish_tick(redis, {"tick": 3, "agent_events": []})

    assert ok is True
    stream, fields, maxlen, approximate = redis.xadds[0]
    assert stream == STREAM
    assert json.loads(fields["tick"]) == {"tick": 3, "agent_events": []}
    assert maxlen == 1024 and approximate is True


@pytest.mark.asyncio
async def test_publish_tick_no_redis_is_noop():
    assert await publish_tick(None, {"tick": 1}) is False


@pytest.mark.asyncio
async def test_publish_tick_swallows_failure():
    redis = _FakeRedis()
    redis.fail_xadd = True
    assert await publish_tick(redis, {"tick": 1}) is False


# -- WtBroadcastConsumer -------------------------------------------------


@pytest.mark.asyncio
async def test_consumer_tails_and_broadcasts_decoded_tick():
    entry = ("123-0", {"tick": json.dumps({"tick": 5, "agent_events": []})})
    redis = _FakeRedis(read_batches=[[(STREAM, [entry])]])

    got = []
    consumer = WtBroadcastConsumer(redis, None, block_ms=1)

    async def broadcast(tick_data):
        got.append(tick_data)
        consumer.stop()  # end the loop after the first tick

    consumer._broadcast = broadcast
    await consumer.run()

    assert got == [{"tick": 5, "agent_events": []}]
    assert consumer._last_id == "123-0"  # advanced past the consumed entry
    # Tailing via XREAD, started from the stream tail "$" (not a consumer group).
    assert redis.xread_args[0] == {STREAM: "$"}


@pytest.mark.asyncio
async def test_consumer_idle_without_redis():
    got = []
    consumer = WtBroadcastConsumer(None, lambda td: got.append(td))
    await consumer.run()  # returns immediately
    assert got == []


@pytest.mark.asyncio
async def test_consumer_skips_malformed_entries():
    entries = [
        ("1-0", {}),                       # no "tick" field
        ("2-0", {"tick": "not json{"}),    # unparseable
        ("3-0", {"tick": json.dumps({"tick": 9})}),  # good
    ]
    redis = _FakeRedis(read_batches=[[(STREAM, entries)]])

    got = []
    consumer = WtBroadcastConsumer(redis, None, block_ms=1)

    async def broadcast(tick_data):
        got.append(tick_data)
        consumer.stop()

    consumer._broadcast = broadcast
    await consumer.run()

    assert got == [{"tick": 9}]
    assert consumer._last_id == "3-0"  # advances even past skipped entries
