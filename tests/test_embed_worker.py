"""Tests for the resolved-fact embed worker (S89)."""

from types import SimpleNamespace

import pytest

from world.embed_worker import EmbedWorker


class _FakeProxy:
    def __init__(self, facts):
        self._facts = facts
        self.requested = None

    async def _get_facts(self, key):
        self.requested = key
        return self._facts


class _FakeStore:
    def __init__(self):
        self.added = None

    async def add_facts(self, facts, expires_at):
        self.added = {"facts": facts, "expires_at": expires_at}
        return len(facts)


@pytest.mark.asyncio
async def test_process_reads_facts_and_pushes_to_store():
    facts = [SimpleNamespace(text="x")]
    proxy = _FakeProxy(facts)
    store = _FakeStore()
    w = EmbedWorker(redis=object(), earth_proxy=proxy, world_store=store)

    await w._process({"key": "earthlink:location:5", "expires_at": "1234.5"})

    assert proxy.requested == "earthlink:location:5"
    assert store.added["facts"] is facts
    assert store.added["expires_at"] == 1234.5


@pytest.mark.asyncio
async def test_process_noop_without_key():
    store = _FakeStore()
    w = EmbedWorker(redis=object(), earth_proxy=_FakeProxy([]), world_store=store)
    await w._process({})
    assert store.added is None


@pytest.mark.asyncio
async def test_ensure_group_false_without_redis():
    w = EmbedWorker(redis=None, earth_proxy=_FakeProxy([]), world_store=_FakeStore())
    assert await w.ensure_group() is False
