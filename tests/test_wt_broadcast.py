"""Tests for the S68/S69 WebTransport broadcast topology (api/wt.py).

No live QUIC stack: the H3 connection and protocol are faked, so we exercise
the routing logic — splitting a tick into per-class reliable streams vs
position datagrams, length-prefixed framing, and per-class stream reuse. The
aioquic wire behaviour is verified at deploy (S70 parity test).
"""

import json

from api.wt import (
    CHANNEL_AGENTS,
    CHANNEL_META,
    CHANNEL_WORLD,
    _WtSession,
    split_tick,
)


def _tick(**overrides):
    data = {
        "tick": 7,
        "time": {"iso": "2026-05-29T12:00:00"},
        "rotation": {"gmst": 1.0},
        "orbital": None,
        "solar_activity": None,
        "earth_proxy_resolves": 42,
        "weather_updated": True,
        "wind_updated": False,
        "agent_events": [
            {"agent_id": "a1", "lat": 51.5, "lng": -0.1, "location_id": 3, "moved": True, "energy": 90},
            {"agent_id": "a2", "lat": None, "lng": None, "location_id": 5, "moved": False, "energy": 80},
        ],
    }
    data.update(overrides)
    return data


# -- split_tick ----------------------------------------------------------


def test_split_meta_always_present_and_carries_world_state():
    channels, positions = split_tick(_tick())
    meta = channels[CHANNEL_META]
    assert meta["tick"] == 7
    assert meta["earth_proxy_resolves"] == 42
    assert meta["rotation"] == {"gmst": 1.0}


def test_split_world_events_only_for_set_flags():
    channels, _ = split_tick(_tick())
    # weather_updated True is included; wind_updated False is omitted.
    assert channels[CHANNEL_WORLD] == {"tick": 7, "weather_updated": True}


def test_split_world_events_absent_when_no_refresh():
    channels, _ = split_tick(_tick(weather_updated=False))
    assert CHANNEL_WORLD not in channels


def test_split_positions_only_for_movers_with_coords():
    _, positions = split_tick(_tick())
    # a1 moved with coords → datagram; a2 stationary → excluded.
    assert positions == [["a1", 51.5, -0.1, 3]]


def test_split_agents_stream_carries_all_events():
    channels, _ = split_tick(_tick())
    events = channels[CHANNEL_AGENTS]["events"]
    assert [e["agent_id"] for e in events] == ["a1", "a2"]


def test_split_no_agents_channel_when_empty():
    channels, positions = split_tick(_tick(agent_events=[]))
    assert CHANNEL_AGENTS not in channels
    assert positions == []


# -- _WtSession routing --------------------------------------------------


class _FakeQuic:
    def __init__(self):
        self.sent = []  # (stream_id, data, end_stream)

    def send_stream_data(self, stream_id, data, end_stream):
        self.sent.append((stream_id, data, end_stream))


class _FakeConnection:
    def __init__(self):
        self._quic = _FakeQuic()
        self._next = 100
        self.created = []      # (session_id, is_unidirectional)
        self.datagrams = []    # (session_id, data)

    def create_webtransport_stream(self, session_id, is_unidirectional):
        self.created.append((session_id, is_unidirectional))
        stream_id = self._next
        self._next += 4
        return stream_id

    def send_datagram(self, session_id, data):
        self.datagrams.append((session_id, data))


class _FakeProtocol:
    def __init__(self):
        self.transmits = 0

    def transmit(self):
        self.transmits += 1


def _session():
    proto, conn = _FakeProtocol(), _FakeConnection()
    return _WtSession(protocol=proto, connection=conn, session_id=4), proto, conn


def _unframe(data: bytes):
    length = int.from_bytes(data[:4], "big")
    body = data[4 : 4 + length]
    assert len(body) == length  # frame length prefix matches body
    return json.loads(body)


def test_send_tick_routes_to_streams_and_datagram():
    session, proto, conn = _session()
    session.send_tick(_tick())

    # Three reliable channels this tick (meta, world_events, agents), one stream each.
    assert len(conn.created) == 3
    assert {sid for sid, _ in conn.created} == {4}  # all under this session
    assert len(conn._quic.sent) == 3
    # One position datagram for the single mover.
    assert len(conn.datagrams) == 1
    assert _unframe(conn.datagrams[0][1]) == [["a1", 51.5, -0.1, 3]]
    # transmit() fired once per stream send + once for the datagram.
    assert proto.transmits == 4


def test_streams_reused_across_ticks_not_reopened():
    session, _, conn = _session()
    session.send_tick(_tick())
    session.send_tick(_tick())

    # Same three channels both ticks → streams opened once, reused the second time.
    assert len(conn.created) == 3
    assert len(conn._quic.sent) == 6  # 3 channels x 2 ticks
    # Each channel's two frames land on the same stream id.
    by_stream = {}
    for stream_id, data, end in conn._quic.sent:
        by_stream.setdefault(stream_id, []).append(data)
        assert end is False  # per-class stream stays open
    assert sorted(len(frames) for frames in by_stream.values()) == [2, 2, 2]


def test_frames_are_length_prefixed_json():
    session, _, conn = _session()
    session.send_event(CHANNEL_META, {"tick": 1, "hello": "world"})
    assert _unframe(conn._quic.sent[0][1]) == {"tick": 1, "hello": "world"}


def test_no_datagram_when_no_movers():
    session, _, conn = _session()
    session.send_tick(_tick(agent_events=[
        {"agent_id": "a2", "lat": None, "lng": None, "location_id": 5, "moved": False},
    ]))
    assert conn.datagrams == []
