"""WebTransport (QUIC/HTTP3) stream endpoint for world tick broadcasting.

This module runs an aioquic server alongside FastAPI and exposes:
  CONNECT /wt/world  (:protocol = webtransport)

Broadcast topology (S68/S69):
  - Agent positions ride unreliable WT **datagrams** (latest-wins): a dropped
    position is harmless because the next tick supersedes it, and datagrams
    avoid head-of-line blocking entirely.
  - Everything else rides **independent reliable unidirectional streams, one
    per event class** (`meta`, `agents`, `world_events`, and the reserved
    `social_events`/`traces`). A stalled or congested class can't block another
    because each is its own QUIC stream. Each stream is long-lived and carries a
    sequence of length-prefixed JSON frames (4-byte big-endian length + body),
    so a client demuxes per class without re-opening streams every tick.

  The earlier spike sent the whole tick as one JSON blob on a fresh stream per
  message; that is what S68/S69 replace.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from email.utils import formatdate
from urllib.parse import urlsplit
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)


# -- S68 stream topology -------------------------------------------------
#
# Event classes, each an independent reliable WT stream. Agent positions are
# NOT here — they go over datagrams (see split_tick).
CHANNEL_META = "meta"            # world heartbeat: tick, time, rotation, orbital, solar
CHANNEL_AGENTS = "agents"        # per-agent state deltas (authoritative, reliable)
CHANNEL_WORLD = "world_events"   # environment refreshes (weather/wind/atmosphere/…)
CHANNEL_SOCIAL = "social_events" # reserved: dialogue/teaching once the tick surfaces it
CHANNEL_TRACES = "traces"        # reserved: diagnostic traces
CHANNELS = (CHANNEL_META, CHANNEL_AGENTS, CHANNEL_WORLD, CHANNEL_SOCIAL, CHANNEL_TRACES)

# Domains whose "<x>_updated" flags, when set, become a world_events payload.
_WORLD_REFRESH_FLAGS = (
    "weather_updated",
    "wind_updated",
    "atmosphere_updated",
    "astronomy_updated",
    "data_feeds_updated",
)


def split_tick(tick_data: dict) -> tuple[dict[str, dict], list[list]]:
    """Split one world tick into per-channel reliable payloads + position datagrams.

    Returns ``(channels, positions)`` where ``channels`` maps an event class to
    its JSON payload (classes with nothing to say this tick are omitted) and
    ``positions`` is a compact latest-wins list ``[[agent_id, lat, lng,
    location_id], …]`` for agents that moved. Positions are intentionally also
    reflected in the authoritative ``agents`` stream; the datagram is the fast,
    lossy path for live map movement, the stream is the reliable record.

    ``social_events``/``traces`` are part of the topology but not emitted yet —
    dialogue/teaching currently live on agent knowledge, not the tick payload —
    so they slot in here unchanged once the world tick surfaces them.
    """
    tick = tick_data.get("tick")
    channels: dict[str, dict] = {
        CHANNEL_META: {
            "tick": tick,
            "time": tick_data.get("time"),
            "rotation": tick_data.get("rotation"),
            "orbital": tick_data.get("orbital"),
            "solar_activity": tick_data.get("solar_activity"),
            "earth_proxy_resolves": tick_data.get("earth_proxy_resolves"),
        }
    }

    refreshed = {flag: True for flag in _WORLD_REFRESH_FLAGS if tick_data.get(flag)}
    if refreshed:
        channels[CHANNEL_WORLD] = {"tick": tick, **refreshed}

    positions: list[list] = []
    agent_events = tick_data.get("agent_events") or []
    for event in agent_events:
        if event.get("moved") and event.get("lat") is not None and event.get("lng") is not None:
            positions.append(
                [event.get("agent_id"), event.get("lat"), event.get("lng"), event.get("location_id")]
            )
    if agent_events:
        channels[CHANNEL_AGENTS] = {"tick": tick, "events": agent_events}

    return channels, positions


_aioquic_available = True
try:
    from aioquic.asyncio import QuicConnectionProtocol, serve
    from aioquic.h3.connection import H3_ALPN, H3Connection
    from aioquic.h3.events import H3Event, HeadersReceived, WebTransportStreamDataReceived
    from aioquic.quic.configuration import QuicConfiguration
    from aioquic.quic.events import ProtocolNegotiated, QuicEvent
    from aioquic.tls import SessionTicket
except Exception:  # pragma: no cover - optional runtime dependency
    _aioquic_available = False
    QuicConnectionProtocol = object  # type: ignore[assignment,misc]
    H3Connection = object  # type: ignore[assignment,misc]
    H3Event = object  # type: ignore[assignment,misc]
    HeadersReceived = object  # type: ignore[assignment,misc]
    WebTransportStreamDataReceived = object  # type: ignore[assignment,misc]
    ProtocolNegotiated = object  # type: ignore[assignment,misc]
    QuicEvent = object  # type: ignore[assignment,misc]
    QuicConfiguration = object  # type: ignore[assignment,misc]
    H3_ALPN = []  # type: ignore[assignment]
    SessionTicket = object  # type: ignore[assignment,misc]


SERVER_NAME = "earthlink-wt"


def _wt_world_path() -> str:
    path = settings.wt_path.strip() or "/wt/world"
    if not path.startswith("/"):
        path = f"/{path}"
    return path


@dataclass
class _WtSession:
    protocol: "WtServerProtocol"
    connection: H3Connection
    session_id: int
    # One long-lived unidirectional stream per event class, created on first use.
    _streams: dict[str, int] = field(default_factory=dict)

    def _stream_for(self, channel: str) -> int:
        """Return this session's stream id for an event class, opening it once."""
        stream_id = self._streams.get(channel)
        if stream_id is None:
            stream_id = self.connection.create_webtransport_stream(
                self.session_id, is_unidirectional=True
            )
            self._streams[channel] = stream_id
        return stream_id

    @staticmethod
    def _frame(payload: dict[str, Any] | list) -> bytes:
        """Length-prefixed JSON frame: 4-byte big-endian length + body."""
        raw = json.dumps(payload, default=str).encode("utf-8")
        return len(raw).to_bytes(4, "big") + raw

    def send_event(self, channel: str, payload: dict[str, Any]) -> None:
        """Append one framed message to the channel's reliable stream."""
        stream_id = self._stream_for(channel)
        # aioquic sends WT stream bytes via the underlying QUIC stream API.
        # end_stream=False keeps the per-class stream open for the next frame.
        self.connection._quic.send_stream_data(stream_id, self._frame(payload), end_stream=False)
        self.protocol.transmit()

    def send_datagram(self, positions: list) -> None:
        """Send agent positions over an unreliable WT datagram (latest-wins)."""
        self.connection.send_datagram(self.session_id, self._frame(positions))
        self.protocol.transmit()

    def send_tick(self, tick_data: dict[str, Any]) -> None:
        """Route one world tick across the S68 streams + position datagrams."""
        channels, positions = split_tick(tick_data)
        for channel, payload in channels.items():
            self.send_event(channel, payload)
        if positions:
            self.send_datagram(positions)


class _WtConnectionManager:
    def __init__(self) -> None:
        self._sessions: dict[int, _WtSession] = {}

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def add(self, session: _WtSession) -> None:
        self._sessions[session.session_id] = session
        logger.info("WebTransport connected. Active sessions: %s", len(self._sessions))

    def remove(self, session_id: int) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
        logger.info("WebTransport disconnected. Active sessions: %s", len(self._sessions))

    def remove_protocol_sessions(self, protocol: "WtServerProtocol") -> None:
        stale = [sid for sid, sess in self._sessions.items() if sess.protocol is protocol]
        for sid in stale:
            self.remove(sid)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self._sessions:
            return
        stale: list[int] = []
        for session_id, session in list(self._sessions.items()):
            try:
                session.send_tick(payload)
            except Exception:
                stale.append(session_id)
        for session_id in stale:
            self.remove(session_id)


_manager = _WtConnectionManager()


class _SessionTicketStore:
    """Simple in-memory session ticket store."""

    def __init__(self) -> None:
        self.tickets: dict[bytes, SessionTicket] = {}

    def add(self, ticket: SessionTicket) -> None:
        self.tickets[ticket.ticket] = ticket

    def pop(self, label: bytes) -> Optional[SessionTicket]:
        return self.tickets.pop(label, None)


class WtServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._http: Optional[H3Connection] = None
        self._wt_session_ids: set[int] = set()

    def _accept_webtransport(self, event: HeadersReceived) -> None:
        assert self._http is not None
        headers = [
            (b":status", b"200"),
            (b"server", SERVER_NAME.encode()),
            (b"date", formatdate(time.time(), usegmt=True).encode()),
            (b"sec-webtransport-http3-draft", b"draft02"),
        ]
        self._http.send_headers(stream_id=event.stream_id, headers=headers)
        self.transmit()
        _manager.add(_WtSession(protocol=self, connection=self._http, session_id=event.stream_id))
        self._wt_session_ids.add(event.stream_id)

    def _reject_request(self, stream_id: int, status: bytes = b"404") -> None:
        if self._http is None:
            return
        self._http.send_headers(stream_id=stream_id, headers=[(b":status", status)], end_stream=True)
        self.transmit()

    def _handle_headers(self, event: HeadersReceived) -> None:
        method = ""
        path = ""
        protocol = ""
        for header, value in event.headers:
            if header == b":method":
                method = value.decode()
            elif header == b":path":
                path = value.decode()
            elif header == b":protocol":
                protocol = value.decode()

        request_path = urlsplit(path).path if path else ""
        if method == "CONNECT" and protocol == "webtransport" and request_path == _wt_world_path():
            self._accept_webtransport(event)
            return

        self._reject_request(event.stream_id, status=b"404")

    def _handle_http_event(self, event: H3Event) -> None:
        if isinstance(event, HeadersReceived):
            if event.stream_id not in self._wt_session_ids:
                self._handle_headers(event)
            return

        if isinstance(event, WebTransportStreamDataReceived):
            # Incoming stream data from client currently unused.
            # Do not treat stream end as session close; sessions can open many streams.
            return

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, ProtocolNegotiated):
            if event.alpn_protocol in H3_ALPN:
                self._http = H3Connection(self._quic, enable_webtransport=True)
            return

        if self._http is not None:
            for http_event in self._http.handle_event(event):
                self._handle_http_event(http_event)

    def connection_lost(self, exc: Exception | None) -> None:
        super().connection_lost(exc)
        _manager.remove_protocol_sessions(self)
        self._wt_session_ids.clear()


_wt_server = None


async def start_wt_server() -> None:
    """Start QUIC/WT listener for world streaming.

    The cert is provisioned by Caddy via ACME DNS-01 (Let's Encrypt) and
    written to a shared volume. On a fresh stack `caddy` and `server`
    start in parallel; the cert may not exist for the first ~30s while
    Caddy completes its DNS-01 challenge. We poll for cert availability
    instead of crashing the server.
    """
    global _wt_server

    if _wt_server is not None:
        return

    if not _aioquic_available:
        raise RuntimeError("WebTransport startup failed: aioquic is not installed")

    cert_path = settings.wt_cert_path.strip()
    key_path = settings.wt_key_path.strip()
    if not cert_path or not key_path:
        raise RuntimeError(
            "WebTransport startup failed: set EARTHLINK_WT_CERT_PATH and EARTHLINK_WT_KEY_PATH"
        )

    # Wait for the cert to appear (Caddy DNS-01 takes ~10–30s on first run).
    # Cap the wait at 5 min — beyond that something is wrong with provisioning
    # and we want a loud failure rather than an infinite hang.
    import os as _os
    wait_deadline = asyncio.get_running_loop().time() + 300
    while not (_os.path.isfile(cert_path) and _os.path.isfile(key_path)):
        if asyncio.get_running_loop().time() > wait_deadline:
            raise RuntimeError(
                f"WebTransport startup failed: cert not available after 5 min "
                f"(cert_path={cert_path}, key_path={key_path}). "
                "Check Caddy ACME logs."
            )
        logger.info(
            "WebTransport waiting for ACME cert at %s (Caddy may still be issuing)…",
            cert_path,
        )
        await asyncio.sleep(5)

    try:
        configuration = QuicConfiguration(
            alpn_protocols=H3_ALPN,
            is_client=False,
            max_datagram_frame_size=65536,
        )
        configuration.load_cert_chain(cert_path, key_path)

        ticket_store = _SessionTicketStore()
        _wt_server = await serve(
            settings.wt_host,
            settings.wt_port,
            configuration=configuration,
            create_protocol=WtServerProtocol,
            session_ticket_fetcher=ticket_store.pop,
            session_ticket_handler=ticket_store.add,
            retry=False,
        )
        logger.info(
            "WebTransport listening on %s:%s%s",
            settings.wt_host,
            settings.wt_port,
            _wt_world_path(),
        )
    except Exception as exc:
        _wt_server = None
        raise RuntimeError(f"WebTransport startup failed: {exc}") from exc


async def stop_wt_server() -> None:
    """Stop QUIC/WT listener if running."""
    global _wt_server
    if _wt_server is None:
        return
    try:
        _wt_server.close()
    except Exception:
        pass
    _wt_server = None
    logger.info("WebTransport server stopped")


# -- S90 Redis-Streams fan-out -------------------------------------------
#
# When Redis is available the tick is published once to a stream and every
# WT-serving process (including this one) serves it via its own consumer; with
# no Redis we broadcast directly in-process. See api/wt_fanout.py.
from api.wt_fanout import WtBroadcastConsumer, publish_tick  # noqa: E402

_fanout_redis = None
_fanout_consumer: Optional[WtBroadcastConsumer] = None
_fanout_task: Optional[asyncio.Task] = None


async def start_wt_fanout(redis) -> None:
    """Enable Redis-Streams broadcast fan-out. No-op (in-process only) without Redis."""
    global _fanout_redis, _fanout_consumer, _fanout_task
    if redis is None:
        logger.info("WT fan-out disabled — no Redis; broadcasting in-process")
        return
    _fanout_redis = redis
    _fanout_consumer = WtBroadcastConsumer(redis, _manager.broadcast)
    _fanout_task = asyncio.create_task(_fanout_consumer.run())
    logger.info("WT fan-out enabled via Redis stream")


async def stop_wt_fanout() -> None:
    global _fanout_redis, _fanout_consumer, _fanout_task
    if _fanout_consumer is not None:
        _fanout_consumer.stop()
    if _fanout_task is not None:
        _fanout_task.cancel()
        try:
            await _fanout_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    _fanout_redis = None
    _fanout_consumer = None
    _fanout_task = None


async def on_tick_wt(tick_data: dict) -> None:
    """World tick callback for WT clients.

    With fan-out enabled, publish once to the stream and let the consumer(s)
    serve it; if publishing fails, fall back to an in-process broadcast so a
    transient Redis hiccup never drops a tick for local subscribers.
    """
    if _fanout_redis is not None and await publish_tick(_fanout_redis, tick_data):
        return
    await _manager.broadcast(tick_data)
