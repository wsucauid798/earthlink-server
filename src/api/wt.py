"""WebTransport (QUIC/HTTP3) stream endpoint for world tick broadcasting.

This module runs an aioquic server alongside FastAPI and exposes:
  CONNECT /wt/world  (:protocol = webtransport)

Broadcast model (current spike):
  - Every world tick is serialized as JSON.
  - Payload is sent on a dedicated WebTransport unidirectional stream.
  - Clients read one full JSON message per stream.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from email.utils import formatdate
from urllib.parse import urlsplit
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)


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

    def send_json(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, default=str).encode("utf-8")
        stream_id = self.connection.create_webtransport_stream(
            self.session_id, is_unidirectional=True
        )
        # aioquic currently sends WT stream bytes via the underlying QUIC stream API.
        self.connection._quic.send_stream_data(stream_id, raw, end_stream=True)
        self.protocol.transmit()


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
                session.send_json(payload)
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
    """Start QUIC/WT listener for world streaming."""
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


async def on_tick_wt(tick_data: dict) -> None:
    """World tick callback for WT clients."""
    await _manager.broadcast(tick_data)
