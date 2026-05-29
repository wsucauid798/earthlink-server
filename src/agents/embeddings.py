"""TEI embedding client — the single embedding service for EarthLink.

Hugging Face Text Embeddings Inference (TEI) serves the chosen embedding
model (S71: BAAI/bge-m3, 1024 dims) as a network service. All embedding in
the server routes through here — no in-process sentence-transformers, no
external embedding API. See server-design-plan S72.

Consumers:
  - SemanticFactRetriever (agents/retrieval.py) — query/fact ranking
  - PgVectorFactStore (S75) — write/read path for durable agent memory
  - EarthProxy embed-on-resolve (S80) — world semantic layer

The client degrades gracefully: if TEI is unreachable, embed() returns None
and callers fall back (token-overlap ranking, deferred embedding, etc.).
This mirrors the chroma timeout hardening in S98 — a slow or dead embedding
service must never wedge the tick loop.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

# Connect fast (local docker network); allow a longer read for the first
# call, which can block while TEI loads the model into memory on cold start.
# Once warm, a small CPU batch embeds in tens of milliseconds.
_CONNECT_TIMEOUT = 2.0
_READ_TIMEOUT = 10.0


class TEIEmbedder:
    """Synchronous + async client for a TEI ``/embed`` endpoint.

    One instance is shared process-wide via :func:`get_embedder`. Failures
    return ``None`` rather than raising, so callers can degrade gracefully.
    """

    def __init__(self, url: str | None = None, expected_dims: int | None = None):
        self._url = (url or settings.tei_url).rstrip("/")
        self._expected_dims = expected_dims or settings.tei_embedding_dims
        self._client: httpx.Client | None = None
        self._aclient: httpx.AsyncClient | None = None
        self._warned_dims = False

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(_READ_TIMEOUT, connect=_CONNECT_TIMEOUT)

    # -- sync ------------------------------------------------------------

    def _sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self._url, timeout=self._timeout())
        return self._client

    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Embed a batch of texts.

        Returns one vector per non-empty input (empty/whitespace inputs are
        dropped before the call), an empty list if there is nothing to embed,
        or ``None`` if TEI is unavailable.
        """
        clean = [t.strip() for t in texts if t and t.strip()]
        if not clean:
            return []
        try:
            resp = self._sync_client().post("/embed", json=self._payload(clean))
            resp.raise_for_status()
            vectors = resp.json()
        except Exception as e:  # connection, timeout, HTTP error, bad JSON
            logger.debug(f"TEI embed failed ({self._url}): {e}")
            return None
        return self._validate(vectors)

    def embed_one(self, text: str) -> Optional[list[float]]:
        result = self.embed([text])
        return result[0] if result else None

    # -- async (for the S75/S76/S80 write paths) -------------------------

    async def embed_async(self, texts: list[str]) -> Optional[list[list[float]]]:
        clean = [t.strip() for t in texts if t and t.strip()]
        if not clean:
            return []
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(base_url=self._url, timeout=self._timeout())
        try:
            resp = await self._aclient.post("/embed", json=self._payload(clean))
            resp.raise_for_status()
            vectors = resp.json()
        except Exception as e:
            logger.debug(f"TEI embed_async failed ({self._url}): {e}")
            return None
        return self._validate(vectors)

    # -- internals -------------------------------------------------------

    @staticmethod
    def _payload(inputs: list[str]) -> dict:
        # normalize=True → L2-normalized vectors, so a dot product is the
        # cosine similarity. truncate=True → long inputs are clipped to the
        # model's max sequence length instead of erroring.
        return {"inputs": inputs, "normalize": True, "truncate": True}

    def _validate(self, vectors) -> Optional[list[list[float]]]:
        if not isinstance(vectors, list) or not vectors:
            return None
        first = vectors[0]
        if (
            not self._warned_dims
            and isinstance(first, list)
            and len(first) != self._expected_dims
        ):
            logger.warning(
                f"TEI returned {len(first)}-dim embeddings, expected "
                f"{self._expected_dims} — embedding model mismatch (see S71)"
            )
            self._warned_dims = True
        return vectors

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


_EMBEDDER: TEIEmbedder | None = None


def get_embedder() -> TEIEmbedder:
    """Return the process-wide shared TEI embedder."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TEIEmbedder()
    return _EMBEDDER
