"""Tests for the TEI embedding client (S72) and the TEI-backed retriever."""

import httpx
import pytest

from agents.embeddings import TEIEmbedder
from agents.retrieval import SemanticFactRetriever


def _transport(handler):
    """Wrap a request handler in a TEIEmbedder whose HTTP client is mocked."""
    embedder = TEIEmbedder(url="http://tei:80", expected_dims=4)
    embedder._client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://tei:80"
    )
    return embedder


def test_embed_parses_tei_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        return httpx.Response(200, json=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    embedder = _transport(handler)
    vectors = embedder.embed(["hello", "world"])

    assert vectors == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]


def test_embed_drops_blank_inputs_and_short_circuits_empty():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        import json

        payload = json.loads(request.content)
        # Blank inputs are stripped before the request is sent.
        assert payload["inputs"] == ["real"]
        return httpx.Response(200, json=[[1.0, 0.0, 0.0, 0.0]])

    embedder = _transport(handler)
    assert embedder.embed(["", "  ", "real"]) == [[1.0, 0.0, 0.0, 0.0]]

    # All-blank input never hits the network.
    assert embedder.embed(["", "   "]) == []
    assert calls["n"] == 1


def test_embed_returns_none_when_tei_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    embedder = _transport(handler)
    assert embedder.embed(["anything"]) is None


def test_embed_returns_none_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    embedder = _transport(handler)
    assert embedder.embed(["anything"]) is None


class _StubEmbedder:
    """Deterministic embedder so we can assert cosine ranking without TEI."""

    def __init__(self, table: dict[str, list[float]] | None = None, fail: bool = False):
        self._table = table or {}
        self._fail = fail

    def embed(self, texts):
        if self._fail:
            return None
        return [self._table[t] for t in texts]


def test_retriever_ranks_by_cosine_when_tei_available():
    # Query aligns with fact B; orthogonal to fact A.
    table = {
        "where is the river": [1.0, 0.0],
        "fact A about mountains": [0.0, 1.0],
        "fact B about the river": [1.0, 0.0],
    }
    retriever = SemanticFactRetriever(embedder=_StubEmbedder(table))
    ranked = retriever.rank(
        "where is the river",
        ["fact A about mountains", "fact B about the river"],
        top_k=2,
    )

    assert retriever.backend == "tei"
    # Best match (river fact, idx 1) ranks first.
    assert ranked[0][1] == 1
    assert ranked[0][0] > ranked[1][0]


def test_retriever_falls_back_to_token_overlap_when_tei_down():
    retriever = SemanticFactRetriever(embedder=_StubEmbedder(fail=True))
    ranked = retriever.rank(
        "population of London",
        ["London is the capital", "population of London is large", "weather today"],
        top_k=2,
    )

    assert retriever.backend == "token-overlap"
    # Token overlap favours the fact sharing the most query tokens.
    assert ranked[0][1] == 1


def test_retriever_empty_facts_returns_empty():
    retriever = SemanticFactRetriever(embedder=_StubEmbedder(fail=True))
    assert retriever.rank("anything", []) == []
