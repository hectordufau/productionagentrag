import json
from urllib.error import URLError
from urllib.request import Request

import pytest

from agentic_rag.retrieval.qdrant import QdrantClient, QdrantHTTPError, QdrantUnavailable


class _Response:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.status = status
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._payload


def test_qdrant_http_adapter_covers_collection_points_and_payload(monkeypatch):
    calls: list[tuple[str, str, dict | None]] = []
    collection = {"result": {"config": {"params": {"vectors": {"size": 3, "distance": "Cosine"}}}}}

    def fake_urlopen(request: Request, timeout: float):
        body = json.loads(request.data.decode()) if request.data else None
        calls.append((request.method, request.full_url, body))
        if request.method == "GET" and request.full_url.endswith("/collections/demo"):
            return _Response(collection)
        if request.method == "POST" and request.full_url.endswith("/points/search"):
            return _Response(
                {"result": [{"id": "p1", "score": 0.9, "payload": {"chunk_id": "c1"}}]}
            )
        return _Response({"status": "ok"})

    monkeypatch.setattr("agentic_rag.retrieval.qdrant.urlopen", fake_urlopen)
    client = QdrantClient("http://qdrant.test")
    assert client.ensure_collection("demo", 3) == {"status": "already_exists", "collection": "demo"}
    assert client.validate_collection("demo", 3)
    assert client.upsert_vectors(
        "demo", [{"id": "p1", "vector": [1, 0, 0], "payload": {"chunk_id": "c1"}}]
    )
    assert client.search_vectors("demo", [1, 0, 0])[0]["payload"]["chunk_id"] == "c1"
    assert client.update_vectors(
        "demo", [{"id": "p1", "vector": [0, 1, 0], "payload": {"version": 2}}]
    )
    assert client.delete_vectors("demo", ["p1"])
    assert any(method == "POST" and "/points/search" in url for method, url, _ in calls)


def test_qdrant_collection_creation_and_404_is_not_unavailable(monkeypatch):
    calls = []

    def fake_urlopen(request: Request, timeout: float):
        calls.append(request.method)
        if request.method == "GET":
            from urllib.error import HTTPError

            raise HTTPError(request.full_url, 404, "missing", {}, None)
        return _Response({"result": True})

    monkeypatch.setattr("agentic_rag.retrieval.qdrant.urlopen", fake_urlopen)
    client = QdrantClient("http://qdrant.test")
    assert client.collection_exists("new") is False
    assert client.ensure_collection("new", 4)["result"] is True
    assert calls == ["GET", "GET", "PUT"]


def test_qdrant_transport_and_http_errors_are_explicit(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr("agentic_rag.retrieval.qdrant.urlopen", unavailable)
    with pytest.raises(QdrantUnavailable, match="unavailable"):
        QdrantClient("http://qdrant.test").health()

    def broken(*_args, **_kwargs):
        from urllib.error import HTTPError

        raise HTTPError("http://qdrant.test/healthz", 500, "boom", {}, None)

    monkeypatch.setattr("agentic_rag.retrieval.qdrant.urlopen", broken)
    with pytest.raises(QdrantHTTPError) as error:
        QdrantClient("http://qdrant.test").health()
    assert error.value.status == 500
