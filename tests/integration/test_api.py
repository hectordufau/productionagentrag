from fastapi.testclient import TestClient

from agentic_rag.api.app import app

client = TestClient(app)


def test_health_ready_version() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}
    assert client.get("/version").json()["version"] == "0.1.0"


def test_ingest_then_query() -> None:
    response = client.post(
        "/v1/ingest",
        json={"source": "test", "filename": "guide.md", "content": "Qdrant stores vectors"},
    )
    assert response.status_code == 200
    response = client.post("/v1/query", json={"query": "Where are vectors stored?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["citations"]
