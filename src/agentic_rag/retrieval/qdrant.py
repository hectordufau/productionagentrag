"""Small dependency-free Qdrant HTTP adapter.

The adapter deliberately uses Qdrant's REST API instead of importing the optional
qdrant-client package.  Transport failures become :class:`QdrantUnavailable`;
HTTP responses with an error status become :class:`QdrantHTTPError`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class QdrantUnavailable(RuntimeError):
    """Qdrant cannot be reached (DNS, connection, timeout, or transport error)."""


class QdrantHTTPError(RuntimeError):
    """Qdrant returned an HTTP error response."""

    def __init__(self, status: int, body: str, method: str, path: str) -> None:
        self.status = status
        self.body = body
        self.method = method
        self.path = path
        super().__init__(f"Qdrant HTTP {status} for {method} {path}: {body[:500]}")


class QdrantClient:
    def __init__(self, url: str = "http://localhost:6333", timeout: float = 2.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = Request(
            f"{self.url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                if not raw:
                    return {"status": response.status}
                try:
                    return json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    if method == "GET" and path == "/healthz" and response.status == 200:
                        return {"status": raw.decode("utf-8", "replace")}
                    raise QdrantHTTPError(
                        response.status, raw.decode("utf-8", "replace"), method, path
                    ) from exc
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            raise QdrantHTTPError(exc.code, raw, method, path) from exc
        except (OSError, URLError, TimeoutError) as exc:
            raise QdrantUnavailable(f"Qdrant unavailable at {self.url}: {exc}") from exc

    @staticmethod
    def _collection_path(collection: str) -> str:
        return f"/collections/{quote(collection, safe='')}"

    def health(self) -> dict[str, Any]:
        result = self._request("GET", "/healthz")
        return result if isinstance(result, dict) else {"result": result}

    def collection_exists(self, collection: str) -> bool:
        try:
            self._request("GET", self._collection_path(collection))
            return True
        except QdrantHTTPError as exc:
            if exc.status == 404:
                return False
            raise

    def create_collection(
        self, collection: str, vector_size: int, distance: str = "Cosine"
    ) -> dict[str, Any]:
        if vector_size < 1:
            raise ValueError("vector_size must be positive")
        return self._request(
            "PUT",
            self._collection_path(collection),
            {"vectors": {"size": vector_size, "distance": distance}},
        )

    def ensure_collection(
        self, collection: str, vector_size: int, distance: str = "Cosine"
    ) -> dict[str, Any]:
        if self.collection_exists(collection):
            self.validate_collection(collection, vector_size, distance)
            return {"status": "already_exists", "collection": collection}
        return self.create_collection(collection, vector_size, distance)

    def validate_collection(
        self, collection: str, vector_size: int, distance: str = "Cosine"
    ) -> bool:
        info = self._request("GET", self._collection_path(collection))
        result = info.get("result", info) if isinstance(info, dict) else {}
        config = result.get("config", {}) if isinstance(result, dict) else {}
        params = config.get("params", {}) if isinstance(config, dict) else {}
        vectors = params.get("vectors", {}) if isinstance(params, dict) else {}
        # Qdrant returns either a single vector config or a named-vector mapping.
        if isinstance(vectors, dict) and "size" in vectors:
            actual_size = vectors.get("size")
            actual_distance = vectors.get("distance")
        elif isinstance(vectors, dict) and vectors:
            first = next(iter(vectors.values()))
            actual_size = first.get("size")
            actual_distance = first.get("distance")
        else:
            actual_size = None
            actual_distance = None
        if actual_size != vector_size or str(actual_distance).lower() != distance.lower():
            raise QdrantHTTPError(
                409,
                f"collection vector config is {actual_size}/{actual_distance}, expected {vector_size}/{distance}",
                "GET",
                self._collection_path(collection),
            )
        return True

    def upsert_vectors(
        self, collection: str, points: Iterable[dict[str, Any]], wait: bool = True
    ) -> dict[str, Any]:
        payload = {"points": list(points)}
        path = f"{self._collection_path(collection)}/points?wait={'true' if wait else 'false'}"
        return self._request("PUT", path, payload)

    def search_vectors(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        query_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        body: dict[str, Any] = {"vector": vector, "limit": limit, "with_payload": True}
        if query_filter is not None:
            body["filter"] = query_filter
        result = self._request("POST", f"{self._collection_path(collection)}/points/search", body)
        return list(result.get("result", [])) if isinstance(result, dict) else []

    def delete_vectors(
        self, collection: str, point_ids: Iterable[str | int], wait: bool = True
    ) -> dict[str, Any]:
        path = (
            f"{self._collection_path(collection)}/points/delete?wait={'true' if wait else 'false'}"
        )
        return self._request("POST", path, {"points": list(point_ids)})

    def update_vectors(
        self, collection: str, points: Iterable[dict[str, Any]], wait: bool = True
    ) -> dict[str, Any]:
        """Replace point vectors and payloads; Qdrant upsert is the update strategy."""
        return self.upsert_vectors(collection, points, wait)

    # Short aliases mirror the REST concepts for simple adapter callers.
    def upsert(self, collection: str, points: Iterable[dict[str, Any]], wait: bool = True) -> dict[str, Any]:
        return self.upsert_vectors(collection, points, wait)

    def search(self, collection: str, vector: list[float], limit: int = 5, query_filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.search_vectors(collection, vector, limit, query_filter)

    def delete(self, collection: str, point_ids: Iterable[str | int], wait: bool = True) -> dict[str, Any]:
        return self.delete_vectors(collection, point_ids, wait)

    def update(self, collection: str, points: Iterable[dict[str, Any]], wait: bool = True) -> dict[str, Any]:
        return self.update_vectors(collection, points, wait)
