"""Optional Qdrant HTTP integration. It never silently falls back."""
from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


class QdrantUnavailable(RuntimeError):
    pass


class QdrantClient:
    def __init__(self, url: str = "http://localhost:6333", timeout: float = 2.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, object]:
        request = Request(f"{self.url}/healthz")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
                return json.loads(payload.decode()) if payload else {"status": response.status}
        except (OSError, URLError, TimeoutError) as exc:
            raise QdrantUnavailable(f"Qdrant unavailable at {self.url}: {exc}") from exc

    def collection_exists(self, collection: str) -> bool:
        request = Request(f"{self.url}/collections/{collection}")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return bool(response.status == 200)
        except (OSError, URLError, TimeoutError) as exc:
            raise QdrantUnavailable(f"Qdrant unavailable at {self.url}: {exc}") from exc
