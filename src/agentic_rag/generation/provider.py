"""Provider abstraction and real local Ollama implementation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMProviderError(RuntimeError):
    """Base class for explicit provider failures."""


class LLMUnavailableError(LLMProviderError):
    """Provider cannot be reached or is not configured."""


class LLMTimeoutError(LLMProviderError):
    """Provider exceeded the configured timeout."""


@dataclass(frozen=True)
class LLMResponse:
    answer: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LLMProvider(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str, *, timeout_s: float = 60.0) -> LLMResponse: ...


class OllamaProvider:
    """Ollama HTTP provider; failures are surfaced, never replaced by fake text."""

    provider = "ollama"

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://127.0.0.1:11434") -> None:
        self.model, self.base_url = model, base_url.rstrip("/")

    def generate(self, prompt: str, *, timeout_s: float = 60.0) -> LLMResponse:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False, "think": False, "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 256}}).encode()
        started = time.perf_counter()
        try:
            request = Request(self.base_url + "/api/generate", data=payload, headers={"content-type": "application/json"})
            with urlopen(request, timeout=timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise LLMTimeoutError(f"Ollama timed out after {timeout_s}s") from exc
        except (URLError, HTTPError, OSError, json.JSONDecodeError) as exc:
            raise LLMUnavailableError(f"Ollama unavailable at {self.base_url}: {exc}") from exc
        latency = (time.perf_counter() - started) * 1000
        answer = str(body.get("response", "")).strip()
        if not answer:
            raise LLMProviderError("Ollama returned an empty response")
        return LLMResponse(
            answer=answer, provider=self.provider, model=self.model,
            latency_ms=latency, prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            total_tokens=(body.get("prompt_eval_count") or 0) + (body.get("eval_count") or 0), raw=body,
        )
