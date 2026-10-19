"""
The provider interface behind /v1.

Requests and responses are plain dicts in the OpenAI Chat Completions and
Embeddings shapes. `model` arguments are the provider's own id, without the
"<provider>/" prefix.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol

import httpx


class ProviderError(Exception):
    """An error with an HTTP status and an OpenAI-style error body."""

    def __init__(self, status: int, message: str, type: str = "upstream_error", code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.code = code

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": None, "code": self.code}}


class Provider(Protocol):
    name: str

    async def list_models(self) -> list[dict]: ...

    async def chat(self, model: str, body: dict, opts: dict) -> dict: ...

    def chat_stream(self, model: str, body: dict, opts: dict) -> AsyncIterator[dict]: ...

    async def embed(self, model: str, body: dict, opts: dict) -> dict:
        """An OpenAI embeddings response with float vectors; /v1 handles encoding_format."""
        ...

    async def health(self) -> bool: ...


def upstream_status(status: int) -> int:
    """
    Status to return to the caller for an upstream error. An upstream 401/403 means
    LLMP's own provider key is wrong, which the caller can't fix, so it becomes 502.
    """
    if status in (400, 402, 404, 408, 413, 422, 429):
        return status
    return 502


def error_type(status: int) -> str:
    return {
        400: "invalid_request_error",
        402: "insufficient_quota",
        404: "not_found_error",
        413: "invalid_request_error",
        422: "invalid_request_error",
        429: "rate_limit_error",
    }.get(status, "upstream_error")


def connection_error(provider: str, exc: Exception) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(504, f"{provider} timed out: {type(exc).__name__}", "timeout")
    return ProviderError(503, f"{provider} is unreachable: {type(exc).__name__}: {exc}", "service_unavailable")


def usage_dict(prompt: Any, completion: Any) -> dict:
    p, c = prompt or 0, completion or 0
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
