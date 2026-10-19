"""
OpenRouter adapter. OpenRouter already speaks Chat Completions, so this is a
pass-through that strips the "openrouter/" prefix, asks for usage accounting
(which returns the cost), and maps errors to OpenAI-style bodies.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
from typing import AsyncIterator

import httpx

from llmp.config import Settings
from llmp.providers.base import ProviderError, connection_error, error_type, upstream_status

PREFIX = "openrouter/"


class OpenRouterProvider:
    name = "openrouter"

    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self._client = client
        self._base = settings.openrouter_url
        self._key = settings.openrouter_key
        self._allowlist = settings.openrouter_allowlist
        self._sem = asyncio.Semaphore(settings.openrouter_max_concurrency)

    def _headers(self) -> dict:
        if not self._key:
            raise ProviderError(503, "OpenRouter is not configured: ORT_API_KEY is missing", "service_unavailable")
        return {"Authorization": f"Bearer {self._key}", "X-Title": "LLMP"}

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self._base}/key", headers=self._headers(), timeout=5)
            return r.status_code == 200
        except (httpx.HTTPError, ProviderError):
            return False

    async def list_models(self) -> list[dict]:
        try:
            r = await self._client.get(f"{self._base}/models", headers=self._headers(), timeout=30)
        except httpx.HTTPError as e:
            raise connection_error("OpenRouter", e) from e
        if r.status_code != 200:
            raise _error(r.status_code, r.content)
        out = []
        for m in r.json().get("data") or []:
            if self._allowlist and not any(fnmatch.fnmatchcase(m["id"], pat) for pat in self._allowlist):
                continue
            params = set(m.get("supported_parameters") or [])
            modalities = set((m.get("architecture") or {}).get("input_modalities") or [])
            out.append({
                "id": f"{PREFIX}{m['id']}",
                "object": "model",
                "created": m.get("created") or 0,
                "owned_by": m["id"].split("/")[0],
                "name": m.get("name"),
                "provider": "openrouter",
                "local": False,
                "loaded": None,
                "context_window": m.get("context_length"),
                "max_output": (m.get("top_provider") or {}).get("max_completion_tokens"),
                "capabilities": {
                    "tools": "tools" in params,
                    "vision": "image" in modalities,
                    "reasoning": "reasoning" in params,
                    "json_schema": "structured_outputs" in params,
                    "chat": True,
                    "embeddings": False,
                },
                "pricing": _pricing(m.get("pricing") or {}),
            })
        return out

    def _payload(self, model: str, body: dict, stream: bool) -> dict:
        payload = {k: v for k, v in body.items() if k != "llmp"}
        payload["model"] = model
        payload["usage"] = {"include": True}
        payload["stream"] = stream
        if stream:
            payload["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        return payload

    async def chat(self, model: str, body: dict, opts: dict) -> dict:
        payload = self._payload(model, body, stream=False)
        async with self._sem:
            try:
                r = await self._client.post(f"{self._base}/chat/completions", headers=self._headers(), json=payload)
            except httpx.HTTPError as e:
                raise connection_error("OpenRouter", e) from e
        if r.status_code != 200:
            raise _error(r.status_code, r.content)
        data = r.json()
        if data.get("error"):
            raise _error_from_body(data["error"])
        data["llmp"] = {"gen_id": data.get("id"), "upstream": data.get("provider")}
        data["model"] = PREFIX + (data.get("model") or model)
        return data

    async def chat_stream(self, model: str, body: dict, opts: dict) -> AsyncIterator[dict]:
        payload = self._payload(model, body, stream=True)
        async with self._sem:
            try:
                async with self._client.stream(
                    "POST", f"{self._base}/chat/completions", headers=self._headers(), json=payload
                ) as r:
                    if r.status_code != 200:
                        raise _error(r.status_code, await r.aread())
                    async for line in r.aiter_lines():
                        # SSE: "data: {...}" lines; ": OPENROUTER PROCESSING" keep-alive comments.
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        chunk = json.loads(data)
                        if chunk.get("error"):
                            raise _error_from_body(chunk["error"])
                        chunk["model"] = PREFIX + (chunk.get("model") or model)
                        if chunk.get("usage"):
                            chunk["llmp"] = {"gen_id": chunk.get("id"), "upstream": chunk.get("provider")}
                        yield chunk
            except httpx.HTTPError as e:
                raise connection_error("OpenRouter", e) from e

    async def embed(self, model: str, body: dict, opts: dict) -> dict:
        raise ProviderError(
            400, "Embeddings are not available for OpenRouter models yet; use an ollama/ embedding model",
            "invalid_request_error", "model_not_supported",
        )


def _pricing(p: dict) -> dict:
    """OpenRouter prices are USD per token, as strings. Return USD per million tokens."""

    def per_mtok(key: str) -> float | None:
        try:
            return round(float(p[key]) * 1e6, 6)
        except (KeyError, TypeError, ValueError):
            return None

    return {
        "unit": "usd_per_mtok",
        "input": per_mtok("prompt"),
        "output": per_mtok("completion"),
        "cache_read": per_mtok("input_cache_read"),
        "cache_write": per_mtok("input_cache_write"),
    }


def _error_from_body(err: dict | str, status: int | None = None) -> ProviderError:
    if isinstance(err, str):
        err = {"message": err}
    code = err.get("code")
    status = status or (code if isinstance(code, int) else 502)
    message = err.get("message") or "OpenRouter error"
    raw = (err.get("metadata") or {}).get("raw")
    if raw:
        message = f"{message}: {raw if isinstance(raw, str) else json.dumps(raw)}"[:2000]
    out = upstream_status(status)
    return ProviderError(out, f"OpenRouter: {message}", error_type(out), str(code) if code is not None else None)


def _error(status: int, content: bytes) -> ProviderError:
    try:
        err = json.loads(content).get("error") or {}
    except (ValueError, AttributeError):
        err = {"message": content.decode(errors="replace")[:2000]}
    return _error_from_body(err, status)
