"""
Ollama adapter. Translates OpenAI Chat Completions to and from Ollama's native
/api/chat, which (unlike Ollama's /v1 endpoint) takes num_ctx and keep_alive per
request, and OpenAI embeddings to and from the native /api/embed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import AsyncIterator

import httpx

from llmp.config import Settings
from llmp.providers.base import ProviderError, connection_error, usage_dict


class OllamaProvider:
    name = "ollama"

    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self._client = client
        self._base = settings.ollama_url
        self._keep_alive = settings.ollama_keep_alive
        self._num_ctx = settings.ollama_num_ctx
        self._embed_num_ctx = settings.ollama_embed_num_ctx
        self._num_ctx_per_model = settings.ollama_num_ctx_per_model
        self._max_output = settings.ollama_max_output_per_model
        self._sem = asyncio.Semaphore(settings.ollama_max_concurrency)
        # Embedding calls are short and use a small, separate model, so they get their
        # own slots instead of queueing behind long chat generations.
        self._embed_sem = asyncio.Semaphore(settings.ollama_max_concurrency)
        self._show_cache: dict[str, dict] = {}

    # ── Catalog ────────────────────────────────────────────────────────────────

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self._base}/api/version", timeout=3)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def show(self, model: str, refresh: bool = False) -> dict:
        if refresh or model not in self._show_cache:
            self._show_cache[model] = await self._request("POST", "/api/show", json={"model": model})
        return self._show_cache[model]

    async def loaded_models(self) -> set[str]:
        ps = await self._request("GET", "/api/ps")
        return {m.get("name") for m in ps.get("models") or []}

    async def list_models(self) -> list[dict]:
        tags, loaded = await asyncio.gather(self._request("GET", "/api/tags"), self.loaded_models())
        models = tags.get("models") or []
        shows = await asyncio.gather(*(self.show(m["name"], refresh=True) for m in models), return_exceptions=True)
        out = []
        for tag, show in zip(models, shows):
            name = tag["name"]
            show = show if isinstance(show, dict) else {}
            caps = set(show.get("capabilities") or [])
            details = tag.get("details") or {}
            out.append({
                "id": f"ollama/{name}",
                "object": "model",
                "created": 0,
                "owned_by": "ollama",
                "provider": "ollama",
                "local": True,
                "loaded": name in loaded,
                "context_window": _context_length(show),
                "max_output": self._max_output.get(name),
                "default_num_ctx": self._num_ctx_for(
                    name, show, None, self._embed_num_ctx if "embedding" in caps and "completion" not in caps else None),
                "capabilities": {
                    "tools": "tools" in caps,
                    "vision": "vision" in caps,
                    "reasoning": "thinking" in caps,
                    "json_schema": "completion" in caps,
                    "chat": "completion" in caps,
                    "embeddings": "embedding" in caps,
                },
                "pricing": None,
                "details": {
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization_level": details.get("quantization_level"),
                    "size_bytes": tag.get("size"),
                },
            })
        return out

    # ── Chat ───────────────────────────────────────────────────────────────────

    async def chat(self, model: str, body: dict, opts: dict) -> dict:
        payload = await self._payload(model, body, opts, stream=False)
        async with self._sem:
            data = await self._request("POST", "/api/chat", json=payload)
        msg = data.get("message") or {}
        calls = _tool_calls(msg.get("tool_calls"), 0)
        message: dict = {"role": "assistant", "content": msg.get("content") or (None if calls else "")}
        if calls:
            message["tool_calls"] = calls
        if msg.get("thinking"):
            message["reasoning"] = msg["thinking"]
        return {
            "id": _completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"ollama/{model}",
            "choices": [{
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": "tool_calls" if calls else _finish_reason(data.get("done_reason")),
            }],
            "usage": _usage(data),
            "llmp": _extras(data, payload),
        }

    async def chat_stream(self, model: str, body: dict, opts: dict) -> AsyncIterator[dict]:
        payload = await self._payload(model, body, opts, stream=True)
        cid, created, model_id = _completion_id(), int(time.time()), f"ollama/{model}"

        def chunk(delta: dict, finish: str | None) -> dict:
            return {
                "id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
            }

        async with self._sem:
            try:
                async with self._client.stream("POST", f"{self._base}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        raise _error(r.status_code, await r.aread())
                    n_calls, sent_role = 0, False
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("error"):
                            raise ProviderError(502, f"Ollama: {data['error']}")
                        msg = data.get("message") or {}
                        delta: dict = {}
                        if not sent_role:
                            delta["role"] = "assistant"
                            sent_role = True
                        if msg.get("content"):
                            delta["content"] = msg["content"]
                        if msg.get("thinking"):
                            delta["reasoning"] = msg["thinking"]
                        if msg.get("tool_calls"):
                            delta["tool_calls"] = _tool_calls(msg["tool_calls"], n_calls, with_index=True)
                            n_calls += len(msg["tool_calls"])
                        if delta:
                            yield chunk(delta, None)
                        if data.get("done"):
                            yield chunk({}, "tool_calls" if n_calls else _finish_reason(data.get("done_reason")))
                            final = chunk({}, None)
                            final["choices"] = []
                            final["usage"] = _usage(data)
                            final["llmp"] = _extras(data, payload)
                            yield final
                            return
            except httpx.HTTPError as e:
                raise connection_error("Ollama", e) from e

    # ── Embeddings ─────────────────────────────────────────────────────────────

    async def embed(self, model: str, body: dict, opts: dict) -> dict:
        inputs = body.get("input")
        inputs = [inputs] if isinstance(inputs, str) else inputs
        if not inputs:
            raise ProviderError(400, "input must be a string or a non-empty list of strings", "invalid_request_error")
        if not all(isinstance(x, str) for x in inputs):
            raise ProviderError(400, "Token-id input is not supported for Ollama models; send text", "invalid_request_error")
        show = await self.show(model)  # also turns an unknown model into a 404
        if "embedding" not in (show.get("capabilities") or []):
            raise ProviderError(
                400, f"ollama/{model} is not an embedding model", "invalid_request_error", "model_not_supported")

        # Embedding inputs are short, and the context is VRAM held for as long as the model stays loaded.
        num_ctx = self._num_ctx_for(model, show, opts.get("num_ctx"), default=self._embed_num_ctx)
        payload: dict = {
            "model": model,
            "input": inputs,
            "keep_alive": self._keep_alive if opts.get("keep_alive") is None else opts["keep_alive"],
            "options": {**(opts.get("options") or {}), "num_ctx": num_ctx},
        }
        if body.get("dimensions"):
            payload["dimensions"] = body["dimensions"]
        if opts.get("truncate") is not None:  # Ollama truncates long inputs unless told not to
            payload["truncate"] = bool(opts["truncate"])
        async with self._embed_sem:
            data = await self._request("POST", "/api/embed", json=payload)

        tokens = data.get("prompt_eval_count") or 0
        load = data.get("load_duration")
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(data.get("embeddings") or [])],
            "model": f"ollama/{model}",
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens, "cost": 0.0},
            "llmp": {
                "load_ms": round(load / 1e6) if load is not None else None,
                "num_ctx": payload["options"]["num_ctx"],
                "keep_alive": payload["keep_alive"],
            },
        }

    # ── Translation ────────────────────────────────────────────────────────────

    async def _payload(self, model: str, body: dict, opts: dict, stream: bool) -> dict:
        if (body.get("n") or 1) > 1:
            raise ProviderError(400, "n > 1 is not supported for Ollama models", "invalid_request_error")
        show = await self.show(model)  # also turns an unknown model into a 404
        caps = set(show.get("capabilities") or [])

        payload: dict = {
            "model": model,
            "messages": await self._messages(body["messages"]),
            "stream": stream,
            "keep_alive": self._keep_alive if opts.get("keep_alive") is None else opts["keep_alive"],  # 0 unloads
        }
        if body.get("tools") and body.get("tool_choice") != "none":
            payload["tools"] = body["tools"]
        fmt = _format(body.get("response_format"))
        if fmt is not None:
            payload["format"] = fmt
        think = _think(body, model)
        if think is not None and "thinking" in caps:
            payload["think"] = think

        options: dict = {}
        for key in ("temperature", "top_p", "seed", "frequency_penalty", "presence_penalty"):
            if body.get(key) is not None:
                options[key] = body[key]
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        if max_tokens:
            options["num_predict"] = max_tokens
        if body.get("stop"):
            options["stop"] = [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
        options.update(opts.get("options") or {})
        options["num_ctx"] = self._num_ctx_for(model, show, opts.get("num_ctx") or options.get("num_ctx"))
        payload["options"] = options
        return payload

    def _num_ctx_for(self, model: str, show: dict, requested: int | None, default: int | None = None) -> int:
        num_ctx = int(requested or self._num_ctx_per_model.get(model) or default or self._num_ctx)
        limit = _context_length(show)
        return min(num_ctx, limit) if limit else num_ctx

    async def _messages(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        call_names: dict[str, str] = {}  # tool_call_id -> function name, latest assistant turn wins
        for m in messages:
            role = "system" if m.get("role") == "developer" else m.get("role")
            text, images = await self._content(m.get("content"))
            msg: dict = {"role": role, "content": text}
            if images:
                msg["images"] = images
            if role == "assistant":
                calls = []
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    call_names[tc.get("id")] = fn.get("name")
                    calls.append({"function": {"name": fn.get("name"), "arguments": _parse_arguments(fn.get("arguments"))}})
                if calls:
                    msg["tool_calls"] = calls
                thinking = m.get("reasoning") or m.get("reasoning_content")
                if thinking:
                    msg["thinking"] = thinking
            elif role == "tool":
                name = m.get("name") or call_names.get(m.get("tool_call_id"))
                if name:
                    msg["tool_name"] = name
            out.append(msg)
        return out

    async def _content(self, content) -> tuple[str, list[str]]:
        if content is None:
            return "", []
        if isinstance(content, str):
            return content, []
        texts, images = [], []
        for part in content:
            kind = part.get("type")
            if kind == "text":
                texts.append(part.get("text") or "")
            elif kind == "image_url":
                image = part.get("image_url")
                images.append(await self._image_b64(image.get("url") if isinstance(image, dict) else image))
            else:
                raise ProviderError(400, f"Content part type {kind!r} is not supported for Ollama models", "invalid_request_error")
        return "\n".join(texts), images

    async def _image_b64(self, url: str | None) -> str:
        if url and url.startswith("data:"):
            header, _, data = url.partition(",")
            if ";base64" not in header:
                raise ProviderError(400, "Image data URLs must be base64-encoded", "invalid_request_error")
            return data
        if url and url.startswith(("http://", "https://")):
            try:
                r = await self._client.get(url, timeout=30, follow_redirects=True)
            except httpx.HTTPError as e:
                raise ProviderError(400, f"Could not fetch image {url}: {type(e).__name__}", "invalid_request_error") from e
            if r.status_code != 200:
                raise ProviderError(400, f"Could not fetch image {url}: HTTP {r.status_code}", "invalid_request_error")
            return base64.b64encode(r.content).decode("ascii")
        raise ProviderError(400, "image_url must be a data: URL or an http(s) URL", "invalid_request_error")

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            r = await self._client.request(method, f"{self._base}{path}", **kwargs)
        except httpx.HTTPError as e:
            raise connection_error("Ollama", e) from e
        if r.status_code != 200:
            raise _error(r.status_code, r.content)
        return r.json()


def _error(status: int, content: bytes) -> ProviderError:
    try:
        message = json.loads(content).get("error") or content.decode(errors="replace")
    except (ValueError, AttributeError):
        message = content.decode(errors="replace")
    if status == 404:
        return ProviderError(404, f"Ollama: {message}", "not_found_error", "model_not_found")
    if status == 400:
        return ProviderError(400, f"Ollama: {message}", "invalid_request_error")
    return ProviderError(502, f"Ollama returned HTTP {status}: {message}")


def _context_length(show: dict) -> int | None:
    for key, value in (show.get("model_info") or {}).items():
        if key.endswith(".context_length"):
            return int(value)
    return None


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _finish_reason(done_reason: str | None) -> str:
    return "length" if done_reason == "length" else "stop"


def _parse_arguments(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_calls(raw: list[dict] | None, start: int, with_index: bool = False) -> list[dict]:
    """Ollama returns arguments as an object; OpenAI clients expect a JSON string and an id."""
    out = []
    for i, tc in enumerate(raw or [], start):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        call = {
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": fn.get("name"),
                "arguments": args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False),
            },
        }
        out.append({"index": i, **call} if with_index else call)
    return out


def _format(response_format: dict | None):
    if not response_format:
        return None
    kind = response_format.get("type")
    if kind == "json_object":
        return "json"
    if kind == "json_schema":
        return (response_format.get("json_schema") or {}).get("schema") or "json"
    return None


def _think(body: dict, model: str):
    """`reasoning` / `reasoning_effort` -> Ollama's `think` (an effort level for gpt-oss, else a bool)."""
    reasoning, effort = body.get("reasoning"), body.get("reasoning_effort")
    if reasoning is None and effort is None:
        return None
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            return False
        effort = reasoning.get("effort") or effort
    if effort == "none":
        return False
    if model.startswith("gpt-oss") and effort in ("minimal", "low", "medium", "high"):
        return "low" if effort == "minimal" else effort
    return True


def _usage(data: dict) -> dict:
    return {**usage_dict(data.get("prompt_eval_count"), data.get("eval_count")), "cost": 0.0}


def _extras(data: dict, payload: dict) -> dict:
    def ms(key: str) -> int | None:
        return round(data[key] / 1e6) if data.get(key) is not None else None

    load, prompt_eval = ms("load_duration"), ms("prompt_eval_duration")
    return {
        "load_ms": load,
        "prompt_eval_ms": prompt_eval,
        "eval_ms": ms("eval_duration"),
        "ttft_ms": (load or 0) + (prompt_eval or 0) if load is not None or prompt_eval is not None else None,
        "num_ctx": payload["options"].get("num_ctx"),
        "keep_alive": payload["keep_alive"],
    }
