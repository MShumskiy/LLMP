"""
OpenAI-compatible API under /v1: chat completions (plain and SSE), embeddings,
the model catalog, health and usage totals. Model ids are "<provider>/<model>",
e.g. "ollama/gpt-oss:20b" or "openrouter/qwen/qwen3-235b-a22b-2507"; the prefix
picks the provider adapter.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio
import httpx
import psycopg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field

from llmp.auth import authenticate
from llmp.chat_log import (
    USAGE_FILTERS, USAGE_GROUPS, ChatLogWriter, SessionPrefixes, embedding_request_record, request_record,
    usage_result, usage_sql,
)
from llmp.config import Settings
from llmp.providers import OllamaProvider, OpenRouterProvider, Provider, ProviderError

log = logging.getLogger("llmp.v1")


class LlmpOptions(BaseModel):
    """The `llmp` extra-body field: caller metadata and Ollama overrides."""

    model_config = ConfigDict(extra="allow")
    src: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    agent: str | None = None
    tags: list[str] | dict | None = None
    keep_alive: str | int | None = None
    num_ctx: int | None = None
    options: dict | None = None  # raw Ollama options, e.g. {"top_k": 20}
    truncate: bool | None = None  # embeddings: False makes Ollama reject inputs longer than num_ctx
    log: Literal["full", "last_message", "meta"] | None = None


class ChatCompletionRequest(BaseModel):
    """Validated for errors and docs; the raw JSON body is what gets forwarded."""

    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[dict] = Field(min_length=1)
    stream: bool = False
    llmp: LlmpOptions | None = None


class EmbeddingRequest(BaseModel):
    """Validated for errors and docs; the raw JSON body is what gets forwarded."""

    model_config = ConfigDict(extra="allow")
    model: str
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = Field(None, gt=0)
    llmp: LlmpOptions | None = None


@dataclass
class Runtime:
    settings: Settings
    client: httpx.AsyncClient
    providers: dict[str, Provider]
    chat_log: ChatLogWriter
    prefixes: SessionPrefixes = field(default_factory=SessionPrefixes)
    catalog: dict[str, tuple[float, list[dict]]] = field(default_factory=dict)


def start_runtime(settings: Settings) -> Runtime:
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=600, write=60, pool=600))
    providers: list[Provider] = [OllamaProvider(client, settings), OpenRouterProvider(client, settings)]
    chat_log = ChatLogWriter(settings.db_conninfo)
    chat_log.start()
    return Runtime(settings, client, {p.name: p for p in providers}, chat_log)


async def stop_runtime(rt: Runtime) -> None:
    await asyncio.to_thread(rt.chat_log.close)
    await rt.client.aclose()


router = APIRouter(prefix="/v1", dependencies=[Depends(authenticate)])


def _rt(request: Request) -> Runtime:
    return request.app.state.llmp


def _error_response(e: ProviderError) -> JSONResponse:
    return JSONResponse(e.body(), status_code=e.status)


# ── Chat completions ───────────────────────────────────────────────────────────


@router.post("/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    rt = _rt(request)
    raw = await request.json()
    opts = body.llmp.model_dump(exclude_none=True) if body.llmp else {}
    call = _Call(rt, request, raw, opts)
    try:
        provider, model = _resolve(rt, body.model)
    except ProviderError as e:
        call.finish("error", e.status, error=e.message)
        return _error_response(e)
    if body.stream:
        return await _stream(call, provider.chat_stream(model, raw, opts))

    try:
        resp = await provider.chat(model, raw, opts)
    except ProviderError as e:
        call.finish("error", e.status, error=e.message)
        return _error_response(e)
    except Exception as e:
        log.exception("chat completion failed")
        call.finish("error", 500, error=f"{type(e).__name__}: {e}")
        return _error_response(ProviderError(500, f"LLMP internal error: {type(e).__name__}: {e}", "server_error"))

    extras = resp.pop("llmp", None) or {}
    gen_id = extras.get("gen_id") or resp.get("id")
    ttft_ms = extras.get("ttft_ms")
    resp["llmp"] = call.llmp_fields(extras, gen_id, ttft_ms)
    choice = (resp.get("choices") or [{}])[0]
    call.finish(
        "ok", 200,
        response={"message": choice.get("message"), "finish_reason": choice.get("finish_reason")},
        usage=resp.get("usage"), gen_id=gen_id, ttft_ms=ttft_ms,
    )
    return JSONResponse(resp)


async def _stream(call: _Call, chunks) -> JSONResponse | StreamingResponse:
    # Pull the first chunk before answering, so errors raised before any output
    # (unknown model, bad request, provider down) still get a proper HTTP status.
    try:
        first = await anext(chunks, None)
    except ProviderError as e:
        call.finish("error", e.status, error=e.message)
        return _error_response(e)
    except Exception as e:
        log.exception("chat completion stream failed")
        call.finish("error", 500, error=f"{type(e).__name__}: {e}")
        return _error_response(ProviderError(500, f"LLMP internal error: {type(e).__name__}: {e}", "server_error"))
    first_at = time.monotonic()

    async def sse():
        acc = _Accumulator()
        ttft_ms = None
        status, http_status, error = "ok", 200, None
        try:
            chunk, received_at = first, first_at
            while chunk is not None:
                if acc.add(chunk) and ttft_ms is None:
                    ttft_ms = round((received_at - call.t0) * 1000)
                if chunk.get("usage") is not None:
                    chunk["llmp"] = call.llmp_fields(chunk.get("llmp") or {}, acc.gen_id(), ttft_ms)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                chunk = await anext(chunks, None)
                received_at = time.monotonic()
            yield "data: [DONE]\n\n"
        except ProviderError as e:
            status, http_status, error = "error", e.status, e.message
            yield f"data: {json.dumps(e.body())}\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            status, http_status, error = "cancelled", 499, "client disconnected"
            raise
        except Exception as e:
            log.exception("chat completion stream failed")
            status, http_status, error = "error", 500, f"{type(e).__name__}: {e}"
            yield f"data: {json.dumps(ProviderError(500, f'LLMP internal error: {error}', 'server_error').body())}\n\n"
        finally:
            with anyio.CancelScope(shield=True):
                await chunks.aclose()  # releases the provider's semaphore and upstream connection
            call.finish(
                status, http_status,
                response=acc.record(), usage=acc.usage, gen_id=acc.gen_id(), ttft_ms=ttft_ms, error=error,
            )

    return StreamingResponse(
        sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


def _resolve(rt: Runtime, model_id: str) -> tuple[Provider, str]:
    prefix, _, model = model_id.partition("/")
    provider = rt.providers.get(prefix)
    if provider is None or not model:
        raise ProviderError(
            404,
            f"Unknown model {model_id!r}. Model ids look like '<provider>/<model>', "
            f"with provider one of: {', '.join(rt.providers)}",
            "not_found_error",
            "model_not_found",
        )
    return provider, model


class _Call:
    """One /v1 chat or embeddings call: its timing, metadata and chat_log row."""

    def __init__(self, rt: Runtime, request: Request, raw: dict, opts: dict, endpoint: str = "chat/completions"):
        self.rt = rt
        self.id = uuid.uuid4()
        self.ts = datetime.now(timezone.utc)
        self.t0 = time.monotonic()
        self.opts = opts
        self.endpoint = endpoint
        self.model = raw.get("model")
        self.provider = (self.model or "").partition("/")[0]
        self.stream = bool(raw.get("stream"))
        self.key_name = getattr(request.state, "key_name", None)
        self.client_ip = request.client.host if request.client else None
        self.log_mode = opts.get("log") or rt.settings.log_default_mode
        if endpoint == "embeddings":
            self.request = embedding_request_record(raw, self.log_mode)
        else:
            self.request = request_record(raw, self.log_mode, opts.get("session_id"), rt.prefixes)

    def latency_ms(self) -> int:
        return round((time.monotonic() - self.t0) * 1000)

    def llmp_fields(self, extras: dict, gen_id: str | None, ttft_ms: int | None) -> dict:
        rest = {k: v for k, v in extras.items() if k not in ("gen_id", "ttft_ms")}
        return {
            "provider": self.provider, "gen_id": gen_id, "log_id": str(self.id),
            "latency_ms": self.latency_ms(), "ttft_ms": ttft_ms, **rest,
        }

    def finish(self, status: str, http_status: int, *, response: dict | None = None, usage: dict | None = None,
               gen_id: str | None = None, ttft_ms: int | None = None, error: str | None = None) -> None:
        usage = usage or {}
        self.rt.chat_log.enqueue({
            "id": self.id, "ts": self.ts, "key_name": self.key_name,
            "src": self.opts.get("src"), "session_id": self.opts.get("session_id"),
            "run_id": self.opts.get("run_id"), "agent": self.opts.get("agent"), "tags": self.opts.get("tags"),
            "client_ip": self.client_ip, "endpoint": self.endpoint,
            "provider": self.provider, "model": self.model, "stream": self.stream,
            "log_mode": self.log_mode, "gen_id": gen_id,
            "request": self.request, "response": None if self.log_mode == "meta" else response,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost": usage.get("cost"),
            "latency_ms": self.latency_ms(), "ttft_ms": ttft_ms,
            "status": status, "http_status": http_status, "error": error,
        })


class _Accumulator:
    """Rebuilds the assistant message from stream chunks, for the log and TTFT."""

    def __init__(self):
        self.id: str | None = None
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.extras: dict = {}

    def add(self, chunk: dict) -> bool:
        """Record a chunk; return True if it carried output (text, reasoning or a tool call)."""
        self.id = self.id or chunk.get("id")
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        if chunk.get("llmp"):
            self.extras.update(chunk["llmp"])
        output = False
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.content.append(delta["content"])
                output = True
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning:
                self.reasoning.append(reasoning)
                output = True
            for tc in delta.get("tool_calls") or []:
                output = True
                slot = self.tool_calls.setdefault(
                    tc.get("index", len(self.tool_calls)),
                    {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
                )
                fn = tc.get("function") or {}
                slot["id"] = slot["id"] or tc.get("id")
                slot["function"]["name"] = slot["function"]["name"] or fn.get("name")
                slot["function"]["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return output

    def gen_id(self) -> str | None:
        return self.extras.get("gen_id") or self.id

    def record(self) -> dict:
        message: dict = {"role": "assistant", "content": "".join(self.content) or None}
        if self.tool_calls:
            message["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        if self.reasoning:
            message["reasoning"] = "".join(self.reasoning)
        return {"message": message, "finish_reason": self.finish_reason}


# ── Embeddings ─────────────────────────────────────────────────────────────────


@router.post("/embeddings")
async def embeddings(request: Request, body: EmbeddingRequest):
    rt = _rt(request)
    raw = await request.json()
    opts = body.llmp.model_dump(exclude_none=True) if body.llmp else {}
    call = _Call(rt, request, raw, opts, endpoint="embeddings")
    try:
        provider, model = _resolve(rt, body.model)
        resp = await provider.embed(model, raw, opts)
    except ProviderError as e:
        call.finish("error", e.status, error=e.message)
        return _error_response(e)
    except Exception as e:
        log.exception("embeddings call failed")
        call.finish("error", 500, error=f"{type(e).__name__}: {e}")
        return _error_response(ProviderError(500, f"LLMP internal error: {type(e).__name__}: {e}", "server_error"))

    data = resp["data"]
    resp["llmp"] = call.llmp_fields(resp.pop("llmp", None) or {}, None, None)
    call.finish("ok", 200, usage=resp.get("usage"), response={
        "n": len(data), "dimensions": len(data[0]["embedding"]) if data else None,
    })
    if body.encoding_format == "base64":
        # What the OpenAI SDK asks for by default: little-endian float32, base64-encoded.
        for item in data:
            vector = item["embedding"]
            item["embedding"] = base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
    return JSONResponse(resp)


# ── Models and health ──────────────────────────────────────────────────────────


@router.get("/models")
async def list_models(request: Request, refresh: bool = False):
    """Models from every provider. Cached for `catalog.ttl_s` (10 min); `?refresh=1` forces a refresh."""
    rt = _rt(request)
    results = await asyncio.gather(*(_catalog(rt, p, refresh) for p in rt.providers.values()))
    data, errors = [], {}
    for (name, provider), (models, error) in zip(rt.providers.items(), results):
        if hasattr(provider, "loaded_models") and models and not error:
            # The catalog is cached, but which local models are in memory changes by the minute.
            try:
                loaded = {f"{name}/{m}" for m in await provider.loaded_models()}
                models = [{**m, "loaded": m["id"] in loaded} for m in models]
            except ProviderError:
                pass
        data += models
        if error:
            errors[name] = error
    out: dict = {"object": "list", "data": data}
    if errors:
        out["errors"] = errors
    return out


@router.get("/models/{model_id:path}")
async def get_model(request: Request, model_id: str):
    rt = _rt(request)
    provider = rt.providers.get(model_id.partition("/")[0])
    if provider is not None:
        models, _ = await _catalog(rt, provider, False)
        for m in models:
            if m["id"] == model_id:
                return m
    return _error_response(ProviderError(404, f"Model {model_id!r} not found", "not_found_error", "model_not_found"))


async def _catalog(rt: Runtime, provider: Provider, refresh: bool) -> tuple[list[dict], str | None]:
    cached = rt.catalog.get(provider.name)
    if cached and not refresh and time.monotonic() - cached[0] < rt.settings.catalog_ttl_s:
        return cached[1], None
    try:
        models = await provider.list_models()
    except ProviderError as e:
        return (cached[1] if cached else []), e.message
    rt.catalog[provider.name] = (time.monotonic(), models)
    return models, None


@router.get("/health")
async def health(request: Request):
    rt = _rt(request)
    *provider_ok, db_ok = await asyncio.gather(
        *(p.health() for p in rt.providers.values()), asyncio.to_thread(rt.chat_log.ping)
    )
    ok = all(provider_ok) and db_ok
    return {
        "status": "ok" if ok else "degraded",
        "providers": {name: "ok" if up else "down" for name, up in zip(rt.providers, provider_ok)},
        "db": "ok" if db_ok else "down",
    }


# ── Usage ──────────────────────────────────────────────────────────────────────


class UsageQuery(BaseModel):
    """Query parameters of /v1/usage. List parameters take repeated or comma-separated values."""

    model_config = ConfigDict(extra="forbid")
    since: datetime | None = Field(None, description="Inclusive start; a time without an offset is read in `tz`")
    until: datetime | None = Field(None, description="Exclusive end; a time without an offset is read in `tz`")
    group_by: list[str] = Field([], description=f"Any of: {', '.join(USAGE_GROUPS)}")
    tz: str = Field("UTC", description="IANA time zone for `day` and for times without an offset")
    limit: int = Field(1000, ge=1, le=10_000, description="Most groups to return")
    src: list[str] = []
    key_name: list[str] = []
    session_id: list[str] = []
    run_id: list[str] = []
    agent: list[str] = []
    provider: list[str] = []
    model: list[str] = []
    endpoint: list[str] = []


@router.get("/usage")
async def usage(request: Request, q: Annotated[UsageQuery, Query()]):
    """Calls, tokens, cost and errors from chat_log, optionally grouped, e.g. ?src=myapp&since=…&group_by=model,day"""
    rt = _rt(request)

    def split(values: list[str]) -> list[str]:
        return list(dict.fromkeys(v.strip() for value in values for v in value.split(",") if v.strip()))

    group_by = split(q.group_by)
    unknown = [g for g in group_by if g not in USAGE_GROUPS]
    if unknown:
        return _error_response(ProviderError(
            400, f"Unknown group_by {', '.join(unknown)}; allowed: {', '.join(USAGE_GROUPS)}", "invalid_request_error"))
    try:
        zone = ZoneInfo(q.tz)
    except (ZoneInfoNotFoundError, ValueError):
        return _error_response(ProviderError(400, f"Unknown time zone {q.tz!r}", "invalid_request_error"))
    since, until = (t if t is None or t.tzinfo else t.replace(tzinfo=zone) for t in (q.since, q.until))
    filters = {name: split(getattr(q, name)) for name in USAGE_FILTERS}
    filters = {name: values for name, values in filters.items() if values}

    statements, params = usage_sql(group_by, filters, since, until, q.tz, q.limit)
    try:
        results = await asyncio.to_thread(rt.chat_log.query, statements, params)
    except (psycopg.OperationalError, PoolTimeout) as e:
        log.warning("usage query failed: %s: %s", type(e).__name__, e)
        return _error_response(ProviderError(503, f"The chat_log database is unavailable ({type(e).__name__})",
                                             "service_unavailable"))
    return {
        "object": "usage",
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
        "tz": q.tz,
        "filters": filters,
        "group_by": group_by,
        **usage_result(group_by, results, q.limit),
    }
