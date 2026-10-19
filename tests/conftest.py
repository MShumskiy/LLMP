"""
Offline tests for the /v1 adapters: upstreams are httpx.MockTransport handlers,
and chat_log rows go to a list instead of Postgres.
"""

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llmp import v1
from llmp.auth import authenticate
from llmp.chat_log import SessionPrefixes
from llmp.config import load_settings
from llmp.providers import OllamaProvider, OpenRouterProvider

SHOW = {
    "llama3.2:latest": {"capabilities": ["completion", "tools"], "model_info": {"llama.context_length": 131072}},
    "gpt-oss:20b": {"capabilities": ["completion", "tools", "thinking"], "model_info": {"gptoss.context_length": 131072}},
    "tiny:1b": {"capabilities": ["completion"], "model_info": {"llama.context_length": 8192}},
    "nomic-embed-text:latest": {"capabilities": ["embedding"], "model_info": {"nomic-bert.context_length": 2048}},
}


def fake_vector(text: str, dimensions: int = 4) -> list[float]:
    """What FakeOllama returns for `text`; exact in float32, so base64 round trips compare equal."""
    return [float(len(text)), 0.5, -0.25, 1.0][:dimensions]


class FakeOllama:
    """Records /api/chat and /api/embed payloads; chat answers with `reply` (a message dict)."""

    def __init__(self):
        self.payloads: list[dict] = []
        self.embed_payloads: list[dict] = []
        self.reply: dict = {"role": "assistant", "content": "hello"}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/show":
            name = json.loads(request.content)["model"]
            if name not in SHOW:
                return httpx.Response(404, json={"error": f"model '{name}' not found"})
            return httpx.Response(200, json=SHOW[name])
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n, "details": {}} for n in SHOW]})
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0"})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "gpt-oss:20b"}]})
        if path == "/api/chat":
            payload = json.loads(request.content)
            self.payloads.append(payload)
            done = {"done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5,
                    "load_duration": 1_000_000, "prompt_eval_duration": 2_000_000, "eval_duration": 3_000_000}
            if not payload["stream"]:
                return httpx.Response(200, json={"message": self.reply, **done})
            lines = []
            if self.reply.get("thinking"):
                lines.append({"message": {"role": "assistant", "content": "", "thinking": self.reply["thinking"]}, "done": False})
            for word in (self.reply.get("content") or "").split(" "):
                if word:
                    lines.append({"message": {"role": "assistant", "content": word + " "}, "done": False})
            if self.reply.get("tool_calls"):
                lines.append({"message": {"role": "assistant", "content": "", "tool_calls": self.reply["tool_calls"]}, "done": False})
            lines.append({"message": {"role": "assistant", "content": ""}, **done})
            return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
        if path == "/api/embed":
            payload = json.loads(request.content)
            self.embed_payloads.append(payload)
            return httpx.Response(200, json={
                "model": payload["model"],
                "embeddings": [fake_vector(text, payload.get("dimensions", 4)) for text in payload["input"]],
                "prompt_eval_count": sum(len(text.split()) for text in payload["input"]),
                "load_duration": 1_000_000, "total_duration": 5_000_000,
            })
        return httpx.Response(404, json={"error": "unexpected path"})


class FakeOpenRouter:
    def __init__(self):
        self.payloads: list[dict] = []
        self.status = 200
        self.error_body: dict | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            payload = json.loads(request.content)
            self.payloads.append(payload)
            if self.status != 200:
                return httpx.Response(self.status, json=self.error_body or {"error": {"code": self.status, "message": "nope"}})
            usage = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9, "cost": 0.0001,
                     "prompt_tokens_details": {"cached_tokens": 4}}
            if not payload["stream"]:
                return httpx.Response(200, json={
                    "id": "gen-1", "model": payload["model"], "provider": "Fake", "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                    "usage": usage,
                })
            events = [
                ": OPENROUTER PROCESSING",
                "data: " + json.dumps({"id": "gen-2", "model": payload["model"], "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}]}),
                "data: " + json.dumps({"id": "gen-2", "model": payload["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
                "data: " + json.dumps({"id": "gen-2", "model": payload["model"], "choices": [], "usage": usage}),
                "data: [DONE]",
            ]
            return httpx.Response(200, content="\n\n".join(events).encode())
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [
                {"id": "qwen/qwen3-235b-a22b-2507", "context_length": 262144, "supported_parameters": ["tools", "structured_outputs"],
                 "architecture": {"input_modalities": ["text"]}, "pricing": {"prompt": "0.00000009", "completion": "0.00000055"},
                 "top_provider": {"max_completion_tokens": 16384}},
                {"id": "other/model", "context_length": 1000, "supported_parameters": [], "pricing": {}},
            ]})
        return httpx.Response(404)


class ListLog:
    """Stands in for ChatLogWriter: rows go to a list, and `query` returns `results`."""

    def __init__(self):
        self.rows: list[dict] = []
        self.queries: list[tuple[list[str], dict]] = []
        self.results: list[list[dict]] = []

    def enqueue(self, row: dict) -> None:
        self.rows.append(row)

    def ping(self) -> bool:
        return True

    def query(self, statements: list[str], params: dict) -> list[list[dict]]:
        self.queries.append((statements, params))
        return self.results[: len(statements)]


@pytest.fixture
def fakes():
    return FakeOllama(), FakeOpenRouter()


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("ORT_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("url", "http://ollama.test/api/")
    monkeypatch.setenv("LLMP_CONFIG", "does-not-exist.toml")  # built-in defaults, not the repo's llmp.toml
    return load_settings()


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def ollama(fakes, settings):
    return OllamaProvider(make_client(fakes[0]), settings)


@pytest.fixture
def openrouter(fakes, settings):
    return OpenRouterProvider(make_client(fakes[1]), settings)


@pytest.fixture
def api(fakes, settings):
    """A TestClient for the /v1 router with fake upstreams. Returns (client, log rows)."""
    fake_ollama, fake_openrouter = fakes

    def route(request: httpx.Request) -> httpx.Response:
        return fake_ollama(request) if request.url.host == "ollama.test" else fake_openrouter(request)

    http = make_client(route)
    log = ListLog()
    app = FastAPI()
    app.include_router(v1.router)
    app.dependency_overrides[authenticate] = lambda: "test"
    app.state.llmp = v1.Runtime(
        settings, http, {"ollama": OllamaProvider(http, settings), "openrouter": OpenRouterProvider(http, settings)},
        log, SessionPrefixes(),
    )
    with TestClient(app) as client:
        yield client, log.rows
