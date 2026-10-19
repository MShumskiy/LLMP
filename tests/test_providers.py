import asyncio
import json

import pytest

from llmp.providers import ProviderError

PNG = "data:image/png;base64,iVBORw0KGgo="


def collect(agen) -> list[dict]:
    async def run():
        return [chunk async for chunk in agen]

    return asyncio.run(run())


# ── Ollama: request translation ────────────────────────────────────────────────


def test_ollama_translates_messages(ollama, fakes):
    body = {"model": "x", "messages": [
        {"role": "developer", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "what is this?"}, {"type": "image_url", "image_url": {"url": PNG}}]},
        {"role": "assistant", "content": None, "reasoning": "think",
         "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "2"},
        # a later turn reusing the id call_0 for a different tool: the nearest assistant turn wins
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "mul", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "3"},
    ]}
    asyncio.run(ollama.chat("llama3.2:latest", body, {}))
    msgs = fakes[0].payloads[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert msgs[1] == {"role": "user", "content": "what is this?", "images": ["iVBORw0KGgo="]}
    assert msgs[2]["tool_calls"] == [{"function": {"name": "add", "arguments": {"a": 1}}}]
    assert msgs[2]["thinking"] == "think"
    assert msgs[3] == {"role": "tool", "content": "2", "tool_name": "add"}
    assert msgs[5]["tool_name"] == "mul"


def test_ollama_payload_defaults_and_options(ollama, fakes):
    tools = [{"type": "function", "function": {"name": "add", "parameters": {}}}]
    asyncio.run(ollama.chat("llama3.2:latest", {
        "messages": [{"role": "user", "content": "hi"}], "tools": tools, "temperature": 0.2, "max_tokens": 50,
        "stop": "END", "seed": 1, "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
    }, {}))
    p = fakes[0].payloads[0]
    assert p["keep_alive"] == "30m"
    assert p["tools"] == tools
    assert p["format"] == {"type": "object"}
    assert p["options"] == {"temperature": 0.2, "seed": 1, "num_predict": 50, "stop": ["END"], "num_ctx": 32768}
    assert "think" not in p


def test_ollama_overrides_and_clamping(ollama, fakes):
    msgs = [{"role": "user", "content": "hi"}]
    asyncio.run(ollama.chat("tiny:1b", {"messages": msgs, "tool_choice": "none", "tools": [{}]},
                            {"num_ctx": 65536, "keep_alive": "5m", "options": {"top_k": 20}}))
    p = fakes[0].payloads[0]
    assert p["options"]["num_ctx"] == 8192  # clamped to the model's context length
    assert p["options"]["top_k"] == 20
    assert p["keep_alive"] == "5m"
    assert "tools" not in p


def test_ollama_think_only_for_thinking_models(ollama, fakes):
    msgs = [{"role": "user", "content": "hi"}]
    asyncio.run(ollama.chat("gpt-oss:20b", {"messages": msgs, "reasoning": {"effort": "high"}}, {}))
    asyncio.run(ollama.chat("llama3.2:latest", {"messages": msgs, "reasoning": {"effort": "high"}}, {}))
    asyncio.run(ollama.chat("gpt-oss:20b", {"messages": msgs, "reasoning_effort": "none"}, {}))
    assert fakes[0].payloads[0]["think"] == "high"
    assert "think" not in fakes[0].payloads[1]
    assert fakes[0].payloads[2]["think"] is False


def test_ollama_unknown_model_is_404(ollama):
    with pytest.raises(ProviderError) as e:
        asyncio.run(ollama.chat("nope:1b", {"messages": [{"role": "user", "content": "hi"}]}, {}))
    assert e.value.status == 404 and e.value.code == "model_not_found"


# ── Ollama: response translation ───────────────────────────────────────────────


def test_ollama_tool_calls_become_openai_shape(ollama, fakes):
    fakes[0].reply = {"role": "assistant", "content": "", "thinking": "hmm", "tool_calls": [
        {"function": {"name": "add", "arguments": {"a": 1, "b": 2}}},
        {"function": {"name": "mul", "arguments": {"a": 3}}},
    ]}
    r = asyncio.run(ollama.chat("gpt-oss:20b", {"messages": [{"role": "user", "content": "hi"}]}, {}))
    choice = r["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["reasoning"] == "hmm"
    assert choice["message"]["tool_calls"] == [
        {"id": "call_0", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
        {"id": "call_1", "type": "function", "function": {"name": "mul", "arguments": '{"a": 3}'}},
    ]
    assert r["model"] == "ollama/gpt-oss:20b"
    assert r["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0}
    assert r["llmp"]["load_ms"] == 1 and r["llmp"]["ttft_ms"] == 3


def test_ollama_stream_maps_ndjson_to_chunks(ollama, fakes):
    fakes[0].reply = {"role": "assistant", "content": "two words", "thinking": "hmm",
                      "tool_calls": [{"function": {"name": "add", "arguments": {"a": 1}}}]}
    chunks = collect(ollama.chat_stream("gpt-oss:20b", {"messages": [{"role": "user", "content": "hi"}]}, {}))
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert deltas[0] == {"role": "assistant", "reasoning": "hmm"}
    assert "".join(d.get("content", "") for d in deltas) == "two words "
    calls = [tc for d in deltas for tc in d.get("tool_calls", [])]
    assert calls == [{"index": 0, "id": "call_0", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}]
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 15
    assert len({c["id"] for c in chunks}) == 1


def test_ollama_catalog(ollama):
    models = {m["id"]: m for m in asyncio.run(ollama.list_models())}
    assert models["ollama/gpt-oss:20b"]["capabilities"]["reasoning"] is True
    assert models["ollama/gpt-oss:20b"]["loaded"] is True
    assert models["ollama/tiny:1b"]["context_window"] == 8192
    assert models["ollama/tiny:1b"]["default_num_ctx"] == 8192
    assert models["ollama/llama3.2:latest"]["capabilities"]["tools"] is True


# ── OpenRouter ─────────────────────────────────────────────────────────────────


def test_openrouter_pass_through(openrouter, fakes):
    body = {"model": "openrouter/qwen/q", "messages": [{"role": "user", "content": "hi"}], "llmp": {"src": "x"},
            "reasoning": {"effort": "low"}}
    r = asyncio.run(openrouter.chat("qwen/q", body, {}))
    p = fakes[1].payloads[0]
    assert p["model"] == "qwen/q" and "llmp" not in p and p["usage"] == {"include": True}
    assert p["reasoning"] == {"effort": "low"} and p["stream"] is False
    assert r["model"] == "openrouter/qwen/q"
    assert r["llmp"]["gen_id"] == "gen-1"


def test_openrouter_stream(openrouter, fakes):
    chunks = collect(openrouter.chat_stream("qwen/q", {"messages": [{"role": "user", "content": "hi"}]}, {}))
    assert fakes[1].payloads[0]["stream_options"] == {"include_usage": True}
    assert [c["choices"][0]["delta"].get("content") for c in chunks if c["choices"]] == ["hi", None]
    assert chunks[-1]["usage"]["cost"] == 0.0001 and chunks[-1]["llmp"]["gen_id"] == "gen-2"
    assert all(c["model"] == "openrouter/qwen/q" for c in chunks)


@pytest.mark.parametrize("upstream, expected", [(401, 502), (429, 429), (400, 400), (500, 502)])
def test_openrouter_error_mapping(openrouter, fakes, upstream, expected):
    fakes[1].status = upstream
    with pytest.raises(ProviderError) as e:
        asyncio.run(openrouter.chat("qwen/q", {"messages": [{"role": "user", "content": "hi"}]}, {}))
    assert e.value.status == expected
    assert e.value.body()["error"]["message"].startswith("OpenRouter:")


def test_openrouter_catalog_allowlist_and_pricing(fakes, settings):
    from dataclasses import replace

    from conftest import make_client
    from llmp.providers import OpenRouterProvider

    provider = OpenRouterProvider(make_client(fakes[1]), replace(settings, openrouter_allowlist=["qwen/*"]))
    models = asyncio.run(provider.list_models())
    assert [m["id"] for m in models] == ["openrouter/qwen/qwen3-235b-a22b-2507"]
    m = models[0]
    assert m["capabilities"]["tools"] and m["capabilities"]["json_schema"] and not m["capabilities"]["vision"]
    assert m["pricing"]["input"] == 0.09 and m["pricing"]["output"] == 0.55
    assert m["context_window"] == 262144 and m["max_output"] == 16384
    assert json.dumps(m)  # serialisable
