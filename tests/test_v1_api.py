import json


def sse_events(text: str) -> list:
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


def test_chat_completion_adds_llmp_fields_and_logs(api):
    client, rows = api
    r = client.post("/v1/chat/completions", json={
        "model": "ollama/llama3.2:latest", "messages": [{"role": "user", "content": "hi"}],
        "llmp": {"src": "test", "session_id": "s1", "tags": ["a"]},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["llmp"]["provider"] == "ollama" and body["llmp"]["log_id"] and body["llmp"]["latency_ms"] >= 0
    assert body["llmp"]["gen_id"] == body["id"]
    (row,) = rows
    assert (row["src"], row["session_id"], row["tags"], row["status"]) == ("test", "s1", ["a"], "ok")
    assert (row["prompt_tokens"], row["completion_tokens"], row["cost"]) == (10, 5, 0.0)
    assert row["key_name"] is None or isinstance(row["key_name"], str)


def test_stream_is_sse_with_usage_last(api):
    client, rows = api
    r = client.post("/v1/chat/completions", json={
        "model": "openrouter/qwen/q", "stream": True, "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    assert events[-1] == "[DONE]"
    last = json.loads(events[-2])
    assert last["usage"]["cost"] == 0.0001 and last["llmp"]["provider"] == "openrouter" and last["llmp"]["gen_id"] == "gen-2"
    (row,) = rows
    assert row["stream"] and row["cost"] == 0.0001 and row["cached_tokens"] == 4
    assert row["response"]["message"]["content"] == "hi" and row["response"]["finish_reason"] == "stop"


def test_errors_before_streaming_get_http_status(api):
    client, rows = api
    for stream in (False, True):
        r = client.post("/v1/chat/completions", json={
            "model": "ollama/nope:1b", "stream": stream, "messages": [{"role": "user", "content": "hi"}],
        })
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"
    r = client.post("/v1/chat/completions", json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert [row["status"] for row in rows] == ["error", "error", "error"]


def test_upstream_error_mid_request(api, fakes):
    client, rows = api
    fakes[1].status = 429
    r = client.post("/v1/chat/completions", json={"model": "openrouter/qwen/q", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429 and r.json()["error"]["type"] == "rate_limit_error"


def test_models_and_health(api):
    client, _ = api
    data = client.get("/v1/models").json()["data"]
    ids = {m["id"] for m in data}
    assert "ollama/gpt-oss:20b" in ids and "openrouter/qwen/qwen3-235b-a22b-2507" in ids
    assert client.get("/v1/models/ollama/gpt-oss:20b").json()["loaded"] is True
    assert client.get("/v1/models/ollama/missing").status_code == 404
    assert client.get("/v1/health").json() == {"status": "ok", "providers": {"ollama": "ok", "openrouter": "ok"}, "db": "ok"}
