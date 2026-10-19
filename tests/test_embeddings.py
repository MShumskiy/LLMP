import base64
import struct

import pytest
from openai import BadRequestError, NotFoundError, OpenAI

from tests.conftest import fake_vector

EMBED = "ollama/nomic-embed-text:latest"


def test_embeddings_float_response_and_log(api, fakes):
    client, rows = api
    r = client.post("/v1/embeddings", json={
        "model": EMBED, "input": ["hello world", "abc"], "encoding_format": "float",
        "llmp": {"src": "test", "session_id": "s1"},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list" and body["model"] == EMBED
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert body["data"][0] == {"object": "embedding", "index": 0, "embedding": fake_vector("hello world")}
    assert body["usage"] == {"prompt_tokens": 3, "total_tokens": 3, "cost": 0.0}
    assert body["llmp"]["provider"] == "ollama" and body["llmp"]["log_id"] and body["llmp"]["load_ms"] == 1

    (payload,) = fakes[0].embed_payloads
    assert payload["input"] == ["hello world", "abc"]
    assert payload["options"]["num_ctx"] == 2048  # the model's context, below the 32768 default
    assert payload["keep_alive"] == "30m" and "dimensions" not in payload

    (row,) = rows
    assert (row["endpoint"], row["provider"], row["model"], row["status"]) == ("embeddings", "ollama", EMBED, "ok")
    assert (row["src"], row["session_id"], row["prompt_tokens"], row["cost"]) == ("test", "s1", 3, 0.0)
    assert row["request"]["input"] == ["hello world", "abc"] and row["request"]["n_inputs"] == 2
    assert row["response"] == {"n": 2, "dimensions": 4}  # never the vectors


def test_embeddings_use_a_smaller_default_context(api, fakes):
    client, _ = api
    for llmp in ({}, {"num_ctx": 16384}):
        assert client.post("/v1/embeddings", json={"model": "ollama/qwen3-embedding:0.6b", "input": "x",
                                                   "llmp": llmp}).status_code == 200
    assert [p["options"]["num_ctx"] for p in fakes[0].embed_payloads] == [8192, 16384]
    catalog = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert catalog["ollama/qwen3-embedding:0.6b"]["default_num_ctx"] == 8192
    assert catalog["ollama/llama3.2:latest"]["default_num_ctx"] == 32768


def test_embeddings_meta_mode_stores_no_input(api):
    client, rows = api
    r = client.post("/v1/embeddings", json={"model": EMBED, "input": "secret text", "llmp": {"log": "meta"}})
    assert r.status_code == 200
    (row,) = rows
    assert "input" not in row["request"] and row["request"]["n_inputs"] == 1 and row["request"]["input_sha256"]
    assert row["response"] is None and row["log_mode"] == "meta"


def test_embeddings_base64(api):
    client, _ = api
    r = client.post("/v1/embeddings", json={"model": EMBED, "input": "hi", "encoding_format": "base64"})
    raw = base64.b64decode(r.json()["data"][0]["embedding"])
    assert list(struct.unpack(f"<{len(raw) // 4}f", raw)) == fake_vector("hi")


def test_openai_sdk_round_trip(api, fakes):
    """The stock SDK asks for base64 unless told otherwise, and decodes it itself."""
    client, rows = api
    sdk = OpenAI(base_url="http://testserver/v1", api_key="test", http_client=client)
    r = sdk.embeddings.create(model=EMBED, input=["one", "three"], dimensions=2,
                              extra_body={"llmp": {"src": "sdk", "truncate": False}})
    assert [d.embedding for d in r.data] == [fake_vector("one", 2), fake_vector("three", 2)]
    assert r.usage.prompt_tokens == 2 and r.model == EMBED
    assert fakes[0].embed_payloads[0]["dimensions"] == 2 and fakes[0].embed_payloads[0]["truncate"] is False
    assert sdk.embeddings.create(model=EMBED, input="x", encoding_format="float").data[0].embedding == fake_vector("x")

    with pytest.raises(BadRequestError):
        sdk.embeddings.create(model="ollama/llama3.2:latest", input="x")
    with pytest.raises(NotFoundError):
        sdk.embeddings.create(model="ollama/missing:1b", input="x")
    assert [row["status"] for row in rows] == ["ok", "ok", "error", "error"]


@pytest.mark.parametrize("body, status, code", [
    ({"model": "ollama/llama3.2:latest", "input": "x"}, 400, "model_not_supported"),  # a chat model
    ({"model": "ollama/missing:1b", "input": "x"}, 404, "model_not_found"),
    ({"model": "nope", "input": "x"}, 404, "model_not_found"),
    ({"model": "openrouter/openai/text-embedding-3-small", "input": "x"}, 400, "model_not_supported"),
    ({"model": EMBED, "input": [1, 2, 3]}, 400, None),  # token ids
    ({"model": EMBED, "input": []}, 400, None),
])
def test_embeddings_errors_are_openai_shaped_and_logged(api, body, status, code):
    client, rows = api
    r = client.post("/v1/embeddings", json=body)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    (row,) = rows
    assert (row["endpoint"], row["status"], row["http_status"]) == ("embeddings", "error", status)


def test_embeddings_rejects_bad_encoding_format(api):
    client, rows = api
    r = client.post("/v1/embeddings", json={"model": EMBED, "input": "x", "encoding_format": "int8"})
    assert r.status_code in (400, 422) and not rows
