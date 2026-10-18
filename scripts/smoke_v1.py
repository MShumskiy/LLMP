"""
End-to-end check of LLMP's /v1 API with nothing but the stock OpenAI SDK.

For each model it runs a two-step tool round trip (the model calls add_numbers,
the script returns the sum, the model answers) with streaming off and then on.
With --image it also asks each model to describe a generated picture sent as a
data: URL.

    set LLMP_API_KEY=<key>
    python scripts/smoke_v1.py --base-url http://localhost:8000/v1 ollama/gpt-oss:20b openrouter/qwen/qwen3-235b-a22b-2507
    python scripts/smoke_v1.py --image ollama/gemma3:4b openrouter/qwen/qwen3.7-flash
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import uuid

from openai import AsyncOpenAI

TOOLS = [{
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers and return their sum.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
}]
SYSTEM = "You are a calculator assistant. Always use the add_numbers tool to add numbers; never add them yourself."
QUESTION = "What is 1234.5 plus 8765.25?"
ANSWER = "9999.75"


async def step(client: AsyncOpenAI, model: str, messages: list, stream: bool, session: str) -> tuple[dict, str, object]:
    """One model call. Returns (assistant message dict, finish_reason, usage)."""
    kwargs = dict(model=model, messages=messages, tools=TOOLS, temperature=0,
                  extra_body={"llmp": {"src": "smoke_v1", "session_id": session}})
    if not stream:
        r = await client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        calls = [tc.model_dump(exclude_none=True) for tc in msg.tool_calls or []]
        return {"role": "assistant", "content": msg.content, "tool_calls": calls}, r.choices[0].finish_reason, r.usage

    content, calls, finish, usage = [], {}, None, None
    async for chunk in await client.chat.completions.create(**kwargs, stream=True, stream_options={"include_usage": True}):
        usage = chunk.usage or usage
        for choice in chunk.choices:
            delta = choice.delta
            content.append(delta.content or "")
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                slot["id"] = slot["id"] or tc.id
                if tc.function:
                    slot["function"]["name"] = slot["function"]["name"] or tc.function.name
                    slot["function"]["arguments"] += tc.function.arguments or ""
            finish = choice.finish_reason or finish
    message = {"role": "assistant", "content": "".join(content) or None, "tool_calls": [calls[i] for i in sorted(calls)]}
    return message, finish, usage


async def tool_round_trip(client: AsyncOpenAI, model: str, stream: bool) -> str:
    session = f"smoke-{uuid.uuid4().hex[:8]}"
    messages: list = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]

    msg, finish, _ = await step(client, model, messages, stream, session)
    if finish != "tool_calls" or not msg["tool_calls"]:
        raise AssertionError(f"step 1: expected a tool call, got finish_reason={finish!r}, content={msg['content']!r}")
    messages.append({k: v for k, v in msg.items() if v})
    for call in msg["tool_calls"]:
        if call["function"]["name"] != "add_numbers":
            raise AssertionError(f"step 1: unexpected tool {call['function']['name']!r}")
        args = json.loads(call["function"]["arguments"])
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"sum": args["a"] + args["b"]})})

    msg, finish, usage = await step(client, model, messages, stream, session)
    answer = (msg["content"] or "").replace(",", "")
    if finish != "stop" or ANSWER not in answer:
        raise AssertionError(f"step 2: expected an answer with {ANSWER}, got finish_reason={finish!r}, content={msg['content']!r}")
    cost = (usage.model_extra or {}).get("cost") if usage else None
    return f"{len(msg['content'])} chars, usage={usage.prompt_tokens if usage else '?'}+{usage.completion_tokens if usage else '?'} tokens, cost={cost}"


def red_circle_png() -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(img).ellipse((48, 48, 208, 208), fill=(220, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


async def describe_image(client: AsyncOpenAI, model: str) -> str:
    r = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Describe this image in one short sentence: what shape is it and what colour?"},
            {"type": "image_url", "image_url": {"url": red_circle_png()}},
        ]}],
        extra_body={"llmp": {"src": "smoke_v1"}},
    )
    text = r.choices[0].message.content or ""
    if "red" not in text.lower() or not any(w in text.lower() for w in ("circle", "circular", "round", "disc", "dot")):
        raise AssertionError(f"expected a red circle, got {text!r}")
    return text.strip()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--base-url", default=os.getenv("LLMP_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--image", action="store_true", help="run the vision check instead of the tool round trip")
    args = parser.parse_args()
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ["LLMP_API_KEY"], max_retries=0, timeout=600)

    failures = 0
    for model in args.models:
        checks = [("image", describe_image(client, model))] if args.image else [
            ("tools, no stream", tool_round_trip(client, model, stream=False)),
            ("tools, stream", tool_round_trip(client, model, stream=True)),
        ]
        for label, check in checks:
            try:
                print(f"PASS {model} [{label}]: {await check}")
            except Exception as e:
                failures += 1
                print(f"FAIL {model} [{label}]: {type(e).__name__}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
