"""
The chat_log table, written by a background thread so logging never slows generation.

Routes call `ChatLogWriter.enqueue(row)`, which never blocks. One writer thread
drains the queue in batches through a psycopg 3 connection pool. (A thread with the
sync driver rather than psycopg's async mode, because psycopg's async mode doesn't
run on the Proactor event loop that uvicorn uses on Windows.)

The `log` option controls how much of each request is stored:
  full          the whole request
  last_message  the messages added since the previous call in the same session_id,
                plus the length and a hash of the prefix they extend
  meta          parameters and metrics only
Image data URLs are replaced by their hash and size in every mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from collections import OrderedDict

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

log = logging.getLogger("llmp.chat_log")

LOG_MODES = ("full", "last_message", "meta")

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chat_log (
    id UUID PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    key_name TEXT,
    src TEXT,
    session_id TEXT,
    run_id TEXT,
    agent TEXT,
    tags JSONB,
    client_ip TEXT,
    provider TEXT,
    model TEXT,
    stream BOOLEAN,
    log_mode TEXT,
    gen_id TEXT,
    request JSONB,
    response JSONB,
    prompt_tokens INT,
    completion_tokens INT,
    cached_tokens INT,
    reasoning_tokens INT,
    cost NUMERIC,
    latency_ms INT,
    ttft_ms INT,
    status TEXT,
    http_status INT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS chat_log_ts_idx ON chat_log (ts);
CREATE INDEX IF NOT EXISTS chat_log_session_idx ON chat_log (session_id, ts);
CREATE INDEX IF NOT EXISTS chat_log_src_ts_idx ON chat_log (src, ts);
"""

COLUMNS = (
    "id", "ts", "key_name", "src", "session_id", "run_id", "agent", "tags", "client_ip",
    "provider", "model", "stream", "log_mode", "gen_id", "request", "response",
    "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "cost",
    "latency_ms", "ttft_ms", "status", "http_status", "error",
)
JSON_COLUMNS = {"tags", "request", "response"}
INSERT = f"INSERT INTO chat_log ({', '.join(COLUMNS)}) VALUES ({', '.join(['%s'] * len(COLUMNS))})"

_STOP = object()


def _dumps(obj) -> str:
    # Postgres jsonb rejects \u0000
    return json.dumps(obj, ensure_ascii=False, default=str).replace("\\u0000", "")


class ChatLogWriter:
    def __init__(self, conninfo: dict, maxsize: int = 10_000, batch_size: int = 100):
        self._pool = ConnectionPool(
            make_conninfo(**{k: v for k, v in conninfo.items() if v}),
            min_size=1, max_size=4, open=False, name="llmp-chat-log",
            kwargs={"connect_timeout": 5},
        )
        self._queue: queue.Queue = queue.Queue(maxsize)
        self._batch_size = batch_size
        self._thread = threading.Thread(target=self._run, name="llmp-chat-log", daemon=True)
        self._table_ready = False
        self._dropped = 0

    def start(self) -> None:
        self._pool.open(wait=False)
        self._thread.start()

    def close(self, timeout: float = 10) -> None:
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout)
        self._pool.close(timeout=5)

    def enqueue(self, row: dict) -> None:
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                log.error("chat_log queue full; %d rows dropped so far", self._dropped)

    def ping(self) -> bool:
        try:
            with self._pool.connection(timeout=3) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ── Writer thread ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            batch = [item]
            while len(batch) < self._batch_size:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STOP:
                    self._write(batch)
                    return
                batch.append(item)
            self._write(batch)

    def _write(self, batch: list[dict]) -> None:
        for attempt in range(5):
            try:
                with self._pool.connection(timeout=10) as conn:
                    if not self._table_ready:
                        conn.execute(CREATE_TABLE)
                        self._table_ready = True
                    with conn.cursor() as cur:
                        cur.executemany(INSERT, [self._values(row) for row in batch])
                return
            except (psycopg.OperationalError, PoolTimeout) as e:
                log.warning("chat_log write failed (attempt %d): %s", attempt + 1, e)
                time.sleep(min(2 ** attempt, 15))
            except Exception:
                log.exception("chat_log write failed; dropping %d rows", len(batch))
                return
        log.error("chat_log unreachable; dropping %d rows", len(batch))

    @staticmethod
    def _values(row: dict) -> tuple:
        values = []
        for col in COLUMNS:
            v = row.get(col)
            if col in JSON_COLUMNS and v is not None:
                v = Jsonb(v, dumps=_dumps)
            elif isinstance(v, str):
                v = v.replace("\x00", "")
            values.append(v)
        return tuple(values)


# ── What to store ──────────────────────────────────────────────────────────────


class SessionPrefixes:
    """Remembers, per session_id, how many messages the last call had and their hash."""

    def __init__(self, max_sessions: int = 5_000):
        self._seen: OrderedDict[str, tuple[int, str]] = OrderedDict()
        self._max = max_sessions

    def split(self, session_id: str | None, messages: list[dict]) -> tuple[int, str | None, list[dict]]:
        """Return (prefix_len, prefix_hash, new_messages) and remember this call."""
        hashes = _chain_hashes(messages)
        start = 0
        if session_id is None:
            start = max(len(messages) - 1, 0)
        elif session_id in self._seen:
            n, h = self._seen[session_id]
            if n <= len(messages) and (hashes[n] if n else "") == h:
                start = n
        if session_id is not None:
            self._seen[session_id] = (len(messages), hashes[len(messages)])
            self._seen.move_to_end(session_id)
            while len(self._seen) > self._max:
                self._seen.popitem(last=False)
        return start, (hashes[start] if start else None), messages[start:]


def _chain_hashes(messages: list[dict]) -> list[str]:
    """hashes[i] is a hash of messages[:i]; hashes[0] is ''."""
    out, h = [""], ""
    for m in messages:
        h = hashlib.sha256((h + json.dumps(m, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
        out.append(h)
    return out


def elide_images(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in content):
            parts = []
            for p in content:
                image = p.get("image_url") if isinstance(p, dict) else None
                url = image.get("url") or "" if isinstance(image, dict) else ""
                if url.startswith("data:"):
                    header, _, data = url.partition(",")
                    digest = hashlib.sha256(data.encode()).hexdigest()[:16]
                    p = {**p, "image_url": {**p["image_url"], "url": f"{header},<sha256:{digest} {len(data)} chars>"}}
                parts.append(p)
            m = {**m, "content": parts}
        out.append(m)
    return out


def request_record(body: dict, mode: str, session_id: str | None, prefixes: SessionPrefixes) -> dict:
    params = {k: v for k, v in body.items() if k not in ("messages", "tools", "llmp")}
    messages = body.get("messages") or []
    tools = body.get("tools")
    record: dict = {"params": params, "n_messages": len(messages)}
    if tools:
        record["tool_names"] = [(t.get("function") or {}).get("name") for t in tools]
        record["tools_sha256"] = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
    if mode == "full":
        record["messages"] = elide_images(messages)
        if tools:
            record["tools"] = tools
    elif mode == "last_message":
        prefix_len, prefix_hash, new = prefixes.split(session_id, messages)
        record.update(prefix_len=prefix_len, prefix_sha256=prefix_hash, messages=elide_images(new))
    return record
