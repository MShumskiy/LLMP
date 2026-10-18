"""
LLMP settings.

Secrets and connection details come from `.env` (as before). Everything else
(defaults, limits, the OpenRouter allowlist) comes from `llmp.toml` at the repo
root, or the file named by the LLMP_CONFIG environment variable.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_toml() -> dict:
    path = Path(os.getenv("LLMP_CONFIG") or REPO_ROOT / "llmp.toml")
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _ollama_root(url: str | None) -> str:
    """`url` in .env is the legacy form `http://host:11434/api/`; adapters want the root."""
    root = (url or "http://127.0.0.1:11434").rstrip("/")
    return root[: -len("/api")] if root.endswith("/api") else root


@dataclass(frozen=True)
class Settings:
    # Ollama
    ollama_url: str
    ollama_keep_alive: str
    ollama_num_ctx: int
    ollama_num_ctx_per_model: dict[str, int]
    ollama_max_output_per_model: dict[str, int]
    ollama_max_concurrency: int
    # OpenRouter
    openrouter_url: str
    openrouter_key: str | None
    openrouter_max_concurrency: int
    openrouter_allowlist: list[str]
    # Catalog and logging
    catalog_ttl_s: int
    log_default_mode: str
    db_conninfo: dict = field(repr=False)


def load_settings() -> Settings:
    cfg = _load_toml()
    ollama = cfg.get("ollama", {})
    openrouter = cfg.get("openrouter", {})
    return Settings(
        ollama_url=_ollama_root(os.getenv("url")),
        ollama_keep_alive=str(ollama.get("keep_alive", "30m")),
        ollama_num_ctx=int(ollama.get("num_ctx", 32768)),
        ollama_num_ctx_per_model=dict(ollama.get("num_ctx_per_model", {})),
        ollama_max_output_per_model=dict(ollama.get("max_output_per_model", {})),
        ollama_max_concurrency=int(ollama.get("max_concurrency", 2)),
        openrouter_url=str(openrouter.get("url", "https://openrouter.ai/api/v1")).rstrip("/"),
        openrouter_key=os.getenv("ORT_API_KEY") or None,
        openrouter_max_concurrency=int(openrouter.get("max_concurrency", 16)),
        openrouter_allowlist=list(openrouter.get("allowlist", [])),
        catalog_ttl_s=int(cfg.get("catalog", {}).get("ttl_s", 600)),
        log_default_mode=str(cfg.get("logging", {}).get("default_mode", "last_message")),
        db_conninfo={
            "user": os.getenv("db_user"),
            "password": os.getenv("db_password"),
            "host": os.getenv("db_host"),
            "port": os.getenv("db_port"),
            "dbname": os.getenv("db_database"),
        },
    )


settings = load_settings()
