"""
API-key and IP authentication, shared by the legacy routes and /v1.

Keys come from .env:
  authentication_key=...      the original key, logged under the name "default"
  LLMP_KEY_<NAME>=...         one named key per caller, e.g. LLMP_KEY_MYAPP

The key may be sent bare (`Authorization: <key>`, as older clients do) or as
`Authorization: Bearer <key>` (as the OpenAI SDK does).

ALLOWED_IPS is a comma-separated list of addresses and CIDR ranges
(`127.0.0.1,192.168.1.0/24,::1`). `*` allows any address. When it is unset,
only loopback addresses are allowed.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os

from fastapi import Header, HTTPException, Request

import llmp.config  # noqa: F401  (loads .env before the keys below are read)

log = logging.getLogger("llmp.auth")

IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network
_LOOPBACK = [ipaddress.ip_network("127.0.0.0/8"), ipaddress.ip_network("::1/128")]


def load_keys(environ=os.environ) -> dict[str, str]:
    """Return {key: name}."""
    keys: dict[str, str] = {}
    if environ.get("authentication_key"):
        keys[environ["authentication_key"]] = "default"
    for var, value in environ.items():
        if var.upper().startswith("LLMP_KEY_") and value:
            keys[value] = var[len("LLMP_KEY_"):].lower()
    return keys


def parse_allowed_ips(raw: str | None) -> list[IPNetwork]:
    if raw is None or not raw.strip():
        log.warning("ALLOWED_IPS is not set; only loopback clients are allowed")
        return list(_LOOPBACK)
    networks: list[IPNetwork] = []
    for entry in (e.strip() for e in raw.split(",")):
        if not entry:
            continue
        if entry == "*":
            networks += [ipaddress.ip_network("0.0.0.0/0"), ipaddress.ip_network("::/0")]
        elif entry.lower() == "localhost":
            networks += _LOOPBACK
        else:
            try:
                networks.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                log.warning("Ignoring invalid ALLOWED_IPS entry %r", entry)
    return networks


def ip_allowed(host: str | None, networks: list[IPNetwork]) -> bool:
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in net for net in networks)


def key_name(credentials: str, keys: dict[str, str]) -> str | None:
    token = credentials.strip()
    if token[:7].lower() == "bearer ":
        token = token[7:].strip()
    name = None
    for key, key_owner in keys.items():
        # compare against every key so timing doesn't reveal which prefix matched
        if hmac.compare_digest(token.encode(), key.encode()):
            name = key_owner
    return name


_KEYS = load_keys()
_NETWORKS = parse_allowed_ips(os.getenv("ALLOWED_IPS"))


def authenticate(request: Request, credentials: str | None = Header(None, alias="Authorization")) -> str:
    """FastAPI dependency. Returns the caller's key name and stores it on request.state."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing credentials")
    name = key_name(credentials, _KEYS)
    if name is None:
        raise HTTPException(status_code=403, detail="Invalid credentials")
    client_ip = request.client.host if request.client else None
    if not ip_allowed(client_ip, _NETWORKS):
        raise HTTPException(status_code=403, detail=f"Access denied for IP: {client_ip}")
    request.state.key_name = name
    return name
