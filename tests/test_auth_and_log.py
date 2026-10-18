from llmp.auth import ip_allowed, key_name, load_keys, parse_allowed_ips
from llmp.chat_log import SessionPrefixes, elide_images, request_record


def test_keys_bearer_and_bare():
    keys = load_keys({"authentication_key": "k-default", "LLMP_KEY_VEGA": "k-vega", "LLMP_KEY_ORION": ""})
    assert keys == {"k-default": "default", "k-vega": "vega"}
    assert key_name("k-default", keys) == "default"
    assert key_name("Bearer k-vega", keys) == "vega"
    assert key_name("bearer   k-vega ", keys) == "vega"
    assert key_name("Bearer wrong", keys) is None
    assert key_name("k-vega-extra", keys) is None


def test_allowed_ips_cidr():
    nets = parse_allowed_ips("127.0.0.1, 192.168.1.0/24,fd00::/8, not-an-ip")
    assert ip_allowed("192.168.1.77", nets)
    assert ip_allowed("::ffff:192.168.1.5", nets)  # IPv4-mapped IPv6
    assert ip_allowed("fd12::1", nets)
    assert not ip_allowed("192.168.2.1", nets)
    assert not ip_allowed("testclient", nets)
    assert ip_allowed("127.0.0.5", parse_allowed_ips(None))  # unset: loopback only
    assert not ip_allowed("10.0.0.1", parse_allowed_ips(""))
    assert ip_allowed("8.8.8.8", parse_allowed_ips("*"))


def test_last_message_logs_only_new_messages():
    prefixes = SessionPrefixes()
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}]
    first = request_record({"model": "x", "messages": m}, "last_message", "s1", prefixes)
    assert first["prefix_len"] == 0 and len(first["messages"]) == 2

    m2 = m + [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}]
    second = request_record({"model": "x", "messages": m2}, "last_message", "s1", prefixes)
    assert second["prefix_len"] == 2 and second["messages"] == m2[2:] and second["prefix_sha256"]

    # history rewritten (e.g. compaction): the whole request is new again
    m3 = [{"role": "system", "content": "s"}, {"role": "user", "content": "summary"}, {"role": "user", "content": "u3"}]
    third = request_record({"model": "x", "messages": m3}, "last_message", "s1", prefixes)
    assert third["prefix_len"] == 0 and len(third["messages"]) == 3

    # no session: just the last message
    lone = request_record({"model": "x", "messages": m2}, "last_message", None, prefixes)
    assert lone["prefix_len"] == 3 and lone["messages"] == m2[3:]


def test_full_and_meta_modes():
    body = {"model": "x", "temperature": 0, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "add"}}], "llmp": {"src": "t"}}
    full = request_record(body, "full", None, SessionPrefixes())
    assert full["messages"] == body["messages"] and full["tools"] == body["tools"]
    assert full["params"] == {"model": "x", "temperature": 0} and full["tool_names"] == ["add"]
    meta = request_record(body, "meta", None, SessionPrefixes())
    assert "messages" not in meta and meta["n_messages"] == 1


def test_images_are_elided():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    out = elide_images(msgs)
    assert out[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,<sha256:")
    assert msgs[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA"  # input untouched
