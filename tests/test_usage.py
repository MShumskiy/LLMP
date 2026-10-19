from datetime import date, datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from llmp.chat_log import usage_result, usage_sql

ZERO = {"calls": 0, "errors": 0, "cancelled": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "cached_tokens": 0, "reasoning_tokens": 0, "cost": Decimal("0")}


def metrics(**kw) -> dict:
    return {**ZERO, **kw}


def test_usage_sql_binds_values_and_groups_by_position():
    since = datetime(2026, 10, 1, tzinfo=timezone.utc)
    statements, params = usage_sql(["model", "day"], {"src": ["vega"], "run_id": ["r1", "r2"]},
                                   since, None, "Europe/Lisbon", 50)
    totals, grouped = statements
    assert "GROUP BY" not in totals and "LIMIT" not in totals
    assert "src = ANY(%(src)s) AND run_id = ANY(%(run_id)s) AND ts >= %(since)s" in totals
    assert grouped.startswith("SELECT model AS model, (ts AT TIME ZONE %(tz)s)::date AS day, count(*) AS calls")
    assert "GROUP BY 1, 2 ORDER BY 1, 2 LIMIT %(limit)s" in grouped
    assert params == {"src": ["vega"], "run_id": ["r1", "r2"], "since": since, "tz": "Europe/Lisbon", "limit": 51}

    statements, params = usage_sql([], {}, None, None, "UTC", 50)
    assert len(statements) == 1 and statements[0].endswith("FROM chat_log") and params == {}


@pytest.mark.parametrize("group_by, filters", [
    (["model; DROP TABLE chat_log"], {}),
    ([], {"status": ["ok"]}),
])
def test_usage_sql_rejects_names_outside_the_whitelist(group_by, filters):
    with pytest.raises(ValueError):
        usage_sql(group_by, filters, None, None, "UTC", 10)


def test_usage_result_shapes_rows():
    totals = metrics(calls=3, errors=1, prompt_tokens=100, completion_tokens=20, cost=Decimal("0.0125"))
    rows = [{"model": "ollama/a", "day": date(2026, 10, 18), **metrics(calls=2, prompt_tokens=60, completion_tokens=5)},
            {"model": "openrouter/b", "day": date(2026, 10, 18), **metrics(calls=1, prompt_tokens=40, completion_tokens=15)}]
    out = usage_result(["model", "day"], [[totals], rows], limit=1)
    assert out["totals"]["cost"] == 0.0125 and out["totals"]["total_tokens"] == 120
    assert list(out["totals"])[:6] == ["calls", "errors", "cancelled", "prompt_tokens", "completion_tokens", "total_tokens"]
    assert out["data"] == [{"model": "ollama/a", "day": "2026-10-18", **metrics(calls=2, prompt_tokens=60,
                            completion_tokens=5, cost=0.0), "total_tokens": 65}]
    assert out["truncated"] is True

    out = usage_result([], [[totals]], limit=10)
    assert out["data"] == [out["totals"]] and out["truncated"] is False


def test_usage_route(api):
    client, _ = api
    log = client.app.state.llmp.chat_log
    log.results = [[metrics(calls=2, prompt_tokens=30, completion_tokens=3, cost=Decimal("0.5"))],
                   [{"model": "ollama/a", "day": date(2026, 10, 2), **metrics(calls=2, prompt_tokens=30, completion_tokens=3)}]]
    r = client.get("/v1/usage", params=[
        ("src", "vega"), ("model", "ollama/a,ollama/b"), ("model", "ollama/c"), ("group_by", "model,day"),
        ("since", "2026-10-01"), ("until", "2026-10-31T00:00:00Z"), ("tz", "Europe/Lisbon"),
    ])
    assert r.status_code == 200
    body = r.json()
    (statements, params), = log.queries
    assert params["src"] == ["vega"] and params["model"] == ["ollama/a", "ollama/b", "ollama/c"]
    assert params["since"] == datetime(2026, 10, 1, tzinfo=ZoneInfo("Europe/Lisbon"))  # no offset: read in tz
    assert params["until"] == datetime(2026, 10, 31, tzinfo=timezone.utc)
    assert body["since"] == "2026-10-01T00:00:00+01:00" and body["group_by"] == ["model", "day"]
    assert body["filters"] == {"src": ["vega"], "model": ["ollama/a", "ollama/b", "ollama/c"]}
    assert body["totals"]["calls"] == 2 and body["totals"]["cost"] == 0.5 and body["totals"]["total_tokens"] == 33
    assert body["data"][0]["day"] == "2026-10-02" and body["truncated"] is False

    log.results = [[metrics(calls=7)]]
    body = client.get("/v1/usage").json()
    assert body["data"] == [body["totals"]] and body["totals"]["calls"] == 7 and body["since"] is None


@pytest.mark.parametrize("query, status", [
    ("group_by=model,status", 400),
    ("group_by=day&tz=Mars/Olympus", 400),
    ("limit=0", None),
    ("since=yesterday", None),
    ("grop_by=model", None),  # typo: unknown parameters are rejected, not ignored
])
def test_usage_rejects_bad_parameters(api, query, status):
    client, _ = api
    r = client.get(f"/v1/usage?{query}")
    assert r.status_code == status if status else r.status_code in (400, 422)  # 400 in the real app, 422 here
    assert not client.app.state.llmp.chat_log.queries
