import asyncio

import httpx
import pytest

from config import Settings
from core.errors import AppError, ErrorCode
from core.llm import LLMClient, _classify_http_error
from core.resilience import CircuitBreaker, retry_async
from rag.chunking import split_text
from rag.pipeline import extract_json


async def _no_sleep(_delay):
    return None


# ---------------- 熔断器 ----------------
def test_breaker_opens_after_threshold_and_half_opens(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("core.resilience._now", lambda: now[0])
    b = CircuitBreaker("t", failure_threshold=3, recovery_timeout_s=10)

    for _ in range(2):
        b.record_failure()
    assert b.state == "closed" and b.allow()
    b.record_failure()
    assert b.state == "open" and not b.allow()

    now[0] += 10
    assert b.state == "half_open"
    assert b.allow()          # 放行一次试探
    assert not b.allow()      # 试探未结束前不再放行
    b.record_success()
    assert b.state == "closed"


def test_breaker_failed_probe_reopens(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("core.resilience._now", lambda: now[0])
    b = CircuitBreaker("t", failure_threshold=1, recovery_timeout_s=5)
    b.record_failure()
    now[0] += 5
    assert b.allow()
    b.record_failure()
    assert b.state == "open"


# ---------------- 重试 ----------------
def test_retry_only_retryable(monkeypatch):
    monkeypatch.setattr("core.resilience._sleep", _no_sleep)
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise AppError(ErrorCode.UPSTREAM_LLM_TIMEOUT)
        return "ok"

    assert asyncio.run(retry_async(flaky, attempts=3)) == "ok"
    assert calls["n"] == 3

    calls["n"] = 0

    async def quota():
        calls["n"] += 1
        raise AppError(ErrorCode.UPSTREAM_LLM_QUOTA)

    with pytest.raises(AppError) as ei:
        asyncio.run(retry_async(quota, attempts=3))
    assert ei.value.code == ErrorCode.UPSTREAM_LLM_QUOTA
    assert calls["n"] == 1  # 额度不足不重试


# ---------------- LLM 错误分类 ----------------
@pytest.mark.parametrize(
    "status,body,expected",
    [
        (402, '{"error":{"message":"Insufficient Balance"}}', ErrorCode.UPSTREAM_LLM_QUOTA),
        (403, '{"error":{"code":"AllocationQuota.FreeTierOnly"}}', ErrorCode.UPSTREAM_LLM_QUOTA),
        (401, '{"error":"invalid api key"}', ErrorCode.UPSTREAM_LLM_AUTH),
        (429, '{"error":"rate limit"}', ErrorCode.UPSTREAM_LLM_RATE),
        (503, "busy", ErrorCode.UPSTREAM_LLM_UNAVAILABLE),
        (400, "bad", ErrorCode.UPSTREAM_LLM_BAD_RESPONSE),
    ],
)
def test_classify_http_error(status, body, expected):
    assert _classify_http_error(status, body) == expected


def _client_with_transport(handler, attempts=3):
    settings = Settings(llm_api_key="sk-test", llm={"retry_attempts": attempts, "breaker_failure_threshold": 2})
    client = LLMClient(settings)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def test_llm_client_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("core.resilience._sleep", _no_sleep)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"choices": [{"message": {"content": " 你好 "}}], "usage": {}})

    client = _client_with_transport(handler)
    assert asyncio.run(client.chat([{"role": "user", "content": "hi"}])) == "你好"
    assert calls["n"] == 2
    assert client.breaker.state == "closed"


def test_llm_client_quota_not_retried(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    client = _client_with_transport(handler)
    with pytest.raises(AppError) as ei:
        asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
    assert ei.value.code == ErrorCode.UPSTREAM_LLM_QUOTA
    assert calls["n"] == 1
    assert client.breaker.state == "closed"  # 额度问题不触发熔断


def test_llm_client_breaker_opens_on_repeated_failures(monkeypatch):
    monkeypatch.setattr("core.resilience._sleep", _no_sleep)

    def handler(request):
        return httpx.Response(500, text="down")

    client = _client_with_transport(handler, attempts=1)
    for _ in range(2):
        with pytest.raises(AppError):
            asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
    assert client.breaker.state == "open"


def test_llm_disabled_without_key():
    client = LLMClient(Settings(llm_api_key=""))
    with pytest.raises(AppError) as ei:
        asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
    assert ei.value.code == ErrorCode.UPSTREAM_LLM_AUTH


# ---------------- 配置 ----------------
def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("LLM__MODEL", "override-model")
    s = Settings()
    assert s.llm.model == "override-model"
    assert s.rag.llm_model  # 未单独配置时沿用 llm.model 或 yaml 中的值
    assert s.rag.db_dir.is_absolute()
    assert "llm_api_key" not in s.public_view()


# ---------------- 工具函数 ----------------
def test_split_text_respects_size_and_overlap():
    text = "。".join(f"第{i}句话内容比较长一些" for i in range(200))
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0][-5:] in text


def test_extract_json_handles_wrapped_text():
    assert extract_json('结果如下：{"label": "Food_Adventure", "score": 0.9} 以上') == {
        "label": "Food_Adventure",
        "score": 0.9,
    }
    assert extract_json("no json") is None
