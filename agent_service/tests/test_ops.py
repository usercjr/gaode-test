"""P8：指标、缓存、/metrics 接口。"""
import time

from core.cache import TTLCache
from core.metrics import Metrics, percentile


def test_percentile_nearest_rank():
    vals = list(range(1, 101))
    assert percentile(vals, 50) in (50, 51)
    assert percentile(vals, 95) in (95, 96)
    assert percentile([], 95) is None


def test_ttl_cache_lru_ttl_and_stats(monkeypatch):
    c = TTLCache(maxsize=2, ttl_s=10)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1          # a 变为最近使用
    c.set("c", 3)                   # 淘汰最久未用的 b
    assert c.get("b") is None
    assert c.stats()["hits"] == 1 and c.stats()["misses"] == 1
    now = time.monotonic()
    monkeypatch.setattr("core.cache.time.monotonic", lambda: now + 11)
    assert c.get("a") is None       # 过期
    c.clear()
    assert c.stats()["size"] == 0


def test_metrics_snapshot_and_prometheus():
    m = Metrics()
    for ms in (10, 20, 30):
        m.request_started()
        m.request_finished("POST", "/retrieve", 200, ms)
    m.request_started()
    m.request_finished("POST", "/chat", 503, 5, error_code="UPSTREAM_LLM_UNAVAILABLE")
    m.llm_call("chat.tool_qa", True, 800, prompt_tokens=100, completion_tokens=20)
    m.llm_call("chat.tool_qa", False, 30)
    m.register_gauge("caches", lambda: {"retrieval": {"hits": 3, "misses": 1}})
    m.register_gauge("broken", lambda: 1 / 0)  # 数据源出错不影响整体

    snap = m.snapshot()
    assert snap["requests_total"] == 4 and snap["errors_5xx_total"] == 1 and snap["in_flight"] == 0
    assert snap["routes"]["POST /retrieve"]["latency_ms"]["max"] == 30
    assert snap["error_codes"] == {"UPSTREAM_LLM_UNAVAILABLE": 1}
    assert snap["llm"]["by_purpose"]["chat.tool_qa"] == {"ok": 1, "failed": 1, "latency_ms": snap["llm"]["by_purpose"]["chat.tool_qa"]["latency_ms"]}
    assert snap["llm"]["tokens"] == {"prompt": 100, "completion": 20}
    assert "error" in snap["broken"]

    text = m.prometheus()
    assert 'wanderai_http_requests_total{method="POST",route="/chat",status="503"} 1' in text
    assert 'wanderai_llm_tokens_total{kind="prompt"} 100' in text
    assert 'wanderai_cache_hits_total{cache="retrieval"} 3' in text


def test_retriever_cache_returns_copies(monkeypatch):
    from config import RetrievalConfig
    from rag.retriever import Retriever

    calls = []

    class FakeRetriever(Retriever):
        ready = True

        def _retrieve_uncached(self, query, *args):
            calls.append(query)
            return {"hits": [{"chunk_id": "c1", "text": "t"}], "places": [], "filtered": False, "timings_ms": {}}

    r = FakeRetriever.__new__(FakeRetriever)
    r.cfg = RetrievalConfig()
    r.cache = TTLCache(16, 60)
    first = r.retrieve("成都美食")
    first["hits"][0]["mutated"] = True   # 调用方修改结果不能污染缓存
    second = r.retrieve("成都美食")
    assert calls == ["成都美食"]
    assert "mutated" not in second["hits"][0]
    assert second["timings_ms"] == {"cache_hit": 1}
    r.retrieve("成都美食", top_k=3)       # 参数不同 → 不命中
    assert len(calls) == 2


def test_open_circuit_is_not_retried(monkeypatch):
    """熔断打开时立即失败：压测发现拒绝被当成可重试错误，白白退避 1~1.5 秒。"""
    import asyncio

    import pytest

    from core.errors import ErrorCode
    from core.resilience import CircuitOpenError, retry_async

    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("core.resilience._sleep", fake_sleep)
    calls = {"n": 0}

    async def rejected():
        calls["n"] += 1
        raise CircuitOpenError(ErrorCode.UPSTREAM_LLM_UNAVAILABLE, "llm")

    with pytest.raises(CircuitOpenError) as ei:
        asyncio.run(retry_async(rejected, attempts=3))
    assert ei.value.code == ErrorCode.UPSTREAM_LLM_UNAVAILABLE and ei.value.retryable  # 对用户仍是“稍后再试”
    assert calls["n"] == 1 and slept == []
