"""进程内指标：请求量、延迟分位、错误码、LLM 调用与 token、缓存命中、熔断状态。

单进程服务，不引入 Prometheus 客户端库：计数直接累加，延迟保留每个路由最近 N 次用于计算分位数。
GET /metrics 返回 JSON（监控页使用），GET /metrics?format=prometheus 返回 Prometheus 文本格式。
"""
import threading
import time
from collections import Counter, defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional

WINDOW = 2000  # 每个序列保留最近多少次耗时


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


def _summary(latencies: Deque[float]) -> Dict[str, Optional[float]]:
    vals = list(latencies)
    return {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "p99": percentile(vals, 99),
            "max": max(vals) if vals else None}


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.in_flight = 0
        self.requests: Counter = Counter()              # (method, route, status) -> 次数
        self.errors: Counter = Counter()                # 错误码 -> 次数
        self.latency: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.llm_calls: Counter = Counter()             # (purpose, ok) -> 次数
        self.llm_tokens: Counter = Counter()            # prompt / completion
        self.llm_latency: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.tool_calls: Counter = Counter()            # (tool, ok) -> 次数
        # 由各组件注册的“现场读取”数据源：缓存命中、熔断状态等
        self._gauges: Dict[str, Callable[[], dict]] = {}

    # ---------- 记录 ----------
    def request_started(self) -> None:
        with self._lock:
            self.in_flight += 1

    def request_finished(self, method: str, route: str, status: int, latency_ms: float,
                         error_code: Optional[str] = None) -> None:
        with self._lock:
            self.in_flight -= 1
            self.requests[(method, route, status)] += 1
            self.latency[f"{method} {route}"].append(latency_ms)
            if error_code:
                self.errors[error_code] += 1

    def error(self, code: str) -> None:
        with self._lock:
            self.errors[code] += 1

    def llm_call(self, purpose: str, ok: bool, latency_ms: float, prompt_tokens: int = 0,
                 completion_tokens: int = 0) -> None:
        with self._lock:
            self.llm_calls[(purpose, ok)] += 1
            self.llm_latency[purpose].append(latency_ms)
            self.llm_tokens["prompt"] += prompt_tokens
            self.llm_tokens["completion"] += completion_tokens

    def tool_call(self, tool: str, ok: bool) -> None:
        with self._lock:
            self.tool_calls[(tool, ok)] += 1

    def register_gauge(self, name: str, fn: Callable[[], dict]) -> None:
        self._gauges[name] = fn

    # ---------- 读取 ----------
    def snapshot(self) -> dict:
        with self._lock:
            routes: Dict[str, dict] = {}
            for (method, route, status), n in self.requests.items():
                r = routes.setdefault(f"{method} {route}", {"count": 0, "errors_5xx": 0, "errors_4xx": 0})
                r["count"] += n
                if status >= 500:
                    r["errors_5xx"] += n
                elif status >= 400:
                    r["errors_4xx"] += n
            for key, r in routes.items():
                r["latency_ms"] = _summary(self.latency[key])
            total = sum(self.requests.values())
            errors_5xx = sum(n for (_, _, s), n in self.requests.items() if s >= 500)
            llm: Dict[str, dict] = {}
            for (purpose, ok), n in self.llm_calls.items():
                d = llm.setdefault(purpose, {"ok": 0, "failed": 0})
                d["ok" if ok else "failed"] += n
            for purpose, d in llm.items():
                d["latency_ms"] = _summary(self.llm_latency[purpose])
            tools: Dict[str, dict] = {}
            for (tool, ok), n in self.tool_calls.items():
                d = tools.setdefault(tool, {"ok": 0, "failed": 0})
                d["ok" if ok else "failed"] += n
            snap = {
                "uptime_s": round(time.time() - self.started_at),
                "in_flight": self.in_flight,
                "requests_total": total,
                "errors_5xx_total": errors_5xx,
                "routes": dict(sorted(routes.items())),
                "error_codes": dict(self.errors.most_common()),
                "llm": {"by_purpose": dict(sorted(llm.items())), "tokens": dict(self.llm_tokens)},
                "tools": dict(sorted(tools.items())),
            }
        for name, fn in self._gauges.items():
            try:
                snap[name] = fn()
            except Exception as e:  # 指标读取失败不能影响接口本身
                snap[name] = {"error": str(e)[:100]}
        return snap

    def prometheus(self) -> str:
        """Prometheus 文本格式。延迟以 summary 的分位数形式输出（窗口内估计，非累计直方图）。"""
        lines: List[str] = []

        def esc(v: str) -> str:
            return str(v).replace("\\", "\\\\").replace('"', '\\"')

        snap = self.snapshot()
        lines += ["# TYPE wanderai_up gauge", "wanderai_up 1",
                  "# TYPE wanderai_uptime_seconds gauge", f"wanderai_uptime_seconds {snap['uptime_s']}",
                  "# TYPE wanderai_in_flight_requests gauge", f"wanderai_in_flight_requests {snap['in_flight']}",
                  "# TYPE wanderai_http_requests_total counter"]
        with self._lock:
            items = list(self.requests.items())
        for (method, route, status), n in sorted(items):
            lines.append(f'wanderai_http_requests_total{{method="{method}",route="{esc(route)}",status="{status}"}} {n}')
        lines.append("# TYPE wanderai_http_latency_ms summary")
        for key, r in snap["routes"].items():
            method, route = key.split(" ", 1)
            for q in ("p50", "p95", "p99"):
                v = r["latency_ms"][q]
                if v is not None:
                    lines.append(f'wanderai_http_latency_ms{{method="{method}",route="{esc(route)}",'
                                 f'quantile="0.{q[1:]}"}} {v:.1f}')
        lines.append("# TYPE wanderai_errors_total counter")
        for code, n in snap["error_codes"].items():
            lines.append(f'wanderai_errors_total{{code="{code}"}} {n}')
        lines.append("# TYPE wanderai_llm_calls_total counter")
        for purpose, d in snap["llm"]["by_purpose"].items():
            for outcome in ("ok", "failed"):
                lines.append(f'wanderai_llm_calls_total{{purpose="{esc(purpose)}",outcome="{outcome}"}} {d[outcome]}')
        lines.append("# TYPE wanderai_llm_tokens_total counter")
        for kind, n in snap["llm"]["tokens"].items():
            lines.append(f'wanderai_llm_tokens_total{{kind="{kind}"}} {n}')
        lines += ["# TYPE wanderai_cache_hits_total counter", "# TYPE wanderai_cache_misses_total counter"]
        for name, cache in (snap.get("caches") or {}).items():
            if isinstance(cache, dict) and "hits" in cache:
                lines.append(f'wanderai_cache_hits_total{{cache="{name}"}} {cache["hits"]}')
                lines.append(f'wanderai_cache_misses_total{{cache="{name}"}} {cache["misses"]}')
        lines.append("# TYPE wanderai_breaker_state gauge")  # 0 关闭 / 1 半开 / 2 打开
        for b in snap.get("breakers") or []:
            state = {"closed": 0, "half_open": 1, "open": 2}.get(b.get("state"), -1)
            lines.append(f'wanderai_breaker_state{{name="{b.get("name")}"}} {state}')
        return "\n".join(lines) + "\n"


metrics = Metrics()
