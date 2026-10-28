"""压测：网关 + agent_service 的吞吐、延迟、限流、过载保护与熔断。

LLM 换成本地模拟服务（eval/mock_llm.py，固定延迟），测的是本系统的并发与保护能力，
不包含 DeepSeek 的速度与费用。脚本自己启动 模拟 LLM / agent_service / 网关（独立端口与日志目录），
按场景重启网关以切换限流配置，结束后全部关闭。

python eval/load_test.py            # 约 3~4 分钟，输出 eval/load_test.md
"""
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
MOCK_PORT, AGENT_PORT, GW_PORT = 8099, 8012, 3010
MOCK = f"http://127.0.0.1:{MOCK_PORT}"
AGENT = f"http://127.0.0.1:{AGENT_PORT}"
GW = f"http://127.0.0.1:{GW_PORT}"
REPORT = ROOT / "eval" / "load_test.md"
RESULTS = ROOT / "eval" / "data" / "load_test_results.json"
LOG_DIR = ROOT / "logs" / "loadtest"

CITIES = ["成都", "重庆", "西安", "广州", "厦门", "北京", "上海", "杭州", "南京", "长沙", "昆明", "大理",
          "青岛", "苏州", "武汉", "贵阳", "桂林", "拉萨", "哈尔滨", "三亚"]
TOPICS = ["美食推荐", "两天怎么玩", "亲子景点", "夜景去哪看", "小众景点", "住哪里方便", "博物馆", "拍照打卡"]
CHAT_MESSAGES = ["你好", "谢谢你", "你好呀", "好的谢谢", "哈哈好的", "你是谁"]


# ---------------- 进程管理 ----------------
def start(cmd: List[str], cwd: Path, env: Dict[str, str], name: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = open(LOG_DIR / f"{name}.out.log", "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **env}, stdout=out, stderr=subprocess.STDOUT)


def stop(p: Optional[subprocess.Popen]) -> None:
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


async def wait_ready(url: str, timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline:
            try:
                if (await c.get(url, timeout=2)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise RuntimeError(f"{url} not ready after {timeout_s}s")


def start_gateway(**limits) -> subprocess.Popen:
    env = {"PORT": str(GW_PORT), "AGENT_BASE": AGENT,
           "RATE_LIMIT_API_PER_MIN": str(limits.get("api", 0)),
           "RATE_LIMIT_HEAVY_PER_MIN": str(limits.get("heavy", 0)),
           "HEAVY_MAX_IN_FLIGHT": str(limits.get("in_flight", 0))}
    return start(["node", "server.js"], ROOT, env, "gateway")


# ---------------- 压测原语 ----------------
async def one(client: httpx.AsyncClient, method: str, path: str, body: Optional[dict]) -> Dict:
    t0 = time.perf_counter()
    try:
        r = await client.request(method, GW + path, json=body, timeout=120)
        code = ""
        if r.status_code >= 400:
            try:
                code = r.json()["error"]["code"]
            except Exception:
                code = "?"
        route = ""
        if r.status_code == 200 and path == "/api/chat":
            route = r.json().get("route", "")
        return {"status": r.status_code, "code": code, "ms": (time.perf_counter() - t0) * 1000, "route": route,
                "retry_after": r.headers.get("retry-after")}
    except httpx.HTTPError as e:
        return {"status": 0, "code": type(e).__name__, "ms": (time.perf_counter() - t0) * 1000, "route": ""}


async def run_load(path: str, bodies: List[dict], concurrency: int) -> Dict:
    """固定并发：concurrency 个 worker 依次领取请求，直到 bodies 用完。"""
    queue = list(bodies)
    results: List[Dict] = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        async def worker():
            while queue:
                results.append(await one(client, "POST", path, queue.pop()))
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    return summarize(results, wall, concurrency)


def pct(values: List[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))] if s else 0.0


def summarize(results: List[Dict], wall: float, concurrency: int) -> Dict:
    ok = [r for r in results if r["status"] == 200]
    lat = [r["ms"] for r in ok]
    return {"concurrency": concurrency, "requests": len(results), "ok": len(ok),
            "throughput_rps": round(len(ok) / wall, 2) if wall else 0, "wall_s": round(wall, 2),
            "p50_ms": round(pct(lat, 50)), "p95_ms": round(pct(lat, 95)), "max_ms": round(max(lat)) if lat else 0,
            "status": dict(Counter(f"{r['status']} {r['code']}".strip() for r in results)),
            "routes": dict(Counter(r["route"] for r in ok if r["route"]))}


async def mock(**mode) -> Dict:
    async with httpx.AsyncClient() as c:
        return (await c.post(MOCK + "/mode", json=mode)).json()


async def agent_metrics() -> Dict:
    async with httpx.AsyncClient() as c:
        return (await c.get(AGENT + "/metrics")).json()


# ---------------- 场景 ----------------
async def scenario_retrieval() -> Dict:
    """检索：纯本地 CPU 计算（embedding + BM25 + RRF，不重排）。先全是新查询，再重复查询看缓存。"""
    unique = [f"{c}{t}" for t in TOPICS for c in CITIES]  # 160 条互不相同
    out = {"cold": [], "warm": []}
    i = 0
    for conc in (1, 4, 16):
        n = 40
        bodies = [{"query": q, "top_k": 5, "use_rerank": False} for q in unique[i:i + n]]
        i += n
        out["cold"].append(await run_load("/api/rag/retrieve", bodies, conc))
    hot = unique[:20]  # 已经查过 → 缓存命中
    for conc in (1, 4, 16):
        bodies = [{"query": hot[k % 20], "top_k": 5, "use_rerank": False} for k in range(80)]
        out["warm"].append(await run_load("/api/rag/retrieve", bodies, conc))
    out["cache"] = (await agent_metrics())["caches"]["retrieval"]
    return out


async def scenario_chat_capacity() -> Dict:
    """聊天（闲聊路由）：模拟 LLM 每次 1 秒。网关不限流，看 LLM 并发闸门（8）如何决定吞吐。"""
    await mock(delay=1.0, fail=False)
    rows = []
    for conc in (1, 8, 16, 32):
        await mock(reset_stats=True)
        n = max(12, conc * 3)
        bodies = [{"message": CHAT_MESSAGES[k % len(CHAT_MESSAGES)]} for k in range(n)]
        row = await run_load("/api/chat", bodies, conc)
        st = await mock()
        row["llm_calls"] = st["calls"]
        row["llm_max_in_flight"] = st["max_in_flight"]
        rows.append(row)
    return {"rows": rows, "llm_concurrency_limit": (await agent_metrics())["llm_concurrency"]["limit"]}


async def scenario_overload() -> Dict:
    """过载保护：同时打进 40 个聊天请求，网关只放行 8 个，其余立即返回 SERVER_BUSY。"""
    await mock(delay=2.0, fail=False)
    bodies = [{"message": "你好"} for _ in range(40)]
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=50)) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one(client, "POST", "/api/chat", b) for b in bodies])
        wall = time.perf_counter() - t0
    rejected = [r["ms"] for r in results if r["code"] == "SERVER_BUSY"]
    return {"summary": summarize(results, wall, 40),
            "rejected_p95_ms": round(pct(rejected, 95)) if rejected else None}


async def scenario_rate_limit() -> Dict:
    """限流：同一 IP 连续 15 个重请求，每分钟配额 10。"""
    await mock(delay=0.05, fail=False)
    rows = []
    async with httpx.AsyncClient() as client:
        for k in range(15):
            r = await one(client, "POST", "/api/chat", {"message": "你好"})
            rows.append({"i": k + 1, "status": r["status"], "code": r["code"], "retry_after": r["retry_after"]})
    return {"rows": rows, "counts": dict(Counter(f"{r['status']} {r['code']}".strip() for r in rows))}


async def scenario_breaker() -> Dict:
    """LLM 故障：模拟 LLM 全部返回 500 → 重试 → 熔断后快速失败、不再打上游 → 恢复后半开探测成功。"""
    await mock(delay=0.05, fail=True, reset_stats=True)
    timeline = []
    async with httpx.AsyncClient() as client:
        for k in range(8):
            before = (await mock())["calls"]
            r = await one(client, "POST", "/api/chat", {"message": "你好"})
            after = (await mock())["calls"]
            m = await agent_metrics()
            llm_state = next(b["state"] for b in m["breakers"] if b["name"] == "llm")
            timeline.append({"i": k + 1, "status": r["status"], "code": r["code"], "ms": round(r["ms"]),
                             "upstream_calls": after - before, "breaker": llm_state})
        await mock(fail=False)
        await asyncio.sleep(6)  # 冷却时间配置为 5 秒
        for k in range(3):
            before = (await mock())["calls"]
            r = await one(client, "POST", "/api/chat", {"message": "你好"})
            after = (await mock())["calls"]
            m = await agent_metrics()
            llm_state = next(b["state"] for b in m["breakers"] if b["name"] == "llm")
            timeline.append({"i": f"恢复后 {k + 1}", "status": r["status"], "code": r["code"], "ms": round(r["ms"]),
                             "upstream_calls": after - before, "breaker": llm_state})
    return {"timeline": timeline}


# ---------------- 主流程 ----------------
async def main() -> None:
    mock_p = agent_p = gw_p = None
    results: Dict = {}
    try:
        mock_p = start([PY, str(ROOT / "eval" / "mock_llm.py"), "--port", str(MOCK_PORT)], ROOT, {}, "mock_llm")
        agent_p = start([PY, "app.py"], ROOT / "agent_service", {
            "SERVER__PORT": str(AGENT_PORT), "LLM__BASE_URL": MOCK, "LLM_API_KEY": "mock-key",
            "LLM__BREAKER_RECOVERY_S": "5", "LOGGING__DIR": str(LOG_DIR), "LOGGING__LEVEL": "WARNING"}, "agent")
        await wait_ready(MOCK + "/stats")
        await wait_ready(AGENT + "/healthz")

        gw_p = start_gateway()  # 容量测试：网关不限流
        await wait_ready(GW + "/api/readyz")
        print("retrieval ...", flush=True)
        results["retrieval"] = await scenario_retrieval()
        print("chat capacity ...", flush=True)
        results["chat"] = await scenario_chat_capacity()
        stop(gw_p)

        gw_p = start_gateway(in_flight=8)
        await wait_ready(GW + "/api/readyz")
        print("overload ...", flush=True)
        results["overload"] = await scenario_overload()
        stop(gw_p)

        gw_p = start_gateway(heavy=10)
        await wait_ready(GW + "/api/readyz")
        print("rate limit ...", flush=True)
        results["rate_limit"] = await scenario_rate_limit()
        stop(gw_p)

        gw_p = start_gateway()
        await wait_ready(GW + "/api/readyz")
        print("breaker ...", flush=True)
        results["breaker"] = await scenario_breaker()
    finally:
        for p in (gw_p, agent_p, mock_p):
            stop(p)
    results["env"] = {"date": date.today().isoformat(), "os": platform.platform(), "cpu": platform.processor(),
                      "cpu_count": os.cpu_count(), "python": platform.python_version(),
                      "node": subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()}
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(results)


def write_report(res: Dict) -> None:
    env = res["env"]
    L = ["# 压测报告", "",
         f"- 运行日期：{env['date']}；单机：{env['os']}，{env['cpu_count']} 逻辑核；Python {env['python']}，Node {env['node']}",
         "- 压测客户端与服务在同一台机器上；LLM 为本地模拟服务（固定延迟），**不包含 DeepSeek 的真实延迟和限流**",
         "- 链路：压测客户端 → Node 网关 → agent_service（单进程 uvicorn）→ 模拟 LLM",
         "- 复现：`python eval/load_test.py`（原始数据 `eval/data/load_test_results.json`）", ""]

    def table(rows, extra=()):
        head = "| 并发 | 请求 | 成功 | 吞吐 (次/秒) | P50 | P95 | 最大 |" + "".join(f" {h} |" for h, _ in extra)
        sep = "|---" * (7 + len(extra)) + "|"
        out = [head, sep]
        for r in rows:
            out.append(f"| {r['concurrency']} | {r['requests']} | {r['ok']} | {r['throughput_rps']} | {r['p50_ms']} ms | "
                       f"{r['p95_ms']} ms | {r['max_ms']} ms |" + "".join(f" {fn(r)} |" for _, fn in extra))
        return out

    rt = res["retrieval"]
    L += ["## 1. 检索接口（本地 CPU：embedding + BM25 + RRF，不重排）", "",
          "**首次查询（全部是没查过的问题）**", ""] + table(rt["cold"]) + [
          "", "**重复查询（命中检索缓存）**", ""] + table(rt["warm"]) + [
          "", f"检索缓存累计：命中 {rt['cache']['hits']}、未命中 {rt['cache']['misses']}。", ""]

    ch = res["chat"]
    L += ["## 2. 聊天接口（模拟 LLM 每次 1 秒）", "",
          f"agent_service 的 LLM 并发上限为 {ch['llm_concurrency_limit']}；网关不限流。", ""] + table(
        ch["rows"], extra=[("LLM 调用数", lambda r: r["llm_calls"]),
                           ("上游最大并发", lambda r: r["llm_max_in_flight"]),
                           ("路由", lambda r: ", ".join(f"{k} {v}" for k, v in r["routes"].items()))]) + [""]

    ov = res["overload"]
    s = ov["summary"]
    L += ["## 3. 过载保护（同时 40 个聊天请求，模拟 LLM 2 秒，网关重请求并发上限 8）", "",
          f"- 结果分布：{s['status']}",
          f"- 被拒绝的请求 P95 返回时间：{ov['rejected_p95_ms']} ms（立即告知“当前使用人数较多”，而不是排队到超时）",
          f"- 放行请求 P50 {s['p50_ms']} ms、P95 {s['p95_ms']} ms", ""]

    rl = res["rate_limit"]
    L += ["## 4. 限流（同一 IP 连续 15 个重请求，配额 10 次/分钟）", "",
          f"- 结果分布：{rl['counts']}",
          "- 被限流的响应带 `Retry-After`：" + ", ".join(str(r["retry_after"]) + "s" for r in rl["rows"] if r["retry_after"]), ""]

    br = res["breaker"]
    L += ["## 5. LLM 故障与熔断（模拟 LLM 全部返回 500，之后恢复）", "",
          "| 请求 | 状态 | 错误码 | 耗时 | 本次打到上游的次数 | 熔断器状态（请求后） |", "|---|---|---|---|---|---|"]
    for t in br["timeline"]:
        L.append(f"| {t['i']} | {t['status']} | {t['code'] or '—'} | {t['ms']} ms | {t['upstream_calls']} | {t['breaker']} |")
    L += ["", "## 说明", "",
          "- 吞吐和延迟受本机 CPU 影响，换机器数值会变；这里关注的是**随并发变化的趋势**和**保护机制是否按设计生效**。",
          "- 聊天吞吐的上限由 LLM 并发闸门决定（约 并发上限 ÷ 单次 LLM 耗时），真实环境下还受 DeepSeek 自身限流影响。",
          "- 检索是 CPU 密集计算，在线程池中执行；单进程下并发提高后吞吐有限，扩容需要多进程 / 多实例。"]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    asyncio.run(main())
