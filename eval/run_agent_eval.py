"""Agent 规划评测：同一批出行需求分别走“新流程（Agent + 校验自纠）”和“旧流程（文本 + 正则解析 + 三级地理编码）”。

可中断续跑：结果逐条写入 eval/data/agent_results.jsonl，已完成的用例跳过。
python eval/run_agent_eval.py [--limit N]
"""
import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional

from common import DATA_DIR, ROOT, get_settings, read_jsonl

from agent.orchestrator import PlanAgent
from agent.schemas import PlanRequest
from agent.tools import ToolExecutor
from core.amap import AmapClient, region_prefix
from core.errors import AppError
from core.llm import LLMClient
from core.logging import setup_logging
from legacy_flow import run_legacy
from rag.metadata import note_places
from rag.retriever import Retriever
from rag.store import VectorStore

CASES_PATH = DATA_DIR / "agent_cases.jsonl"
RESULTS_PATH = DATA_DIR / "agent_results.jsonl"
REPORT_PATH = ROOT / "eval" / "agent_report.md"
MAX_LEG = 60


def in_region(adcode: str, dest_prefix: str) -> bool:
    return bool(adcode) and (region_prefix(adcode).startswith(dest_prefix) or adcode.startswith(dest_prefix))


async def leg_minutes(amap: AmapClient, a: Dict[str, float], b: Dict[str, float], city: str) -> Optional[int]:
    """与校验器相同的口径：公交/地铁，无方案时退回步行。"""
    try:
        r = await amap.route(a, b, "transit", city) or await amap.route(a, b, "walking")
        return r["duration_min"] if r else None
    except AppError:
        return None


async def eval_legacy(llm, amap, retriever, case, dest_geo) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        out = await run_legacy(llm, amap, retriever, case)
    except AppError as e:
        return {"ok": False, "error": e.code.value}
    prefix = region_prefix(dest_geo["adcode"], dest_geo.get("level", ""))
    stops = resolved = in_city = legs = long_legs = 0
    via = Counter()
    for day in out["days"]:
        prev = None
        for p in day["places"]:
            stops += 1
            geo = p.get("geo")
            if not geo:
                prev = None
                continue
            resolved += 1
            via[geo["via"]] += 1
            if in_region(geo["adcode"], prefix):
                in_city += 1
            if prev is not None:
                m = await leg_minutes(amap, prev, geo["location"], case["destination"])
                if m is not None:
                    legs += 1
                    long_legs += m > MAX_LEG
            prev = geo["location"]
    return {
        "ok": True, "days_ok": len(out["days"]) == case["days"], "stops": stops, "resolved": resolved,
        "in_city": in_city, "legs": legs, "long_legs": long_legs, "via": dict(via),
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


async def eval_agent(agent: PlanAgent, case) -> Dict[str, Any]:
    req = PlanRequest(destination=case["destination"], days=case["days"],
                      preferences=case.get("preferences", []), message=case.get("message", ""))
    try:
        r = await agent.plan(req)
    except AppError as e:
        return {"ok": False, "error": e.code.value}
    v = r.validation
    return {
        "ok": True, "passed": r.passed, "first_pass": r.first_pass, "fix_rounds": r.fix_rounds,
        "llm_rounds": r.llm_rounds, "days_ok": len(r.itinerary["days"]) == case["days"],
        "stops": v.get("stops", 0), "resolved": v.get("stops_resolved", 0), "in_city": v.get("stops_in_city", 0),
        "legs": v.get("legs_verified", 0), "long_legs": v.get("long_legs", 0),
        "tool_calls": [{"name": c["name"], "ok": c["ok"], "empty": c["empty"]} for c in r.tool_calls],
        "tokens": r.tokens, "latency_ms": r.latency_ms, "issues": r.issues,
    }


async def main(limit: int) -> None:
    settings = get_settings()
    setup_logging("INFO", settings.logging.dir)
    cases = read_jsonl(CASES_PATH)[: limit or None]
    done = {r["cid"]: r for r in read_jsonl(RESULTS_PATH)}
    llm, amap = LLMClient(settings), AmapClient(settings)
    retriever = Retriever(VectorStore(settings.rag), settings.retrieval)
    retriever.load()
    agent = PlanAgent(llm, amap, ToolExecutor(amap, retriever, settings.agent.notes_top_k), settings.agent)
    corpus_places = {p for meta in retriever.metadata.values() for p in note_places(meta)}

    try:
        for case in cases:
            if case["cid"] in done:
                continue
            dest_geo = await amap.geocode(case["destination"])
            calls_before = amap.calls
            new = await eval_agent(agent, case)
            calls_agent = amap.calls - calls_before
            old = await eval_legacy(llm, amap, retriever, case, dest_geo)
            row = {**case, "in_corpus": case["destination"] in corpus_places, "agent": new, "legacy": old,
                   "amap_calls_agent": calls_agent}
            done[case["cid"]] = row
            with RESULTS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            a, l = new, old
            print(f"{case['cid']} {case['destination']}: agent ok={a.get('ok')} pass={a.get('passed')} "
                  f"first={a.get('first_pass')} {a.get('resolved')}/{a.get('stops')} "
                  f"| legacy {l.get('resolved')}/{l.get('stops')} in_city={l.get('in_city')}", flush=True)
    finally:
        await llm.aclose()
        await amap.aclose()

    write_report([done[c["cid"]] for c in cases if c["cid"] in done])


def pct(n, d):
    return f"{n / d:.1%}" if d else "—"


def quantile(values: List[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))] if values else 0.0


def summarize(rows: List[dict], key: str) -> Dict[str, Any]:
    ok = [r[key] for r in rows if r[key].get("ok")]
    s = lambda f: sum(x.get(f, 0) for x in ok)
    return {
        "runs": len(rows), "ok": len(ok), "days_ok": sum(1 for x in ok if x.get("days_ok")),
        "stops": s("stops"), "resolved": s("resolved"), "in_city": s("in_city"),
        "legs": s("legs"), "long_legs": s("long_legs"),
        "lat_p50": statistics.median([x["latency_ms"] for x in ok]) if ok else 0,
        "lat_p95": quantile([x["latency_ms"] for x in ok], 0.95),
    }


def write_report(rows: List[dict]) -> None:
    new, old = summarize(rows, "agent"), summarize(rows, "legacy")
    agent_ok = [r["agent"] for r in rows if r["agent"].get("ok")]
    calls = [c for a in agent_ok for c in a["tool_calls"]]
    by_tool: Dict[str, List[dict]] = {}
    for c in calls:
        by_tool.setdefault(c["name"], []).append(c)
    first_pass = sum(1 for a in agent_ok if a["first_pass"])
    passed = sum(1 for a in agent_ok if a["passed"])
    tokens = [a["tokens"]["prompt"] + a["tokens"]["completion"] for a in agent_ok]
    via = Counter()
    for r in rows:
        via.update(r["legacy"].get("via", {}))
    in_corpus = [r for r in rows if r["in_corpus"]]
    out_corpus = [r for r in rows if not r["in_corpus"]]

    lines = [
        "# Agent 规划评测报告", "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- 用例：{len(rows)} 个出行需求（{len(set(r['destination'] for r in rows))} 个城市，其中 {len(out_corpus)} 个城市不在笔记语料中），见 `eval/data/agent_cases.jsonl`",
        "- 新流程：Agent（Function Calling：笔记检索 / 高德 POI / 路线 / 天气）→ 规则校验 → 问题回填自纠（最多 2 次）",
        "- 旧流程：同一 LLM 按固定格式输出文本 → 正则解析 → 三级地理编码（同城 POI → 全国 POI → 地址编码），逐行复刻前端实现（`eval/legacy_flow.py`）",
        "- 两边的“超长路段”用同一口径计算：相邻两站公交/地铁时长（无方案时步行）> 60 分钟", "",
        "## 新旧流程对比", "",
        "| 指标 | 旧流程 | 新流程（Agent） |", "|---|---|---|",
        f"| 成功产出行程 | {old['ok']}/{old['runs']} | {new['ok']}/{new['runs']} |",
        f"| 天数符合要求 | {pct(old['days_ok'], old['ok'])} | {pct(new['days_ok'], new['ok'])} |",
        f"| 地点匹配到坐标 | {pct(old['resolved'], old['stops'])}（{old['resolved']}/{old['stops']}） | {pct(new['resolved'], new['stops'])}（{new['resolved']}/{new['stops']}） |",
        f"| 坐标位于目的地城市（占全部地点） | {pct(old['in_city'], old['stops'])} | {pct(new['in_city'], new['stops'])} |",
        f"| 超过 60 分钟的路段 | {pct(old['long_legs'], old['legs'])}（{old['long_legs']}/{old['legs']}） | {pct(new['long_legs'], new['legs'])}（{new['long_legs']}/{new['legs']}） |",
        f"| 端到端耗时 P50 / P95 | {old['lat_p50'] / 1000:.0f}s / {old['lat_p95'] / 1000:.0f}s | {new['lat_p50'] / 1000:.0f}s / {new['lat_p95'] / 1000:.0f}s |",
        "",
        f"旧流程坐标来源：同城 POI {via.get('poi_city', 0)}，全国 POI {via.get('poi_national', 0)}，地址编码 {via.get('geocode', 0)}。"
        "“全国 POI”和“地址编码”兜底是旧流程坐标落到外地的主要来源。", "",
        "## Agent 指标", "",
        "| 指标 | 数值 |", "|---|---|",
        f"| 工具调用执行成功率（无报错） | {pct(sum(c['ok'] for c in calls), len(calls))}（{sum(c['ok'] for c in calls)}/{len(calls)}） |",
        f"| 工具调用有效率（成功且返回非空） | {pct(sum(c['ok'] and not c['empty'] for c in calls), len(calls))} |",
        f"| 首次提交即通过校验 | {pct(first_pass, len(agent_ok))}（{first_pass}/{len(agent_ok)}） |",
        f"| 自纠后通过校验 | {pct(passed, len(agent_ok))}（{passed}/{len(agent_ok)}） |",
        f"| 平均修正次数 / 平均 LLM 轮数 | {statistics.mean([a['fix_rounds'] for a in agent_ok]) if agent_ok else 0:.2f} / {statistics.mean([a['llm_rounds'] for a in agent_ok]) if agent_ok else 0:.1f} |",
        f"| 平均 token / 次规划 | {statistics.mean(tokens) if tokens else 0:,.0f} |",
        f"| 平均高德调用 / 次规划（不含缓存命中） | {statistics.mean([r['amap_calls_agent'] for r in rows]) if rows else 0:.0f} |",
        "", "| 工具 | 调用次数 | 执行成功率 | 返回为空 |", "|---|---|---|---|",
    ]
    for name, cs in sorted(by_tool.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {name} | {len(cs)} | {pct(sum(c['ok'] for c in cs), len(cs))} | {pct(sum(c['empty'] for c in cs), len(cs))} |")

    def sub(rs, label):
        n, o = summarize(rs, "agent"), summarize(rs, "legacy")
        return f"| {label}（{len(rs)}） | {pct(o['in_city'], o['stops'])} | {pct(n['in_city'], n['stops'])} |"

    lines += ["", "## 按城市是否在笔记语料中拆分（坐标位于目的地城市）", "",
              "| 分组 | 旧流程 | 新流程 |", "|---|---|---|", sub(in_corpus, "语料内城市"), sub(out_corpus, "语料外城市")]

    failed = [r for r in rows if not r["agent"].get("ok") or not r["agent"].get("passed")]
    lines += ["", "## 未通过 / 失败的用例", ""]
    if not failed:
        lines.append("无。")
    for r in failed:
        a = r["agent"]
        lines.append(f"- {r['cid']} {r['destination']}：" + (a.get("error") or "；".join(a.get("issues", [])[:3])))
    lines += ["", "## 说明与局限", "",
              f"- 用例 {len(rows)} 个、每个只跑一次，LLM 有随机性，数字的波动范围未评估。",
              "- “地点匹配到坐标”对旧流程是宽松口径：兜底编码几乎总能返回一个坐标，但不保证是对的地点，所以更应看“坐标位于目的地城市”。",
              "- 新流程的地点一致性由校验器保证（坐标只能来自 POI 检索结果），该指标衡量的是校验规则是否生效，而不是模型本身的能力。",
              "- 未评估行程内容质量（是否好玩、文案是否准确），只评估可执行性。"]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    asyncio.run(main(parser.parse_args().limit))
