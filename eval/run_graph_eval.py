"""知识图谱对“地点关系类”问题的作用：同一问题分别用“带图谱工具”和“不带图谱工具（P5 的工具集）”回答。

指标：
- 回答里出现的“游记中提到过的地点”数量（图谱 POI 名称出现在回答中）
- 延迟、token
- LLM 盲评偏好：A/B 两种顺序各评一次，两次一致才算胜，否则算平（抵消位置偏差）

python eval/run_graph_eval.py
"""
import asyncio
import json
import statistics
import time
from datetime import date
from typing import Dict, List

from common import DATA_DIR, ROOT, get_settings

from agent.qa import answer_with_tools
from agent.router import SPATIAL_SYSTEM
from agent.tools import PlanContext, ToolExecutor
from core.amap import AmapClient
from core.llm import LLMClient
from core.logging import setup_logging
from graph.client import GraphClient
from rag.pipeline import extract_json
from rag.retriever import Retriever
from rag.store import VectorStore

REPORT = ROOT / "eval" / "graph_report.md"
RESULTS = DATA_DIR / "graph_eval_results.json"

QUESTIONS = [
    ("成都", "去了熊猫基地还能顺路去哪", "co_visit"), ("重庆", "逛完洪崖洞附近还有什么值得去", "co_visit"),
    ("广州", "去完沙面还能去哪逛", "co_visit"), ("北京", "看完故宫还能顺便去哪", "co_visit"),
    ("厦门", "去了鼓浪屿的人一般还去哪", "co_visit"), ("西安", "逛完回民街还能去哪", "co_visit"),
    ("成都", "宽窄巷子附近有什么好吃的", "nearby"), ("成都", "春熙路附近有什么值得去的", "nearby"),
    ("重庆", "解放碑附近吃什么", "nearby"), ("广州", "北京路步行街附近有什么好吃的", "nearby"),
    ("西安", "钟楼附近有什么推荐的", "nearby"), ("成都", "成都有哪些推荐的博物馆", "by_tag"),
    ("广州", "广州有什么值得逛的街区", "by_tag"), ("重庆", "重庆有什么好吃的老店", "by_tag"),
    ("厦门", "厦门有哪些推荐的景点", "by_tag"), ("西安", "西安有哪些值得吃的馆子", "by_tag"),
]

# P5 版本的系统提示词（不带图谱）
P5_SPATIAL = """你是旅行助手，回答地点关系类问题（某地附近有什么、两地顺不顺路、去了 A 还能去哪）。
目的地：{dest}。{trip}
必须用工具查证：附近的店用 nearby_search，两地距离和时长用 route_plan（先用 poi_search 拿到 poi_id）。
不要编造距离、评分和人均；回答简洁，列出 3~5 个具体结果（名称、距离、评分或人均），最后给一句建议。"""

JUDGE = """两位旅行助手回答了同一个问题。站在游客角度，哪个回答更有帮助（推荐是否具体可信、是否切题、是否有真实依据）？
只返回 JSON：{{"better": "A" 或 "B" 或 "tie", "reason": "一句话"}}

问题：{q}

回答 A：
{a}

回答 B：
{b}"""


async def answer(llm, executor, amap, city, question, with_graph: bool) -> Dict:
    ctx = PlanContext(destination=city)
    ctx.dest_geo = await amap.geocode(city)
    if with_graph:
        system = SPATIAL_SYSTEM.format(dest=city, trip="")
        tools = ["graph_query", "poi_search", "nearby_search", "route_plan", "search_notes"]
    else:
        system = P5_SPATIAL.format(dest=city, trip="")
        tools = ["poi_search", "nearby_search", "route_plan", "search_notes"]
    before = dict(llm.usage_total)
    started = time.perf_counter()
    res = await answer_with_tools(llm, executor, ctx, system, question, tools)
    return {"answer": res["answer"], "latency_ms": round((time.perf_counter() - started) * 1000),
            "tokens": llm.usage_total["prompt_tokens"] + llm.usage_total["completion_tokens"]
                      - before["prompt_tokens"] - before["completion_tokens"],
            "tools": [c.name for c in ctx.records]}


async def judge(llm, q: str, a: str, b: str) -> str:
    raw = await llm.chat([{"role": "user", "content": JUDGE.format(q=q, a=a[:1500], b=b[:1500])}],
                         temperature=0.0, max_tokens=120, purpose="eval.graph_judge")
    return str((extract_json(raw) or {}).get("better", "tie"))


async def main() -> None:
    settings = get_settings()
    setup_logging("WARNING", settings.logging.dir, name="eval")
    llm, amap, graph = LLMClient(settings), AmapClient(settings), GraphClient(settings)
    retriever = Retriever(VectorStore(settings.rag), settings.retrieval)
    retriever.load()
    executor = ToolExecutor(amap, retriever, settings.agent.notes_top_k, graph)
    names = [r["name"] for r in await graph.run("MATCH (p:POI) WHERE COUNT { (:Note)-[:MENTIONS]->(p) } > 0 "
                                                "RETURN p.name AS name")]
    rows = []
    try:
        for city, q, kind in QUESTIONS:
            with_g = await answer(llm, executor, amap, city, q, True)
            without = await answer(llm, executor, amap, city, q, False)
            for r in (with_g, without):
                r["note_places"] = sorted({n for n in names if len(n) >= 2 and n in r["answer"]})
            # 两种顺序各评一次
            v1 = await judge(llm, q, with_g["answer"], without["answer"])   # A=图谱
            v2 = await judge(llm, q, without["answer"], with_g["answer"])   # A=无图谱
            verdict = "graph" if (v1 == "A" and v2 == "B") else "no_graph" if (v1 == "B" and v2 == "A") else "tie"
            rows.append({"city": city, "q": q, "kind": kind, "with": with_g, "without": without,
                         "judge": [v1, v2], "verdict": verdict})
            print(f"[{kind}] {q}: graph places={len(with_g['note_places'])} vs {len(without['note_places'])} "
                  f"verdict={verdict} {v1}/{v2}", flush=True)
    finally:
        await llm.aclose()
        await amap.aclose()
        await graph.close()
    RESULTS.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(rows)


def write_report(rows: List[Dict]) -> None:
    mean = lambda xs: statistics.mean(xs) if xs else 0
    v = {k: sum(r["verdict"] == k for r in rows) for k in ("graph", "no_graph", "tie")}
    used = sum("graph_query" in r["with"]["tools"] for r in rows)
    lines = ["# 知识图谱效果评测", "", f"- 运行日期：{date.today().isoformat()}",
             f"- {len(rows)} 个人工编写的地点关系类问题（6 个城市；“去了 A 还能去哪”“附近”“某类推荐”三类），"
             "同一问题分别用“带图谱工具”和“P5 工具集（高德周边搜索等）”回答",
             f"- 带图谱时实际调用了 graph_query 的问题：{used}/{len(rows)}", "",
             "## 结果", "", "| 指标 | 不带图谱 | 带图谱 |", "|---|---|---|",
             f"| 回答中“游记里提到过的地点”平均个数 | {mean([len(r['without']['note_places']) for r in rows]):.1f} | "
             f"{mean([len(r['with']['note_places']) for r in rows]):.1f} |",
             f"| 平均延迟 | {mean([r['without']['latency_ms'] for r in rows]) / 1000:.1f}s | "
             f"{mean([r['with']['latency_ms'] for r in rows]) / 1000:.1f}s |",
             f"| 平均 token | {mean([r['without']['tokens'] for r in rows]):,.0f} | {mean([r['with']['tokens'] for r in rows]):,.0f} |",
             "", f"LLM 盲评（两种顺序各评一次，一致才算胜）：带图谱胜 **{v['graph']}**，不带图谱胜 **{v['no_graph']}**，平 **{v['tie']}**。",
             "", "| 类型 | 问题 | 游记地点数（无 / 有） | 盲评 |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['kind']} | {r['q']} | {len(r['without']['note_places'])} / {len(r['with']['note_places'])} | "
                     f"{ {'graph': '带图谱胜', 'no_graph': '不带图谱胜', 'tie': '平'}[r['verdict']]} |")
    lines += ["", "## 样例", ""]
    for r in rows[:3]:
        lines += [f"**{r['q']}**", "", f"- 不带图谱：{r['without']['answer'][:300].replace(chr(10), ' ')}",
                  f"- 带图谱：{r['with']['answer'][:300].replace(chr(10), ' ')}", ""]
    lines += ["## 说明", "",
              "- “游记里提到过的地点”按图谱中被至少一篇游记提到的 POI 名称在回答中的出现计数；带图谱的回答天然更容易命中，"
              "这个指标衡量的是“推荐是否有真实旅行者依据”，不是正确性。",
              "- 盲评由 LLM 完成，可能与人工判断不一致；问题由开发者编写，规模小（16 条）。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    asyncio.run(main())
