"""动态路由评测：同一份成都行程上下文，40 条人工编写的聊天消息（每类 8 条）分别走
  - 旧方式：每条消息都交给 Agent 在原行程上重新规划（P5 之前的前端行为）
  - 新方式：先路由，再交给对应流程
对比延迟、token、行程被改动的比例与路由准确率。可中断续跑。

python eval/run_router_eval.py [--limit N] [--skip-before]
"""
import argparse
import asyncio
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Dict, List

from common import DATA_DIR, ROOT, get_settings, read_jsonl

from agent.orchestrator import PlanAgent
from agent.router import ChatRouter
from agent.schemas import ChatRequest, Itinerary, PlanRequest
from agent.tools import ToolExecutor
from classifier.service import ClassifierService
from core.amap import AmapClient
from core.errors import AppError
from core.llm import LLMClient
from core.logging import setup_logging
from rag.pipeline import RagPipeline
from rag.retriever import Retriever
from rag.store import VectorStore

MESSAGES = DATA_DIR / "router_messages.jsonl"
CONTEXT = DATA_DIR / "router_context.json"
RESULTS = DATA_DIR / "router_results.jsonl"
REPORT = ROOT / "eval" / "router_report.md"
ROUTES = ["plan", "notes_qa", "graph_qa", "realtime", "chitchat"]


def stops_of(it: Dict[str, Any]) -> List[str]:
    return [s["poi_id"] for d in it["days"] for s in d["stops"]]


async def timed(llm: LLMClient, coro):
    before = dict(llm.usage_total)
    started = time.perf_counter()
    try:
        result, error = await coro, None
    except AppError as e:
        result, error = None, e.code.value
    tokens = (llm.usage_total["prompt_tokens"] - before["prompt_tokens"]
              + llm.usage_total["completion_tokens"] - before["completion_tokens"])
    return result, error, round((time.perf_counter() - started) * 1000), tokens


async def main(limit: int, skip_before: bool) -> None:
    settings = get_settings()
    setup_logging("INFO", settings.logging.dir, name="eval")
    llm, amap = LLMClient(settings), AmapClient(settings)
    retriever = Retriever(VectorStore(settings.rag), settings.retrieval)
    retriever.load()
    retriever.reranker.ensure_ready()
    executor = ToolExecutor(amap, retriever, settings.agent.notes_top_k)
    agent = PlanAgent(llm, amap, executor, settings.agent)
    classifier = ClassifierService(llm, settings.classifier.models_dir, settings.classifier.style_model,
                                   settings.classifier.route_model, route_confidence=0.0)  # 本报告记录的是阈值 0（全部本地）
    classifier.load()
    router = ChatRouter(classifier, llm, RagPipeline(retriever, llm, settings.rag.llm_model), agent, executor, amap)

    try:
        if not CONTEXT.exists():  # 固定上下文：生成一次后复用，保证每次评测的起点相同
            plan = await agent.plan(PlanRequest(destination="成都", days=2, preferences=["美食探店"]))
            CONTEXT.write_text(json.dumps(plan.itinerary, ensure_ascii=False), encoding="utf-8")
        context = Itinerary.model_validate(json.loads(CONTEXT.read_text(encoding="utf-8")))
        base_stops = stops_of(context.model_dump())

        done = {r["message"] for r in read_jsonl(RESULTS)}
        for item in read_jsonl(MESSAGES)[: limit or None]:
            if item["message"] in done:
                continue
            req = ChatRequest(message=item["message"], destination="成都", days=2, previous_itinerary=context)
            row = {**item}
            if not skip_before:
                res, err, ms, tok = await timed(llm, agent.plan(PlanRequest(
                    destination="成都", days=2, message=item["message"], previous_itinerary=context)))
                row["before"] = {"ok": err is None, "error": err, "latency_ms": ms, "tokens": tok,
                                 "changed": bool(res) and stops_of(res.itinerary) != base_stops}
            res, err, ms, tok = await timed(llm, router.handle(req))
            row["after"] = {"ok": err is None, "error": err, "latency_ms": ms, "tokens": tok,
                            "route": res["route"] if res else None, "route_source": res.get("route_source") if res else None,
                            "changed": bool(res and res.get("plan")) and stops_of(res["plan"]["itinerary"]) != base_stops,
                            "answer": (res or {}).get("answer", "")[:400]}
            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            a = row["after"]
            print(f"[{item['expected']:>8}] {item['message']} → {a['route']} {a['latency_ms']}ms {a['tokens']}tok"
                  + (f" | before {row['before']['latency_ms']}ms {row['before']['tokens']}tok" if "before" in row else ""),
                  flush=True)
    finally:
        await llm.aclose()
        await amap.aclose()
    write_report(read_jsonl(RESULTS))


def write_report(rows: List[dict]) -> None:
    def stat(rs, mode, key):
        vals = [r[mode][key] for r in rs if mode in r and r[mode]["ok"]]
        return statistics.mean(vals) if vals else 0

    acc = sum(r["after"]["route"] == r["expected"] for r in rows) / len(rows)
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r["expected"]].append(r)
    confusion = Counter((r["expected"], r["after"]["route"]) for r in rows if r["after"]["route"] != r["expected"])
    has_before = any("before" in r for r in rows)
    non_plan = [r for r in rows if r["expected"] != "plan"]

    lines = ["# 动态路由评测报告", "",
             f"- 运行日期：{date.today().isoformat()}",
             f"- 上下文：固定的成都 2 天行程（`eval/data/router_context.json`）",
             f"- 消息：{len(rows)} 条人工编写的聊天消息（每类 8 条，`eval/data/router_messages.jsonl`），期望路由为人工标注",
             "- 旧方式：每条消息都让 Agent 在原行程上重新规划（P5 之前聊天框的行为）；新方式：先路由再处理", "",
             "## 总体", "", "| 指标 | 旧方式 | 新方式（路由） |", "|---|---|---|"]
    if has_before:
        lines += [
            f"| 平均延迟 | {stat(rows, 'before', 'latency_ms') / 1000:.1f}s | {stat(rows, 'after', 'latency_ms') / 1000:.1f}s |",
            f"| 平均 token / 条 | {stat(rows, 'before', 'tokens'):,.0f} | {stat(rows, 'after', 'tokens'):,.0f} |",
            f"| 非修改类消息（32 条）中行程被改动 | {sum(r['before']['changed'] for r in non_plan if 'before' in r)}/{len(non_plan)} | "
            f"{sum(r['after']['changed'] for r in non_plan)}/{len(non_plan)} |",
            f"| 修改类消息（8 条）中行程被改动 | {sum(r['before']['changed'] for r in by_exp['plan'] if 'before' in r)}/8 | "
            f"{sum(r['after']['changed'] for r in by_exp['plan'])}/8 |",
        ]
    lines += [f"| 路由准确率（人工标注） | — | {acc:.1%}（{sum(r['after']['route'] == r['expected'] for r in rows)}/{len(rows)}） |", "",
              "## 按消息类别", "", "| 类别 | 路由正确 | 旧方式 延迟 / token | 新方式 延迟 / token |", "|---|---|---|---|"]
    for route in ROUTES:
        rs = by_exp[route]
        if not rs:
            continue
        before = f"{stat(rs, 'before', 'latency_ms') / 1000:.1f}s / {stat(rs, 'before', 'tokens'):,.0f}" if has_before else "—"
        lines.append(f"| {route} | {sum(r['after']['route'] == route for r in rs)}/{len(rs)} | {before} | "
                     f"{stat(rs, 'after', 'latency_ms') / 1000:.1f}s / {stat(rs, 'after', 'tokens'):,.0f} |")
    lines += ["", "## 路由错误", ""]
    lines += [f"- {e} → {p}：{n} 条（{'；'.join(r['message'] for r in rows if r['expected'] == e and r['after']['route'] == p)}）"
              for (e, p), n in confusion.most_common()] or ["无。"]
    lines += ["", "## 回答样例（每类 2 条）", ""]
    for route in ROUTES:
        for r in by_exp[route][:2]:
            lines.append(f"- **{r['message']}**（路由 {r['after']['route']}）：{r['after']['answer'][:160].replace(chr(10), ' ')}")
    lines += ["", "## 说明", "",
              "- 40 条消息由开发者编写并标注，规模小、单人标注，只能说明趋势；路由模型本身的大规模评测见 `eval/classifier_report.md`。",
              "- “行程被改动”= 新行程的地点序列与原行程不同；旧方式下即使是“谢谢”也会触发一次完整规划。",
              "- 回答质量未做量化评估，仅列样例。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-before", action="store_true")
    args = ap.parse_args()
    asyncio.run(main(args.limit, args.skip_before))
