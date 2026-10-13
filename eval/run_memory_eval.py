"""记忆评测：
A. 抽取质量：18 个人工编写的脚本用例（含临时需求、敏感信息、改口等陷阱），统计召回、误记与敏感信息泄漏
B. 约束遵守：同一规划请求，带 / 不带长期记忆各跑一次，检查行程是否违反记忆里的约束
C. 摘要保留：长对话摘要后关键信息是否还在

python eval/run_memory_eval.py [--skip-plans]
"""
import argparse
import asyncio
import json
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, List

from common import DATA_DIR, ROOT, get_settings, read_jsonl

from agent.orchestrator import PlanAgent
from agent.schemas import PlanRequest
from agent.tools import ToolExecutor
from core.amap import AmapClient
from core.errors import AppError
from core.llm import LLMClient
from core.logging import setup_logging
from memory.service import MemoryService
from memory.session import SessionContext
from memory.store import MemoryStore
from rag.pipeline import extract_json
from rag.retriever import Retriever
from rag.store import VectorStore

REPORT = ROOT / "eval" / "memory_report.md"
PLAN_CACHE = DATA_DIR / "memory_plan_results.json"
SENSITIVE = re.compile(r"\d{11,}")

# B：约束场景。check=deterministic 时用规则判断，否则 LLM 判分
SCENARIOS = [
    {"sid": "s1", "memory": [("dislikes", "博物馆")], "dest": "西安", "check": "no_stop_contains", "arg": "博物馆",
     "desc": "不喜欢博物馆"},
    {"sid": "s2", "memory": [("diet", "不吃辣")], "dest": "成都", "check": "llm", "desc": "不吃辣"},
    {"sid": "s3", "memory": [("mobility", "膝盖不好不能爬山爬楼梯")], "dest": "重庆", "check": "llm", "desc": "膝盖不好，不能爬山爬楼梯"},
    {"sid": "s4", "memory": [("companion", "带5岁孩子"), ("mobility", "推婴儿车")], "dest": "北京", "check": "llm",
     "desc": "带 5 岁孩子、推婴儿车"},
    {"sid": "s5", "memory": [("budget", "学生党预算低")], "dest": "上海", "check": "llm", "desc": "学生党，预算低"},
    {"sid": "s6", "memory": [("diet", "素食")], "dest": "广州", "check": "llm", "desc": "素食"},
]

JUDGE = """用户有一条长期约束：{desc}。下面是为他生成的一日行程。
判断行程是否违反了这条约束（例如推荐了用户不能吃的菜、需要爬山的景点、明显超出预算的高价餐厅等）。
只返回 JSON：{{"violated": true/false, "reason": "一句话"}}

行程：
{text}"""

LONG_DIALOG = [
    ("user", "我想 11 月 8 号去成都玩三天"), ("assistant", "好的，成都三天，11 月 8 日出发。"),
    ("user", "第一天想去熊猫基地"), ("assistant", "已安排第一天上午熊猫基地。"),
    ("user", "都江堰太远了，不去了"), ("assistant", "好的，去掉都江堰。"),
    ("user", "晚上想吃火锅"), ("assistant", "已加入火锅。"),
    ("user", "住宿还没定，帮我留意春熙路附近"), ("assistant", "好的，会优先考虑春熙路附近。"),
    ("user", "第二天想去宽窄巷子"), ("assistant", "已安排。"),
    ("user", "第三天自由活动"), ("assistant", "好的。"),
]
# 只检查被折叠进摘要的前 8 轮里的事实（最近 6 轮保留原文，不算摘要的功劳）
SUMMARY_FACTS = {"目的地成都": ["成都"], "日期 11 月 8 日": ["11月8", "11 月 8", "11-08", "11.8"],
                 "三天": ["三天", "3天", "3 天"], "熊猫基地": ["熊猫"], "否决都江堰": ["都江堰"], "晚上吃火锅": ["火锅"]}


def matches(mem: Dict, rule: Dict) -> bool:
    return mem["key"] == rule["key"] and any(k in mem["value"] or k in (mem.get("evidence") or "") for k in rule["any"])


async def part_a(llm: LLMClient) -> Dict:
    tmp = Path(tempfile.mkdtemp()) / "eval_memory.sqlite3"
    svc = MemoryService(MemoryStore(tmp), llm)
    cases = read_jsonl(DATA_DIR / "memory_cases.jsonl")
    rows, exp_total, exp_hit, stored_total, stored_ok, forbidden_hits, leaks = [], 0, 0, 0, 0, 0, 0
    for case in cases:
        uid = case["cid"]
        for msg in case["messages"]:
            await svc.learn(uid, msg, "eval")
        active = svc.store.list(uid, ("active",))
        pending = svc.store.list(uid, ("pending",))
        hit = [r for r in case["expect"] if any(matches(m, r) for m in active)]
        ok_stored = [m for m in active if any(matches(m, r) for r in case["expect"])]
        forb = [m for m in active for r in case.get("forbid", []) if matches(m, r)]
        leak = [m for m in active + pending if SENSITIVE.search(m["value"] + (m.get("evidence") or ""))]
        exp_total += len(case["expect"])
        exp_hit += len(hit)
        stored_total += len(active)
        stored_ok += len(ok_stored)
        forbidden_hits += len(forb)
        leaks += len(leak)
        rows.append({"cid": uid, "messages": case["messages"],
                     "active": [f"{m['key']}:{m['value']}" for m in active],
                     "pending": [f"{m['key']}:{m['value']}" for m in pending],
                     "missed": [r["key"] for r in case["expect"] if r not in hit],
                     "extra": [f"{m['key']}:{m['value']}" for m in active if m not in ok_stored]})
    svc.store.close()
    return {"rows": rows, "recall": exp_hit / exp_total, "precision": stored_ok / max(1, stored_total),
            "exp_total": exp_total, "exp_hit": exp_hit, "stored_total": stored_total, "stored_ok": stored_ok,
            "forbidden": forbidden_hits, "leaks": leaks,
            "fp_cases": sum(1 for c, r in zip(cases, rows) if not c["expect"] and r["active"])}


async def part_b(llm: LLMClient, agent: PlanAgent, repeat: int) -> List[Dict]:
    """每个场景带 / 不带记忆各规划一次，重复 repeat 轮（LLM 有随机性，单次结果不足以下结论）。"""
    results = []
    for rep in range(1, repeat + 1):
        for sc in SCENARIOS:
            row = {"sid": sc["sid"], "desc": sc["desc"], "dest": sc["dest"], "rep": rep}
            for mode in ("without", "with"):
                row[mode] = await plan_and_check(llm, agent, sc, mode)
                print(f"[r{rep} {sc['sid']} {mode}] violated={row[mode]['violated']} {row[mode]['reason'][:40]}",
                      flush=True)
            results.append(row)
    return results


async def plan_and_check(llm: LLMClient, agent: PlanAgent, sc: Dict, mode: str) -> Dict:
    memory = [f"{k}：{v}" for k, v in sc["memory"]] if mode == "with" else []
    try:
        plan = await agent.plan(PlanRequest(destination=sc["dest"], days=1, memory=memory))
    except AppError as e:  # 规划失败单独记录，不中断整个评测
        return {"violated": None, "reason": f"规划失败：{e.code.value}", "stops": []}
    stops = [s["name"] for d in plan.itinerary["days"] for s in d["stops"]]
    if sc["check"] == "no_stop_contains":
        violated = any(sc["arg"] in s for s in stops)
        return {"violated": violated, "reason": ("地点名包含" if violated else "地点名均不含") + sc["arg"], "stops": stops}
    raw = await llm.chat([{"role": "user", "content": JUDGE.format(desc=sc["desc"], text=plan.text)}],
                         temperature=0.0, max_tokens=120, purpose="eval.memory_judge")
    obj = extract_json(raw) or {}
    return {"violated": bool(obj.get("violated")), "reason": str(obj.get("reason", ""))[:80], "stops": stops}


async def part_c(llm: LLMClient) -> Dict:
    tmp = Path(tempfile.mkdtemp()) / "eval_session.sqlite3"
    store = MemoryStore(tmp)
    for role, text in LONG_DIALOG:
        store.add_turn("s", "u", role, text)
    ctx = SessionContext(store, llm)
    await ctx.maybe_summarize("s")
    summary = store_summary(ctx.context_text("s"))
    kept = {fact: any(k in summary for k in keys) for fact, keys in SUMMARY_FACTS.items()}
    store.close()
    return {"kept": kept, "summary": summary}


def store_summary(text: str) -> str:
    return text.split("【最近对话】")[0].strip()


async def main(skip_plans: bool, repeat: int) -> None:
    settings = get_settings()
    setup_logging("WARNING", settings.logging.dir, name="eval")
    llm, amap = LLMClient(settings), AmapClient(settings)
    try:
        a = await part_a(llm)
        print(f"[A] recall={a['recall']:.2f} precision={a['precision']:.2f} leaks={a['leaks']}", flush=True)
        c = await part_c(llm)
        print(f"[C] kept={c['kept']}", flush=True)
        if skip_plans and PLAN_CACHE.exists():  # B 要规划 12 次，结果缓存后可单独重跑 A / C
            b = json.loads(PLAN_CACHE.read_text(encoding="utf-8"))
        else:
            retriever = Retriever(VectorStore(settings.rag), settings.retrieval)
            retriever.load()
            agent = PlanAgent(llm, amap, ToolExecutor(amap, retriever, settings.agent.notes_top_k), settings.agent)
            b = await part_b(llm, agent, repeat)
            PLAN_CACHE.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        await llm.aclose()
        await amap.aclose()
    write_report(a, b, c)


def write_report(a: Dict, b: List[Dict], c: Dict) -> None:
    lines = ["# 记忆评测报告", "", f"- 运行日期：{date.today().isoformat()}", "",
             "## A. 抽取质量（18 个人工编写用例，`eval/data/memory_cases.jsonl`）", "",
             "| 指标 | 数值 |", "|---|---|",
             f"| 应记住的信息被记住（召回） | {a['recall']:.0%}（{a['exp_hit']}/{a['exp_total']}） |",
             f"| 记住的信息是对的（精确率） | {a['precision']:.0%}（{a['stored_ok']}/{a['stored_total']}） |",
             f"| 不该记的用例却记了（临时需求、闲聊等） | {a['fp_cases']} 个用例 |",
             f"| 改口后旧值仍生效 | {a['forbidden']} 条 |",
             f"| 敏感信息（手机号/证件号）被存储 | {a['leaks']} 条 |", "",
             "| 用例 | 用户说 | 生效的记忆 | 待确认 | 漏记 | 多记 |", "|---|---|---|---|---|---|"]
    for r in a["rows"]:
        lines.append(f"| {r['cid']} | {' → '.join(r['messages'])} | {'、'.join(r['active']) or '—'} | "
                     f"{'、'.join(r['pending']) or '—'} | {'、'.join(r['missed']) or '—'} | {'、'.join(r['extra']) or '—'} |")
    if b:
        reps = max(r["rep"] for r in b)
        def rate(rows, mode):
            done = [r[mode] for r in rows if r[mode]["violated"] is not None]
            return sum(x["violated"] for x in done), len(done)
        w, wn = rate(b, "with")
        wo, won = rate(b, "without")
        lines += ["", f"## B. 约束遵守（同一请求，带 / 不带长期记忆各规划一次，共 {reps} 轮）", "",
                  f"违反约束：不带记忆 **{wo}/{won}**，带记忆 **{w}/{wn}**。"
                  "s1 用规则判断（地点名是否含“博物馆”），其余由 LLM 判分。", "",
                  "| 场景 | 目的地 | 不带记忆违反 | 带记忆违反 | 带记忆仍违反的原因 |", "|---|---|---|---|---|"]
        for sc in SCENARIOS:
            rows = [r for r in b if r["sid"] == sc["sid"]]
            a1, n1 = rate(rows, "without")
            a2, n2 = rate(rows, "with")
            why = "；".join(r["with"]["reason"] for r in rows if r["with"]["violated"])[:120] or "—"
            lines.append(f"| {sc['desc']} | {sc['dest']} | {a1}/{n1} | {a2}/{n2} | {why} |")
    kept = c["kept"]
    lines += ["", "## C. 摘要保留（14 轮对话，前 8 轮被压缩进摘要，只检查摘要文本）", "",
              f"关键信息保留 {sum(kept.values())}/{len(kept)}：" + "，".join(f"{k}{'✅' if v else '❌'}" for k, v in kept.items()),
              "", "摘要内容：", "", "```", c["summary"], "```", "",
              "## 说明", "", "- 用例与期望由开发者编写，规模小，只说明趋势。",
              "- m12 多记的“不喜欢一天赶多个景点”可视为对原话的正确理解，是期望标签写得偏严；为避免事后调整，未修改标签。",
              "- m10（先说喜欢博物馆、后改口不喜欢）在第一版中两条记忆同时生效：LLM 把动词写进了 value，改口规则按字面比较没匹配上。"
              "修复为比较去掉“喜欢/不喜欢/逛”等词后的对象，并在提示词中要求 value 只写对象。",
              "- B 中除 s1 外由 LLM 判分，可能与人工判断不一致；每个场景只跑一次。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-plans", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    asyncio.run(main(args.skip_plans, args.repeat))
