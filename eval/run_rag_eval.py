"""RAG 检索消融评测：在 eval/data/rag_eval.jsonl 上对比多种检索配置，输出 eval/rag_report.md。

指标均在“笔记级”计算（同一笔记的多个 chunk 只算一次）：
- Recall@5：前 5 篇中高度相关（2 分）笔记数 / 该查询全部高度相关笔记数
- MRR@10：第一篇高度相关笔记排名的倒数（前 10 内）
- nDCG@5：按 0/1/2 分级增益计算
- P@5(≥1)：前 5 篇中至少部分相关的比例
没有任何 2 分笔记的查询不参与 Recall/MRR 统计（但参与 nDCG、P@5）。

python eval/run_rag_eval.py
"""
import argparse
import asyncio
import math
import statistics
import time
from datetime import date
from typing import Dict, List

from common import DATA_DIR, ROOT, get_settings, notes_from_hits, read_jsonl, write_jsonl

from core.llm import LLMClient
from core.logging import setup_logging
from rag.pipeline import RagPipeline
from rag.retriever import Retriever
from rag.store import VectorStore

EVAL_PATH = DATA_DIR / "rag_eval.jsonl"
REWRITE_CACHE = DATA_DIR / "rag_rewrites.jsonl"
REPORT_PATH = ROOT / "eval" / "rag_report.md"

CONFIGS = [
    ("向量", dict(mode="vector", rerank="none", city_filter=False), False),
    ("BM25", dict(mode="bm25", rerank="none", city_filter=False), False),
    ("混合（RRF）", dict(mode="hybrid", rerank="none", city_filter=False), False),
    ("向量 + 地名过滤", dict(mode="vector", rerank="none", city_filter=True), False),
    ("混合 + 地名过滤", dict(mode="hybrid", rerank="none", city_filter=True), False),
    ("混合 + 地名过滤 + bge-reranker", dict(mode="hybrid", rerank="cross_encoder", city_filter=True), False),
    ("混合 + 地名过滤 + bge-reranker + LLM 改写", dict(mode="hybrid", rerank="cross_encoder", city_filter=True), True),
]


def dcg(gains: List[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def score_query(ranked: List[str], grades: Dict[str, int]) -> Dict[str, float]:
    relevant = {n for n, g in grades.items() if g == 2}
    top5 = ranked[:5]
    gains5 = [grades.get(n, 0) for n in top5]
    ideal = sorted(grades.values(), reverse=True)[:5]
    result = {
        "ndcg5": dcg(gains5) / dcg(ideal) if dcg(ideal) > 0 else 0.0,
        "p5": sum(1 for g in gains5 if g >= 1) / 5,
        "judged5": sum(1 for n in top5 if n in grades) / max(1, len(top5)),
    }
    if relevant:
        result["recall5"] = len(relevant & set(top5)) / len(relevant)
        rr = 0.0
        for i, n in enumerate(ranked[:10], start=1):
            if n in relevant:
                rr = 1.0 / i
                break
        result["mrr10"] = rr
    return result


async def load_rewrites(pipeline: RagPipeline, items: List[dict]) -> Dict[str, str]:
    cache = {r["qid"]: r["rewrite"] for r in read_jsonl(REWRITE_CACHE)}
    todo = [it for it in items if it["qid"] not in cache]

    async def one(it):
        degraded: List[str] = []
        intent = await pipeline.intent_understand(it["query"], 10, degraded)
        return it["qid"], intent.get("query_rewrite") or it["query"]

    for qid, rewrite in await asyncio.gather(*[one(it) for it in todo]):
        cache[qid] = rewrite
    write_jsonl(REWRITE_CACHE, [{"qid": k, "rewrite": v} for k, v in cache.items()])
    return cache


def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p * (len(values) - 1))))]


async def main(limit: int = 0) -> None:
    settings = get_settings()
    setup_logging("WARNING")
    items = read_jsonl(EVAL_PATH)
    if limit:
        items = items[:limit]
    retriever = Retriever(VectorStore(settings.rag), settings.retrieval)
    retriever.load()
    retriever.reranker.ensure_ready()
    llm = LLMClient(settings)
    pipeline = RagPipeline(retriever, llm, settings.rag.llm_model)
    rewrites = await load_rewrites(pipeline, items)
    await llm.aclose()

    rows = []
    for name, opts, use_rewrite in CONFIGS:
        per_query, latencies = [], []
        by_type: Dict[str, List[Dict[str, float]]] = {}
        for it in items:
            query = rewrites[it["qid"]] if use_rewrite else it["query"]
            t0 = time.perf_counter()
            hits = retriever.retrieve(query, top_k=40, **opts)["hits"]
            latencies.append((time.perf_counter() - t0) * 1000)
            s = score_query(notes_from_hits(hits, 10), it["grades"])
            per_query.append(s)
            by_type.setdefault(it["type"], []).append(s)

        def avg(metric, scores=per_query):
            vals = [s[metric] for s in scores if metric in s]
            return statistics.mean(vals) if vals else 0.0

        rows.append({
            "name": name,
            "recall5": avg("recall5"), "mrr10": avg("mrr10"), "ndcg5": avg("ndcg5"), "p5": avg("p5"),
            "judged5": avg("judged5"),
            "grounded_ndcg5": avg("ndcg5", by_type.get("grounded", [])),
            "open_ndcg5": avg("ndcg5", by_type.get("open", [])),
            "p50": statistics.median(latencies), "p95": pct(latencies, 0.95),
        })
        print(f"{name}: R@5={rows[-1]['recall5']:.3f} MRR@10={rows[-1]['mrr10']:.3f} nDCG@5={rows[-1]['ndcg5']:.3f} p95={rows[-1]['p95']:.0f}ms")

    write_report(rows, items, settings)


def write_report(rows: List[dict], items: List[dict], settings) -> None:
    n_rel = sum(1 for it in items if any(g == 2 for g in it["grades"].values()))
    n_types = {t: sum(1 for it in items if it["type"] == t) for t in ("grounded", "open")}
    avg_pool = statistics.mean(len(it["grades"]) for it in items)
    judged_by = items[0].get("judged_by", "?") if items else "?"
    verified = sum(1 for it in items if it.get("verified"))
    base = rows[0]

    lines = [
        "# RAG 检索评测报告",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- 语料：`data/travel_notes.csv`，217 篇笔记 / 382 个 chunk",
        f"- 评测集：{len(items)} 条查询（基于笔记生成 {n_types['grounded']} 条，开放式 {n_types['open']} 条），"
        f"其中 {n_rel} 条存在高度相关笔记，参与 Recall/MRR 统计",
        f"- 相关性标注：4 种检索方式各取前 10 篇 + 来源笔记组成候选池（平均 {avg_pool:.1f} 篇/查询），"
        f"由 `{judged_by}` 按 0/1/2 判分；人工核验 {verified} 条（抽检表 `eval/data/rag_review.csv`）",
        f"- Embedding：`{settings.rag.embedding_model}`；Reranker：`{settings.retrieval.reranker_model}`"
        f"（重排前 {settings.retrieval.rerank_k} 条，max_length={settings.retrieval.reranker_max_length}，CPU）",
        "",
        "## 结果（笔记级）",
        "",
        "| 配置 | Recall@5 | MRR@10 | nDCG@5 | P@5(≥1) | nDCG@5 基于笔记 / 开放式 | 延迟 P50 / P95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['recall5']:.3f} | {r['mrr10']:.3f} | {r['ndcg5']:.3f} | {r['p5']:.3f} | "
            f"{r['grounded_ndcg5']:.3f} / {r['open_ndcg5']:.3f} | {r['p50']:.0f} / {r['p95']:.0f} ms |"
        )
    best = max(rows, key=lambda r: r["ndcg5"])
    lines += [
        "",
        "## 结论",
        "",
        f"- nDCG@5 最高的配置：**{best['name']}**（{best['ndcg5']:.3f}，向量基线 {base['ndcg5']:.3f}，"
        f"相对提升 {(best['ndcg5'] / base['ndcg5'] - 1) * 100 if base['ndcg5'] else 0:+.1f}%）。",
        f"- Recall@5：向量基线 {base['recall5']:.3f} → 最佳 {max(r['recall5'] for r in rows):.3f}。",
        "- 延迟为单机 CPU 检索耗时（不含 LLM 回答生成）；带 LLM 改写的配置不含改写本身的耗时（约 1~2 秒）。",
        "",
        "## 方法局限",
        "",
        "- 判分由 LLM 完成，尚未全部人工核验；建议抽检 `rag_review.csv` 后在报告中注明一致率。",
        "- 候选池只来自部分检索方式，池外笔记一律按不相关计；`判分覆盖率` 列见下表，越接近 1 越可靠。",
        f"- {n_types['grounded'] / len(items):.0%} 的查询由某篇笔记生成，词面上可能偏向该笔记，对 BM25 略有利；表中单列了两类查询的 nDCG@5。",
        "- 判分覆盖率低于 1 的配置（如带 LLM 改写的配置）会召回候选池外的笔记，这些笔记按不相关计，分数偏保守。",
        "- 语料只有 217 篇，结论的统计显著性有限；扩充语料后需要重跑。",
        "",
        "| 配置 | 前 5 判分覆盖率 |",
        "|---|---|",
    ]
    lines += [f"| {r['name']} | {r['judged5']:.3f} |" for r in rows]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report -> {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条（调试用）")
    asyncio.run(main(parser.parse_args().limit))
