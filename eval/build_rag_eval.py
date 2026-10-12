"""构建 RAG 检索评测集。

步骤（每步结果落盘，重跑时跳过已完成的部分）：
1. 生成查询：约 100 条基于具体笔记的用户问题 + 约 50 条开放式问题
2. 候选池：对每条查询取 4 种检索方式各自前 10 篇笔记 + 来源笔记，取并集
3. LLM 判分：对候选池内每篇笔记打 0/1/2 分
4. 导出人工抽检表 eval/data/rag_review.csv

python eval/build_rag_eval.py
"""
import asyncio
import csv
import random
from collections import Counter
from typing import Dict, List

from common import DATA_DIR, get_settings, load_corpus, notes_from_hits, read_jsonl, write_jsonl

from core.errors import AppError
from core.llm import LLMClient
from core.logging import setup_logging
from rag.metadata import note_places
from rag.pipeline import extract_json
from rag.retriever import Retriever
from rag.store import VectorStore

QUERIES_PATH = DATA_DIR / "rag_queries.jsonl"
POOL_PATH = DATA_DIR / "rag_pool.jsonl"
JUDGED_PATH = DATA_DIR / "rag_eval.jsonl"
REVIEW_PATH = DATA_DIR / "rag_review.csv"
REVIEW_KEY_PATH = DATA_DIR / "rag_review_key.jsonl"

SEED = 20261012
N_GROUNDED = 100
OPEN_FOCUSES = [
    "美食、小吃、餐厅",
    "拍照出片、机位、vlog",
    "带老人、带孩子、情侣等特定人群",
    "预算、省钱、穷游、季节和天气",
    "路线安排、天数、交通、避坑",
]
POOL_SYSTEMS = [
    ("vector", dict(mode="vector", rerank="none", city_filter=False)),
    ("bm25", dict(mode="bm25", rerank="none", city_filter=False)),
    ("hybrid_filter", dict(mode="hybrid", rerank="none", city_filter=True)),
    ("hybrid_filter_ce", dict(mode="hybrid", rerank="cross_encoder", city_filter=True)),
]
POOL_DEPTH = 10

GROUNDED_PROMPT = """下面是几篇旅行笔记。请为每篇笔记写 1 个真实用户可能会提的问题，这篇笔记能很好地回答它。
要求：
- 像用户在旅行 App 里打字提问，口语化，10~30 字
- 不要照抄标题，不要出现"这篇笔记"之类的说法
- 大约七成问题带上目的地（城市或省份），其余不带目的地、只描述需求
- 关注笔记里最有价值的具体信息（吃什么、怎么安排、注意什么、去哪拍照等）
只返回 JSON：{{"queries": [{{"i": 笔记编号, "query": "问题"}}]}}

{notes}"""

OPEN_PROMPT = """你在为旅行问答系统构造测试问题。知识库覆盖的目的地包括：{places}。
请写 10 个不同的用户问题，主题聚焦在：{focus}。
要求：口语化，10~30 字；一半问题带上上面列表中的目的地，一半不带具体目的地；问题之间不要重复。
只返回 JSON：{{"queries": ["问题1", "问题2"]}}"""

JUDGE_PROMPT = """你是检索相关性标注员。判断每篇旅行笔记对用户问题的有用程度：
2 = 高度相关：直接回答问题，或包含问题所需的具体信息（目的地也对得上）
1 = 部分相关：目的地或主题有一项对得上，能提供一些参考，但不能直接回答
0 = 不相关
问题：{query}

候选笔记：
{notes}

只返回 JSON：{{"grades": {{"笔记编号": 分数}}}}，每篇都要给分。"""


def snippet(note: dict, limit: int = 500) -> str:
    return f"标题：{note['title']}\n正文：{note['text'][:limit]}"


async def generate_queries(llm: LLMClient, corpus: Dict[str, dict], retriever: Retriever) -> List[dict]:
    existing = read_jsonl(QUERIES_PATH)
    if existing:
        print(f"[queries] reuse {len(existing)}")
        return existing

    rng = random.Random(SEED)
    note_ids = sorted(corpus)
    sampled = rng.sample(note_ids, N_GROUNDED)
    queries: List[dict] = []

    async def grounded_batch(batch: List[str]) -> List[dict]:
        notes = "\n\n".join(f"[{i}] {snippet(corpus[nid], 400)}" for i, nid in enumerate(batch))
        raw = await llm.chat([{"role": "user", "content": GROUNDED_PROMPT.format(notes=notes)}],
                             temperature=0.7, max_tokens=800, purpose="eval.gen")
        out = []
        for item in (extract_json(raw) or {}).get("queries", []):
            try:
                idx = int(item["i"])
                q = str(item["query"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= idx < len(batch) and q:
                out.append({"query": q, "type": "grounded", "source_note": batch[idx]})
        return out

    places = Counter(p for meta in retriever.metadata.values() for p in note_places(meta))
    place_list = "、".join(p for p, _ in places.most_common(40))

    async def open_batch(focus: str) -> List[dict]:
        raw = await llm.chat([{"role": "user", "content": OPEN_PROMPT.format(places=place_list, focus=focus)}],
                             temperature=0.9, max_tokens=800, purpose="eval.gen")
        return [{"query": str(q).strip(), "type": "open", "source_note": None, "focus": focus}
                for q in (extract_json(raw) or {}).get("queries", []) if str(q).strip()]

    batches = [sampled[i:i + 5] for i in range(0, len(sampled), 5)]
    results = await asyncio.gather(*[grounded_batch(b) for b in batches], *[open_batch(f) for f in OPEN_FOCUSES])
    seen = set()
    for group in results:
        for q in group:
            if q["query"] not in seen:
                seen.add(q["query"])
                queries.append(q)
    for i, q in enumerate(queries):
        q["qid"] = f"q{i:03d}"
    write_jsonl(QUERIES_PATH, queries)
    print(f"[queries] generated {len(queries)} ({Counter(q['type'] for q in queries)})")
    return queries


def build_pool(queries: List[dict], retriever: Retriever) -> List[dict]:
    existing = {p["qid"]: p for p in read_jsonl(POOL_PATH)}
    pools = []
    for n, q in enumerate(queries, start=1):
        if q["qid"] in existing:
            pools.append(existing[q["qid"]])
            continue
        pool: Dict[str, List[str]] = {}
        for name, opts in POOL_SYSTEMS:
            hits = retriever.retrieve(q["query"], top_k=40, **opts)["hits"]
            pool[name] = notes_from_hits(hits, POOL_DEPTH)
        union = sorted({nid for ids in pool.values() for nid in ids} | ({q["source_note"]} if q["source_note"] else set()))
        pools.append({"qid": q["qid"], "systems": pool, "candidates": union})
        if n % 10 == 0:
            print(f"[pool] {n}/{len(queries)}")
            write_jsonl(POOL_PATH, pools)
    write_jsonl(POOL_PATH, pools)
    print(f"[pool] done, avg candidates {sum(len(p['candidates']) for p in pools) / len(pools):.1f}")
    return pools


async def judge(llm: LLMClient, queries: List[dict], pools: List[dict], corpus: Dict[str, dict]) -> List[dict]:
    existing = {j["qid"]: j for j in read_jsonl(JUDGED_PATH)}
    pool_by_qid = {p["qid"]: p for p in pools}
    model = llm.default_model

    async def judge_one(q: dict) -> dict:
        cands = pool_by_qid[q["qid"]]["candidates"]
        notes = "\n\n".join(f"[{i}] {snippet(corpus[nid])}" for i, nid in enumerate(cands))
        raw = await llm.chat([{"role": "user", "content": JUDGE_PROMPT.format(query=q["query"], notes=notes)}],
                             temperature=0.0, max_tokens=600, purpose="eval.judge")
        grades_raw = (extract_json(raw) or {}).get("grades", {})
        grades = {}
        for i, nid in enumerate(cands):
            try:
                g = int(grades_raw.get(str(i), grades_raw.get(i, 0)))
            except (TypeError, ValueError):
                g = 0
            grades[nid] = max(0, min(2, g))
        return {**q, "grades": grades, "judged_by": model, "verified": False}

    todo = [q for q in queries if q["qid"] not in existing]
    print(f"[judge] reuse {len(existing)}, judging {len(todo)}")
    results = await asyncio.gather(*[judge_one(q) for q in todo], return_exceptions=True)
    failed = 0
    for q, r in zip(todo, results):
        if isinstance(r, AppError) or isinstance(r, Exception):
            failed += 1
            continue
        existing[q["qid"]] = r
    judged = [existing[q["qid"]] for q in queries if q["qid"] in existing]
    write_jsonl(JUDGED_PATH, judged)
    print(f"[judge] done {len(judged)}, failed {failed}")
    return judged


REVIEW_PER_GRADE = 20  # 每个 LLM 分数档抽 20 对，共 60 对，人工几分钟可完成


def export_review(judged: List[dict], corpus: Dict[str, dict]) -> None:
    """按 LLM 分数分层抽样（查询, 笔记）对，供人工核验判分一致率。已有人工填写时不覆盖。"""
    if REVIEW_PATH.exists():
        with REVIEW_PATH.open(encoding="utf-8-sig") as f:
            if any(row[-1].strip() for row in list(csv.reader(f))[1:]):
                print(f"[review] {REVIEW_PATH} already has human grades, skip")
                return
    rng = random.Random(SEED)
    pairs_by_grade: Dict[int, List[tuple]] = {0: [], 1: [], 2: []}
    for j in judged:
        for nid, g in j["grades"].items():
            pairs_by_grade[g].append((j, nid, g))
    sample = []
    for g in (2, 1, 0):
        sample += rng.sample(pairs_by_grade[g], min(REVIEW_PER_GRADE, len(pairs_by_grade[g])))
    rng.shuffle(sample)
    # 盲标：表里不出现 LLM 分数，避免人工被暗示；LLM 分数另存，计算一致率时再对照
    with REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "query", "note_title", "note_excerpt", "your_grade(0/1/2)"])
        for i, (j, nid, _g) in enumerate(sample, start=1):
            note = corpus[nid]
            excerpt = note["text"][:150].replace("\n", " ")
            w.writerow([i, j["query"], note["title"], excerpt, ""])
    write_jsonl(REVIEW_KEY_PATH, [{"no": i, "qid": j["qid"], "note_id": nid, "llm_grade": g}
                                  for i, (j, nid, g) in enumerate(sample, start=1)])
    print(f"[review] {len(sample)} pairs -> {REVIEW_PATH}")


async def main() -> None:
    settings = get_settings()
    setup_logging("WARNING")
    corpus = load_corpus()
    retriever = Retriever(VectorStore(settings.rag), settings.retrieval)
    retriever.load()
    llm = LLMClient(settings)
    try:
        queries = await generate_queries(llm, corpus, retriever)
        pools = build_pool(queries, retriever)
        judged = await judge(llm, queries, pools, corpus)
        export_review(judged, corpus)
        dist = Counter(g for j in judged for g in j["grades"].values())
        no_rel = sum(1 for j in judged if not any(g == 2 for g in j["grades"].values()))
        print(f"[summary] queries={len(judged)} grade_dist={dict(dist)} queries_without_grade2={no_rel}")
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
