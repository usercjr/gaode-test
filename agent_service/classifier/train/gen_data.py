"""用 LLM 构造意图分类数据（风格多标签 + 路由单标签）。

- 训练集与测试集分开生成：不同提示词（测试集更口语、带错别字/否定/省略），且测试集只用训练集中没出现过的城市。
- 去重：同一集合内 bge 余弦相似度 ≥ 0.92 视为重复；测试集里与任何训练样本 ≥ 0.90 的样本剔除，防止泄漏。
- 标签来自“生成时指定的意图”，人工抽检见 classifier/data/review_*.csv。

python -m classifier.train.gen_data
"""
import asyncio
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from classifier.labels import ROUTE_DESC, ROUTE_LABELS, STYLE_DESC, STYLE_LABELS, format_input
from config import get_settings
from core.llm import LLMClient
from core.logging import setup_logging
from rag.pipeline import extract_json
from rag.store import VectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 20261012

TRAIN_CITIES = ["成都", "重庆", "西安", "北京", "上海", "广州", "厦门", "贵阳", "昆明", "大理", "丽江", "乌鲁木齐",
                "长沙", "深圳", "泉州", "杭州", "南京", "苏州", "青岛", "天津", "武汉", "哈尔滨", "桂林", "三亚",
                "洛阳", "拉萨", "敦煌", "张家界", "景德镇", "潮州", "喀什", "伊犁", "黄山", "西双版纳", "凤凰古城"]
TEST_CITIES = ["郑州", "济南", "合肥", "福州", "南宁", "兰州", "银川", "呼和浩特", "大连", "宁波", "扬州", "绍兴",
               "开封", "大同", "北海"]
CHIPS = ["独自出行", "家庭出行", "情侣出行", "朋友出行", "老人同行", "文化体验", "经典必去", "自然风光", "城市景观",
         "历史古迹", "紧凑", "宽松", "特种兵快游", "深度文化", "美食探店", "自然疗愈", "Citywalk", "Vlog 创作", "拍照出片"]

STYLE_PROMPT = """你在为旅行助手的“风格识别”模型造训练数据。请写 {n} 条用户在旅行 App 里提出的出行需求。

本批所有样本的风格必须恰好是：{labels}
（风格定义：{desc}）
{none_hint}
每条样本是 JSON 对象，字段：
- destination：目的地，从 [{cities}] 中选，约 2 成样本留空
- preferences：从可选标签 [{chips}] 中选 0~3 个，必须与风格一致或无关（如同行人、节奏），不能与风格矛盾
- text：用户原话 5~40 字，可以为空字符串（但 preferences 和 text 不能同时为空）
输入形式要混合：约 4 成只有 text，4 成 preferences + text，2 成只有 preferences。
{style_extra}
只返回 JSON：{{"items": [{{"destination": "", "preferences": [], "text": ""}}]}}"""

STYLE_TRAIN_EXTRA = "表达要多样：直接说、间接暗示（“想拍好看的照片”）、带人群（带爸妈/带娃/情侣）、带预算和天数。"
STYLE_TEST_EXTRA = ("像真实用户随手打字：口语化、简写、偶尔错别字、会用否定或排除（“不想太累”“不吃辣”），"
                    "可以只给很少的信息。不要和常见模板雷同。")

ROUTE_PROMPT = """你在为旅行助手的“意图路由”模型造数据。请写 {n} 条用户在旅行对话中发出的单句消息，
本批全部属于类别：{label}（{desc}）。

所有类别定义（帮助你把握边界，不要写成其他类别）：
{all_desc}

要求：消息 3~40 字；城市从 [{cities}] 中选，部分消息不带城市；{extra}
只返回 JSON：{{"items": ["消息1", "消息2"]}}"""

ROUTE_TRAIN_EXTRA = "覆盖多种问法；也要包含容易混淆的边界样本（如“成都三天怎么玩”是 plan，“成都三天够不够”是 notes_qa）。"
ROUTE_TEST_EXTRA = "像真实用户随手打字：口语、省略主语、错别字、多轮对话里的追问（“那第二天呢”“换个近点的”）。"


def sample_label_sets(rng: random.Random, n_batches: int) -> List[List[str]]:
    sets = []
    for _ in range(n_batches):
        r = rng.random()
        k = 0 if r < 0.10 else 1 if r < 0.60 else 2 if r < 0.95 else 3
        sets.append(sorted(rng.sample(STYLE_LABELS, k)))
    return sets


async def gen_style(llm: LLMClient, rng: random.Random, n_batches: int, cities: List[str], extra: str,
                    label_sets: List[List[str]] = None) -> List[dict]:
    async def one(labels: List[str]) -> List[dict]:
        desc = "；".join(STYLE_DESC[l] for l in labels) if labels else "无明显风格偏好"
        none_hint = "注意：这批是“综合 / 无明显风格”的需求，不要体现任何单一风格（如“经典景点都去看看”“随便逛逛”）。" if not labels else ""
        prompt = STYLE_PROMPT.format(n=10, labels="、".join(labels) or "无（综合）", desc=desc, none_hint=none_hint,
                                     cities="、".join(rng.sample(cities, min(10, len(cities)))),
                                     chips="、".join(CHIPS), style_extra=extra)
        raw = await llm.chat([{"role": "user", "content": prompt}], temperature=0.95, max_tokens=1500, purpose="clf.gen")
        out = []
        for it in (extract_json(raw) or {}).get("items", []):
            if not isinstance(it, dict):
                continue
            text = format_input(str(it.get("destination") or ""), [str(p) for p in it.get("preferences") or []],
                                str(it.get("text") or ""))
            if text:
                out.append({"text": text, "labels": labels})
        return out

    label_sets = label_sets if label_sets is not None else sample_label_sets(rng, n_batches)
    results = await asyncio.gather(*[one(ls) for ls in label_sets], return_exceptions=True)
    return [x for r in results if isinstance(r, list) for x in r]


async def gen_route(llm: LLMClient, rng: random.Random, per_label_batches: int, cities: List[str], extra: str) -> List[dict]:
    all_desc = "\n".join(f"- {k}：{v}" for k, v in ROUTE_DESC.items())

    async def one(label: str) -> List[dict]:
        prompt = ROUTE_PROMPT.format(n=12, label=label, desc=ROUTE_DESC[label], all_desc=all_desc,
                                     cities="、".join(rng.sample(cities, min(10, len(cities)))), extra=extra)
        raw = await llm.chat([{"role": "user", "content": prompt}], temperature=0.95, max_tokens=1200, purpose="clf.gen")
        return [{"text": str(t).strip(), "label": label}
                for t in (extract_json(raw) or {}).get("items", []) if str(t).strip()]

    jobs = [one(l) for l in ROUTE_LABELS for _ in range(per_label_batches)]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    return [x for r in results if isinstance(r, list) for x in r]


def dedupe(items: List[dict], store: VectorStore, threshold: float, against: np.ndarray = None) -> List[dict]:
    """贪心去重：与已保留样本（及 against 中任何样本）相似度 ≥ threshold 的丢弃。"""
    if not items:
        return items
    emb = np.array(store.embed([it["text"] for it in items]))
    kept_idx: List[int] = []
    kept_emb: List[np.ndarray] = []
    for i, v in enumerate(emb):
        if against is not None and len(against) and float(np.max(against @ v)) >= 0.90:
            continue
        if kept_emb and float(np.max(np.stack(kept_emb) @ v)) >= threshold:
            continue
        kept_idx.append(i)
        kept_emb.append(v)
    return [items[i] for i in kept_idx]


def write(name: str, items: List[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / name).open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{name}: {len(items)}")


async def main() -> None:
    settings = get_settings()
    setup_logging("WARNING")
    rng = random.Random(SEED)
    llm = LLMClient(settings)
    store = VectorStore(settings.rag)
    try:
        style_train, style_test, route_train, route_test = await asyncio.gather(
            gen_style(llm, rng, 160, TRAIN_CITIES, STYLE_TRAIN_EXTRA),
            gen_style(llm, rng, 24, TEST_CITIES, STYLE_TEST_EXTRA),
            gen_route(llm, rng, 20, TRAIN_CITIES, ROUTE_TRAIN_EXTRA),
            gen_route(llm, rng, 4, TEST_CITIES, ROUTE_TEST_EXTRA),
        )
    finally:
        await llm.aclose()

    for name, train, test in (("style", style_train, style_test), ("route", route_train, route_test)):
        print(f"[{name}] raw train={len(train)} test={len(test)}")
        train = dedupe(train, store, 0.92)
        train_emb = np.array(store.embed([it["text"] for it in train]))
        test = dedupe(test, store, 0.92, against=train_emb)
        rng.shuffle(train)
        write(f"{name}_train.jsonl", train)
        write(f"{name}_test.jsonl", test)


async def append_none_test(n_batches: int = 4) -> None:
    """测试集首轮生成时没抽到“综合”批次，按同样的测试集流程补生成，追加到 style_test.jsonl 末尾（不改变已有行号）。"""
    settings = get_settings()
    setup_logging("WARNING")
    rng = random.Random(SEED + 1)
    llm = LLMClient(settings)
    store = VectorStore(settings.rag)
    try:
        items = await gen_style(llm, rng, n_batches, TEST_CITIES, STYLE_TEST_EXTRA,
                                label_sets=[[] for _ in range(n_batches)])
    finally:
        await llm.aclose()
    path = DATA_DIR / "style_test.jsonl"
    existing = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    train = [json.loads(l) for l in (DATA_DIR / "style_train.jsonl").open(encoding="utf-8") if l.strip()]
    against = np.array(store.embed([it["text"] for it in train + existing]))
    new = dedupe(items, store, 0.92, against=against)
    with path.open("a", encoding="utf-8") as f:
        for it in new:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"appended {len(new)} none-label items (raw {len(items)})")


if __name__ == "__main__":
    import sys

    asyncio.run(append_none_test() if "none-test" in sys.argv else main())
