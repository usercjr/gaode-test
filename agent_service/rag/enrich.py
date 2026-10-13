"""用 LLM 为每篇笔记抽取结构化元数据，增量写入 jsonl（已抽取过的笔记跳过）。

python -m rag.enrich --csv ../data/travel_notes.csv
"""
import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

from config import get_settings
from core.errors import AppError
from core.llm import LLMClient
from core.logging import get_logger, setup_logging
from rag.ingest import load_csv_records
from rag.metadata import load_metadata
from rag.pipeline import extract_json

log = get_logger("rag.enrich")

PROMPT = """你是旅行笔记信息抽取器。阅读笔记，只返回 JSON，不要解释。字段：
- cities: 笔记实际涉及的城市（地级市或知名旅游城市，如"成都""大理"），不要写国家
- provinces: 涉及的省份/自治区（如"云南""新疆"）
- pois: 提到的具体景点、街区、餐厅名（最多 10 个，保持原文写法）
- audience: 适用人群，从 [情侣, 亲子, 老人, 朋友, 独自, 学生] 中选，可多选，没有明确提到就空数组
- season: 提到的季节或月份（字符串，没有则空字符串）
- budget: low / mid / high / ""（没有提到预算就空字符串）

标题：{title}
正文：{text}"""


def _clean_list(value, limit=10):
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


async def enrich_one(llm: LLMClient, record: dict) -> dict:
    raw = await llm.chat(
        [{"role": "user", "content": PROMPT.format(title=record["title"], text=record["text"][:1500])}],
        temperature=0.0,
        max_tokens=400,
        purpose="rag.enrich",
    )
    obj = extract_json(raw) or {}
    return {
        "id": str(record["row_id"]),
        "cities": _clean_list(obj.get("cities")),
        "provinces": _clean_list(obj.get("provinces")),
        "pois": _clean_list(obj.get("pois")),
        "audience": _clean_list(obj.get("audience")),
        "season": str(obj.get("season") or ""),
        "budget": str(obj.get("budget") or ""),
        "extracted_by": llm.default_model,
        "extracted_at": date.today().isoformat(),
    }


async def main_async(csv_path: Path, out_path: Path) -> None:
    settings = get_settings()
    llm = LLMClient(settings)
    existing = load_metadata(out_path)
    records = [r for r in load_csv_records(csv_path, "title", "text") if str(r["row_id"]) not in existing]
    print(f"existing={len(existing)} to_extract={len(records)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    with out_path.open("a", encoding="utf-8") as f:
        # LLMClient 内部有并发闸门，这里直接全部并发提交
        tasks = [asyncio.create_task(enrich_one(llm, r)) for r in records]
        for task in asyncio.as_completed(tasks):
            try:
                item = await task
            except AppError as e:
                failed += 1
                log.warning("enrich failed", extra={"code": e.code.value})
                continue
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
    await llm.aclose()
    print(f"done={done} failed={failed} -> {out_path}")


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Extract note metadata with LLM")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default=str(settings.retrieval.metadata_path))
    args = parser.parse_args()
    setup_logging("WARNING")
    asyncio.run(main_async(Path(args.csv), Path(args.out)))


if __name__ == "__main__":
    main()
