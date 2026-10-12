"""评测脚本公共部分：导入 agent_service、读取语料、笔记级去重。"""
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent_service"))

from config import get_settings  # noqa: E402
from rag.ingest import load_csv_records  # noqa: E402

DATA_DIR = ROOT / "eval" / "data"
CORPUS_CSV = ROOT / "data" / "travel_notes.csv"


def load_corpus() -> Dict[str, dict]:
    return {r["row_id"]: r for r in load_csv_records(CORPUS_CSV, "title", "text")}


def notes_from_hits(hits: Iterable[dict], limit: int) -> List[str]:
    """chunk 级结果 → 笔记级排名（同一笔记只保留第一次出现的位置）。"""
    seen: List[str] = []
    for h in hits:
        rid = h.get("row_id")
        if rid and rid not in seen:
            seen.append(rid)
            if len(seen) >= limit:
                break
    return seen


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


__all__ = ["ROOT", "DATA_DIR", "get_settings", "load_corpus", "notes_from_hits", "read_jsonl", "write_jsonl"]
