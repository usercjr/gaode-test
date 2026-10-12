"""笔记元数据（城市、景点、人群等）：由 enrich.py 用 LLM 抽取后缓存到 jsonl，入库与检索时读取。"""
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

# 抽取结果里城市名的常见后缀，统一去掉，便于与查询匹配
_SUFFIXES = ("特别行政区", "自治州", "地区", "市", "省", "县")


def normalize_place(name: str) -> str:
    name = (name or "").strip()
    for suffix in _SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix) + 1:
            return name[: -len(suffix)]
    return name


def load_metadata(path: Path) -> Dict[str, dict]:
    """读取 {note_id: meta}。文件不存在时返回空字典（检索退化为不按城市过滤）。"""
    if not path.exists():
        return {}
    result: Dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result[str(item["id"])] = item
    return result


def note_places(meta: dict) -> Set[str]:
    """一篇笔记涉及的地名集合（城市 + 省份），已归一化。"""
    places = {normalize_place(c) for c in meta.get("cities", [])}
    places |= {normalize_place(p) for p in meta.get("provinces", [])}
    return {p for p in places if p}


class PlaceMatcher:
    """从查询中识别地名：基于语料中出现过的城市/省份词典做最长匹配，不调用 LLM。"""

    def __init__(self, places: Iterable[str]):
        self.places: List[str] = sorted({p for p in places if len(p) >= 2}, key=len, reverse=True)

    @classmethod
    def from_metadata(cls, metadata: Dict[str, dict]) -> "PlaceMatcher":
        places: Set[str] = set()
        for meta in metadata.values():
            places |= note_places(meta)
        return cls(places)

    def find(self, query: str) -> List[str]:
        found: List[str] = []
        text = query or ""
        for place in self.places:
            if place in text and not any(place in f for f in found):
                found.append(place)
        return found
