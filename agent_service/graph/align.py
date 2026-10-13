"""实体对齐：把笔记元数据里的地点名映射到高德 POI（poi_id 作为图谱主键）。

规则（宁缺毋滥，图谱里的错误节点比缺失节点危害更大）：
- 泛称（“博物馆”“小吃”“夜市”）和过短的名称不对齐
- 依次在笔记涉及的各个城市里搜索；交通设施、地名地址类结果排在真实场所之后
- 名称必须对得上：去掉括号与空格后，查询词包含于 POI 名，或 POI 名主体包含于查询词
- 遇到高德日配额用尽立即停止，已完成的结果缓存在 data/graph/poi_alignment.jsonl，可续跑

python -m graph.align
"""
import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from config import get_settings
from core.amap import AmapClient, region_prefix
from core.errors import AppError
from core.logging import setup_logging
from rag.metadata import load_metadata, normalize_place

GENERIC = {"博物馆", "景区", "古镇", "古城", "老街", "火锅", "小吃", "美食", "公园", "夜市", "商场", "步行街", "酒店",
           "民宿", "餐厅", "咖啡", "咖啡馆", "茶馆", "寺庙", "江边", "海边", "山顶", "市区", "景点", "广场", "大学"}
LOW_PRIORITY_TYPES = ("交通设施服务", "地名地址信息", "通行设施")
_BRACKETS = re.compile(r"[（(].*?[)）]")


class _CallBudgetReached(Exception):
    """达到本次运行的高德调用上限（在协程里不能用 StopIteration，会被转换成 RuntimeError）。"""


def clean(name: str) -> str:
    return re.sub(r"\s+", "", _BRACKETS.sub("", name or "")).strip("·-—")


def names_match(query: str, poi_name: str) -> bool:
    q, p = clean(query), clean(poi_name)
    if not q or not p:
        return False
    return q in p or (len(p) >= 2 and p in q)


def pick(query: str, pois: List[Dict], city_prefix: str) -> Optional[Dict]:
    """在候选里选：名称对得上、在该城市内，真实场所优先于交通站点/地址点。"""
    ok = [p for p in pois if isinstance(p.get("poi_id"), str) and p["poi_id"] and names_match(query, p["name"])
          and (not city_prefix or region_prefix(p["adcode"]).startswith(city_prefix))]
    ok.sort(key=lambda p: (p["type"].startswith(LOW_PRIORITY_TYPES), len(p["name"])))
    return ok[0] if ok else None


async def align_all(out_path: Path, max_calls: int = 1500) -> Dict[str, int]:
    settings = get_settings()
    amap = AmapClient(settings)
    metadata = load_metadata(settings.retrieval.metadata_path)
    done = {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[(r["query"], r["note_city"])] = r
    city_geo: Dict[str, Optional[Dict]] = {}
    stats = {"queries": 0, "aligned": 0, "skipped_generic": 0, "not_found": 0, "cached": len(done)}

    async def prefix_of(city: str) -> str:
        if city not in city_geo:
            g = await amap.geocode(city)
            city_geo[city] = g
        g = city_geo[city]
        return region_prefix(g["adcode"], g.get("level", "")) if g else ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for note_id, meta in metadata.items():
                cities = [normalize_place(c) for c in meta.get("cities", [])] or \
                         [normalize_place(p) for p in meta.get("provinces", [])]
                for raw in meta.get("pois", []):
                    query = clean(raw)
                    key = (query, cities[0] if cities else "")
                    if key in done:
                        continue
                    record = {"query": query, "note_city": key[1], "poi": None, "reason": ""}
                    if len(query) < 2 or query in GENERIC:
                        record["reason"] = "generic"
                        stats["skipped_generic"] += 1
                    else:
                        for city in cities[:3]:
                            if amap.calls >= max_calls:
                                raise _CallBudgetReached
                            stats["queries"] += 1
                            hit = pick(query, await amap.search_poi(query, city, limit=5), await prefix_of(city))
                            if hit:
                                record.update(poi=hit, city=city, reason="ok")
                                stats["aligned"] += 1
                                break
                        if not record["poi"]:
                            record["reason"] = "not_found"
                            stats["not_found"] += 1
                    done[key] = record
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
    except _CallBudgetReached:
        stats["stopped"] = "max_calls"
    except AppError as e:
        if "OVER_LIMIT" in str(e.detail) and "QPS" not in str(e.detail):
            stats["stopped"] = f"amap quota: {e.detail}"  # 日配额用尽：停止，明天续跑
        else:
            raise
    finally:
        stats["amap_calls"] = amap.calls
        await amap.aclose()
    return stats


def main():
    settings = get_settings()
    setup_logging("WARNING")
    out = settings.rag.data_dir / "graph" / "poi_alignment.jsonl"
    print(asyncio.run(align_all(out)))


if __name__ == "__main__":
    main()
