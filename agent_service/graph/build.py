"""构建知识图谱：笔记元数据 + 实体对齐结果 → Neo4j。

    (:City {name})  (:POI {poi_id, name, lng, lat, district, rating, cost, branch_guess})
    (:Tag {name})   (:Note {note_id, title})

    (POI)-[:LOCATED_IN]->(City)          (POI)-[:HAS_TAG]->(Tag)
    (Note)-[:MENTIONS]->(POI)            (POI)-[:CO_MENTIONED {count}]->(POI)   同一篇笔记里同时出现
    (POI)-[:NEAR {distance_m}]->(POI)    同城、直线距离 ≤ near_radius_m，且两端都不是“猜测的连锁分店”

python -m graph.build [--reset]
"""
import argparse
import asyncio
import itertools
import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from config import get_settings
from core.logging import setup_logging
from graph.align import clean
from graph.client import GraphClient
from rag.ingest import load_csv_records
from rag.metadata import load_metadata, normalize_place

# 名称不完全一致时，这些类别基本是误配（“黑芝麻胡同” → “黑芝麻胡同小学”）
REJECT_TYPES = ("科教文化服务;学校", "医疗保健服务", "公司企业", "商务住宅", "政府机构及社会团体",
                "金融保险服务", "汽车服务", "汽车销售", "汽车维修", "摩托车服务")
COMMERCIAL_TYPES = ("住宿服务", "餐饮服务", "购物服务", "生活服务", "体育休闲服务")
TOP_TAG = {"餐饮服务": "美食", "风景名胜": "景点", "科教文化服务": "文化", "购物服务": "购物",
           "住宿服务": "住宿", "体育休闲服务": "休闲"}
SUB_TAG = {"咖啡厅": "咖啡", "火锅店": "火锅", "博物馆": "博物馆", "公园广场": "公园", "特色商业街": "街区",
           "茶艺馆": "茶馆", "冷饮店": "饮品", "糕饼店": "甜品", "美术馆": "展馆", "展览馆": "展馆",
           "寺庙道观": "寺庙", "酒吧": "夜生活", "夜总会": "夜生活", "海滩": "自然", "国家级景点": "景点"}


def haversine_m(a: Dict, b: Dict) -> float:
    r = 6371000.0
    la1, la2 = math.radians(a["lat"]), math.radians(b["lat"])
    dla, dlo = la2 - la1, math.radians(b["lng"] - a["lng"])
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def tags_of(poi_type: str) -> List[str]:
    tags = set()
    for part in (poi_type or "").split("|"):
        segs = part.split(";")
        if segs and segs[0] in TOP_TAG:
            tags.add(TOP_TAG[segs[0]])
        tags |= {SUB_TAG[s] for s in segs[1:] if s in SUB_TAG}
    return sorted(tags)


def accept(record: Dict) -> Tuple[bool, bool]:
    """返回（是否采用，是否为猜测的分店）。"""
    poi = record["poi"]
    if not poi.get("poi_id") or not isinstance(poi["poi_id"], str):  # 高德偶尔返回没有 id 的条目
        return False, False
    q, pc = clean(record["query"]), clean(poi["name"])
    same_core = q == pc  # 去掉括号后相同
    if not same_core and poi["type"].startswith(REJECT_TYPES):
        return False, False
    # 以下两条来自人工抽检（30 条中 9 条错）：
    # 1) 高德名称明显比查询短：只匹配到了泛称片段（“都匀东黄记地摊火锅” → “地摊火锅”）
    if pc in q and len(pc) < len(q) - 1:
        return False, False
    # 2) 查询被包含在更长的商户名里：多半是借名的店（“洱海” → “大理白族洱海渔庄店”、“蓝月谷” → “…温泉酒店”）
    if q in pc and len(pc) - len(q) >= 3 and poi["type"].startswith(COMMERCIAL_TYPES):
        return False, False
    # 查询没写分店、结果是“XX(某某店)”：多半是连锁品牌，具体哪家店是猜的。
    # 这里必须比较完整名称：clean() 会去掉括号，“紫光园”与“紫光园(隆福寺店)”去括号后相同
    full_equal = "".join(record["query"].split()) == "".join(poi["name"].split())
    branch_guess = ("(" in poi["name"] or "（" in poi["name"]) and "店" in poi["name"] and not full_equal
    return True, branch_guess


def build_graph_data(settings) -> Dict:
    metadata = load_metadata(settings.retrieval.metadata_path)
    titles = {r["row_id"]: r["title"] for r in load_csv_records(settings.rag.data_dir / "travel_notes.csv", "title", "text")}
    align_path = settings.rag.data_dir / "graph" / "poi_alignment.jsonl"
    records = {}
    for line in align_path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            records[(r["query"], r["note_city"])] = r

    stats = Counter()
    pois: Dict[str, Dict] = {}
    mentions: List[Tuple[str, str]] = []
    note_pois: Dict[str, set] = defaultdict(set)
    for note_id, meta in metadata.items():
        cities = [normalize_place(c) for c in meta.get("cities", [])] or \
                 [normalize_place(p) for p in meta.get("provinces", [])]
        for raw in meta.get("pois", []):
            r = records.get((clean(raw), cities[0] if cities else ""))
            if r is None:
                stats["not_aligned_yet"] += 1
                continue
            if r["reason"] != "ok":
                stats[f"skip_{r['reason']}"] += 1
                continue
            ok, branch_guess = accept(r)
            if not ok:
                stats["rejected_type"] += 1
                continue
            p = r["poi"]
            node = pois.setdefault(p["poi_id"], {
                "poi_id": p["poi_id"], "name": p["name"], "city": r["city"], "district": p.get("district", ""),
                "lng": p["location"]["lng"], "lat": p["location"]["lat"], "rating": p.get("rating", ""),
                "cost": p.get("cost", ""), "type": p["type"], "tags": tags_of(p["type"]), "branch_guess": branch_guess})
            node["branch_guess"] = node["branch_guess"] and branch_guess  # 任何一次精确命中即视为确定
            mentions.append((note_id, p["poi_id"]))
            note_pois[note_id].add(p["poi_id"])
            stats["mentions_linked"] += 1

    co = Counter()
    for ids in note_pois.values():
        for a, b in itertools.combinations(sorted(ids), 2):
            co[(a, b)] += 1

    near = []
    by_city: Dict[str, List[Dict]] = defaultdict(list)
    for p in pois.values():
        if not p["branch_guess"]:
            by_city[p["city"]].append(p)
    for city_pois in by_city.values():
        for a, b in itertools.combinations(city_pois, 2):
            d = haversine_m(a, b)
            if d <= settings.graph.near_radius_m:
                near.append({"a": a["poi_id"], "b": b["poi_id"], "d": round(d)})

    notes = [{"note_id": n, "title": titles.get(n, "")} for n in note_pois]
    return {"pois": list(pois.values()), "notes": notes, "mentions": [{"n": n, "p": p} for n, p in set(mentions)],
            "co": [{"a": a, "b": b, "c": c} for (a, b), c in co.items()], "near": near, "stats": dict(stats)}


SCHEMA = [
    "CREATE CONSTRAINT poi_id IF NOT EXISTS FOR (p:POI) REQUIRE p.poi_id IS UNIQUE",
    "CREATE CONSTRAINT note_id IF NOT EXISTS FOR (n:Note) REQUIRE n.note_id IS UNIQUE",
    "CREATE CONSTRAINT city_name IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
    "CREATE INDEX poi_name IF NOT EXISTS FOR (p:POI) ON (p.name)",
]

LOAD = [
    ("pois", """UNWIND $rows AS r
        MERGE (p:POI {poi_id: r.poi_id})
        SET p.name = r.name, p.district = r.district, p.lng = r.lng, p.lat = r.lat, p.rating = r.rating,
            p.cost = r.cost, p.type = r.type, p.branch_guess = r.branch_guess
        MERGE (c:City {name: r.city}) MERGE (p)-[:LOCATED_IN]->(c)
        WITH p, r UNWIND r.tags AS tag MERGE (t:Tag {name: tag}) MERGE (p)-[:HAS_TAG]->(t)"""),
    ("notes", "UNWIND $rows AS r MERGE (n:Note {note_id: r.note_id}) SET n.title = r.title"),
    ("mentions", """UNWIND $rows AS r MATCH (n:Note {note_id: r.n}), (p:POI {poi_id: r.p})
        MERGE (n)-[:MENTIONS]->(p)"""),
    ("co", """UNWIND $rows AS r MATCH (a:POI {poi_id: r.a}), (b:POI {poi_id: r.b})
        MERGE (a)-[e:CO_MENTIONED]->(b) SET e.count = r.c"""),
    ("near", """UNWIND $rows AS r MATCH (a:POI {poi_id: r.a}), (b:POI {poi_id: r.b})
        MERGE (a)-[e:NEAR]->(b) SET e.distance_m = r.d"""),
]


async def load(graph: GraphClient, data: Dict, reset: bool) -> Dict:
    if reset:
        await graph.run("MATCH (n) DETACH DELETE n", write=True)
    for stmt in SCHEMA:
        await graph.run(stmt, write=True)
    for key, cypher in LOAD:
        rows = data[key]
        for i in range(0, len(rows), 500):
            await graph.run(cypher, {"rows": rows[i:i + 500]}, write=True)
    counts = await graph.run("""
        CALL () { MATCH (n) RETURN labels(n)[0] AS k, count(*) AS c
                  UNION ALL MATCH ()-[r]->() RETURN type(r) AS k, count(*) AS c }
        RETURN k, c""")
    return {r["k"]: r["c"] for r in counts}


async def main_async(reset: bool) -> None:
    settings = get_settings()
    data = build_graph_data(settings)
    print("prepared:", {k: len(v) for k, v in data.items() if isinstance(v, list)}, data["stats"])
    graph = GraphClient(settings)
    try:
        counts = await load(graph, data, reset)
    finally:
        await graph.close()
    out = {"prepared": data["stats"], "graph": counts,
           "branch_guess_pois": sum(p["branch_guess"] for p in data["pois"])}
    (settings.rag.data_dir / "graph" / "build_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("loaded:", counts)


if __name__ == "__main__":
    setup_logging("WARNING")
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    asyncio.run(main_async(ap.parse_args().reset))
