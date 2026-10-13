"""图谱查询模板。LLM 只选模板、填参数，不直接写 Cypher（防注入、结果可预期）。

所有模板都返回 mentions（被多少篇笔记提到）作为“真实旅行者推荐度”，这是高德搜索给不了的信息。
"""
from typing import Any, Dict, List

from graph.client import GraphClient

# 用名称找 POI：优先完全相等，其次包含；同名时取被提到最多的
_FIND = """
MATCH (p:POI) WHERE p.name = $place OR p.name CONTAINS $place OR $place CONTAINS p.name
OPTIONAL MATCH (p)-[:LOCATED_IN]->(c:City)
WITH p, c, COUNT { (:Note)-[:MENTIONS]->(p) } AS m,
     CASE WHEN p.name = $place THEN 0 ELSE 1 END AS exactness
WHERE $city = '' OR c.name = $city
RETURN p ORDER BY exactness, m DESC LIMIT 1
"""

TEMPLATES: Dict[str, str] = {
    # 某地附近（≤1.5km）被游记提到过的地点，可按标签过滤
    "nearby": """
        CALL () { """ + _FIND + """ }
        MATCH (p)-[n:NEAR]-(q:POI)
        WHERE $tag = '' OR EXISTS { (q)-[:HAS_TAG]->(:Tag {name: $tag}) }
        RETURN p.name AS center, q.name AS name, n.distance_m AS distance_m,
               COUNT { (:Note)-[:MENTIONS]->(q) } AS mentions, q.rating AS rating, q.cost AS cost
        ORDER BY mentions DESC, distance_m LIMIT $limit""",
    # 和某地同一篇游记里一起出现过的地点（“去了 A 的人还去了哪”）
    "co_visit": """
        CALL () { """ + _FIND + """ }
        MATCH (p)-[c:CO_MENTIONED]-(q:POI)
        OPTIONAL MATCH (p)-[n:NEAR]-(q)
        RETURN p.name AS center, q.name AS name, c.count AS together, n.distance_m AS distance_m,
               COUNT { (:Note)-[:MENTIONS]->(q) } AS mentions
        ORDER BY together DESC, mentions DESC LIMIT $limit""",
    # 某城市某类地点，按被提到次数排序
    "by_tag": """
        MATCH (c:City {name: $city})<-[:LOCATED_IN]-(q:POI)-[:HAS_TAG]->(:Tag {name: $tag})
        RETURN q.name AS name, q.district AS district,
               COUNT { (:Note)-[:MENTIONS]->(q) } AS mentions, q.rating AS rating, q.cost AS cost
        ORDER BY mentions DESC LIMIT $limit""",
    # 提到某地的游记（可回到 RAG 看原文）
    "evidence": """
        CALL () { """ + _FIND + """ }
        MATCH (n:Note)-[:MENTIONS]->(p)
        RETURN p.name AS place, n.note_id AS note_id, n.title AS title LIMIT $limit""",
}

TAGS = ["美食", "景点", "文化", "购物", "住宿", "休闲", "咖啡", "火锅", "博物馆", "公园", "街区", "茶馆", "饮品",
        "甜品", "展馆", "寺庙", "夜生活", "自然"]


async def run_template(graph: GraphClient, template: str, place: str = "", city: str = "", tag: str = "",
                       limit: int = 8) -> List[Dict[str, Any]]:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template}")
    return await graph.run(TEMPLATES[template], {"place": place, "city": city, "tag": tag,
                                                 "limit": max(1, min(limit, 15))})
