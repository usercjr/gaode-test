"""旧流程的 Python 复刻，仅用于评测对比：
LLM 按固定格式输出文本 → 正则解析出景点 → 三级地理编码（同城 POI → 全国 POI → 地址编码）。

提示词、解析正则、候选名生成、POI 打分都逐行对应 ai-chat.html / output-map.html 中的实现。
"""
import re
from typing import Any, Dict, List, Optional

from core.amap import AmapClient
from core.errors import AppError
from core.llm import LLMClient
from rag.retriever import Retriever

INSTRUCTION = """你是一个资深旅行博主兼行程规划师。你需要像小红书热门攻略一样，用生动、有温度的语言输出行程。

【输出格式要求】
严格按如下格式输出，不要额外解释或添加 markdown 代码块：

第1天 标题（标题要有画面感，如”老成都的烟火气与味蕾暴击🔥”）
- [城市] 景点名（实用备注）
- [城市] 景点名（实用备注）

第2天 标题
- [城市] 景点名（实用备注）
...

格式规则：
- 标题以”第N天 …”开头，标题要吸引人，可带1个emoji
- 每个景点一行，以”- “开头
- 城市写在中括号内，如[成都]
- 备注写在中文括号内

【内容风格要求】
1. 像朋友在分享旅行经历一样写——不要百科式介绍，要有体验感和温度
2. 每个景点备注要包含实用信息：最佳到达时间、人均花费、停留建议、拍照tips或避坑提醒，至少写2-3条
3. 推荐理由要具体生动：不说”风景优美”，要说”日落时分站在九眼桥上能看到整条锦江的金色光带✨”
4. 适当使用emoji增加可读性（每个景点1-2个），但不要堆砌
5. 行程安排要合理：考虑景点之间的距离和交通，标注交通方式和大约耗时
6. 不要输出除行程文本以外的任何内容"""

DAY_PATTERNS = [
    re.compile(r"^Day\s*(\d+)\s*[:：]?\s*(.*)$", re.I),
    re.compile(r"^D\s*(\d+)\s*(.*)$", re.I),
    re.compile(r"^第\s*(\d+)\s*天\s*[:：]?\s*(.*)$"),
]
PLACE_LINE = re.compile(r"^[-•·]\s*(.+)$")
BRACKET_CITY = re.compile(r"^\[(?P<city>[^\]\[]+)\]\s*(?P<name>[^()（）]+?)(?:[（(](?P<notes>.*?)[)）])?$")
SEP_CITY = re.compile(r"^(?P<city>[^·:]+?)\s*[·:]\s*(?P<name>[^()（）]+?)(?:[（(](?P<notes>.*?)[)）])?$")


def parse_itinerary_text(text: str) -> List[Dict[str, Any]]:
    days: List[Dict[str, Any]] = []
    current = None
    for line in (l.strip() for l in (text or "").splitlines()):
        if not line:
            continue
        matched = False
        for pat in DAY_PATTERNS:
            m = pat.match(line)
            if m:
                current = {"day": int(m.group(1)), "city": None, "places": []}
                days.append(current)
                matched = True
                break
        if matched:
            continue
        pm = PLACE_LINE.match(line)
        if pm and current is not None:
            body = pm.group(1).strip()
            m = BRACKET_CITY.match(body) or SEP_CITY.match(body)
            if m:
                city, name = m.group("city").strip(), m.group("name").strip()
            else:
                city = None
                note = re.search(r"[（(](.*?)[)）]\s*$", body)
                name = re.sub(r"[（(].*[)）]\s*$", "", body).strip() if note else body
            current["places"].append({"name": name, "city": city})
            if not current["city"] and city:
                current["city"] = city
    return days


def _dedupe(items):
    out = []
    for v in items:
        v = (v or "").strip()
        if v and v not in out:
            out.append(v)
    return out


def clean_place_name(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^[\d一二三四五六七八九十]+[.\-、:：）)]\s*", "", s)
    s = re.sub(r"[（(].*?[)）]\s*$", "", s)
    s = re.sub(r"^[\"“”'‘’`]+|[\"“”'‘’`]+$", "", s)
    return s.strip()


def name_candidates(raw: str) -> List[str]:
    base = clean_place_name(raw)
    no_space = re.sub(r"\s+", "", base)
    no_dot = re.sub(r"[·•]", "", no_space)
    light = re.sub(r"(旅游景区|风景名胜区|风景区|景区|景点|打卡点|网红打卡地|网红景点)$", "", no_dot).strip()
    return _dedupe([base, no_space, no_dot, light])


def city_candidates(hint: str) -> List[str]:
    c = (hint or "").strip()
    if not c:
        return [""]
    no_suffix = re.sub(r"市$", "", c)
    # 与 JS 版 dedupeList 一致：空字符串会被去掉，所以有城市提示时不会做“不限城市”的地址编码
    return _dedupe([c, no_suffix, no_suffix + "市" if no_suffix else ""])


def score_poi(poi: Dict[str, Any], keyword: str, city_hint: str) -> int:
    name, kw = (poi.get("name") or "").strip(), keyword.strip()
    poi_city = re.sub(r"市$", "", poi.get("city") or "")
    hint = re.sub(r"市$", "", city_hint or "")
    score = 0
    if name == kw or name.replace(" ", "") == kw.replace(" ", ""):
        score += 10
    if kw in name or name in kw:
        score += 5
    if hint and poi_city and hint in poi_city:
        score += 6
    if poi.get("district"):
        score += 1
    return score


async def geocode_place(amap: AmapClient, name: str, city_hint: str) -> Optional[Dict[str, Any]]:
    """三级匹配：同城 POI → 全国 POI → 地址地理编码。返回 {location, adcode, level} 或 None。"""
    names, cities = name_candidates(name), city_candidates(city_hint)

    async def best_poi(kw, city):
        try:
            pois = await amap.search_poi(kw, city, limit=10)
        except AppError:
            return None
        return max(pois, key=lambda p: score_poi(p, kw, city_hint)) if pois else None

    for city in [c for c in cities if c]:
        for kw in names:
            p = await best_poi(kw, city)
            if p:
                return {"location": p["location"], "adcode": p["adcode"], "via": "poi_city"}
    for kw in names:
        p = await best_poi(kw, "")
        if p:
            return {"location": p["location"], "adcode": p["adcode"], "via": "poi_national"}
    for city in cities:
        for kw in names:
            try:
                g = await amap.geocode(kw, city)
            except AppError:
                g = None
            if g:
                return {"location": g["location"], "adcode": g["adcode"], "via": "geocode"}
    return None


async def run_legacy(llm: LLMClient, amap: AmapClient, retriever: Retriever, case: Dict[str, Any]) -> Dict[str, Any]:
    """返回 {text, days: [{places: [{name, city, geo}]}]}。"""
    prefs = case.get("preferences", [])
    hits = retriever.retrieve(" ".join([case["destination"], *prefs]), 5, rerank="none")["hits"]
    snippets = "\n\n".join(f"【笔记{i + 1}】「{h['title']}」\n{h['text']}" for i, h in enumerate(hits))
    sys = INSTRUCTION
    if snippets:
        sys += ("\n\n【参考素材——真实旅行者的笔记】\n以下是其他旅行者分享的真实体验，请从中提取有价值的信息"
                f"（如具体店名、实用tips、踩坑经验）融入你的行程推荐，但不要照搬原文：\n\n{snippets}")
    sys += (f"\n\n【固定背景】\n出发地：未填写；目的地：{case['destination']}；日期：未填写；"
            f"偏好：{'、'.join(prefs) or '未选择'}。")
    user = (f"请为我规划前往 {case['destination']} 的行程，旅行偏好：{'、'.join(prefs) or '未选择'}。"
            f"{case.get('message', '')}请生成 {case['days']} 天行程。")
    text = await llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                          temperature=0.7, purpose="eval.legacy")
    days = parse_itinerary_text(text)
    for day in days:
        for place in day["places"]:
            place["geo"] = await geocode_place(amap, place["name"], place["city"] or day["city"] or case["destination"])
    return {"text": text, "days": days}
